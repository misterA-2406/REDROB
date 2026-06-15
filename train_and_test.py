#!/usr/bin/env python3
# train_and_test.py
"""
Redrob Hackathon v4 — Production-Grade Offline LTR Training Script
Corrected for feature evaluation, schema alignment, and safe query grouping.
Usage: python train_and_test.py
"""
import os
import sys
import json
import gzip
import argparse
import logging
import numpy as np
import pickle
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LTRTrainer")

try:
    from ranker import normalize_candidate_schema, prefilter, score_candidate, score_behavioral
    from ranker.config import JD_TEXT, FEATURE_NAMES
    from ranker.ltr_scorer import build_feature_vector, _compute_shipper_ratio
    from ranker.semantic import SemanticRanker
except ImportError as e:
    logger.error(f"Import failed: {e}")
    logger.error("Make sure you run this script from your project root directory.")
    sys.exit(1)

def create_groups(total_count: int, max_group_size: int = 1000) -> List[int]:
    """Splits candidates into queries of safe maximum size to avoid LightGBM limits."""
    groups = []
    remaining = total_count
    while remaining > 0:
        current_size = min(remaining, max_group_size)
        groups.append(current_size)
        remaining -= current_size
    return groups

def load_candidates(file_path: str, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load candidates from JSONL or gzipped JSONL."""
    candidates = []
    open_func = gzip.open if file_path.endswith('.gz') else open
    try:
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    cand = json.loads(line)
                    candidates.append(normalize_candidate_schema(cand))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line {i+1}")
        logger.info(f"Loaded {len(candidates)} candidates from {file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    return candidates

def main():
    parser = argparse.ArgumentParser(description="Offline LTR training with robust label mapping")
    parser.add_argument(
        "--candidates", 
        default=r"C:\Users\Alence\Downloads\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz (full dataset)"
    )
    parser.add_argument("--max-candidates", type=int, default=50000,
                        help="Maximum number of candidates to use. Default 50000")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Fraction of data for validation")
    parser.add_argument("--out-model", default="./ltr_ranker.pkl",
                        help="Output path for trained model")
    parser.add_argument("--n-estimators", type=int, default=150,
                        help="Number of boosting rounds")
    args = parser.parse_args()

    # 1. Load candidates
    logger.info("Loading candidate dataset...")
    all_candidates = load_candidates(args.candidates, max_rows=args.max_candidates)
    if len(all_candidates) < 100:
        logger.error("Too few candidates loaded. Check file path.")
        sys.exit(1)

    # 2. Run prefilter
    logger.info("Running prefilter to keep only plausible candidates...")
    clean_candidates = []
    for c in all_candidates:
        is_clean, _ = prefilter(c)
        if is_clean and not c.get("is_honeypot"):
            clean_candidates.append(c)
    logger.info(f"Kept {len(clean_candidates)} candidates after prefilter.")
    if len(clean_candidates) < 50:
        logger.error("Too few candidates after prefilter. Adjust thresholds.")
        sys.exit(1)

    # 3. Pre-evaluate heuristics to populate base scores
    logger.info("Running baseline scorers to populate feature dictionaries...")
    for c in clean_candidates:
        c.update(score_candidate(c))
        c.update(score_behavioral(c.get("signals") or {}, c))

    # 4. Extract 27-dimensional feature vectors and multi-criterion target labels
    logger.info("Extracting candidate feature arrays...")
    X_list = []
    y_list = []
    for c in clean_candidates:
        try:
            X_list.append(build_feature_vector(c))
            
            # Multi-criterion pseudo-labeling [0 - 4]
            lbl = (0.40 * float(c.get("skill_score", 0)) + 
                   0.30 * float(c.get("semantic_score", 0)) + 
                   0.20 * float(c.get("behavioral_composite", 0)) + 
                   0.10 * _compute_shipper_ratio(c)) * 4.0
            y_list.append(lbl)
        except Exception as e:
            logger.warning(f"Skipping candidate due to feature error: {e}")
            
    X = np.array(X_list, dtype=np.float32)
    y = np.clip(np.round(y_list), 0, 4).astype(np.int32)

    valid = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    X = X[valid]
    y = y[valid]
    logger.info(f"Final feature matrix shape: {X.shape}")

    # 5. Train/validation split
    np.random.seed(42)
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - args.test_split))
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    logger.info(f"Training set: {len(X_train)} candidates")
    logger.info(f"Validation set: {len(X_val)} candidates")

    # 6. Train LightGBM ranker with safe query grouping
    import lightgbm as lgb

    train_groups = create_groups(len(X_train), max_group_size=1000)
    val_groups = create_groups(len(X_val), max_group_size=1000)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10, 50],
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        random_state=42,
        verbose=-1
    )

    logger.info("Starting LightGBM LambdaMART training...")
    model.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        eval_metric=['ndcg'],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # 7. Evaluate on validation set
    y_pred = model.predict(X_val)
    sorted_idx = np.argsort(y_pred)[::-1]
    y_val_sorted = y_val[sorted_idx]

    def dcg(scores, k):
        scores = scores[:k]
        if len(scores) == 0:
            return 0.0
        gains = 2**scores - 1
        discounts = np.log2(np.arange(2, len(scores)+2))
        return np.sum(gains / discounts)

    def ndcg(y_true, k):
        ideal = np.sort(y_true)[::-1]
        dcg_val = dcg(y_true, k)
        idcg_val = dcg(ideal, k)
        return dcg_val / idcg_val if idcg_val > 0 else 0.0

    ndcg10 = ndcg(y_val_sorted, 10)
    ndcg50 = ndcg(y_val_sorted, 50)

    # MAP with relevance threshold (e.g., score >= 2.0)
    threshold = 2.0
    rel = (y_val_sorted >= threshold).astype(int)
    precisions = []
    rel_count = 0
    for i, r in enumerate(rel):
        if r == 1:
            rel_count += 1
            precisions.append(rel_count / (i+1))
    map_score = np.mean(precisions) if precisions else 0.0

    composite = 0.5 * ndcg10 + 0.3 * ndcg50 + 0.15 * map_score

    print("\n" + "="*60)
    print("            LTR MODEL VALIDATION (MULTI-CRITERION LABELS)")
    print("="*60)
    print(f"Validation set size    : {len(y_val)}")
    print(f"NDCG@10                : {ndcg10:.6f}")
    print(f"NDCG@50                : {ndcg50:.6f}")
    print(f"MAP (threshold {threshold}) : {map_score:.6f}")
    print(f"Composite score        : {composite:.6f}")
    print("="*60 + "\n")

    # 8. Save model safely 
    complete_feature_names = FEATURE_NAMES + ["shipper_versus_researcher_ratio"]
    output_data = {
        "model": model,
        "backend": "lightgbm",
        "feature_importance": dict(zip(complete_feature_names, model.feature_importances_))
    }
    with open(args.out_model, "wb") as f:
        pickle.dump(output_data, f)
    logger.info(f"Model saved to {args.out_model}")
    logger.info("Offline training complete. Replace your existing ltr_ranker.pkl with this file.")

if __name__ == "__main__":
    main()