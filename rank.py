#!/usr/bin/env python3
# rank.py
"""
Redrob Hackathon v4 — CLI Entry Point
Usage: python rank.py --candidates ./candidates.jsonl.gz --out ./team_id.csv
"""
import os
import sys
import time
import argparse
import logging

# Enforce strict offline execution models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optimizations for CPU-only execution under shared container quotas
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Safe PyTorch import wrapper (Optimized Issue 3 & 6)
try:
    import torch
    torch.set_num_threads(1)  # Prevent CPU thread thrashing
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RedrobRanker")

def run_pipeline(
    candidates_path: str,
    output_path: str,
    top_k: int = 100,
    semantic_weight: float = 0.45,
    behavioral_strength: float = 0.40,
    ltr_alpha: float = 0.70,
    ltr_model_path: str = "./ltr_ranker.pkl",
    team_id: str = "team",
) -> None:
    start_time = time.time()
    logger.info("Initializing 7-stage candidate parsing sequence.")
    
    from ranker import (
        stream_candidates, prefilter, score_candidate, score_behavioral,
        SemanticRanker, prune_honeypots, detect_duplicates, apply_ltr_stage,
        generate_reasoning, deduplicate_reasonings, write_submission,
        validate_submission_format
    )
    from ranker.config import LTR_CONFIG, JD_TEXT
    from ranker.ltr_scorer import LTRRanker
    
    passed_pool = []
    honeypot_pool = []
    rejected_pool = []  # Preserved for emergency backfill
    
    timeout_limit = start_time + 240.0 # 4-minute maximum hard stop threshold
    
    for cand in stream_candidates(candidates_path):
        if time.time() > timeout_limit:
            logger.warning("Hard time-limit threshold reached during streaming phase.")
            break
            
        is_clean, weight = prefilter(cand)
        if not is_clean:
            cand["is_honeypot"] = False
            cand["is_rejected"] = True
            rejected_pool.append(cand)
            continue
            
        if cand.get("is_honeypot"):
            honeypot_pool.append(cand)
        else:
            passed_pool.append(cand)
            
    logger.info(f"Filtered database. Clean: {len(passed_pool)}, Honeypots: {len(honeypot_pool)}, Rejected: {len(rejected_pool)}")
    
    # Combined check to ensure we have enough profiles to fulfill the 100-row requirement
    total_available = len(passed_pool) + len(honeypot_pool) + len(rejected_pool)
    if total_available == 0:
        logger.error("No candidate profiles found in the source file.")
        sys.exit(1)
        
    passed_pool = detect_duplicates(passed_pool)
    
    scored_candidates = []
    for cand in passed_pool:
        s_report = score_candidate(cand)
        cand.update(s_report)
        
        b_report = score_behavioral(cand.get("signals") or {}, cand)
        cand.update(b_report)
        
        skills_score = float(s_report["skill_score"])
        behavior_mult = float(b_report["behavioral_multiplier"])
        
        # Safe Multiplicative Composite Formula (Optimized Warning 8)
        cand["composite_score"] = skills_score * behavior_mult
        scored_candidates.append(cand)
        
    logger.info("Stage 4: Executing dense-transformer semantic rerank checks...")
    scored_candidates.sort(key=lambda x: (-float(x.get("composite_score", 0.0)), str(x.get("candidate_id", ""))))
    rerank_pool = scored_candidates[:300]
    remainder_pool = scored_candidates[300:]
    
    ranker = SemanticRanker()
    reranked_subset = ranker.rerank(rerank_pool, JD_TEXT, batch_size=32, semantic_weight=semantic_weight)
    
    if remainder_pool:
        ranker._tfidf_fallback(remainder_pool, JD_TEXT, semantic_weight=semantic_weight)
        
    all_scored_candidates = reranked_subset + remainder_pool
    
    import gc
    del ranker
    gc.collect()
    
    # --- DYNAMIC OFFLINE LTR TRAINING ---
    # Train the LTR model on a representative sample of candidates to ensure high feature variance
    train_pool = all_scored_candidates[:2000]
    ltr_ranker = LTRRanker(LTR_CONFIG)
    if not ltr_ranker.load_ranker(ltr_model_path) and len(train_pool) >= 10:
        logger.info(f"Training LTR pairwise model on {len(train_pool)} candidates...")
        ltr_ranker.train_ranker(train_pool)
    
    logger.info("Stage 5: Vetting honeypots and assembling top-100 targets...")
    all_scored_candidates.sort(key=lambda x: (-float(x.get("final_score", 0.0)), str(x.get("candidate_id", ""))))
    top_100_clean = prune_honeypots(all_scored_candidates, all_scored_candidates)
    
    # --- EMERGENCY BACKFILL SYSTEM ---
    # If our clean list has fewer than 100 entries, fill the remaining slots to satisfy format rules
    if len(top_100_clean) < 100 and total_available >= 100:
        logger.warning(f"Clean pool contains only {len(top_100_clean)} profiles. Backfilling to meet the 100-row requirement.")
        seen_ids = {c["candidate_id"] for c in top_100_clean}
        
        # 1. Backfill from honeypots
        for cand in honeypot_pool:
            if cand["candidate_id"] not in seen_ids:
                cand["final_score"] = 0.01
                cand["composite_score"] = 0.01
                cand["skill_score"] = 0.01
                cand["behavioral_composite"] = 0.01
                cand["semantic_score"] = 0.01
                cand["reasoning"] = "Dismantled anomaly profile; included as bottom-tier fallback."
                top_100_clean.append(cand)
                seen_ids.add(cand["candidate_id"])
                if len(top_100_clean) >= 100:
                    break
                    
        # 2. Backfill from rejected pool
        if len(top_100_clean) < 100:
            for cand in rejected_pool:
                if cand["candidate_id"] not in seen_ids:
                    cand["final_score"] = 0.0001
                    cand["composite_score"] = 0.0001
                    cand["skill_score"] = 0.0001
                    cand["behavioral_composite"] = 0.0001
                    cand["semantic_score"] = 0.0001
                    cand["reasoning"] = "Incomplete technical alignment; included for formatting compliance."
                    top_100_clean.append(cand)
                    seen_ids.add(cand["candidate_id"])
                    if len(top_100_clean) >= 100:
                        break

    logger.info("Stage 5.5: Adjusting profiles with XGBoost LTR rankers...")
    top_100_clean = apply_ltr_stage(top_100_clean, alpha=ltr_alpha, model_path=ltr_model_path)
    
    logger.info("Stage 6: Synthesizing explanations...")
    for idx, cand in enumerate(top_100_clean, start=1):
        if "reasoning" not in cand or cand["reasoning"] == "":
            cand["reasoning"] = generate_reasoning(cand, idx)
        elif cand.get("is_rejected") or cand.get("is_honeypot"):
            from ranker.reasoner import _clean_truncate
            cand["reasoning"] = _clean_truncate(cand["reasoning"], 235)
        
    top_100_clean = deduplicate_reasonings(top_100_clean)
    
    logger.info("Stage 7: Writing high-level submission...")
    write_submission(top_100_clean, output_path, team_id=team_id)
    
    report = validate_submission_format(output_path, candidates_path)
    if report["valid"]:
        logger.info(f"Pipeline executed in {time.time() - start_time:.2f}s. Format validation passed.")
    else:
        logger.error(f"Format validation failed: {report['errors']}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Hackathon v4 Candidate Discoverer")
    parser.add_argument("--candidates", required=True, help="Input GZ/JSONL database path")
    parser.add_argument("--out", default="./submission.csv", help="Output path target destination")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--semantic-weight", type=float, default=0.45)
    parser.add_argument("--behavioral-weight", type=float, default=0.40)
    parser.add_argument("--ltr-alpha", type=float, default=0.70)
    parser.add_argument("--ltr-model-path", default="./ltr_ranker.pkl")
    parser.add_argument("--team-id", default="antigravity_v4")
    
    args = parser.parse_args()
    
    run_pipeline(
        candidates_path=args.candidates,
        output_path=args.out,
        top_k=args.top_k,
        semantic_weight=args.semantic_weight,
        behavioral_strength=args.behavioral_weight,
        ltr_alpha=args.ltr_alpha,
        ltr_model_path=args.ltr_model_path,
        team_id=args.team_id,
    )