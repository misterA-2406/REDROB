# ranker/ltr_scorer.py
from __future__ import annotations
import math
import os
import pickle
import logging
from typing import Any, Dict, List, Optional
from ranker.config import (
    LTR_CONFIG, FEATURE_NAMES, CAREER_LEVEL_SCORES,
    PRODUCT_COMPANIES_TIER1, CONSULTING_FIRMS, RECENCY_DECAY_WEIGHTS,
    DEFAULT_LTR_ALPHA
)

logger = logging.getLogger("RedrobRanker.LTRScorer")

def _compute_shipper_ratio(cand: Dict[str, Any]) -> float:
    # Upgrade 3: JD Intent differentiator
    work_history = cand.get("work_history") or []
    text = (str(cand.get("current_title", "")) + " " + " ".join([str(s) for s in cand.get("skills") or []])).lower()
    for job in work_history:
        text += " " + (str(job.get("title", "")) + " " + str(job.get("description", ""))).lower()
        
    shipping_kws = ["shipped", "deployed", "launched", "production", "scale", "latency", "throughput", "real-time", "architecture", "distributed"]
    academic_kws = ["academic", "research", "thesis", "publication", "paper", "fellow", "lab", "university", "professor", "phd"]
    
    ship_count = sum(1 for kw in shipping_kws if kw in text)
    acad_count = sum(1 for kw in academic_kws if kw in text)
    
    if ship_count == 0 and acad_count == 0: 
        return 0.5
    return float(ship_count) / (ship_count + acad_count + 0.001)

def build_feature_vector(cand: Dict[str, Any]) -> List[float]:
    signals = cand.get("signals") or {}
    work_history = cand.get("work_history") or []
    
    skill_score = float(cand.get("skill_score", 0.0))
    semantic_score = float(cand.get("semantic_score", 0.0))
    behavioral_composite = float(cand.get("behavioral_composite", 0.0))
    
    yoe = float(cand.get("total_years_experience", 0.0))
    yoe_norm = 1.0 if 5.0 <= yoe <= 9.0 else (0.6 if 3.0 <= yoe < 5.0 else (0.4 if yoe > 9.0 else 0.1))
    
    exp_fit = float(cand.get("experience_fit", 0.5))
    
    all_text = (str(cand.get("current_title", "")) + " " + " ".join([str(s) for s in cand.get("skills") or []])).lower()
    for job in work_history:
        all_text += " " + (str(job.get("title", "")) + " " + str(job.get("description", ""))).lower()
        
    # Granular counters instead of binary flags (Optimized Warning 4)
    vector_tools = ["faiss", "pinecone", "qdrant", "weaviate", "milvus", "opensearch", "elasticsearch"]
    vec_db_exp = min(1.0, sum(1 for t in vector_tools if t in all_text) / 3.0)
    
    retrieval_terms = ["retrieval", "nearest neighbor", "ann", "knn", "vector similarity", "sparse retrieval"]
    retrieval_exp = min(1.0, sum(1 for t in retrieval_terms if t in all_text) / 3.0)
    
    ranking_terms = ["ranking", "ltr", "learning to rank", "lambdamart", "neural ltr", "recsys", "recommender"]
    ranking_exp = min(1.0, sum(1 for t in ranking_terms if t in all_text) / 3.0)
    
    eval_terms = ["ndcg", "mrr", "map", "eval", "metrics", "a/b testing", "offline eval"]
    eval_exp = min(1.0, sum(1 for t in eval_terms if t in all_text) / 3.0)
    
    # Scale python strength cleanly (Capped at 1.0)
    python_strength = min(1.0, (1.2 if "python" in all_text else 0.0))
    
    llm_fine_tune = min(1.0, sum(1 for t in ["lora", "qlora", "peft", "fine-tuning"] if t in all_text) / 2.0)
    
    company_qual = float(cand.get("company_quality", 0.5))
    github_sc = float(signals.get("github_activity_score", 0)) / 100.0
    endorsement_sc = min(float(signals.get("endorsements_received", 0)) / 50.0, 1.0)
    resp_rate = float(signals.get("recruiter_response_rate", 0.50))
    complete_sc = float(signals.get("profile_completeness_score", 80)) / 100.0
    otw_signal = 1.0 if signals.get("open_to_work_flag") else 0.0
    
    notice_val = signals.get("notice_period_days", 90)
    try:
        n_days = int(notice_val)
        notice_sc = 1.0 if n_days <= 30 else (0.75 if n_days <= 60 else (0.50 if n_days <= 90 else 0.25))
    except (ValueError, TypeError):
        notice_sc = 0.50
        
    career_growth = 0.5
    if len(work_history) >= 2:
        recent_title = str(work_history[0].get("title", "")).lower()
        older_title = str(work_history[-1].get("title", "")).lower()
        recent_lvl = 3
        for k, v in CAREER_LEVEL_SCORES.items():
            if k in recent_title:
                recent_lvl = v
        older_lvl = 3
        for k, v in CAREER_LEVEL_SCORES.items():
            if k in older_title:
                older_lvl = v
        if recent_lvl > older_lvl:
            career_growth = 1.0
        elif recent_lvl < older_lvl:
            career_growth = 0.2
            
    activity_sc = min(float(signals.get("applications_submitted_30d", 0)) / 10.0, 1.0)
    
    email_b = 1.0 if signals.get("verified_email") else 0.0
    phone_b = 1.0 if signals.get("verified_phone") else 0.0
    linkedin_b = 1.0 if signals.get("linkedin_connected") else 0.0
    trust_score = (0.35 * email_b + 0.25 * phone_b + 0.20 * linkedin_b + 0.20 * complete_sc)
    
    honeypot_prob = 1.0 if cand.get("is_honeypot") else 0.0
    
    avail_score = float(cand.get("availability_score", skill_score))
    cred_score = float(cand.get("credibility_score", skill_score))
    reach_score = float(cand.get("reachability_score", skill_score))
    
    feat_vector = [
        skill_score, semantic_score, behavioral_composite, yoe_norm,
        exp_fit, vec_db_exp, retrieval_exp, ranking_exp, eval_exp,
        python_strength, llm_fine_tune, company_qual, github_sc,
        endorsement_sc, resp_rate, complete_sc, otw_signal, notice_sc,
        career_growth, activity_sc, trust_score, honeypot_prob,
        avail_score, cred_score, reach_score, semantic_score,
        _compute_shipper_ratio(cand)  # Index 26: Upgrade 3
    ]
    
    return [max(0.0, min(1.0, float(f))) if idx != 21 else float(f) for idx, f in enumerate(feat_vector)]

class LTRRanker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.backend = "none"
        self._feature_importance: Dict[str, float] = {}

    def load_ranker(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.backend = data["backend"]
                self._feature_importance = data.get("feature_importance", {})
            logger.info(f"Loaded LTR Ranker with backend: {self.backend}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load LTR ranker from {path}: {e}")
            return False

    def save_ranker(self, path: str) -> None:
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "backend": self.backend,
                    "feature_importance": self._feature_importance
                }, f)
            logger.info(f"Saved LTR ranker model to {path}")
        except Exception as e:
            logger.warning(f"Could not save model: {e}")

    def train_ranker(self, candidates: List[Dict[str, Any]], labels: Optional[List[int]] = None, save: bool = True) -> None:
        if len(candidates) < 5:
            self.backend = "fallback"
            return
            
        import numpy as np
        X = np.array([build_feature_vector(c) for c in candidates])
        
        if labels is None:
            y = []
            for c in candidates:
                skill_score = float(c.get("skill_score", 0.0))
                semantic_score = float(c.get("semantic_score", 0.0))
                behavioral_composite = float(c.get("behavioral_composite", 0.0))
                company_qual = float(c.get("company_quality", 0.0))
                
                # Dynamic LTR Formula optimized with Shipper Ratio
                lbl = (0.40 * skill_score + 0.30 * semantic_score +
                       0.20 * behavioral_composite + 0.10 * _compute_shipper_ratio(c)) * 4.0
                if c.get("is_honeypot"):
                    lbl = 0.0
                y.append(lbl)
            y_arr = np.clip(np.round(y), 0, 4).astype(np.int32)
        else:
            y_arr = np.array(labels, dtype=np.int32)

        if self.config.get("use_xgboost", True):
            try:
                import xgboost as xgb
                model = xgb.XGBRanker(
                    objective="rank:ndcg",
                    tree_method="hist",
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.08,
                    min_child_weight=5,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    n_jobs=-1
                )
                qid = np.zeros(len(X), dtype=np.int32)
                model.fit(X, y_arr, qid=qid)
                self.model = model
                self.backend = "xgboost"
                
                importances = model.feature_importances_
                self._feature_importance = {name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)}
                if save:
                    self.save_ranker(self.config.get("model_path", "./ltr_ranker.pkl"))
                return
            except Exception as e:
                logger.warning(f"XGBRanker training failed: {e}. Switching to LightGBM.")

        # Upgrade 4: Pairwise LambdaMART Metric Alignment
        try:
            import lightgbm as lgb
            model = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                ndcg_eval_at=[10, 50],
                n_estimators=200,
                learning_rate=0.05,
                importance_type="gain",
                random_state=42
            )
            group = [len(X)]
            model.fit(X, y_arr, group=group)
            self.model = model
            self.backend = "lightgbm"
            
            importances = model.feature_importances_
            total = sum(importances) if sum(importances) > 0 else 1.0
            self._feature_importance = {name: float(imp / total) for name, imp in zip(FEATURE_NAMES, importances)}
            if save:
                self.save_ranker(self.config.get("model_path", "./ltr_ranker.pkl"))
            return
        except Exception as e:
            logger.warning(f"LGBMRanker training failed: {e}. Switching to Fallback.")
            self.backend = "fallback"

    def predict_rank_score(self, candidates: List[Dict[str, Any]]) -> List[float]:
        if not candidates:
            return []
            
        import numpy as np
        X = np.array([build_feature_vector(c) for c in candidates])
        
        if self.backend in ("xgboost", "lightgbm") and self.model is not None:
            try:
                preds = self.model.predict(X)
                min_p, max_p = preds.min(), preds.max()
                if max_p > min_p:
                    normalized = (preds - min_p) / (max_p - min_p)
                else:
                    normalized = np.zeros_like(preds)
                return normalized.tolist()
            except Exception as e:
                logger.warning(f"Inference failed: {e}. Applying fallback logic.")

        # Standardized 27-dimensional fallback weight mapping (Upgrade 2)
        fallback_weights = [
            0.25, 0.20, 0.15, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02,
            0.05, 0.02, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, -0.20, 0.01, 0.01, 0.01, 0.01, 0.05
        ]
        scores = []
        for vec in X:
            s = sum(f * w for f, w in zip(vec, fallback_weights))
            scores.append(max(0.0, min(1.0, s)))
        return scores
    
    def get_top_signals(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        feats = build_feature_vector(candidate)
        pos = []
        neg = []
        
        if feats[0] > 0.6: pos.append("Excellent structural core skill alignment")
        else: neg.append("Incomplete core skill coverage matches")
        
        if feats[1] > 0.6: pos.append("Strong semantic job profile context")
        if feats[11] > 0.8: pos.append("Outstanding work history tier records")
        if feats[9] > 1.0: pos.append("Demonstrated high proficiency Python history")
        if feats[5] > 0.0: pos.append("Operated vector databases at production scale")
        if feats[10] > 0.0: pos.append("Hands-on LLM tuning exposure (LoRA/PEFT)")
        
        if feats[21] > 0.0:
            neg.append("Flagged timeline profile anomalies")
            
        completeness = feats[15]
        signal_consistency = 1.0 - abs(feats[0] - feats[1])
        hp_risk = feats[21]
        
        confidence = (0.50 * completeness + 0.30 * signal_consistency + 0.20 * (1.0 - hp_risk))
        
        return {
            "positive_signals": pos[:3],
            "negative_signals": neg,
            "confidence_score": float(confidence),
            "feature_completeness": float(completeness)
        }

def apply_ltr_stage(candidates: List[Dict[str, Any]], alpha: float = DEFAULT_LTR_ALPHA, model_path: str = "./ltr_ranker.pkl") -> List[Dict[str, Any]]:
    ranker = LTRRanker(LTR_CONFIG)
    loaded = ranker.load_ranker(model_path)
    
    ltr_scores = ranker.predict_rank_score(candidates)
    
    for cand, l_score in zip(candidates, ltr_scores):
        existing = float(cand.get("final_score", cand.get("composite_score", 0.0)))
        cand["final_score"] = alpha * l_score + (1.0 - alpha) * existing
        cand["ltr_score"] = l_score
        
        signals = ranker.get_top_signals(cand)
        cand["confidence_score"] = signals["confidence_score"]
        cand["feature_importance_summary"] = signals["positive_signals"]
        cand["ltr_backend"] = ranker.backend
        
    candidates.sort(key=lambda x: (-float(x.get("final_score", 0.0)), str(x.get("candidate_id", ""))))
    return candidates