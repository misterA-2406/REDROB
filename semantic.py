# ranker/semantic.py
from __future__ import annotations
import os
import re
import math
import logging
from collections import Counter
from typing import Any, Dict, List, Optional
from ranker.config import DEFAULT_SEMANTIC_WEIGHT
from ranker.skill_scorer import _recency_weight

logger = logging.getLogger("RedrobRanker.Semantic")

_SEMANTIC_MODEL_SINGLETON = None
_SEMANTIC_MODEL_NAME_LOADED: str = ""

def _get_semantic_model(preferred: str = "BAAI/bge-small-en-v1.5"):
    global _SEMANTIC_MODEL_SINGLETON, _SEMANTIC_MODEL_NAME_LOADED
    if _SEMANTIC_MODEL_SINGLETON is not None:
        return _SEMANTIC_MODEL_SINGLETON
        
    for model_name in [preferred, "all-MiniLM-L6-v2", "./local_model_weights"]:
        try:
            from sentence_transformers import SentenceTransformer
            if model_name == "./local_model_weights" and not os.path.exists(model_name):
                continue
            model = SentenceTransformer(model_name)
            _SEMANTIC_MODEL_SINGLETON = model
            _SEMANTIC_MODEL_NAME_LOADED = model_name
            logger.info(f"Loaded Transformer backend Model: {model_name}")
            return model
        except Exception as e:
            logger.warning(f"Unable to load transformer model '{model_name}': {e}")
            continue
    return None

class SemanticRanker:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = _get_semantic_model(model_name)
        self._jd_embedding = None

    def _build_profile_text(self, cand: Dict[str, Any]) -> str:
        title = cand.get("current_title", "")
        yoe = cand.get("total_years_experience", 0.0)
        loc = cand.get("current_location", "")
        
        skills_raw = cand.get("skills") or []
        skills_str = ", ".join([s.get("name", "") if isinstance(s, dict) else str(s) for s in skills_raw])
        
        hist_parts = []
        work_history = cand.get("work_history") or []
        for idx, job in enumerate(work_history):
            weight = _recency_weight(idx)
            j_title = job.get("title", "")
            j_comp = job.get("company", "")
            j_desc = job.get("description", "")
            if weight >= 0.8:
                hist_parts.append(f"{j_title} at {j_comp}: {j_desc}")
            elif weight >= 0.6:
                hist_parts.append(f"{j_title} at {j_comp}")
            else:
                hist_parts.append(j_comp)
                
        history_text = " | ".join(hist_parts)
        
        inferred = []
        for job in work_history:
            jc = str(job.get("company", "")).lower()
            jt = str(job.get("title", "")).lower()
            if any(c in jc for c in ("flipkart", "amazon", "myntra", "meesho", "swiggy", "zomato")):
                inferred += ["large-scale recommendation", "embeddings", "a/b testing"]
            if "search" in jt:
                inferred += ["information retrieval", "bm25", "elasticsearch", "ranking metrics"]
            if "recommendation" in jt or "recsys" in jt:
                inferred += ["collaborative filtering", "retrieval", "embedding", "ranking"]
            if "nlp" in jt:
                inferred += ["transformer models", "embeddings", "nlp"]
            if "platform" in jt and any(a in jc for a in ("ai", "ml", "data")):
                inferred += ["ml infrastructure", "model serving", "feature stores"]
        
        inferred_str = ", ".join(list(set(inferred)))
        
        return (
            f"{title} {yoe}yr {loc} Skills: {skills_str} "
            f"History: {history_text} Inferred: {inferred_str}"
        )

    def _tfidf_fallback(self, candidates: List[Dict[str, Any]], jd_text: str, semantic_weight: float) -> None:
        logger.warning("Triggered fallback TF-IDF alignment algorithm.")
        def tokenize(text: str) -> List[str]:
            return re.findall(r"[a-z0-9_]+", text.lower())

        all_texts = [self._build_profile_text(c) for c in candidates]
        doc_freq = Counter()
        for t in all_texts:
            doc_freq.update(set(tokenize(t)))
            
        N = len(all_texts) if all_texts else 1
        idf = {w: math.log((N + 1) / (df + 1)) for w, df in doc_freq.items()}
        
        jd_tokens = tokenize(jd_text)
        jd_vec = Counter(jd_tokens)
        jd_weighted = {t: cnt * idf.get(t, 1.0) for t, cnt in jd_vec.items()}
        jd_len = math.sqrt(sum(v*v for v in jd_weighted.values())) or 1.0
        
        for c, text in zip(candidates, all_texts):
            c_tokens = tokenize(text)
            c_vec = Counter(c_tokens)
            c_weighted = {t: cnt * idf.get(t, 1.0) for t, cnt in c_vec.items()}
            c_len = math.sqrt(sum(v*v for v in c_weighted.values())) or 1.0
            
            dot = sum(jd_weighted.get(t, 0.0) * v for t, v in c_weighted.items())
            sim = dot / (jd_len * c_len) if (jd_len * c_len) > 0 else 0.0
            
            c["semantic_score"] = float(sim)
            c["final_score"] = (
                float(c.get("composite_score", 0.0)) * (1.0 - semantic_weight)
                + float(sim) * semantic_weight
            )

    def rerank(self, candidates: List[Dict[str, Any]], jd_text: str, batch_size: int = 32, semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT) -> List[Dict[str, Any]]:
        if not candidates:
            return []
            
        if self.model is None:
            self._tfidf_fallback(candidates, jd_text, semantic_weight)
            return candidates
            
        if self._jd_embedding is None:
            self._jd_embedding = self.model.encode(
                jd_text, normalize_embeddings=True, show_progress_bar=False
            )
            
        texts = [self._build_profile_text(c) for c in candidates]
        cand_embs = self.model.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        scores = cand_embs @ self._jd_embedding
        
        for c, s in zip(candidates, scores.tolist()):
            c["semantic_score"] = float(s)
            c["final_score"] = (
                float(c.get("composite_score", 0.0)) * (1.0 - semantic_weight)
                + float(s) * semantic_weight
            )
            
        candidates.sort(key=lambda x: (-float(x.get("final_score", 0.0)), str(x.get("candidate_id", ""))))
        return candidates