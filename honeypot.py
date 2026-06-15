# ranker/honeypot.py
from __future__ import annotations
import logging
from typing import Any, Dict, List

logger = logging.getLogger("RedrobRanker.Honeypot")

def prune_honeypots(top_ranked: List[Dict[str, Any]], clean_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean_top_100 = []
    demoted = []
    
    for cand in top_ranked:
        if len(clean_top_100) >= 100:
            break
        if cand.get("is_honeypot"):
            demoted.append(cand)
        else:
            clean_top_100.append(cand)
            
    if len(clean_top_100) < 100:
        seen_ids = {c["candidate_id"] for c in clean_top_100}
        for cand in clean_pool:
            if cand.get("candidate_id") not in seen_ids:
                clean_top_100.append(cand)
                if len(clean_top_100) >= 100:
                    break
                    
    final = clean_top_100[:100]
    
    # Fill remaining spots with up to 4 demoted honeypots as a fallback
    if len(final) < 100 and demoted:
        n = min(100 - len(final), 4)
        final.extend(demoted[:n])
        
    return final

def detect_duplicates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    skill_fingerprints: Dict[frozenset, str] = {}
    history_fingerprints: Dict[frozenset, str] = {}
    
    for cand in candidates:
        cid = cand.get("candidate_id", "")
        
        skills = frozenset(
            (s.get("name", "") if isinstance(s, dict) else str(s)).strip().lower()
            for s in cand.get("skills") or []
        )
        
        history = frozenset(
            (
                str(j.get("company", "")).strip().lower(),
                str(j.get("title", "")).strip().lower(),
                int(j.get("duration_months") or 0)
            )
            for j in cand.get("work_history") or []
            if isinstance(j, dict)
        )
        
        is_dup = False
        if skills and skills in skill_fingerprints:
            is_dup = True
        if history and history in history_fingerprints:
            is_dup = True
            
        if is_dup:
            for key in ("composite_score", "final_score", "skill_score"):
                if cand.get(key) is not None:
                    cand[key] = float(cand[key]) * 0.95
            cand["is_duplicate"] = True
            logger.info(f"Duplicate signature found for candidate {cid}. Applying score penalty.")
            
        skill_fingerprints[skills] = cid
        history_fingerprints[history] = cid
        
    return candidates