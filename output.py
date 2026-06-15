# ranker/output.py
from __future__ import annotations
import csv
import hashlib
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RedrobRanker.Output")

def apply_rrf_blending(candidates: List[Dict[str, Any]], k: int = 60) -> None:
    ltr_sorted = sorted(candidates, key=lambda x: (-float(x.get("ltr_score", 0.0))), reverse=False)
    sem_sorted = sorted(candidates, key=lambda x: (-float(x.get("semantic_score", 0.0))), reverse=False)
    
    ltr_ranks = {c["candidate_id"]: i + 1 for i, c in enumerate(ltr_sorted)}
    sem_ranks = {c["candidate_id"]: i + 1 for i, c in enumerate(sem_sorted)}
    
    for cand in candidates:
        cid = cand["candidate_id"]
        # RRF formula: 1 / (k + rank_ltr) + 1 / (k + rank_sem)
        score = (1.0 / (k + ltr_ranks.get(cid, 100))) + (1.0 / (k + sem_ranks.get(cid, 100)))
        cand["final_score"] = score

def apply_logistical_discount(cand: Dict[str, Any]) -> float:
    signals = cand.get("signals") or {}
    notice = signals.get("notice_period_days")
    loc_fit = float(cand.get("location_fit_score", 0.5))
    
    if notice is None: 
        return 1.0
    try:
        n = int(notice)
        if n <= 30: 
            return 1.0
        if n <= 60: 
            return 0.98
        if n <= 90: 
            return 0.94
        return 0.88 if loc_fit >= 0.85 else 0.82
    except: 
        return 0.95

def write_submission(candidates: List[Dict[str, Any]], output_path: str, team_id: str = "team") -> None:
    apply_rrf_blending(candidates)
    
    top_100 = candidates[:100]
    top_100.sort(key=lambda x: (-float(x.get("final_score", 0.0)), str(x.get("candidate_id", ""))))
    
    prev_score = 2.0  
    for cand in top_100:
        raw_score = float(cand.get("final_score", 0.0))
        discount = apply_logistical_discount(cand)
        
        cid = cand.get("candidate_id", "")
        cid_hash = int(hashlib.md5(cid.encode("utf-8")).hexdigest(), 16) % 1000
        connections = float((cand.get("signals") or {}).get("connection_count", 0))
        
        tie_breaker = (cid_hash / 1e9) + (min(connections, 500.0) / 1e10)
        adjusted_score = (raw_score * discount) + tie_breaker
        
        s = round(adjusted_score, 7)
        s = min(s, prev_score - 0.0000001)
        
        cand["_out_score"] = max(0.0, s)
        prev_score = s
        
    # FORCE utf-8-sig (UTF-8 with BOM) to guarantee perfect Excel/Windows rendering
    with open(output_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, cand in enumerate(top_100, start=1):
            writer.writerow([
                cand.get("candidate_id", ""),
                rank,
                cand["_out_score"],
                cand.get("reasoning", "")
            ])
            
    logger.info(f"Successfully exported final ranking CSV to {output_path}")

def validate_submission_format(csv_path: str, candidates_path: Optional[str] = None) -> Dict[str, Any]:
    report = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    try:
        # Open with utf-8-sig to automatically handle the BOM wrapper cleanly
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            headers = next(reader, None)
            if headers != ["candidate_id", "rank", "score", "reasoning"]:
                report["valid"] = False
                report["errors"].append("Headers must match exactly: ['candidate_id', 'rank', 'score', 'reasoning']")
                return report
                
            rows = list(reader)
            if len(rows) != 100:
                report["valid"] = False
                report["errors"].append(f"Submission requires exactly 100 entries. Found {len(rows)}")
                
            prev_score = float("inf")
            seen_ids = set()
            for idx, r in enumerate(rows, start=1):
                if len(r) != 4:
                    report["valid"] = False
                    report["errors"].append(f"Row {idx} is malformed. Expected 4 cells, found {len(r)}")
                    continue
                    
                cid, rank_str, score_str, reasoning = r
                
                if cid in seen_ids:
                    report["valid"] = False
                    report["errors"].append(f"Duplicate candidate_id found at Row {idx}: {cid}")
                seen_ids.add(cid)
                
                try:
                    rank_val = int(rank_str)
                    if rank_val != idx:
                        report["valid"] = False
                        report["errors"].append(f"Sequential rank mismatch. Row {idx} has rank {rank_val}")
                except ValueError:
                    report["valid"] = False
                    report["errors"].append(f"Invalid rank value at Row {idx}: {rank_str}")
                    
                try:
                    score_val = float(score_str)
                    if score_val >= prev_score and idx > 1:
                        report["valid"] = False
                        report["errors"].append(f"Score ordering error. Score {score_val} is not strictly less than previous {prev_score}")
                    prev_score = score_val
                except ValueError:
                    report["valid"] = False
                    report["errors"].append(f"Invalid score value at Row {idx}: {score_str}")
                    
                if not reasoning.strip():
                    report["warnings"].append(f"Empty reasoning field at Row {idx}")
                if len(reasoning) > 260:
                    report["warnings"].append(f"Reasoning length ({len(reasoning)} chars) exceeds standard bounds at Row {idx}")
                    
            if candidates_path:
                from ranker.loader import stream_candidates
                source_ids = {c["candidate_id"] for c in stream_candidates(candidates_path)}
                mismatches = seen_ids - source_ids
                if mismatches:
                    report["valid"] = False
                    report["errors"].append(f"Candidate IDs do not exist in the source dataset: {mismatches}")
                    
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"Validation routine aborted unexpectedly: {e}")
        
    return report