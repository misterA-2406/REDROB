# ranker/loader.py
from __future__ import annotations
import gzip
import json
import logging
from typing import Any, Dict, Iterator, Optional
import datetime
import re
from ranker.config import CURRENT_DATE

logger = logging.getLogger("RedrobRanker.Loader")

def _parse_date_safe(date_str: Any) -> Optional[datetime.datetime]:
    if not date_str:
        return None
    val = str(date_str).strip().lower()
    if val in ("present", "current", "none", "null", ""):
        return CURRENT_DATE
    
    # Fast path ISO standard (YYYY-MM-DD)
    if len(val) >= 10 and val[4] == '-' and val[7] == '-':
        try:
            return datetime.datetime.strptime(val[:10], "%Y-%m-%d")
        except ValueError:
            pass
            
    # Try parsing via dateutil library
    try:
        from dateutil import parser
        return parser.parse(val)
    except Exception:
        pass

    # Extract year fallback
    match = re.search(r"\b(19\d{2}|20\d{2})\b", val)
    if match:
        try:
            return datetime.datetime(int(match.group(1)), 1, 1)
        except ValueError:
            pass
            
    return None

def normalize_candidate_schema(cand: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict(cand)
    profile = flat.get("profile") or {}
    signals = flat.get("redrob_signals", flat.get("signals")) or {}
    
    flat["current_title"] = str(profile.get("current_title", flat.get("current_title", ""))).strip()
    flat["current_location"] = str(profile.get("location", flat.get("current_location", ""))).strip()
    
    if "total_years_experience" not in flat or flat["total_years_experience"] is None:
        val = profile.get("years_of_experience", profile.get("total_years_experience", 0.0))
        try:
            flat["total_years_experience"] = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            flat["total_years_experience"] = 0.0
            
    if "career_history" in flat and flat["career_history"]:
        flat["work_history"] = flat["career_history"]
    else:
        flat["work_history"] = flat.get("work_history") or []
        
    if "education" not in flat or flat["education"] is None:
        flat["education"] = []
        
    if "skills" not in flat or flat["skills"] is None:
        flat["skills"] = []
        
    flat["signals"] = signals if isinstance(signals, dict) else {}
    return flat

def stream_candidates(file_path: str) -> Iterator[Dict[str, Any]]:
    is_gz = file_path.endswith(".gz")
    open_func = gzip.open if is_gz else open
    
    err_count = 0
    total = 0
    with open_func(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                cand = json.loads(line_str)
                total += 1
                yield normalize_candidate_schema(cand)
            except json.JSONDecodeError:
                err_count += 1
                if err_count <= 5:
                    logger.warning(f"Malformed JSON line omitted: {line_str[:100]}")
                    
    logger.info(f"Streamed {total:,} lines ({err_count} parse errors skipped).")