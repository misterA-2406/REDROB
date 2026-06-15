# ranker/prefilter.py
from __future__ import annotations
import re
import datetime
import logging
from typing import Any, Dict, List, Tuple
from ranker.config import (
    DISQUALIFYING_TITLES, CONSULTING_FIRMS, COMPANY_FOUNDING_YEARS, CURRENT_DATE
)
from ranker.loader import _parse_date_safe

logger = logging.getLogger("RedrobRanker.Prefilter")

_AI_KEYWORDS_RE = re.compile(
    r"\b(ml|ai|machine.learning|deep.learning|nlp|embeddings?|retrieval|"
    r"vector|ranking|recommendation|pytorch|tensorflow|sklearn|scikit|"
    r"xgboost|lightgbm|transformers?|llm|bert|gpt|rag|faiss|"
    r"qdrant|pinecone|weaviate|elasticsearch)\b",
    re.IGNORECASE,
)

def is_consulting_only(work_history: List[Dict[str, Any]]) -> bool:
    if not work_history:
        return False
    for job in work_history:
        company = str(job.get("company", "")).strip().lower()
        if not company:
            continue
        if not any(cf in company for cf in CONSULTING_FIRMS):
            return False
    return True

def _detect_honeypot(cand: Dict[str, Any], work_history: List[Dict[str, Any]], signals: Dict[str, Any]) -> List[str]:
    reasons = []
    
    # 1. Timeline Impossibility
    stated_yoe = float(cand.get("total_years_experience", 0.0))
    sum_duration_months = 0.0
    for job in work_history:
        dur = job.get("duration_months")
        if dur is not None:
            sum_duration_months += float(dur)
        else:
            s_dt = _parse_date_safe(job.get("start_date"))
            e_dt = _parse_date_safe(job.get("end_date")) or CURRENT_DATE
            if s_dt and e_dt:
                sum_duration_months += max(0.0, (e_dt - s_dt).days / 30.4)
                
    if stated_yoe > (sum_duration_months / 12.0 + 5.0) and stated_yoe > 2.0:
        reasons.append(f"Timeline mismatch: Stated YoE ({stated_yoe}) exceeds calculated duration by over 5 years.")

    # 2. Expert Skill + Zero Duration
    expert_zero_count = 0
    for skill in cand.get("skills") or []:
        if isinstance(skill, dict):
            prof = str(skill.get("proficiency", "")).strip().lower()
            dur_months = skill.get("duration_months") or (skill.get("years_used", 0) * 12)
            if prof in ("expert", "advanced") and (dur_months is None or dur_months == 0):
                expert_zero_count += 1
    if expert_zero_count >= 3:
        reasons.append(f"Expert skill mismatch: {expert_zero_count} advanced skills declared with zero duration.")

    # 3. Assessment Contradiction
    assessments = signals.get("skill_assessment_scores") or {}
    for skill, score in assessments.items():
        try:
            numeric_score = float(score)
        except (ValueError, TypeError):
            continue
        
        match_dur = 0.0
        for s in cand.get("skills") or []:
            if isinstance(s, dict):
                s_name = str(s.get("name", "")).strip().lower()
                if s_name == skill.lower():
                    match_dur = s.get("duration_months") or (s.get("years_used", 0) * 12) or 0.0
                    break
        if numeric_score > 80.0 and match_dur == 0:
            reasons.append(f"Assessment discrepancy: Top scoring assessment {skill} (>80%) has zero duration.")

    # 4. Company Founding Anomaly
    for job in work_history:
        comp_clean = str(job.get("company", "")).strip().lower()
        start_date = _parse_date_safe(job.get("start_date"))
        if start_date:
            for parent_comp, yr in COMPANY_FOUNDING_YEARS.items():
                if parent_comp in comp_clean and start_date.year < yr:
                    reasons.append(f"Timeline anomaly: Start date {start_date.year} for {comp_clean} predates founding ({yr}).")

    # 5. Over-Optimized Behavioral Signals
    max_fields = {
        "recruiter_response_rate": 1.0,
        "profile_completeness_score": 100.0,
        "interview_completion_rate": 1.0,
        "open_to_work_flag": True,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
        "willing_to_relocate": True,
        "offer_acceptance_rate": 1.0,
        "github_activity_score": 100.0,
    }
    present_count = 0
    max_count = 0
    for field, max_val in max_fields.items():
        if field in signals:
            present_count += 1
            if signals[field] == max_val:
                max_count += 1
    if present_count >= 8 and max_count >= (present_count - 1):
        reasons.append("Behavioral metric anomaly: Unrealistic 100% metrics across multiple candidate signals.")

    # 6. Impossible Full-Time Overlaps
    ft_roles = []
    for job in work_history:
        desc = str(job.get("description", "")).lower()
        title = str(job.get("title", "")).lower()
        company = str(job.get("company", "")).lower()
        
        indicators = (
            "freelance", "contractor", "consultant", "adviser", "advisor", 
            "part-time", "founder", "board member", "co-founder", "intern"
        )
        if any(ind in desc or ind in title or ind in company for ind in indicators):
            continue
            
        s_dt = _parse_date_safe(job.get("start_date"))
        e_dt = _parse_date_safe(job.get("end_date")) or CURRENT_DATE
        if s_dt and e_dt:
            ft_roles.append((s_dt, e_dt, title))
            
    overlapping_roles = False
    for i in range(len(ft_roles)):
        for j in range(i + 1, len(ft_roles)):
            s1, e1, t1 = ft_roles[i]
            s2, e2, t2 = ft_roles[j]
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            if overlap_start < overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                if overlap_days > 180:
                    overlapping_roles = True
                    break
    if overlapping_roles:
        reasons.append("Improbable overlap: Overlapping full-time roles discovered for a period >180 days.")

    return reasons

def prefilter(cand: Dict[str, Any]) -> Tuple[bool, float]:
    title = str(cand.get("current_title", "")).strip().lower()
    work_history = cand.get("work_history") or []
    signals = cand.get("signals") or {}
    
    if any(dt in title for dt in DISQUALIFYING_TITLES):
        return False, 0.0
        
    if is_consulting_only(work_history):
        return False, 0.0

    all_corpus = title + " " + " ".join([str(s) for s in cand.get("skills") or []])
    for job in work_history:
        all_corpus += " " + str(job.get("title", "")) + " " + str(job.get("description", ""))
    if not _AI_KEYWORDS_RE.search(all_corpus):
        return False, 0.0

    response_rate = signals.get("recruiter_response_rate")
    if response_rate is not None:
        try:
            if float(response_rate) < 0.05:
                return False, 0.0
        except (ValueError, TypeError):
            pass

    honeypot_reasons = _detect_honeypot(cand, work_history, signals)
    if honeypot_reasons:
        cand["is_honeypot"] = True
        cand["honeypot_reasons"] = honeypot_reasons
        return True, 0.1
        
    cand["is_honeypot"] = False
    return True, 1.0