# ranker/behavioral.py
from __future__ import annotations
import logging
from typing import Any, Dict
from ranker.config import (
    BEHAVIORAL_WEIGHTS, LOCATION_TIER1, JD_CORE_KEYWORDS, CURRENT_DATE
)
from ranker.loader import _parse_date_safe

logger = logging.getLogger("RedrobRanker.Behavioral")

def _compute_availability(sig: Dict[str, Any]) -> float:
    active_date = _parse_date_safe(sig.get("last_active_date"))
    if active_date:
        days = (CURRENT_DATE - active_date).days
        rec_s = 1.0 if days <= 30 else (0.7 if days <= 90 else (0.4 if days <= 180 else 0.1))
    else:
        rec_s = 0.3
        
    otw_s = 1.0 if sig.get("open_to_work_flag") else 0.2
    
    app_count = sig.get("applications_submitted_30d", 0)
    app_s = min(1.0, app_count / 10.0)
    
    resp_rate = float(sig.get("recruiter_response_rate", 0.50))
    
    notice = sig.get("notice_period_days", 90)
    if notice is not None:
        try:
            n_val = int(notice)
            notice_s = 1.0 if n_val <= 30 else (0.75 if n_val <= 60 else (0.50 if n_val <= 90 else 0.25))
        except (ValueError, TypeError):
            notice_s = 0.50
    else:
        notice_s = 0.50
        
    return (0.30 * rec_s + 0.25 * otw_s + 0.20 * app_s + 0.15 * resp_rate + 0.10 * notice_s)

def _compute_credibility(sig: Dict[str, Any]) -> float:
    assessments = sig.get("skill_assessment_scores") or {}
    matching_scores = []
    for k, v in assessments.items():
        if any(kw in k.lower() for kw in JD_CORE_KEYWORDS):
            try:
                matching_scores.append(float(v))
            except (ValueError, TypeError):
                pass
    assess_s = (sum(matching_scores) / len(matching_scores) / 100.0) if matching_scores else 0.45
    
    git_act = sig.get("github_activity_score", -1)
    git_s = 0.40 if git_act == -1 else (float(git_act) / 100.0)
    
    endorsements = float(sig.get("endorsements_received", 0))
    endorse_s = min(1.0, endorsements / 50.0)
    
    saved = float(sig.get("saved_by_recruiters_30d", 0))
    saved_s = min(1.0, saved / 20.0)
    
    completeness = float(sig.get("profile_completeness_score", 80)) / 100.0
    int_rate = float(sig.get("interview_completion_rate", 1.0))
    
    acc_rate = sig.get("offer_acceptance_rate", 0.85)
    if acc_rate == -1: 
        acc_rate = 0.85
    acc_rate = float(acc_rate)
    
    return (
        0.30 * assess_s +
        0.20 * git_s +
        0.15 * endorse_s +
        0.15 * saved_s +
        0.10 * completeness +
        0.05 * int_rate +
        0.05 * acc_rate
    )

def _compute_reachability(sig: Dict[str, Any]) -> float:
    email_s = 1.0 if sig.get("verified_email") else 0.0
    phone_s = 1.0 if sig.get("verified_phone") else 0.0
    linkedin_s = 1.0 if sig.get("linkedin_connected") else 0.0
    
    resp_time = sig.get("avg_response_time_hours", 48.0)
    try:
        val = float(resp_time)
        resp_s = 1.0 if val <= 4.0 else (0.8 if val <= 24.0 else (0.5 if val <= 72.0 else 0.2))
    except (ValueError, TypeError):
        resp_s = 0.5
        
    return (0.35 * email_s + 0.30 * phone_s + 0.20 * linkedin_s + 0.15 * resp_s)

def _compute_location_fit(sig: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    loc = str(candidate.get("current_location", "")).lower().strip()
    mode = str(sig.get("preferred_work_mode", "hybrid")).lower()
    
    preferred_cities = ("pune", "noida")
    if any(p in loc for p in preferred_cities):
        loc_score = 1.00
    elif any(t1 in loc for t1 in LOCATION_TIER1):
        loc_score = 0.85
    elif sig.get("willing_to_relocate"):
        loc_score = 0.70
    else:
        loc_score = 0.30
        
    if mode in ("hybrid", "onsite"):
        mode_score = 1.00 if loc_score >= 0.85 else 0.50
    else:
        mode_score = 0.80
        
    return (0.60 * loc_score + 0.40 * mode_score)

def _compute_market_signal(sig: Dict[str, Any]) -> float:
    views = float(sig.get("profile_views_received_30d", 0))
    views_s = min(1.0, views / 50.0)
    
    search = float(sig.get("search_appearance_30d", 0))
    search_s = min(1.0, search / 100.0)
    
    connections = float(sig.get("connection_count", 0))
    connections_s = min(1.0, connections / 500.0)
    
    return (0.40 * views_s + 0.40 * search_s + 0.20 * connections_s)

def score_behavioral(signals: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    avail = _compute_availability(signals)
    cred = _compute_credibility(signals)
    reach = _compute_reachability(signals)
    loc = _compute_location_fit(signals, candidate)
    market = _compute_market_signal(signals)
    
    composite = (
        BEHAVIORAL_WEIGHTS["availability"]  * avail +
        BEHAVIORAL_WEIGHTS["credibility"]   * cred +
        BEHAVIORAL_WEIGHTS["reachability"]  * reach +
        BEHAVIORAL_WEIGHTS["location_fit"]  * loc +
        BEHAVIORAL_WEIGHTS["market_signal"] * market
    )
    
    if composite < 0.30:
        multiplier = 0.35 + 0.30 * composite
    elif composite < 0.60:
        multiplier = 0.50 + 0.40 * composite
    else:
        multiplier = 0.60 + 0.40 * composite
        
    multiplier_clipped = max(0.35, min(1.0, multiplier))
    
    return {
        "behavioral_multiplier": multiplier_clipped,
        "availability_score": avail,
        "credibility_score": cred,
        "reachability_score": reach,
        "location_fit_score": loc,
        "market_signal_score": market,
        "behavioral_composite": composite,
    }