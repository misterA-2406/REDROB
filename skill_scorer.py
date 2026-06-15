# ranker/skill_scorer.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple, Union
from ranker.config import (
    STAGE_WEIGHTS, JD_REQUIRED_SKILLS, JD_NICE_TO_HAVE,
    SKILL_ALIASES, STRONG_ALIAS_EXPANSIONS, PRODUCT_COMPANIES_TIER1,
    STARTUP_COMPANIES, RESEARCH_LABS, OUTSOURCING_FIRMS, CONSULTING_FIRMS,
    RECENCY_DECAY_WEIGHTS, JD_CORE_KEYWORDS
)
from ranker.prefilter import is_consulting_only

logger = logging.getLogger("RedrobRanker.SkillScorer")

PRODUCTION_KW_WEIGHTED = {
    "kubernetes":       1.5, "kafka":           1.5,
    "airflow":          1.3, "spark":            1.3,
    "feature store":    1.5, "online serving":  1.5,
    "distributed system": 1.4, "microservice":  1.3,
    "deployed":         1.0, "deployment":      1.0,
    "serving":          1.0, "inference":        0.9,
    "real-time":        1.0, "production":       1.0,
    "latency":          0.9, "throughput":       0.9,
    "monitoring":       0.8, "a/b test":         1.0,
    "high scale":       1.0, "launched":         0.8,
    "shipped":          0.8, "live":             0.7,
    "online":           0.6, "traffic":          0.7,
    "million":          0.8, "billion":          1.0,
    "real users":       1.0,
}

def _recency_weight(idx: int) -> float:
    return RECENCY_DECAY_WEIGHTS[min(idx, len(RECENCY_DECAY_WEIGHTS) - 1)]

def _classify_company(company: str) -> Tuple[str, float]:
    comp_clean = company.strip().lower()
    if not comp_clean:
        return "generic_product", 0.75
        
    if comp_clean in PRODUCT_COMPANIES_TIER1:
        return "tier1_product", 1.00
    if comp_clean in STARTUP_COMPANIES:
        return "startup", 0.88
    if comp_clean in RESEARCH_LABS:
        return "research_lab", 0.82
    if any(out in comp_clean for out in ("outsourc", "bpo", "staffing", "manpower")) or comp_clean in OUTSOURCING_FIRMS:
        return "outsourcing", 0.35
    if comp_clean in CONSULTING_FIRMS:
        return "consulting", 0.28
    if any(serv in comp_clean for serv in ("consulting", "services", "solutions", "systems", "infotech")):
        return "services_generic", 0.55
        
    return "generic_product", 0.75

def _score_company_quality(work_history: List[Dict[str, Any]]) -> float:
    if not work_history:
        return 0.50
    weighted_sum = 0.0
    total_weight = 0.0
    all_low_tier = True
    
    for idx, job in enumerate(work_history):
        comp = str(job.get("company", ""))
        tier, score = _classify_company(comp)
        if tier not in ("consulting", "outsourcing", "services_generic"):
            all_low_tier = False
        w = _recency_weight(idx)
        weighted_sum += score * w
        total_weight += w
        
    avg = weighted_sum / total_weight if total_weight > 0 else 0.50
    if all_low_tier:
        avg = min(0.15, avg)
    return avg

def _extract_production_signals(work_history: List[Dict[str, Any]]) -> float:
    if not work_history:
        return 0.0
    total_hits = 0.0
    for idx, job in enumerate(work_history):
        decay = _recency_weight(idx)
        text = (str(job.get("description", "")) + " " + str(job.get("title", ""))).lower()
        for kw, weight in PRODUCTION_KW_WEIGHTED.items():
            if kw in text:
                total_hits += weight * decay
    return min(1.0, total_hits / 8.0)

def _normalize_skills(skills: List[Union[str, Dict[str, Any]]]) -> List[Tuple[str, float]]:
    proficiency_map = {
        "expert": 1.0, "advanced": 0.85,
        "intermediate": 0.65, "beginner": 0.35,
    }
    normalized = []
    for s in skills:
        if isinstance(s, dict):
            name = str(s.get("name", "")).strip().lower()
            prof = str(s.get("proficiency", "")).strip().lower()
            mult = proficiency_map.get(prof, 0.65)
        else:
            name = str(s).strip().lower()
            mult = 0.65
            
        name_clean = name.replace("_", " ")
        canonical = SKILL_ALIASES.get(name_clean, name_clean)
        normalized.append((canonical, mult))
    return normalized

def _compute_skill_depth(skill_item: Dict[str, Any], assessment_scores: Dict[str, Any]) -> float:
    proficiency_map = {
        "expert": 1.0, "advanced": 0.85,
        "intermediate": 0.65, "beginner": 0.35,
    }
    name = str(skill_item.get("name", "")).strip().lower()
    prof = str(skill_item.get("proficiency", "")).strip().lower()
    prof_mult = proficiency_map.get(prof, 0.65)
    
    dur_months = float(skill_item.get("duration_months") or (skill_item.get("years_used", 0) * 12) or 0.0)
    dur_score = min(dur_months / 60.0, 1.0)
    
    endorsements = float(skill_item.get("endorsements", 0))
    endorse_score = min(endorsements / 30.0, 1.0)
    
    assess_score = float(assessment_scores.get(name, 50)) / 100.0
    
    depth = (0.40 * prof_mult + 0.35 * dur_score + 0.15 * endorse_score + 0.10 * assess_score)
    return min(1.0, depth)

def _compute_education_signal(candidate: Dict[str, Any]) -> float:
    tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.5, "tier_4": 0.2}
    education = candidate.get("education") or []
    if not education:
        return 0.35
        
    highest_score = 0.30
    for edu in education:
        tier = edu.get("tier", edu.get("institution_tier", "tier_3"))
        score = tier_map.get(tier, 0.3)
        field = str(edu.get("field_of_study", "")).lower()
        if any(f in field for f in ("computer", "software", "ml", "ai", "information", "data", "math", "stat")):
            score = min(1.0, score + 0.10)
        if score > highest_score:
            highest_score = score
    return highest_score

def _compute_profile_quality(signals: Dict[str, Any]) -> float:
    email = 1.0 if signals.get("verified_email") else 0.0
    phone = 1.0 if signals.get("verified_phone") else 0.0
    linkedin = 1.0 if signals.get("linkedin_connected") else 0.0
    complete = float(signals.get("profile_completeness_score", 80)) / 100.0
    
    profile_quality = (0.35 * email + 0.25 * phone + 0.20 * linkedin + 0.20 * complete)
    return max(0.0, min(1.0, profile_quality))

def score_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    work_history = candidate.get("work_history") or []
    signals = candidate.get("signals") or {}
    raw_skills = candidate.get("skills") or []
    assessment_scores = signals.get("skill_assessment_scores") or {}
    
    # Pre-parse skill depth metrics safely
    depth_scores = []
    for s in raw_skills:
        if isinstance(s, dict):
            depth_scores.append(_compute_skill_depth(s, assessment_scores))
    avg_skill_depth = sum(depth_scores) / len(depth_scores) if depth_scores else 0.50
    
    normalized_declarations = _normalize_skills(raw_skills)
    declared_lookup = {k: v for k, v in normalized_declarations}
    
    core_score_sum = 0.0
    matched_set = set()
    missing_set = set()
    
    for req_sk, weight in JD_REQUIRED_SKILLS.items():
        if req_sk in declared_lookup:
            depth_mult = 1.0
            for s in raw_skills:
                if isinstance(s, dict) and str(s.get("name", "")).strip().lower() == req_sk:
                    depth_mult = _compute_skill_depth(s, assessment_scores)
            term_score = weight * depth_mult
            core_score_sum += term_score
            matched_set.add(req_sk)
        else:
            alias_match = False
            for expansion in STRONG_ALIAS_EXPANSIONS.get(req_sk, []):
                if expansion in declared_lookup:
                    core_score_sum += weight * 0.70
                    matched_set.add(req_sk)
                    alias_match = True
                    break
            if not alias_match:
                missing_set.add(req_sk)
                
    core_match_score = (core_score_sum / sum(JD_REQUIRED_SKILLS.values())) if JD_REQUIRED_SKILLS else 0.0

    title_raw = str(candidate.get("current_title", "")).lower()
    title_score = 0.10
    if "senior" in title_raw or "lead" in title_raw or "principal" in title_raw:
        title_score = 0.70
    if "ai" in title_raw or "ml" in title_raw or "machine learning" in title_raw or "nlp" in title_raw:
        title_score = min(1.0, title_score + 0.30)

    yoe = float(candidate.get("total_years_experience", 0.0))
    if 4.0 <= yoe <= 9.0:
        exp_score = 1.00
    elif 10.0 <= yoe <= 12.0:
        exp_score = 0.80
    elif 3.0 <= yoe < 4.0:
        exp_score = 0.60
    elif yoe >= 13.0:
        exp_score = 0.65
    elif 0.0 < yoe < 3.0:
        exp_score = 0.20
    else:
        exp_score = 0.05

    company_score = _score_company_quality(work_history)
    prod_score = _extract_production_signals(work_history)

    nice_score_sum = 0.0
    for nice_sk, weight in JD_NICE_TO_HAVE.items():
        if nice_sk in declared_lookup:
            nice_score_sum += weight
    nice_score = (nice_score_sum / len(JD_NICE_TO_HAVE)) if JD_NICE_TO_HAVE else 0.0

    os_score = 0.0
    github_sc = float(signals.get("github_activity_score", -1))
    if github_sc != -1:
        os_score = min(1.0, github_sc / 100.0)

    edu_score = _compute_education_signal(candidate)

    base_skills_score = (
        STAGE_WEIGHTS["core_skills_match"]    * core_match_score +
        STAGE_WEIGHTS["title_relevance"]      * title_score      +
        STAGE_WEIGHTS["experience_band_fit"]  * exp_score        +
        STAGE_WEIGHTS["company_type_quality"] * company_score    +
        STAGE_WEIGHTS["production_evidence"]  * prod_score       +
        STAGE_WEIGHTS["nice_to_have_skills"]  * nice_score       +
        STAGE_WEIGHTS["open_source_signal"]   * os_score         +
        STAGE_WEIGHTS["education_signal"]     * edu_score
    )

    profile_q = _compute_profile_quality(signals)
    skills_score_adjusted = base_skills_score * (0.95 + 0.05 * profile_q)

    consult_flag = is_consulting_only(work_history)
    if consult_flag:
        skills_score_adjusted *= 0.70

    return {
        "skill_score": skills_score_adjusted,
        "core_skills": core_match_score,
        "title_relevance": title_score,
        "experience_fit": exp_score,
        "company_quality": company_score,
        "production_evidence": prod_score,
        "nice_to_have": nice_score,
        "open_source": os_score,
        "education_signal": edu_score,
        "profile_quality": profile_q,
        "matched_skills": list(matched_set),
        "missing_skills": list(missing_set),
        "consulting_only": consult_flag,
        "skill_depth_avg": avg_skill_depth,
    }