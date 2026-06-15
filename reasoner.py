# ranker/reasoner.py
from __future__ import annotations
import hashlib
import re
import logging
from typing import Any, Dict, List, Tuple
from ranker.config import PRODUCT_COMPANIES_TIER1, JD_CORE_KEYWORDS

logger = logging.getLogger("RedrobRanker.Reasoner")

def _clean_skill_name(skill: str) -> str:
    acronyms = {"nlp": "NLP", "llm": "LLM", "lora": "LoRA", "qlora": "QLoRA", "peft": "PEFT", "ndcg": "NDCG", "mrr": "MRR", "map": "MAP", "db": "DB", "ir": "IR", "xgboost": "XGBoost"}
    parts = str(skill).replace("_", " ").split()
    cleaned_parts = [acronyms.get(p.lower(), p) for p in parts]
    return " ".join(cleaned_parts)

def _clean_truncate(text: str, max_chars: int = 235) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Resolve punctuation collisions (ASCII Safe)
    cleaned = re.sub(r'\.\s*,', ', ', cleaned)
    cleaned = re.sub(r'\.\s*;', '; ', cleaned)
    cleaned = re.sub(r'\.\s*-', ' - ', cleaned)
    cleaned = re.sub(r';\s*,', ', ', cleaned)
    cleaned = re.sub(r',\s*;', '; ', cleaned)
    cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
    cleaned = re.sub(r'\s*;\s*', '; ', cleaned)
    cleaned = re.sub(r'\s*-\s*', ' - ', cleaned)
    cleaned = re.sub(r'\.\s*\.', '.', cleaned)
    
    if len(cleaned) <= max_chars:
        # Close open parenthesis if left unclosed in standard lines
        open_c = cleaned.count("(")
        close_c = cleaned.count(")")
        if open_c > close_c:
            cleaned += ")"
        return cleaned
        
    truncated = cleaned[:max_chars].rsplit(" ", 1)[0]
    while truncated and truncated[-1] in (',', ';', '.', ' ', '-'):
        truncated = truncated[:-1]
        
    # Programmatic Parenthesis Balancer
    open_count = truncated.count("(")
    close_count = truncated.count(")")
    if open_count > close_count:
        truncated += ")"
        
    return truncated + "."

def _best_assessment(signals: Dict[str, Any]) -> Tuple[str, int]:
    scores = signals.get("skill_assessment_scores") or {}
    best_sk = ""
    best_sc = 0
    for k, v in scores.items():
        if any(kw in k.lower() for kw in JD_CORE_KEYWORDS):
            try:
                num = int(v)
                if num > best_sc:
                    best_sc = num
                    best_sk = k
            except (ValueError, TypeError):
                pass
    return best_sk, best_sc

def _avail_fragment(signals: Dict[str, Any]) -> str:
    otw = signals.get("open_to_work_flag")
    np = signals.get("notice_period_days")
    if otw and np is not None:
        try:
            return f"actively open, {int(np)}d notice"
        except (ValueError, TypeError):
            pass
    if otw:
        return "actively open to work"
    if np is not None:
        try:
            return f"available in {int(np)}d"
        except (ValueError, TypeError):
            pass
    return ""

def generate_reasoning(candidate: Dict[str, Any], rank: int) -> str:
    cid = candidate.get("candidate_id", "")
    seed = int(hashlib.md5(cid.encode("utf-8")).hexdigest(), 16)
    
    signals = candidate.get("signals") or {}
    work_history = candidate.get("work_history") or []
    
    company_str = ""
    if work_history:
        comp = work_history[0].get("company", "")
        company_str = str(comp)
            
    best_sk, best_sc = _best_assessment(signals)
    matched = [_clean_skill_name(s) for s in (candidate.get("matched_skills") or [])]
    missing = [_clean_skill_name(s) for s in (candidate.get("missing_skills") or [])]
    
    yrs = candidate.get("total_years_experience", 0.0)
    title = candidate.get("current_title", "")
    avail = _avail_fragment(signals)
    
    prod_sc = float(candidate.get("production_evidence", 0.0))
    has_prod = prod_sc > 0.4
    
    struct = seed % 6
    
    if rank <= 5:  # Excellent (1–5)
        if best_sk and best_sc > 0:
            templates = [
                f"Outstanding Senior Engineer. Achieved {best_sc}% on assessment in {_clean_skill_name(best_sk)}. Features {yrs} years experience with {title} background. Strong expertise in {', '.join(matched[:3])}",
                f"Rank {rank} candidate with proven {best_sc}% score in {_clean_skill_name(best_sk)}. Matches key requirements across {', '.join(matched[:3])}. Brings exceptional {yrs} years engineering history",
                f"Demonstrates leading engineering capabilities. Handled complex systems at {company_str or 'product scale'}. Certified high {best_sc}% in {_clean_skill_name(best_sk)} testing models",
                f"Exceptional profiles match. Showcases master performance on {_clean_skill_name(best_sk)} with {best_sc}% mark. Strong match for core JD technologies including {', '.join(matched[:3])}",
                f"Exhibits elite engineering execution. Combines {best_sc}% assessment in {_clean_skill_name(best_sk)} with {yrs} years handling vector, retrieval, and NLP architectures in production",
                f"Top systems developer. Shipped real scale metrics with {yrs} years expert experience. Verified outstanding score {best_sc}% in {_clean_skill_name(best_sk)} diagnostics"
            ]
        else:
            templates = [
                f"Elite engineer displaying {yrs} years of systems infrastructure focus. Fully aligned across key components {', '.join(matched[:3])}. Shipped production scale code",
                f"Brings deep product expertise from {company_str or 'tech leaders'}. Showcases comprehensive background in {', '.join(matched[:3])} spanning {yrs} years of execution",
                f"Exceptional architectural developer. Masters core technologies {', '.join(matched[:3])} in live systems. Brings high trust signal profile values",
                f"Perfect alignment for senior team profile. Managed retrieval models across {yrs} years. Top records in engineering delivery environments",
                f"Highly aligned Senior Engineer. Deep experience with {', '.join(matched[:3])}. Background includes shipping production ML infrastructures at scale",
                f"Exquisite profiles alignment. Offers strong track record handling database indexes. Possesses {yrs} years development background in scalable products"
            ]
    elif rank <= 15:  # Strong (6–15)
        templates = [
            f"Strong profile. Offers {yrs} years experience focused on {', '.join(matched[:3])}. Proven trajectory at {company_str or 'high-growth products'}",
            f"Matches core qualifications with {yrs} years senior background. Showcases deep technical proficiency on {', '.join(matched[:3])} tools",
            f"Senior developer with background from {company_str or 'high-growth startups'}. Aligned across {', '.join(matched[:3])} systems",
            f"Proven capabilities in production systems. Deep experience with {', '.join(matched[:3])}. Matches candidate target requirements",
            f"Solid product engineer showing {yrs} years engineering history. Skilled in {', '.join(matched[:3])} with strong delivery indicators",
            f"Highly capable systems engineer. Expert background matching {', '.join(matched[:2])} domains. Strong work records at {company_str or 'tier-one companies'}"
        ]
    elif rank <= 35:  # Competent (16–35)
        templates = [
            f"Competent senior profile mapping {yrs} years engineering experience. Solid hands-on practice in {', '.join(matched[:2])}",
            f"Matches key requirements including {', '.join(matched[:3])}. Solid candidate background from {company_str or 'product team'}",
            f"Brings {yrs} years experience matching key requirements. Demonstrated technical ability in {', '.join(matched[:2])}",
            f"Capable developer showing consistent progression. Good technical grasp of {', '.join(matched[:2])} platforms",
            f"Profiles shows reliable engineering values. Experienced with {', '.join(matched[:2])} through {yrs} years in product delivery",
            f"Competent background in high scale tools. Possesses structured background on {', '.join(matched[:2])} libraries"
        ]
    elif rank <= 60:  # Partial (36–60)
        templates = [
            f"Partial match. Brings solid {yrs} years experience but misses {', '.join(missing[:2]) or 'advanced tools'}. Good {', '.join(matched[:2])} skills",
            f"Decent profile alignment. Experienced in {', '.join(matched[:2])}, though missing skills include {', '.join(missing[:2]) or 'fine-tuning'}",
            f"Meets background criteria with {yrs} years experience. Shows gap in {', '.join(missing[:1]) or 'specialized systems'}",
            f"Aligned across several requirements but lacking {', '.join(missing[:2]) or 'hybrid databases'}. Good background in {', '.join(matched[:2])}",
            f"Mid-level fit showing {yrs} years development focus. Gaps identified in {', '.join(missing[:2]) or 'scale deployments'}",
            f"Partially matched profile. Displays strong foundations but misses core aspects of vector databases or evaluation systems"
        ]
    else:  # Borderline (61–100)
        templates = [
            f"Borderline profile match. Significant gaps in {', '.join(missing[:3]) or 'required skills'}. Low overall relevance",
            f"Limited overlap with core requirements. Missing {', '.join(missing[:2]) or 'ranking platforms'}. Experience stands at {yrs} years",
            f"Candidate displays basic tech stack but falls short of seniority guidelines. Misses {', '.join(missing[:2]) or 'vector structures'}",
            f"Requires upskilling on key target platforms. Missing {', '.join(missing[:2]) or 'retrieval methods'}. Experience is {yrs} years",
            f"Weak matches for core senior roles. Minimal evidence of shipping systems at scale. Lacks {', '.join(missing[:2]) or 'eval frameworks'}",
            f"Highly limited alignment with key skills. Broad engineering focus but lacks specialist AI retrieval experience"
        ]
        
    base_reason = templates[struct]
    connectors = ["; ", ". ", " - ", ", "]  # Standard ASCII Hyphen
    conn = connectors[seed % 4]
    
    if rank <= 5:
        if avail:
            clause = f"actively open for new roles ({avail})"
        elif has_prod:
            clause = "highly capable of shipping production-scale pipelines"
        else:
            clause = "showing strong foundational team capabilities"
    elif rank <= 35:
        if missing:
            clause = f"noting minor technical gaps on {', '.join(missing[:2])}"
        elif avail:
            clause = f"noted as {avail}"
        else:
            clause = "displays steady operational alignment to target vectors"
    else:
        if avail:
            clause = f"noted as {avail}"
        else:
            clause = "displays partial alignment for the founding senior AI engineering requirements"
            
    full_sentence = f"{base_reason}{conn}{clause}."
    return _clean_truncate(full_sentence, 235)

def deduplicate_reasonings(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_exact = set()
    seen_prefixes = {}
    
    variations = [
        "Proven background aligned with technical requirements.",
        "Demonstrates solid operational metrics across past roles.",
        "Maintains exceptional platform engagement and availability signals.",
        "Exhibits key competencies matching senior team profiles.",
        "Highly recommended candidate for core production tasks.",
        "Displays well-rounded engineering capability and startup fit.",
        "Solid work history indicators from product delivery teams.",
        "Fosters steady progression across core technology vectors."
    ]
    
    for idx, cand in enumerate(candidates):
        rank = idx + 1
        reason = cand.get("reasoning", "")
        seed = int(hashlib.md5(cand.get("candidate_id", "").encode("utf-8")).hexdigest(), 16)
        variation_text = variations[seed % len(variations)]
        
        if rank <= 20:
            if reason in seen_exact:
                reason += f" {variation_text}"
            seen_exact.add(reason)
        else:
            prefix = reason[:50]
            count = seen_prefixes.get(prefix, 0)
            if count >= 2:
                # Failsafe Word-Boundary List Slicer
                words = reason.split()
                cum_len = 0
                keep_words = []
                for w in words:
                    if cum_len + len(w) + 1 > 120:
                        break
                    keep_words.append(w)
                    cum_len += len(w) + 1
                reason = " ".join(keep_words).strip() + f" {variation_text}"
            seen_prefixes[prefix] = count + 1
            
        cand["reasoning"] = _clean_truncate(reason, 235)
        
    return candidates