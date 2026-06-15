# ranker/__init__.py
"""
ranker — Redrob Hackathon v4 modular ranking package.
Public API re-exports for use in rank.py and app.py.
"""
from ranker.loader import stream_candidates, normalize_candidate_schema
from ranker.prefilter import prefilter, is_consulting_only
from ranker.skill_scorer import score_candidate
from ranker.behavioral import score_behavioral
from ranker.semantic import SemanticRanker
from ranker.honeypot import prune_honeypots, detect_duplicates
from ranker.ltr_scorer import apply_ltr_stage, LTRRanker
from ranker.reasoner import generate_reasoning, deduplicate_reasonings
from ranker.output import write_submission, validate_submission_format

__all__ = [
    "stream_candidates",
    "normalize_candidate_schema",
    "prefilter",
    "is_consulting_only",
    "score_candidate",
    "score_behavioral",
    "SemanticRanker",
    "prune_honeypots",
    "detect_duplicates",
    "apply_ltr_stage",
    "LTRRanker",
    "generate_reasoning",
    "deduplicate_reasonings",
    "write_submission",
    "validate_submission_format",
]