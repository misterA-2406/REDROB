### `ranker/config.py`

# ranker/config.py
from __future__ import annotations
import datetime
from typing import Any, Dict, List, Set

CURRENT_DATE = datetime.datetime(2026, 6, 4)
CURRENT_YEAR = 2026

# Component weights (sum to 1.0)
STAGE_WEIGHTS: Dict[str, float] = {
    "core_skills_match":    0.30,
    "title_relevance":      0.20,
    "experience_band_fit":  0.15,
    "company_type_quality": 0.15,
    "production_evidence":  0.10,
    "nice_to_have_skills":  0.05,
    "open_source_signal":   0.025,
    "education_signal":     0.025,
}

JD_REQUIRED_SKILLS: Dict[str, float] = {
    "embeddings":             1.00,
    "vector_database":        1.00,
    "retrieval":              0.95,
    "hybrid_search":          0.90,
    "python":                 0.85,
    "evaluation_frameworks":  0.85,
    "ranking_systems":        0.80,
    "semantic_search":        0.80,
    "recommendation_systems": 0.75,
    "nlp":                    0.70,
    "information_retrieval":  0.70,
}

JD_NICE_TO_HAVE: Dict[str, float] = {
    "llm_fine_tuning":          0.80,
    "lora":                     0.80,
    "qlora":                    0.80,
    "peft":                     0.80,
    "learning_to_rank":         0.90,
    "xgboost_ltr":              0.90,
    "neural_ltr":               0.90,
    "lambdamart":               0.90,
    "hr_tech":                  1.00,
    "recruiting_tech":          1.00,
    "distributed_systems":      0.70,
    "inference_optimization":   0.70,
    "open_source_contributors": 0.80,
}

SKILL_ALIASES: Dict[str, str] = {
    "sentence-transformers":        "embeddings",
    "bge":                          "embeddings",
    "e5":                           "embeddings",
    "bi-encoder":                   "embeddings",
    "cross-encoder":                "embeddings",
    "dense retrieval":              "embeddings",
    "text embeddings":              "embeddings",
    "vector embeddings":            "embeddings",
    "pinecone":                     "vector_database",
    "qdrant":                       "vector_database",
    "weaviate":                     "vector_database",
    "faiss":                        "vector_database",
    "milvus":                       "vector_database",
    "opensearch":                   "vector_database",
    "elasticsearch":                "vector_database",
    "annoy":                        "vector_database",
    "scann":                        "vector_database",
    "ndcg":                         "evaluation_frameworks",
    "mrr":                          "evaluation_frameworks",
    "map":                          "evaluation_frameworks",
    "a/b testing":                  "evaluation_frameworks",
    "precision at k":               "evaluation_frameworks",
    "recall at k":                  "evaluation_frameworks",
    "ranking metrics":              "evaluation_frameworks",
    "lora":                         "llm_fine_tuning",
    "qlora":                        "llm_fine_tuning",
    "peft":                         "llm_fine_tuning",
    "learning-to-rank":             "ranking_systems",
    "lambdamart":                   "ranking_systems",
    "xgboost":                      "xgboost_ltr",
    "recsys":                       "recommendation_systems",
    "recommender":                  "recommendation_systems",
    "collaborative filtering":      "recommendation_systems",
    "matrix factorization":         "recommendation_systems",
    "content-based filtering":      "recommendation_systems",
    "hybrid retrieval":             "hybrid_search",
    "sparse dense fusion":          "hybrid_search",
    "rrf":                          "hybrid_search",
    "reciprocal rank fusion":       "hybrid_search",
    "nearest neighbor":             "retrieval",
    "ann":                          "retrieval",
    "knn":                          "retrieval",
    "vector similarity":            "retrieval",
    "sparse retrieval":             "retrieval",
    "dense vector":                 "retrieval",
    "py-spark":                     "distributed_systems",
    "spark":                        "distributed_systems",
    "kubernetes":                   "distributed_systems",
}

STRONG_ALIAS_EXPANSIONS: Dict[str, List[str]] = {
    "embeddings": [
        "sentence-transformers", "bge", "e5", "dense retrieval",
        "neural embeddings", "text embeddings", "vector embeddings",
        "bi-encoder", "cross-encoder",
    ],
    "vector_database": [
        "faiss", "pinecone", "qdrant", "weaviate", "milvus",
        "opensearch", "elasticsearch", "annoy", "scann", "vector store",
    ],
    "evaluation_frameworks": [
        "ndcg", "mrr", "map", "precision at k", "recall at k",
        "a/b testing", "offline eval", "ranking metrics",
    ],
    "recommendation_systems": [
        "recsys", "recommender", "collaborative filtering",
        "matrix factorization", "content-based filtering",
    ],
    "hybrid_search": [
        "bm25", "hybrid retrieval", "sparse dense fusion",
        "rrf", "reciprocal rank fusion", "elasticsearch", "opensearch",
    ],
    "retrieval": [
        "nearest neighbor", "ann", "knn", "vector similarity",
        "sparse retrieval", "dense vector", "personalization", "recall",
    ],
}

PRODUCT_COMPANIES_TIER1: Set[str] = {
    "flipkart", "swiggy", "zomato", "phonepe", "razorpay", "cred",
    "meesho", "sharechat", "dream11", "freshworks", "zoho", "paytm",
    "goibibo", "makemytrip", "ola", "nykaa", "policybazaar", "vedantu",
    "byju", "byjus", "groww", "zepto", "dunzo", "slice", "khatabook",
    "lenskart", "mamaearth", "rapido", "urban company", "urban ladder",
    "inmobi", "moengage", "clevertap", "chargebee", "postman",
    "browserstack", "ola electric", "koo",
    "google", "microsoft", "amazon", "meta", "apple", "netflix",
    "uber", "airbnb", "linkedin", "twitter", "x corp", "stripe",
    "atlassian", "shopify", "twilio", "databricks", "snowflake",
    "salesforce", "workday", "servicenow", "adobe", "quora", "reddit",
}

CONSULTING_FIRMS: Set[str] = {
    "tcs", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl", "hcl technologies",
    "tech mahindra", "l&t infotech", "mindtree", "mphasis", "hexaware",
    "zensar", "niit technologies", "genpact", "kpit technologies",
    "sonata software",
}

STARTUP_COMPANIES: Set[str] = {
    "zepto", "blinkit", "slice", "open", "jupiter", "fi money",
    "niyo", "smallcase", "zerodha", "groww", "upstox", "kreditbee",
    "sarvam", "krutrim", "ola krutrim", "haptik", "vernacular",
    "uniphore", "murf", "yellow ai", "wadhwani ai",
}

RESEARCH_LABS: Set[str] = {
    "deepmind", "google deepmind", "openai", "anthropic", "cohere",
    "microsoft research", "google research", "facebook ai research", "fair",
    "iit", "iim", "iisc", "iiser", "bits pilani", "iiit hyderabad",
    "iiit bangalore",
}

OUTSOURCING_FIRMS: Set[str] = {
    "kforce", "infosys bpo", "wipro bpo", "tcs bpo", "firstsource",
    "mphasis", "hexaware", "igate", "patni", "mastech", "niit",
    "syntel", "cyient", "coforge", "ltimindtree",
}

COMPANY_FOUNDING_YEARS: Dict[str, int] = {
    "flipkart": 2007, "ola": 2010, "swiggy": 2014, "zomato": 2008,
    "phonepe": 2015, "razorpay": 2014, "cred": 2018, "meesho": 2015,
    "sharechat": 2015, "dream11": 2008, "freshworks": 2010, "zoho": 1996,
    "paytm": 2010, "uber": 2009, "google": 1998, "microsoft": 1975,
    "meta": 2004, "amazon": 1994, "netflix": 1997, "apple": 1976,
    "linkedin": 2002, "twitter": 2006, "stripe": 2010, "airbnb": 2008,
    "nykaa": 2012, "groww": 2016, "zepto": 2021, "dunzo": 2015,
    "inmobi": 2007,
}

RECENCY_DECAY_WEIGHTS: List[float] = [1.0, 0.8, 0.6, 0.4, 0.25]

DISQUALIFYING_TITLES: Set[str] = {
    "marketing manager", "product manager", "business analyst",
    "hr recruiter", "finance manager", "sales executive",
    "ux designer", "ui designer", "account manager",
}

LOCATION_TIER1: List[str] = [
    "pune", "noida", "delhi", "delhi ncr", "hyderabad", "mumbai",
    "bangalore", "bengaluru", "gurgaon", "gurugram", "chennai",
]

JD_CORE_KEYWORDS: List[str] = [
    "embedding", "retrieval", "python", "vector", "ranking",
    "nlp", "search", "recommendation", "evaluation", "hybrid",
    "faiss", "qdrant", "pinecone", "ndcg", "mrr", "map",
]

BEHAVIORAL_WEIGHTS: Dict[str, float] = {
    "availability":  0.30,
    "credibility":   0.35,
    "reachability":  0.15,
    "location_fit":  0.10,
    "market_signal": 0.10,
}

DEFAULT_SEMANTIC_WEIGHT: float = 0.45
DEFAULT_LTR_ALPHA: float = 0.70

LTR_CONFIG: Dict[str, Any] = {
    "alpha":            0.70,
    "model_path":       "./ltr_ranker.pkl",
    "use_xgboost":      True,
    "n_estimators":     150,
    "max_depth":        5,
    "learning_rate":    0.08,
    "min_child_weight": 5,
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "random_state":     42,
}

CAREER_LEVEL_SCORES: Dict[str, int] = {
    "intern": 1, "trainee": 1, "junior": 2,
    "associate": 2, "engineer": 3, "developer": 3,
    "senior": 5, "lead": 6, "principal": 7,
    "staff": 7, "architect": 7, "manager": 6,
    "director": 8, "vp": 9, "head": 8, "chief": 10,
}

FEATURE_NAMES: List[str] = [
    "skill_score", "semantic_score", "behavioral_composite", "yoe_normalised",
    "experience_fit", "vector_db_experience", "retrieval_experience",
    "ranking_system_experience", "evaluation_metrics_experience",
    "python_strength", "llm_finetuning_score", "company_quality_score",
    "github_activity_score", "endorsement_score", "recruiter_response_rate",
    "profile_completeness_score", "open_to_work_signal", "notice_period_score",
    "career_growth_score", "candidate_activity_score", "trust_score",
    "honeypot_probability", "availability_score", "credibility_score",
    "reachability_score", "semantic_similarity_score",
]

JD_TEXT = """
We are Redrob AI, a Series A AI-native talent intelligence platform based in
Pune and Noida, India. We are hiring a Senior AI Engineer for our founding
engineering team. We need someone with 5 to 9 years of experience who has
actually shipped production ranking, retrieval, or recommendation systems to
real users at scale — not just experimented with them in research settings.

The role requires deep hands-on experience with embeddings-based retrieval
systems using models like sentence-transformers, BGE, or E5, including
production concerns such as embedding drift, index refresh, and retrieval
quality regression monitoring. You must have operated vector databases or
hybrid search infrastructure such as Pinecone, Weaviate, Qdrant, FAISS,
Milvus, Elasticsearch, or OpenSearch at production scale.

Strong Python is non-negotiable — we care deeply about code quality. You must
have designed rigorous evaluation frameworks for ranking systems, covering
NDCG, MRR, MAP, offline-to-online correlation, and A/B test interpretation.

We also value LLM fine-tuning experience using LoRA, QLoRA, or PEFT methods,
as well as learning-to-rank experience with XGBoost-based LTR, LambdaMART, or
neural LTR approaches. Prior exposure to HR-tech or recruiting technology is a
strong bonus.

We do not want candidates from purely academic or research backgrounds without
any production deployments, or those whose entire career has been at consulting
firms like TCS, Infosys, Wipro, Accenture, or Capgemini with no product company
experience. We need a builder who tilts toward shipping.
"""