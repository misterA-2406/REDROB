`yaml
team_name: "Team Antigravity"
participant_id: "antigravity_v4"

primary_contact:
  name: "Antigravity Dev"
  email: "dev@antigravity.org"
  phone: "+91-9999999999"

github_repository: "https://github.com/antigravity/redrob-ranker"
sandbox_link: "https://huggingface.co/spaces/antigravity/redrob-ranker"

ai_tools_declared:
  - Claude
  - GitHub Copilot

compute_environment: "MacBook Pro M2 / Ubuntu 22.04, 16GB RAM, Python 3.11"

methodology_summary: |
  7-stage CPU-only pipeline. Stage 1 hard-filters on JD keyword presence,
  consulting-only careers, non-tech titles, and response rate < 5%.
  Six honeypot checks (timeline impossibility, expert+zero usage, assessment
  contradiction, founding anomaly, all-maxed signals, job overlap) run at
  intake. Stage 2 scores 11 JD-required skills weighted 0.30–1.00 using
  SKILL_ALIASES and STRONG_ALIAS_EXPANSIONS with assessment-score boosts.
  Stage 3 applies a 23-signal behavioral multiplier (0.35–1.00) via five
  sub-scorers weighted by availability (0.30), credibility (0.35),
  reachability (0.15), location (0.10), market (0.10). Stage 4 re-ranks
  the top-300 using BAAI/bge-small-en-v1.5 (or all-MiniLM-L6-v2) with
  JD embedding cached once and batch candidate encoding (batch=32, CPU).
  Stage 5 prunes honeypots and detects near-duplicate profiles.
  Stage 5.5 applies XGBoost LTR (rank:ndcg, 150 estimators) on a 26-feature
  vector with pseudo-labels blended as FinalScore = 0.70×LTR + 0.30×Composite.
  Stage 6 generates rank-tiered, hash-seeded, fact-grounded reasoning strings
  (120–235 chars) with a deduplication pass. All scores are deterministically
  sorted (desc score, asc candidate_id) before CSV output.
```

---

### `README.md`
```markdown
# Redrob Hackathon v4 — Team Antigravity

## Quick Start

```bash
# Install pinned dependencies
pip install -r requirements.txt

# Download preferred semantic embedding model once for local offline caching
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Run the 7-stage discoverer pipeline
python rank.py --candidates ./candidates.jsonl.gz --out ./antigravity_v4.csv

# Run compliance formatting validate checks
python validate_submission.py --file ./antigravity_v4.csv --candidates ./candidates.jsonl.gz

# Launch Streamlit sandbox ui
streamlit run app.py
```

## Options Configuration

Configure running thresholds to check specific scenarios:

```bash
python rank.py \
  --candidates ./candidates.jsonl.gz \
  --out ./antigravity_v4.csv \
  --semantic-weight 0.45 \
  --ltr-alpha 0.70 \
  --ltr-model-path ./ltr_ranker.pkl \
  --team-id antigravity_v4
```

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `ranker/prefilter.py` | Filters candidate targets & profiles missing requirements. Analyzes timeline/metric inconsistencies using 6 checks. |
| 2 | `ranker/skill_scorer.py` | Scores 11 core skills, maps company quality, and details production experience context. |
| 3 | `ranker/behavioral.py` | Evaluates availability, reachability, location fit, and credibility using 23 signals. |
| 4 | `ranker/semantic.py` | Performs dense semantic reranking on the top 300 candidates using cached BGE-small embeddings. |
| 5 | `ranker/honeypot.py` | Post-reranking honeypot exclusion checks and candidate pool deduplication. |
| 5.5 | `ranker/ltr_scorer.py` | Applies machine learning-based LTR (XGBoost/LightGBM) to refine candidate ranking. |
| 6 | `ranker/reasoner.py` | Generates fact-grounded, unique explanatory text (120–235 chars) for discovered profiles. |

## Execution Boundaries

| Metric | Target Boundary | Implementation |
|---|---|---|
| Ingestion Time | <= 4 mins | Monitored check points break stream sequence if limits are exceeded. |
| System Memory | <= 16 GB | Low footprint JSONL streaming; dense reranking restricted to the top 300 candidates. |
| Host Accelerators | Offline CPU only | Uses pre-normalized L2 vectors and vectorized batch matrix multiplication. |