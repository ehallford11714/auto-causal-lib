# Insight suite

Library-first insight reports over the AutoCausal loop, with an **optional SLM**
in the guide / synthesize stages and an optional **closed research loop**
(recommend → join/remine → rediscover). Prefer importing `autocausal.insight`
in apps and notebooks; the CLI is a thin wrapper.

> **Epistemic honesty:** exploratory discovery ≠ identification. Edges are
> candidate relationships. Optional SLM text and experiment ideas are
> **generative assistance**, not causal estimates.

## Python API (primary)

```python
from autocausal.insight import InsightSuite, InsightReport, run_insight_loop

# One-shot loop: load → mine → impute → discover → guide → synthesize
report = run_insight_loop(
    "data.csv",
    text="what drives revenue?",
    use_slm=False,          # rule narrator offline
    join="demographics_demo",
)
print(report.summary)
print(report.role_hypotheses.to_dict())
report.write("insight.md")   # or .json
print(report.to_markdown())
print(report.to_dict())
```

### Closed research loop (optional)

```python
from autocausal.insight import InsightSuite, run_slm_research_loop

suite = InsightSuite(use_slm=False)  # soft-fails to rules if True without HF
report = suite.run_loop(
    "data.csv",
    max_rounds=3,
    join_sources=["demographics_demo", "instruments_demo"],
    text="what drives revenue?",
)
print(report.experiments_recommended[:3])
print(report.round_history)
```

### From a pre-built AutoCausal / discovery result

```python
from autocausal import AutoCausal
from autocausal.insight import InsightSuite

ac = AutoCausal.from_csv("data.csv")
ac.mine().impute().discover()

suite = InsightSuite.from_autocausal(ac, use_slm=False)
report = suite.run(text="Does spend cause sales?")
# Or: ac.insight_loop(text="...", use_slm=False)
```

```python
# From DiscoveryResult alone
report = run_insight_loop(discovery=ac.result, use_slm=False)
```

### Notebook / app embedding

```python
from autocausal.insight import run_insight_loop

def build_insight(csv_path: str, question: str) -> dict:
    report = run_insight_loop(csv_path, text=question, use_slm=False)
    return {
        "markdown": report.to_markdown(),
        "json": report.to_dict(),
        "key_edges": report.key_edges,
        "roles": report.role_hypotheses.to_dict(),
        "experiments": report.experiments_recommended,
    }
```

### Offline demo

```python
from autocausal.insight import demo_insight

report = demo_insight(use_slm=False, research_loop=False)
assert report.summary
```

## Pipeline stages

| Stage | Behavior |
|-------|----------|
| load / join | CSV / Parquet / SQL; optional `join_public` |
| mine | Column profiles + associations |
| impute | Soft skip when frame is clean |
| discover | Existing PC-lite / score edges (+ optional IV) |
| guide | Rule or SLM via `ac.guide` / `guides.direct` (soft HF) |
| synthesize | Rule narrator always; optional SLM narrative via `infer` |
| recommend | `ExperimentRecommender` (rule / soft SLM) |
| research loop | join / remine / nlp / behavioral → rediscover (optional) |
| report | `InsightReport` → Markdown / JSON |

## How the SLM is referenced

1. **Guide stage:** `AutoCausal.guide(use_slm=...)` → `autocausal.slm.guide_pipeline` or multi-backend `guides.direct` (soft `llmintent` when selected).
2. **Synthesize stage:** `optional_slm_narrative` / `build_insight_report` calls `get_backend(...).infer(...)` when SLM is requested.
3. **Experiment recommendations:** `ExperimentRecommender` may use the same soft HF path.
4. **Env / flags (soft, no hard crash):**
   - `use_slm=True` or `AUTOCAUSAL_SLM=1`
   - `AUTOCAUSAL_SLM_MODEL` (default `sshleifer/tiny-gpt2` for tests)
   - Missing `torch` / `transformers` → rule narrator retained + notes on report
5. SLM narrative is labeled: *generative assistance (not identification)*.

```python
report = run_insight_loop("data.csv", use_slm=False)  # offline
report = run_insight_loop("data.csv", use_slm=True)   # soft HF
```

Optional NLP hint injection (when `autocausal.nlp` exists) happens inside the research loop via experiment actions, or:

```python
from autocausal.nlp import extract_causal_hints_from_text
hints = extract_causal_hints_from_text("Randomized treatment leads to higher revenue")
# feed text= into run_insight_loop / guide
```

## InsightReport fields

- `summary` — rule-based narrator text
- `key_edges` — top exploratory edges
- `role_hypotheses` — X/Y/Z/W lists
- `data_sources` — load + join ids
- `caveats` — epistemic warnings
- `guide` / `guide_backend` — guide stage payload
- `slm_narrative` / `slm_used` / `slm_label` — optional generative text
- `experiments_recommended` — next measurements / joins / A-B / IV ideas
- `relationships_mined_further` / `round_history` — research-loop deltas
- `stages` — executed pipeline stages
- Writers: `to_markdown()`, `to_dict()`, `to_json()`, `write(path)`

## CLI (secondary)

```bash
python -m autocausal insight run --csv data.csv --join demographics_demo --no-slm --out report.md
python -m autocausal insight run --csv data.csv --slm --text "what drives Y?" -o report.json
python -m autocausal insight loop --csv data.csv --rounds 3 --join-sources demographics_demo --no-slm
python -m autocausal insight demo --no-slm
python -m autocausal insight demo -o demo_insight.md
```

## Public exports

```python
from autocausal.insight import (
    InsightSuite,
    InsightReport,
    RoleHypotheses,
    ExperimentRecommender,
    ExperimentPlan,
    run_insight_loop,
    run_slm_research_loop,
    demo_insight,
    synthesize_insight,
    build_insight_report,
    CAVEATS,
)

# Also lazy-exported from top-level autocausal:
from autocausal import InsightSuite, InsightReport, run_insight_loop
```
