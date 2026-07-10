![AutoCausal](assets/autocausal-logo.png)

# AutoCausalLib (`autocausal`)

Automatically **impute** missing tabular fields and discover *exploratory* causal relationships from CSV / Parquet and SQL databases — with optional **SLM-aided creation/inference**, a shared **tool suite**, and a **physics predictive / autocausal loop** for physical insight grounding.

> Scope is intentionally small: impute → role inference → PC-lite / score edges → optional IV → optional physics rollout.  
> This is **not** a full AutoML OS and does **not** guarantee causal identification.

## Features

- Load from CSV, Parquet, or SQLAlchemy URLs (Postgres, Vertica, DuckDB, and more via extras)
- Auto-imputation (`median_mode`, `knn`, or `auto`) with strategy reporting
- Role inference (treatment / outcome / instrument / confounder candidates)
- Exploratory discovery: PC-lite + scored edges + optional 2SLS
- **Mining** — column profiles, associations, KPI hints
- **SLM** — `RuleBackend` always; optional HuggingFace for *creation* (questions/Z/morphemes) and *inference* (narrative/caveats)
- **Direction guides** — soft-optional `LLMIntent` / `retracement` / Kineteq pivot embeddings → `DirectionPlan` (see [docs/GUIDES.md](docs/GUIDES.md))
- **suite_tools** — registry of causal/NLP/KPI/validation adapters (NLTK, gensim, DoWhy stubs, …)
- **Physics loop** — analytic KPI dynamics (damped oscillator / drift-diffusion / linear ODE), physical insight grounding, `PhysicsCausalSuite.loop`, optional **Streamlit demo** (`physics ui`)
- **KPI ML loop** — SLM/Rule `ModelConstructPlan` → median/sklearn/**PyTorch MLP** impute → discover → FitReport ([docs/ML_KPI_LOOP.md](docs/ML_KPI_LOOP.md))
- **Isolates causal** — soft bridge to IntentIsolates layer motifs → indication vs IV ([docs/LAYER_CAUSAL_IV.md](docs/LAYER_CAUSAL_IV.md))
- **NLP library** (`autocausal.nlp`) — soft-optional NLTK tokenize/POS/sentiment, `TextCausalHints`, `NlpFeatureBuilder` for apps/notebooks ([docs/NLP_AND_BEHAVIORAL_TRACES.md](docs/NLP_AND_BEHAVIORAL_TRACES.md))
- **Behavioral traces** (`autocausal.behavioral`) — habit/nudge/reinforcement demos → panel → mine/discover ([docs/NLP_AND_BEHAVIORAL_TRACES.md](docs/NLP_AND_BEHAVIORAL_TRACES.md))
- **Public causal mining** — multi-source join of bundled/open datasets → mine → discover → report ([docs/PUBLIC_CAUSAL_MINING.md](docs/PUBLIC_CAUSAL_MINING.md))
- **Insight suite** (`autocausal.insight`) — `InsightReport` + optional SLM; **closed research loop** recommends experiments and mines further (`run_loop` / `ExperimentRecommender`) ([docs/INSIGHT_SUITE.md](docs/INSIGHT_SUITE.md))
- **Fabric contracts** — `to_mine_report` / `to_causal_edges` / `to_fabric_bundle` / `to_search_dag` aligned with shared Causal Fabric schemas ([docs/LIBRARY_API.md](docs/LIBRARY_API.md))
- **Discovery stability & ensemble** — bootstrap per-edge stability (honest confidence); multi-method consensus (`pc_lite` + `corr_skeleton` + `mi_stub`)
- **QC gate** — `autocausal.qc.validate_frame` before discover (ID leakage / bad keys)
- **Panel / join / IV handoff** — `PanelSpec`, `join.align`, `to_causaliv_request`, sensitivity + soft refute hooks
- Markdown / JSON reports and a CLI

## Install

```bash
cd research/AutoCausalLib
pip install -e ".[dev]"

# Optional
pip install -e ".[slm]"          # torch + transformers (lazy load)
pip install -e ".[torch]"        # torch only (KPI MLP imputer)
pip install -e ".[ml]"           # torch + scikit-learn
pip install -e ".[nlp]"          # nltk + gensim
pip install -e ".[ui]"           # Streamlit physics demo (+ plotly)
pip install -e ".[streamlit]"    # alias for [ui]
pip install -e ".[postgres]"
pip install -e ".[vertica]"
pip install -e ".[mysql,duckdb,parquet]"
```

Env:

| Variable | Effect |
|----------|--------|
| `AUTOCAUSAL_SLM=1` | Prefer HuggingFace SLM |
| `AUTOCAUSAL_SLM_MODEL` | Model id (default `sshleifer/tiny-gpt2` for tests) |
| `AUTOCAUSAL_TORCH=1` | Prefer PyTorch MLP imputer/predictor when installed |
| `AUTOCAUSAL_TORCH_TEST=1` | Enable gated torch unit tests |
| `AUTOCAUSAL_LLMINTENT_MODEL` | Optional LLMIntent heavy analyzer model |
| `AUTOCAUSAL_KINETEQ_MCP=1` + `KINETEQ_MCP_URL` | Live Kineteq MCP pivot embeddings |

Better instruct SLMs (document only): `Qwen/Qwen2.5-0.5B-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct`, `microsoft/Phi-3-mini-4k-instruct`.

Core deps: `numpy`, `pandas`, `sqlalchemy`. See [docs/CONNECTIONS.md](docs/CONNECTIONS.md). Optional path deps: `pip install -e ../LLMIntent`.

## Quick start

```python
from autocausal import AutoCausal

ac = AutoCausal.from_csv("data.csv")
result = ac.run()          # impute + discover
print(ac.report())         # markdown
print(result.to_json())    # graph + edges + candidates

# 0.8: stability, QC, fabric, NLP→guide
ac.enrich_from_text("Does spend cause sales?")
result = ac.discover(stability=True, bootstrap_n=12, ensemble=True)
print(ac.to_fabric_bundle()["schema"])  # FabricBundle.v1
print(ac.to_causaliv_request()["schema"])

# SLM/rule creation + inference
print(ac.create(text="Does spend cause sales?").to_markdown())
print(ac.interpret().to_markdown())

# Tool suite validation
print(ac.validate_tools(y="y", d="d", z="z").to_markdown())

# Direction steering (LLMIntent / retracement / Kineteq pivots — soft-optional)
plan = ac.direct(
    text="Does spend cause revenue?",
    backends=["llmintent", "retracement", "kineteq_pivot", "rule"],
)
print(plan.to_markdown())

# Physics predictive / autocausal loop
from autocausal.physics import PhysicsCausalSuite

suite = PhysicsCausalSuite.from_csv("data.csv")
phys = suite.loop(horizon=5, text="what drives outcome?")
print(phys.to_markdown())
# Or: ac.physics_loop(horizon=5) / AutoCausal.auto(..., physics=True)

# KPI-mined ML loop (SLM/Rule constructs torch vs median imputer)
from autocausal.ml import KPIMinedCausalLoop

ml = KPIMinedCausalLoop.from_csv("data.csv").run(
    text="what drives Y?", use_slm=False, use_torch=True, horizon=5
)
print(ml.plan.to_markdown())
print(ml.fit.to_markdown())
```

### Real dataset examples (offline)

Bundled Iris, Wine, Titanic, Gapminder subset, Diabetes, and California housing sample — no network required. Licenses/attribution: [DATASETS.md](DATASETS.md). Full walkthrough: [docs/EXAMPLES.md](docs/EXAMPLES.md).

```python
from autocausal import AutoCausal, load_dataset

df = load_dataset("iris")  # Fisher Iris CSV from package data
ac = AutoCausal(df)
ac.mine().impute().discover(use_iv=False, min_abs_corr=0.2)
print(ac.report())  # exploratory edges — not scientific flower-causation claims
```

```bash
python examples/iris_causal.py
python examples/iris_causal.py --insight
python examples/multi_dataset_tour.py
python -m autocausal public load iris
python -m autocausal insight demo --dataset iris --no-slm
```

### NLP hints & behavioral traces (library-first)

These are **importable modules** for apps and notebooks — the CLI is optional.

```python
from autocausal.nlp import extract_causal_hints_from_text, NlpFeatureBuilder
from autocausal.behavioral import BehavioralTraceStore, mine_behavioral_traces

hints = extract_causal_hints_from_text(
    "Randomized treatment leads to higher revenue, associated with age."
)
print(hints.roles.to_dict())          # treatment / outcome / confounder / instrument cues
print(hints.to_guide_context())       # feed guide/discover

features = NlpFeatureBuilder().transform(["because spend increases sales"])

result = mine_behavioral_traces("habit_loop", discover=True)
print(result.report.to_markdown())    # hypothesized stimulus→response / habit→outcome edges
```

See [docs/NLP_AND_BEHAVIORAL_TRACES.md](docs/NLP_AND_BEHAVIORAL_TRACES.md) (Python API first, CLI secondary).


### Public causal mining (library-first)

Join bundled/open demo sources, mine associations, and run exploratory discovery:

```python
from autocausal import AutoCausal, PublicCausalMiner, mine_public

report = AutoCausal.mine_public(
    ["finance_demo", "demographics_demo", "health_demo"],
    join_on="region",
    discover=True,
    use_iv=True,
)
print(report.to_markdown())

# Explicit miner
miner = PublicCausalMiner(["marketing_demo", "instruments_demo", "demographics_demo"])
report = miner.run(discover=True, validate=True)

# Convenience
report = mine_public(["finance_demo", "climate_demo"], discover=True)
```

```bash
python -m autocausal public list --offline
python -m autocausal public mine --sources finance_demo,demographics_demo --discover
python -m autocausal public causal --sources finance_demo,demographics_demo,health_demo -o report.md
```

See [docs/PUBLIC_CAUSAL_MINING.md](docs/PUBLIC_CAUSAL_MINING.md).

### Insight suite (library-first)

```python
from autocausal.insight import InsightSuite, ExperimentRecommender, run_insight_loop, demo_insight

report = run_insight_loop("data.csv", text="what drives revenue?", use_slm=False)
report.write("insight.md")

# Closed loop: mine → guide/SLM → recommend experiments → join/remine → rediscover
suite = InsightSuite(use_slm=False)
report = suite.run_loop(
    "data.csv", max_rounds=3, join_sources=["demographics_demo", "instruments_demo"]
)
print(report.experiments_recommended[:3])
print(report.round_history)

# From a pre-built AutoCausal
from autocausal import AutoCausal
ac = AutoCausal.from_csv("data.csv")
report = InsightSuite.from_autocausal(ac).run(use_slm=False)
```

```bash
python -m autocausal insight run --csv data.csv --no-slm -o report.md
python -m autocausal insight loop --csv data.csv --rounds 3 --no-slm -o loop.md
python -m autocausal insight demo
python -m autocausal insight demo --dataset iris --no-slm
```

See [docs/INSIGHT_SUITE.md](docs/INSIGHT_SUITE.md) and [docs/EXAMPLES.md](docs/EXAMPLES.md).

### Guiding direction with LLMIntent / Retracement / Kineteq pivots

Backends are **soft-optional**: missing packages soft-fail to stubs/fallbacks and never break core discovery.

```bash
python -m autocausal guides list
python -m autocausal auto --csv data.csv --text "what causes revenue?" \
  --guides llmintent,retracement,kineteq_pivot
python -m autocausal direct --csv data.csv --text "..." --guides llmintent
```

See [docs/GUIDES.md](docs/GUIDES.md) for install paths, env vars, and `DirectionPlan` shape.

```bash
python -m autocausal discover --csv data.csv
python -m autocausal create --csv data.csv --text "lottery assignment"
python -m autocausal infer --csv data.csv
python -m autocausal tools list
python -m autocausal tools validate --csv data.csv --y y --d d --z z
python -m autocausal physics loop --csv data.csv --horizon 5 --text "what drives outcome?"
python -m autocausal physics rollout --csv data.csv --horizon 5
python -m autocausal physics ui --port 8518
python -m autocausal auto --csv data.csv --physics --horizon 5
python -m autocausal ml loop --csv data.csv --text "what drives Y?"
python -m autocausal ml loop --csv data.csv --torch --guides rule
python -m autocausal ml fit-imputer --csv data.csv --backend median
python -m autocausal nlp extract --text "treatment leads to revenue"
python -m autocausal behavioral list
python -m autocausal behavioral mine --demo habit_loop --discover
python -m autocausal public list --offline
python -m autocausal public mine --sources finance_demo,demographics_demo --discover
python -m autocausal public causal --sources finance_demo,demographics_demo,health_demo -o report.md
python -m autocausal slm-status
python -m autocausal guides list
python -m autocausal auto --csv data.csv --slm
```

### Physics Streamlit demo

Interactive UI for the physics autocausal loop (trajectory charts, edges, physical insights, energy proxies). Soft-optional — core mine/discover/ml loops do not import Streamlit.

```bash
pip install -e ".[ui]"
python -m autocausal physics ui --port 8518
# or: streamlit run src/autocausal/apps/physics_streamlit.py --server.port 8518
```

See [docs/PHYSICS_DEMO.md](docs/PHYSICS_DEMO.md). **Caveat:** exploratory dynamics only — not true physics ID.

### PostgreSQL

```bash
pip install -e ".[postgres]"
python -m autocausal discover \
  --db "postgresql+psycopg2://user:pass@localhost:5432/mydb" \
  --table events
```


## Epistemic caveats

AutoCausalLib is an **exploratory** toolkit. Please read these before treating outputs as science:

- **Discovery ≠ identification.** PC-lite / scored edges are *candidate* relationships, not proven causal effects.
- **Imputation and joins change the sample.** Missing-data fills and multi-source joins can invent associations (including ecological fallacy on region aggregates).
- **SLM text is assistance only.** Narratives, experiment suggestions, and role hints from rules/HF models are generative — not statistical proof.
- **IV / 2SLS paths are soft.** Optional instruments need human design review (relevance, exclusion); lite F-stats are not a substitute.
- **Bundled public/behavioral tables** include MIT synthetic fixtures *and* real educational CSVs (Iris, etc.) — see [DATASETS.md](DATASETS.md). Exploratory edges on Iris are illustrative, not flower-causation science.
- **Physics / KPI loops** are predictive rollouts and grounding aids — not true physical or causal identification.

Reports (InsightReport, PublicCausalReport, markdown CLI output) repeat these caveats; keep them in downstream apps.

## Docs

- [Library API map (0.8+)](docs/LIBRARY_API.md)
- [Roadmap (P1–P3 shipped)](docs/ROADMAP.md)
- [Examples (Iris + real datasets)](docs/EXAMPLES.md)
- [Dataset licenses & paths](DATASETS.md)
- [Insight suite (library API + optional SLM)](docs/INSIGHT_SUITE.md)
- [Public causal mining (multi-source join)](docs/PUBLIC_CAUSAL_MINING.md)
- [NLP & behavioral traces (library API)](docs/NLP_AND_BEHAVIORAL_TRACES.md)
- [KPI ML loop (SLM → PyTorch)](docs/ML_KPI_LOOP.md)
- [ML Model Hub proposals](../docs/AUTOCAUSAL_ML_MODEL_HUB_PROPOSALS.md)
- [Physics Streamlit demo](docs/PHYSICS_DEMO.md)
- [Physics world models + autocausal loop (SOTA)](docs/SOTA_PHYSICS_WORLD_MODEL_AUTOCAUSAL.md)
- [Direction guides](docs/GUIDES.md)
- [Tool suite registry](docs/SUITE_TOOLS.md)
- [Connection matrix & pip extras](docs/CONNECTIONS.md)
- [SOTA context (PC / GES / NOTEARS, imputation)](docs/SOTA.md)

## Related suite

| Project | Role |
|---------|------|
| [EmotiveVision](https://github.com/ehallford11714/emotivevision) | Emotion/intent streams → autocausal frames |
| [CausalIVSuite](https://github.com/ehallford11714/causal-iv-suite) | IV / DiD / AutoML causal suite |
| [CausalSearch](https://github.com/ehallford11714/causal-search) | Causal evidence search & DAG infill |
| [CausalBridge](https://github.com/ehallford11714/causal-bridge) | Control plane (status shows SLM/tools) |
| [NextFrameSeq](https://github.com/ehallford11714/next-frame-seq) | Vision / next-frame prediction |

## License

MIT
