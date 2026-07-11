![AutoCausal](assets/autocausal-logo.png)

# AutoCausalLib (`auto-causal-lib` / `import autocausal`)

Automatically **impute** missing tabular fields and discover *exploratory* causal relationships from CSV / Parquet and SQL databases - with optional **SLM-aided creation/inference**, a shared **tool suite**, and a **physics predictive / autocausal loop** for physical insight grounding.

> Scope is intentionally small: impute → role inference → PC-lite / score edges → optional IV → optional physics rollout.  
> This is **not** a full AutoML OS and does **not** guarantee causal identification.

## Features

- Load from CSV, Parquet, or SQLAlchemy URLs (Postgres, Vertica, DuckDB, and more via extras)
- Auto-imputation (`median_mode`, `knn`, or `auto`) with strategy reporting
- Role inference (treatment / outcome / instrument / confounder candidates)
- Exploratory discovery: PC-lite + scored edges + optional 2SLS
- **Mining** - column profiles, associations, KPI hints
- **SLM** - `RuleBackend` always; optional HuggingFace for *creation* (questions/Z/morphemes) and *inference* (narrative/caveats)
- **Direction guides** - soft-optional `LLMIntent` / `retracement` / Kineteq pivot embeddings / **GRAIL** → `DirectionPlan` (see [docs/GUIDES.md](docs/GUIDES.md), [docs/GRAIL.md](docs/GRAIL.md))
- **GRAIL** (`autocausal.grail`) - embellished Kineteq Generative Reflective Agentic Imputation Loop; live MCP/module when configured, rich offline stub otherwise; MCP tools `autocausal_grail_*`
- **suite_tools** - registry of causal/NLP/KPI/validation adapters (NLTK, gensim, DoWhy stubs, …)
- **Physics loop** - analytic KPI dynamics (damped oscillator / drift-diffusion / linear ODE), physical insight grounding, `PhysicsCausalSuite.loop`, optional **Streamlit demo** (`physics ui`)
- **KPI ML loop** - SLM/Rule `ModelConstructPlan` → median/sklearn/**PyTorch MLP** impute → discover → FitReport ([docs/ML_KPI_LOOP.md](docs/ML_KPI_LOOP.md))
- **Isolates causal** - soft bridge to IntentIsolates layer motifs → indication vs IV ([docs/LAYER_CAUSAL_IV.md](docs/LAYER_CAUSAL_IV.md))
- **NLP library** (`autocausal.nlp`) - soft-optional NLTK tokenize/POS/sentiment, `TextCausalHints`, `NlpFeatureBuilder` for apps/notebooks ([docs/NLP_AND_BEHAVIORAL_TRACES.md](docs/NLP_AND_BEHAVIORAL_TRACES.md))
- **Behavioral traces** (`autocausal.behavioral`) - habit/nudge/reinforcement demos → panel → mine/discover ([docs/NLP_AND_BEHAVIORAL_TRACES.md](docs/NLP_AND_BEHAVIORAL_TRACES.md))
- **Public causal mining** - multi-source join of bundled/open datasets → mine → discover → report ([docs/PUBLIC_CAUSAL_MINING.md](docs/PUBLIC_CAUSAL_MINING.md))
- **Insight suite** (`autocausal.insight`) - `InsightReport` + optional SLM; **closed research loop** recommends experiments and mines further (`run_loop` / `ExperimentRecommender`) ([docs/INSIGHT_SUITE.md](docs/INSIGHT_SUITE.md))
- **Auto suites** (`autocausal.suites`) - **SLM-directed** `AutoCleanseSuite` / `AutoEDASuite` / `AutoMineSuite` with dedicated action modules + `autocausal.skilling` tool surface ([docs/SUITES.md](docs/SUITES.md), [docs/SLM_SKILLING.md](docs/SLM_SKILLING.md))
- **MCP connective** (`autocausal.mcp` / `autocausal.connective`) - Model Context Protocol stdio server + in-process `AgentHook` so other agents can load/cleanse/mine/discover/report ([docs/MCP.md](docs/MCP.md))
- **Agentic loop** (`autocausal.agentic`) - SLM-guided cyclic FSM + **LangGraph chain** (`run_slm_langgraph_loop` / `ac.slm_loop`): hypothesize → skill → validate → compact → insight → route ([docs/AGENTIC_LOOP.md](docs/AGENTIC_LOOP.md))
- **Local Qwen** - `ensure_local_qwen()` / `python -m autocausal slm setup-qwen` probes hardware and caches a fitting Instruct model
- **Fabric contracts** - `to_mine_report` / `to_causal_edges` / `to_fabric_bundle` / `to_search_dag` aligned with shared Causal Fabric schemas ([docs/LIBRARY_API.md](docs/LIBRARY_API.md))
- **Discovery stability & ensemble** - bootstrap per-edge stability (honest confidence); multi-method consensus (`pc_lite` + `corr_skeleton` + `mi_stub`)
- **QC gate** - `autocausal.qc.validate_frame` before discover (ID leakage / bad keys)
- **Panel / join / IV handoff** - `PanelSpec`, `join.align`, `to_causaliv_request`, sensitivity + soft refute hooks; optional exploratory `auto_instrument`
- **Causal backends** - soft `causal-learn` / LiNGAM / gCastle discovery; DoubleML + EconML estimate; real DoWhy refute ([docs/CAUSAL_BACKENDS.md](docs/CAUSAL_BACKENDS.md))
- **Engines surface** - `autocausal.engines` list/status + CLI `engines` / `estimate` / `refute`; MCP `autocausal_list_engines` / `_estimate` / `_refute`
- Markdown / JSON reports and a CLI

## Install

> **PyPI name:** `auto-causal-lib` · **Import:** `autocausal`  
> `pip install autocausal` is **not** this project (name rejected as too similar to other packages). Always:
>
> ```bash
> pip install auto-causal-lib
> python -c "import autocausal; print(autocausal.__version__)"
> ```

From [PyPI](https://pypi.org/project/auto-causal-lib/):

```bash
pip install auto-causal-lib
# Base includes: numpy, pandas, sqlalchemy, scikit-learn, nltk, httpx, pyarrow,
# torch, transformers, accelerate, huggingface_hub, langgraph, langchain-core, mcp
pip install "auto-causal-lib[all]"   # + ui + causal-extra + drivers + bitsandbytes
```

Optional extras (still soft; core already has SLM/LangGraph/MCP):

```bash
pip install "auto-causal-lib[bitsandbytes]" # CUDA 4-bit Qwen
pip install "auto-causal-lib[ui]"           # Streamlit physics demo (+ plotly)
pip install "auto-causal-lib[causal-extra]" # causal-learn, DoWhy, DoubleML, EconML, lingam, gCastle
pip install "auto-causal-lib[postgres]"     # and other DB drivers - see docs/CONNECTIONS.md
```

```bash
python -m autocausal slm setup-qwen   # probe hardware + download recommended Qwen Instruct
python -m autocausal slm-loop --no-slm
python -m autocausal engines status
python -m autocausal insight --help
python -m autocausal.mcp
```

**Docs:** [docs/INDEX.md](docs/INDEX.md) (full map) · [docs/MODULES.md](docs/MODULES.md) · [docs/CLI.md](docs/CLI.md) · [docs/MCP.md](docs/MCP.md) · [docs/CAUSAL_BACKENDS.md](docs/CAUSAL_BACKENDS.md) · [docs/LIBRARY_API.md](docs/LIBRARY_API.md).

From source (development):

```bash
cd research/AutoCausalLib
pip install -e ".[dev]"
```

Env:

| Variable | Effect |
|----------|--------|
| `AUTOCAUSAL_SLM=1` | Prefer HuggingFace SLM |
| `AUTOCAUSAL_SLM_MODEL` | Model id (set by `slm setup-qwen`; else tiny-gpt2 for tests) |
| `AUTOCAUSAL_SLM_4BIT=1` | Prefer bitsandbytes 4-bit when CUDA available |
| `AUTOCAUSAL_TEST_QWEN=1` | Enable slow Qwen pytest markers |
| `AUTOCAUSAL_TORCH=1` | Prefer PyTorch MLP imputer/predictor when installed |
| `AUTOCAUSAL_TORCH_TEST=1` | Enable gated torch unit tests |
| `AUTOCAUSAL_LLMINTENT_MODEL` | Optional LLMIntent heavy analyzer model |
| `AUTOCAUSAL_KINETEQ_MCP=1` + `KINETEQ_MCP_URL` | Live Kineteq MCP pivot embeddings / GRAIL |
| `AUTOCAUSAL_GRAIL_MCP=1` | Also enables live GRAIL MCP calls |

Recommended local Instruct: `python -m autocausal slm setup-qwen` (CPU ≈ `Qwen2.5-1.5B-Instruct` / `0.5B`; larger VRAM → 3B/7B).

## Quick start

```python
from autocausal import AutoCausal

ac = AutoCausal.from_csv("data.csv")
result = ac.run()          # impute + discover
print(ac.report())         # markdown (AutoCausal.report)
print(result.report())     # same via DiscoveryResult.report()
print(result.to_json())    # graph + edges + candidates

# Engines connectivity (list / status / estimate / refute)
from autocausal.engines import list_engines, engine_status
print(engine_status()["n"], "engines")

# 0.8: stability, QC, fabric, NLP→guide
ac.enrich_from_text("Does spend cause sales?")
result = ac.discover(stability=True, bootstrap_n=12, ensemble=True)
print(ac.to_fabric_bundle()["schema"])      # FabricBundle.v1
print(result.to_fabric_bundle()["schema"])  # same via DiscoveryResult
est = result.estimate(backend="builtin_ols")  # also: ac.estimate(...)
ref = result.refute(method="placebo")         # also: ac.refute(...)
print(ac.to_causaliv_request()["schema"])

# SLM/rule creation + inference
print(ac.create(text="Does spend cause sales?").to_markdown())
print(ac.interpret().to_markdown())

# Tool suite validation
print(ac.validate_tools(y="y", d="d", z="z").to_markdown())

# Direction steering (LLMIntent / retracement / Kineteq pivots / GRAIL - soft-optional)
plan = ac.direct(
    text="Does spend cause revenue?",
    backends=["llmintent", "retracement", "kineteq_pivot", "grail", "rule"],
)
print(plan.to_markdown())

# GRAIL reflective loop (offline stub unless Kineteq MCP configured)
from autocausal.grail import GrailEngine
report = GrailEngine().run("Does spend cause revenue?", context={"text": "Does spend cause revenue?"})
print(report.to_markdown())

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

Bundled Iris, Wine, Titanic, Gapminder subset, Diabetes, and California housing sample - no network required. Licenses/attribution: [DATASETS.md](DATASETS.md). Full walkthrough: [docs/EXAMPLES.md](docs/EXAMPLES.md).

```python
from autocausal import AutoCausal, load_dataset

df = load_dataset("iris")  # Fisher Iris CSV from package data
ac = AutoCausal(df)
ac.mine().impute().discover(use_iv=False, min_abs_corr=0.2)
print(ac.report())  # exploratory edges - not scientific flower-causation claims
```

```bash
python examples/iris_causal.py
python examples/iris_causal.py --insight
python examples/multi_dataset_tour.py
python -m autocausal public load iris
python -m autocausal insight demo --dataset iris --no-slm
```

### NLP hints & behavioral traces (library-first)

These are **importable modules** for apps and notebooks - the CLI is optional.

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

### Auto suites - Cleanse / EDA / Mine (SLM-directed)

Every **auto\*** path is directed by `SLMAutoDirector` when available; rules always work offline.

```python
from autocausal import AutoCausal, AutoCleanseSuite, AutoEDASuite, AutoMineSuite

clean = AutoCleanseSuite(df, use_slm=True).run()
eda = AutoEDASuite(clean.frame, use_slm=True).run()
mine = AutoMineSuite(clean.frame, use_slm=True).run()

ac = AutoCausal.from_dataframe(df).cleanse().eda().automine().discover()
# or: AutoCausal.auto("data.csv", use_slm=True, cleanse=True)
```

```bash
python -m autocausal suite cleanse --csv data.csv --no-slm -o cleanse.md
python -m autocausal suite eda --csv data.csv -o eda.md
python -m autocausal suite mine --csv data.csv --format json -o mine.json
```

See [docs/SUITES.md](docs/SUITES.md) and [docs/SLM_SKILLING.md](docs/SLM_SKILLING.md).

### Agentic causal loop (library-first)

SLM-guided cyclic research loop with compaction + constant-budget memory (SOTA-inspired APIs - not paper clones):

```python
from autocausal import AutoCausal, load_dataset
from autocausal.agentic import AgenticCausalLoop, run_agentic_loop

df = load_dataset("iris")
report = run_agentic_loop(df, text="petal drivers", max_rounds=2, use_slm=False)
print(report.to_markdown())

# Or: AutoCausal(...).agentic_loop(...) / MCP tool autocausal_agentic_loop
```

See [docs/AGENTIC_LOOP.md](docs/AGENTIC_LOOP.md).

### Use from other agents (MCP)

Expose AutoCausal as MCP tools for Cursor, Claude Desktop, and other MCP clients - or call the same surface in-process via `AgentHook` (no `mcp` SDK required).

```bash
pip install -e ".[mcp]"
python -m autocausal.mcp          # stdio server
# or: autocausal-mcp
```

Cursor / Claude Desktop stdio config:

```json
{
  "mcpServers": {
    "autocausal": {
      "command": "python",
      "args": ["-m", "autocausal.mcp"],
      "env": { "PYTHONUNBUFFERED": "1" }
    }
  }
}
```

Library-first (scripts / non-MCP agents):

```python
from autocausal.connective import AgentHook

hook = AgentHook()
hook.call_tool("autocausal_load_dataset", {"dataset_id": "iris"})
hook.call_tool("autocausal_discover", {"use_iv": False})
print(hook.call_tool("autocausal_report", {"format": "markdown"})["markdown"][:400])
```

Tools include `autocausal_load_dataset`, `autocausal_from_csv`, `autocausal_cleanse` / `eda` / `mine`, `autocausal_discover`, `autocausal_insight_loop`, `autocausal_recommend_experiments`, `autocausal_public_mine`, `autocausal_report`, `autocausal_list_datasets`, `autocausal_skilling_list`. Soft-fail if optional suites are missing. Full setup: [docs/MCP.md](docs/MCP.md).

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

### Guiding direction with LLMIntent / Retracement / Kineteq pivots / GRAIL

Backends are **soft-optional**: missing packages soft-fail to stubs/fallbacks and never break core discovery.

```bash
python -m autocausal guides list
python -m autocausal auto --csv data.csv --text "what causes revenue?" \
  --guides llmintent,retracement,kineteq_pivot,grail
python -m autocausal direct --csv data.csv --text "..." --guides grail,rule
```

See [docs/GUIDES.md](docs/GUIDES.md) and [docs/GRAIL.md](docs/GRAIL.md).

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

Interactive UI for the physics autocausal loop (trajectory charts, edges, physical insights, energy proxies). Soft-optional - core mine/discover/ml loops do not import Streamlit.

```bash
pip install -e ".[ui]"
python -m autocausal physics ui --port 8518
# or: streamlit run src/autocausal/apps/physics_streamlit.py --server.port 8518
```

See [docs/PHYSICS_DEMO.md](docs/PHYSICS_DEMO.md). **Caveat:** exploratory dynamics only - not true physics ID.

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
- **SLM text is assistance only.** Narratives, experiment suggestions, and role hints from rules/HF models are generative - not statistical proof.
- **IV / 2SLS paths are soft.** Optional instruments need human design review (relevance, exclusion); lite F-stats are not a substitute.
- **Bundled public/behavioral tables** include MIT synthetic fixtures *and* real educational CSVs (Iris, etc.) - see [DATASETS.md](DATASETS.md). Exploratory edges on Iris are illustrative, not flower-causation science.
- **Physics / KPI loops** are predictive rollouts and grounding aids - not true physical or causal identification.

Reports (InsightReport, PublicCausalReport, markdown CLI output) repeat these caveats; keep them in downstream apps.

## Docs

- [Library API map (0.8+)](docs/LIBRARY_API.md)
- [Roadmap (P1-P3 shipped)](docs/ROADMAP.md)
- [Examples (Iris + real datasets)](docs/EXAMPLES.md)
- [Dataset licenses & paths](DATASETS.md)
- [Insight suite (library API + optional SLM)](docs/INSIGHT_SUITE.md)
- [Auto suites - Cleanse / EDA / Mine (SLM-directed)](docs/SUITES.md)
- [SLM skilling / tool surface](docs/SLM_SKILLING.md)
- [MCP connective (agents / Cursor / Claude)](docs/MCP.md)
- [Agentic causal loop (compact + memory)](docs/AGENTIC_LOOP.md)
- [Public causal mining (multi-source join)](docs/PUBLIC_CAUSAL_MINING.md)
- [NLP & behavioral traces (library API)](docs/NLP_AND_BEHAVIORAL_TRACES.md)
- [KPI ML loop (SLM → PyTorch)](docs/ML_KPI_LOOP.md)
- [ML Model Hub proposals](../docs/AUTOCAUSAL_ML_MODEL_HUB_PROPOSALS.md)
- [Physics Streamlit demo](docs/PHYSICS_DEMO.md)
- [Physics world models + autocausal loop (SOTA)](docs/SOTA_PHYSICS_WORLD_MODEL_AUTOCAUSAL.md)
- [Direction guides](docs/GUIDES.md)
- [GRAIL (Kineteq adaptation)](docs/GRAIL.md)
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
