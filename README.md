![Logo](assets/logo.svg)

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
- **Physics loop** — analytic KPI dynamics (damped oscillator / drift-diffusion / linear ODE), physical insight grounding, `PhysicsCausalSuite.loop`
- **KPI ML loop** — SLM/Rule `ModelConstructPlan` → median/sklearn/**PyTorch MLP** impute → discover → FitReport ([docs/ML_KPI_LOOP.md](docs/ML_KPI_LOOP.md))
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
python -m autocausal auto --csv data.csv --physics --horizon 5
python -m autocausal ml loop --csv data.csv --text "what drives Y?"
python -m autocausal ml loop --csv data.csv --torch --guides rule
python -m autocausal ml fit-imputer --csv data.csv --backend median
python -m autocausal slm-status
python -m autocausal guides list
python -m autocausal auto --csv data.csv --slm
```

### PostgreSQL

```bash
pip install -e ".[postgres]"
python -m autocausal discover \
  --db "postgresql+psycopg2://user:pass@localhost:5432/mydb" \
  --table events
```

## Docs

- [KPI ML loop (SLM → PyTorch)](docs/ML_KPI_LOOP.md)
- [ML Model Hub proposals](../docs/AUTOCAUSAL_ML_MODEL_HUB_PROPOSALS.md)
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
