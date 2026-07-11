# AutoCausalLib — Library API map

Prefer importing from submodules; the top-level package lazy-re-exports common symbols.
CLI (`python -m autocausal`) is a thin wrapper — see [CLI.md](CLI.md).

> **Epistemic honesty:** discovery, mining, NLP hints, SLM text, estimate/refute, and agent loops are **exploratory**. They do not guarantee causal identification. Soft engines soft-skip when missing.

**Install (PyPI name ≠ import name):**

```bash
pip install auto-causal-lib          # PyPI distribution name
python -c "import autocausal; print(autocausal.__version__)"
```

Do **not** rely on `pip install autocausal` — that name was rejected / may resolve to a different project. Always install **`auto-causal-lib`**, then **`import autocausal`**.

Doc index: [INDEX.md](INDEX.md) · Modules: [MODULES.md](MODULES.md) · Backends: [CAUSAL_BACKENDS.md](CAUSAL_BACKENDS.md).

---

## Top-level exports (`import autocausal`)

| Symbol | Source |
|--------|--------|
| `AutoCausal`, `DiscoveryResult`, `AutoResult`, `__version__` | api / results |
| `create_from_context`, `infer_from_results`, `slm_status` | slm |
| `list_tools`, `validate_pipeline`, `refute` | suite_tools |
| `estimate`, `list_engines`, `engine_status` | engines |
| `list_guides`, `direct` | guides |
| `KPIMinedCausalLoop`, `ModelConstructPlan` | ml |
| `PublicCausalMiner`, `PublicCausalReport`, `mine_public` | public_causal |
| `TextCausalHints`, `NlpFeatureBuilder`, `extract_causal_hints_from_text` | nlp |
| `BehavioralTraceStore`, `mine_behavioral_traces` | behavioral |
| `InsightSuite`, `InsightReport`, `ExperimentRecommender`, `run_insight_loop` | insight |
| `load_dataset`, `list_datasets` | datasets |
| `validate_frame`, `QCReport` | qc |
| `align`, `PanelSpec` | join / panel |
| Suites / skilling / grail / agentic / AgentHook | suites, skilling, grail, agentic, connective |
| `doctor_report` | doctor |

---

## Core pipeline

```python
from autocausal import AutoCausal

ac = AutoCausal.from_csv("data.csv")   # or from_parquet / from_sqlalchemy / from_dataframe
result = ac.mine().impute().discover(qc="warn", use_iv=True)
print(ac.report())           # markdown from last DiscoveryResult
print(result.report())       # same via DiscoveryResult.report() alias
print(result.to_markdown())  # explicit
print(ac.result.to_json())
```

### Discovery

```python
ac.discover(stability=True, bootstrap_n=20)
ac.discover(ensemble=True, include_optional=True)  # appends installed soft backends
ac.discover(method="causal_learn_pc")              # soft; falls back to pc_lite
ac.discover(methods=["score_pc_lite", "lingam", "gcastle_notears"], min_methods=1)
ac.discover(method="mi_binned")  # cheap binned NMI (aliases: mi, mi_stub)
```

### GRAIL → discover

```python
ac.discover(qc="off")
report = ac.apply_grail("Does spend cause revenue?")  # focus + boost_edges merge
```

### Estimate / refute / engines

```python
ac.estimate(backend="builtin_ols")     # always
ac.estimate(backend="doubleml")        # soft-skip if missing
ac.estimate(backend="econml")
ac.refute(method="placebo")
ac.refute(method="dowhy")              # real DoWhy when installed
# Same methods on the discover() return value:
result = ac.discover()
result.estimate(backend="builtin_ols")
result.refute(method="placebo")
result.to_fabric_bundle()
result.sensitivity(n_boot=8)
result.to_causaliv_request()
result.engines_status()
from autocausal.engines import list_engines, engine_status, connectivity_map
# also: from autocausal import list_engines, engine_status
engine_status()
```

### QC / NLP / fabric / panel

```python
from autocausal.qc import validate_frame
ac.validate_qc(mode="warn")
ac.enrich_from_text("Does spend cause sales?")
ac.to_fabric_bundle()
result.to_fabric_bundle()   # same contracts from DiscoveryResult
result.to_causal_edges()
result.to_search_dag()
ac.to_causaliv_request()
ac.set_panel("unit_id", "year", outcome="y")
ac.panel_features(["y"], kind="lag")
```

### Orchestrated

```python
AutoCausal.auto("data.csv", text="…", use_slm=False)
ac.insight_loop(text="…")
ac.agentic_loop(text="…", max_rounds=2)
ac.physics_loop(horizon=5)
ac.ml_loop(text="…", use_torch=False)
```

---

## Module groups (summary)

| Group | Import path | See |
|-------|-------------|-----|
| Core tabular | `ingest`, `impute`, `mining`, `discovery`, `roles`, `iv`, `qc`, `join`, `panel` | [MODULES.md](MODULES.md) |
| Soft backends | `autocausal.backends.*`, `autocausal.engines` | [CAUSAL_BACKENDS.md](CAUSAL_BACKENDS.md) |
| Suites / skilling | `autocausal.suites`, `autocausal.skilling` | [SUITES.md](SUITES.md), [SLM_SKILLING.md](SLM_SKILLING.md) |
| Insight / agentic / GRAIL | `insight`, `agentic`, `grail` | area docs |
| MCP | `autocausal.mcp`, `autocausal.connective` | [MCP.md](MCP.md) |
| NLP / behavioral | `nlp`, `behavioral` | [NLP_AND_BEHAVIORAL_TRACES.md](NLP_AND_BEHAVIORAL_TRACES.md) |
| Physics / ML | `physics`, `ml` | [PHYSICS_DEMO.md](PHYSICS_DEMO.md), [ML_KPI_LOOP.md](ML_KPI_LOOP.md) |
| Data | `datasets`, `public_suite`, `public_causal` | [EXAMPLES.md](EXAMPLES.md) |

---

## Session persistence

`AutoCausal` is **in-memory** — the full DataFrame is not pickled in 0.11.x.

- `ac.session_snapshot()` — lightweight metadata (source, shape, columns, `n_edges`, methods) for doctor/MCP debugging
- MCP `SessionStore` — agent multi-step tool sessions
- Agentic `EpisodeStore` / `persist_dir` — loop JSONL persistence
- Full pickle session format is deferred past 0.11.x

---

## Extras

| Extra | Provides |
|-------|----------|
| *(none)* | numpy, pandas, sqlalchemy — full core + insight/mcp code/skilling/cli/engines |
| `causal-extra` | causal-learn, dowhy, DoubleML, econml, lingam, gcastle, sklearn |
| `mcp` | `mcp` SDK for stdio server (`AgentHook` works without it) |
| `nlp` / `slm` / `ml` / `ui` / `web` | NLTK, torch/transformers, sklearn, Streamlit, httpx |
| `all` | union of the above + DB drivers |

---

## Version

`autocausal.__version__` (0.11.4+). Roadmap: [ROADMAP.md](ROADMAP.md).
