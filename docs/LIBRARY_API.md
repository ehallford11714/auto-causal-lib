# AutoCausalLib — Library API map

Library-first public surface for **autocausal** 0.8+. Prefer importing from
submodules in apps/notebooks; the top-level package re-exports the most common
symbols. CLI (`python -m autocausal`) is a thin wrapper.

> Epistemic honesty: discovery, mining, NLP hints, and refute stubs are
> **exploratory**. They do not guarantee causal identification.

## Core

| Import | Role |
|--------|------|
| `from autocausal import AutoCausal` | Load → impute → discover → guide/direct |
| `from autocausal import DiscoveryResult, AutoResult` | Structured outputs |
| `AutoCausal.from_csv` / `from_parquet` / `from_sqlalchemy` / `from_dataframe` | Ingest |
| `ac.impute()` / `ac.mine()` / `ac.discover()` / `ac.run()` | Pipeline steps |
| `AutoCausal.auto(path, …)` | Orchestrated load→mine→impute→discover→guide→ground→sensitivity |

### Discovery (0.8)

```python
ac.discover(stability=True, bootstrap_n=20)          # per-edge stability → honest confidence
ac.discover(ensemble=True)                           # pc_lite + corr_skeleton + mi_stub consensus
ac.discover_ensemble(methods=["score_pc_lite", "corr_skeleton", "mi_stub"])
```

### QC gate

```python
from autocausal.qc import validate_frame
report = validate_frame(df)          # QCReport
ac.validate_qc(mode="warn")          # or mode="block"
ac.discover(qc="warn")               # hooked before discover
```

### NLP → guide/direct

```python
ac.enrich_from_text("Randomized spend increases revenue")
ac.guide(text="…")   # auto-calls enrich_from_text
ac.direct(text="…")
```

## Fabric contracts

Aligned with `research/shared_contracts` (MineReport / CausalEdge / InsightPack).

```python
from autocausal.contracts import fabric_bundle
from autocausal.mining import mine

mr = mine(df).to_mine_report()                 # MineReport.v1
edges = ac.discover().to_causal_edges()        # list[CausalEdge.v1]
bundle = ac.to_fabric_bundle()                 # FabricBundle.v1
dag = ac.result.to_search_dag()                # SearchDAG.v1 (soft CausalSearch)
auto = AutoCausal.auto("data.csv")
bundle2 = auto.to_fabric_bundle()
```

## Panel / join / IV handoff

```python
from autocausal.panel import PanelSpec, panel_lag
from autocausal.join import align

ac.set_panel("unit_id", "year", treatment="t", outcome="y")
ac.panel_features(["y", "x"], kind="lag")

joined, report = align([df_a, df_b], keys=["id"], how="outer")
ac.join_frames(df_b, keys="id")

spec = ac.to_causaliv_request()   # CausalIVRequest.v1 soft dict
```

## Sensitivity & refute

```python
sens = ac.sensitivity(text="physics rollout")
ref = ac.refute(method="placebo")              # builtin
ref2 = ac.refute(method="dowhy")               # soft-skip if missing
from autocausal.suite_tools import refute
```

## SQL chunked / sampled

```python
ac = AutoCausal.from_sqlalchemy(
    "sqlite:///demo.db", table="events",
    chunksize=5000, sample_n=2000, sample_seed=0,
)
```

## NLP / behavioral / public / insight

| Module | Entry |
|--------|-------|
| `autocausal.nlp` | `TextCausalHints`, `extract_causal_hints_from_text`, `NlpFeatureBuilder` |
| `autocausal.behavioral` | `BehavioralTraceStore`, `mine_behavioral_traces` |
| `autocausal.public_suite` / `public_causal` | `load_public`, `mine_public` |
| `autocausal.insight` | `InsightSuite`, `run_insight_loop` |
| `autocausal.suites` | `AutoCleanseSuite`, `AutoEDASuite`, `AutoMineSuite`, `SLMAutoDirector` (SLM-directed; [SUITES.md](SUITES.md)) |
| `autocausal.datasets` | `load_dataset`, `list_datasets` (cache + soft network) |
| `autocausal.physics` | `PhysicsCausalSuite` |
| `autocausal.ml` | `KPIMinedCausalLoop` |

## Imputation diagnostics

`ImputationReport.mechanism_hint` ∈ `{none, MCAR_plausible, MAR_suspected, MNAR_possible, unknown}`
plus `mechanism_notes` / `diagnostics` (heuristic — not Little's MCAR).

## Version

See `autocausal.__version__` (0.9.0+). Roadmap: [ROADMAP.md](ROADMAP.md).
