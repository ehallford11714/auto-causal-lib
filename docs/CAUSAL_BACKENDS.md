# Causal backends (soft-optional)

AutoCausalLib keeps a **numpy/pandas core** and soft-adapts heavy causal libraries.
Install extras when you need them:

```bash
pip install "auto-causal-lib[causal-extra]"
# pulls: causal-learn, dowhy, DoubleML, econml, lingam, gcastle (+ sklearn)
```

> Epistemic honesty: discovery / estimate / refute outputs are **exploratory**.
> Soft engines do not guarantee causal identification.

## Pipeline

```
mine → discover(+causal-learn / lingam / gcastle)
     → estimate(DoubleML | EconML | builtin_ols)
     → refute(DoWhy | placebo)
```

## Discovery backends

| Method id | Package | Notes |
|-----------|---------|-------|
| `score_pc_lite` | builtin | Default PC-lite |
| `corr_skeleton` | builtin | Correlation threshold |
| `mi_stub` | builtin | Binned MI stub |
| `causal_learn_pc` | causal-learn | Soft |
| `causal_learn_ges` | causal-learn | Soft |
| `causal_learn_fci` | causal-learn | Soft (capped columns) |
| `lingam` / `direct_lingam` | lingam | Soft DirectLiNGAM |
| `gcastle_notears` | gCastle | Soft NOTEARS |

```python
from autocausal import AutoCausal

ac = AutoCausal.from_csv("data.csv")
ac.discover(method="causal_learn_pc", qc="off")          # soft-skip → pc_lite fallback
ac.discover(ensemble=True, include_optional=True)        # appends installed soft methods
ac.discover(methods=["score_pc_lite", "causal_learn_ges", "lingam"])
```

## Estimate backends

| Backend | Package | Effect |
|---------|---------|--------|
| `builtin_ols` | numpy | Exploratory OLS association |
| `builtin_2sls` | numpy | 2SLS (needs `z=`) |
| `doubleml` | DoubleML | PLR ATE |
| `econml` / `econml_linear_dml` | EconML | LinearDML CATE |
| `econml_causal_forest` | EconML | CausalForestDML |

```python
ac.mine().discover(qc="off")
est = ac.estimate(backend="doubleml")          # soft-skip if missing
est2 = ac.estimate(backend="econml", y="y", d="x")
from autocausal.engines import estimate
estimate(ac.df, backend="builtin_ols", y="y", d="x")
```

## Refute backends

| Method | Package |
|--------|---------|
| `placebo` | builtin |
| `random_common_cause` | builtin |
| `dowhy` / `dowhy_placebo` | DoWhy `placebo_treatment_refuter` |
| `dowhy_random_common_cause` | DoWhy |
| `dowhy_data_subset` | DoWhy |

```python
ac.refute(method="placebo")
ac.refute(method="dowhy")                 # real CausalModel.refute_estimate when installed
```

## Unified engines surface

```python
from autocausal.engines import list_engines, engine_status, connectivity_map

print(engine_status()["by_kind"]["discovery"])
print(connectivity_map())
```

### How insight / MCP / skilling / CLI reach engines

| Surface | Entry |
|---------|-------|
| Library | `autocausal.engines.*`, `AutoCausal.discover/estimate/refute` |
| CLI | `python -m autocausal engines status\|list` |
| CLI | `python -m autocausal estimate --csv … --backend doubleml` |
| CLI | `python -m autocausal refute --csv … --method dowhy` |
| CLI | `python -m autocausal insight …` / `skilling list` / `mcp --list-tools` |
| MCP | `autocausal_list_engines`, `autocausal_estimate`, `autocausal_refute`, `autocausal_discover`, `autocausal_insight_loop` |
| Connective | `AgentHook.call_tool("autocausal_list_engines", {})` |
| Skilling | suite ToolSurface + `suite_tools` causal adapters |
| Insight | `InsightSuite` / session `AutoCausal` estimate+refute |

## Package modules in the wheel

Always included (no heavy deps required):

- `autocausal.insight` — insight engine
- `autocausal.mcp` / `autocausal.connective` — MCP + AgentHook
- `autocausal.skilling` — SLM tool surface
- `autocausal.cli` / `__main__`
- `autocausal.backends` / `autocausal.engines`
- `autocausal.agentic`, `autocausal.grail`, `autocausal.suites`

Optional SDK: `pip install "auto-causal-lib[mcp]"` for the `mcp` package used by the stdio server.
