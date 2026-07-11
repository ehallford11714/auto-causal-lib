# Auto suites — AutoCleanse / AutoEDA / AutoMine

Library-first suites under `autocausal.suites.*` with **dedicated action
modules**. Every **auto\*** path is **SLM-directed** via
`SLMAutoDirector` + `autocausal.skilling` ToolSurface when available;
deterministic rules always work offline.

> Epistemic honesty: suite reports are hygiene / exploration aids. They do
> **not** identify causal effects. SLM text/tools are generative assistance.

## Module tree

```
autocausal/suites/
  __init__.py
  director.py              # SLMAutoDirector
  action_protocol.py
  base.py
  autocleanse/
    __init__.py
    suite.py               # AutoCleanseSuite
    actions.py             # CleanseActions + CLEANSE_REGISTRY
    report.py
  autoeda/
    __init__.py
    suite.py
    actions.py             # EDAActions
    report.py
  automine/
    __init__.py
    suite.py
    actions.py             # MineActions
    report.py

autocausal/skilling/       # structured SLM tools wrapping suite actions
  surface.py / registry.py / broker.py / catalog.py / trace.py
```

## Public API (preferred)

```python
from autocausal.suites.autocleanse import AutoCleanseSuite, CleanseActions
from autocausal.suites.autoeda import AutoEDASuite, EDAActions
from autocausal.suites.automine import AutoMineSuite, MineActions
from autocausal import AutoCausal

# Direct actions
CleanseActions.impute(df, method="auto")
EDAActions.suggest_roles(df)
MineActions.mine_associations(df)
print(CleanseActions.list())

# SLM-directed suites
clean = AutoCleanseSuite(df, use_slm=True).run()
eda = AutoEDASuite(clean.frame, use_slm=True).run()
mine = AutoMineSuite(clean.frame, use_slm=True).run()

ac = AutoCausal.from_dataframe(df).cleanse().eda().automine()
```

## Dedicated actions

### AutoCleanse (`CleanseActions`)

| Action | Role |
|--------|------|
| `profile_missingness` | Missingness profile (read-only) |
| `coerce_types` | Object → numeric/datetime |
| `drop_duplicates` | Exact duplicate rows |
| `drop_high_null_cols` | Drop near-all-null columns |
| `drop_constant_cols` | Drop constant columns |
| `flag_outliers` | Z-score flag / winsorize |
| `impute` | Via `autocausal.impute` |
| `strip_id_leakage` | ID / leakage-name flags |
| `qc_snapshot` | `validate_frame` |

### AutoEDA (`EDAActions`)

| Action | Role |
|--------|------|
| `summarize_distributions` | Numeric summaries |
| `correlation_matrix` | Pairwise corr |
| `cardinality_report` | nunique / missing |
| `suggest_roles` | X/Y/Z/W hypotheses |
| `qc_snapshot` | QC gate |
| `leakage_hints` | Name + corr leakage |
| `mining_profile` | Soft mining profile |

### AutoMine (`MineActions`)

| Action | Role |
|--------|------|
| `mine_associations` | `autocausal.mining.mine` |
| `mine_kpi_hints` | KPI-like columns |
| `join_public_sources` | Optional public join |
| `mine_behavioral` | Soft behavioral |
| `rank_candidates` | Rank associations |
| `to_mine_report` | Fabric MineReport.v1 |

## SLM skilling

See [SLM_SKILLING.md](SLM_SKILLING.md). Tools are named `autocleanse.impute`,
`autoeda.suggest_roles`, … and bundled into skills
`skill:autocleanse` / `skill:autoeda` / `skill:automine` /
`skill:autocausal_loop`.

## CLI (thin)

```bash
python -m autocausal suite cleanse --csv data.csv --no-slm -o cleanse.md
python -m autocausal suite eda --csv data.csv -o eda.md
python -m autocausal suite mine --csv data.csv --format json -o mine.json
python -m autocausal skilling list
```

## Env

| Variable | Effect |
|----------|--------|
| `AUTOCAUSAL_SLM=1` | Prefer HuggingFace SLM director / tool selection |
| `AUTOCAUSAL_SLM_MODEL` | Model id |
