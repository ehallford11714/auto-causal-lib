# Auto suites — AutoCleanse / AutoEDA / AutoMine

Library-first suites under `autocausal.suites`. Every **auto\*** path is
**SLM-directed** when HuggingFace is available, with a deterministic rule
fallback that never hard-crashes offline.

> Epistemic honesty: suite reports are hygiene / exploration aids. They do
> **not** identify causal effects. SLM text is generative assistance only.

## Install / env

```bash
pip install -e ".[dev]"
pip install -e ".[slm]"   # optional torch + transformers
```

| Variable | Effect |
|----------|--------|
| `AUTOCAUSAL_SLM=1` | Prefer HuggingFace SLM director |
| `AUTOCAUSAL_SLM_MODEL` | Model id (default `sshleifer/tiny-gpt2` for tests) |

Default for suites and `AutoCausal.auto()`: **try SLM**, soft-fail to rules.
Pass `use_slm=False` to force the rule director.

## Public API (preferred)

```python
from autocausal import (
    AutoCausal,
    AutoCleanseSuite,
    AutoEDASuite,
    AutoMineSuite,
    SLMAutoDirector,
)

df = ...  # DataFrame, or pass a path / AutoCausal

# 1) Standalone suites
clean = AutoCleanseSuite(df, use_slm=True).run()
print(clean.report.to_markdown())
eda = AutoEDASuite(clean.frame, use_slm=True).run()
mine = AutoMineSuite(clean.frame, use_slm=True).run()
print(mine.report.to_mine_report()["schema"])  # MineReport.v1

# 2) Fluent AutoCausal chain (use_slm passed through)
ac = (
    AutoCausal.from_dataframe(df)
    .cleanse(use_slm=True)
    .eda(use_slm=True)
    .automine(use_slm=True)
)
result = ac.discover()
assert ac.cleanse_report is not None
assert ac.eda_report is not None
assert ac.mine_report is not None

# 3) Orchestrated auto() — cleanse + automine + discover (SLM by default)
auto = AutoCausal.auto("data.csv", use_slm=True, cleanse=True, eda=False)
```

Also: `from autocausal.suites import auto_cleanse, auto_eda, auto_mine`.

## How SLM directs each suite

Shared director: `autocausal.suites.director.SLMAutoDirector`.

| Suite | Director stage | What SLM/rules propose | What is applied |
|-------|----------------|------------------------|-----------------|
| **AutoCleanseSuite** | `cleanse` | drop / impute / coerce / outlier columns | Feasible hygiene ops + existing `impute_dataframe` + QC |
| **AutoEDASuite** | `eda` | focus columns, role hypotheses (X/Y/Z/W), analyses | Distributions, corr, cardinality, QC, leakage hints |
| **AutoMineSuite** | `mine` | KPI focus, join sources, association priority | `autocausal.mining.mine` + optional public join + soft DataMine |

Reports always include `slm_directives` (dict) and label generative text with
epistemic caveats. Backend is `rule` offline, or `huggingface:…` when SLM runs.

```python
from autocausal.suites import SLMAutoDirector

d = SLMAutoDirector(use_slm=True).direct("cleanse", df, text="prep for IV")
print(d.to_markdown())
print(d.drop_columns, d.impute_columns)
```

## Suite responsibilities

### AutoCleanseSuite

- Missingness profile, type coercion, duplicate/constant drops, outlier winsorize
- Optional impute via existing `autocausal.impute`
- Output: cleaned frame + `CleanseReport` (`to_dict` / `to_markdown` / `write`)
- Hook: `ac.cleanse()` or `AutoCleanseSuite(df).run().to_autocausal()`

### AutoEDASuite

- Distributions, correlations, cardinality, suggested roles, QC, leakage hints
- Soft hook to mining profiles; plots optional (skip without matplotlib)
- Output: `EDAReport`

### AutoMineSuite

- Wraps `autocausal.mining` + optional public joins
- Soft `datamine_adapter` if DataMineLib is on path
- Output: `MineReport` with Fabric `to_mine_report()`

## Insight / broader auto loop

- `AutoCausal.auto(..., use_slm=True)` — default try-SLM; runs cleanse → automine → discover → guide
- `InsightSuite(use_slm=True).run_loop(...)` — recommended; constructor still defaults `False` for 0.8 compat, but **auto\*** means SLM-directed when you opt in / set `AUTOCAUSAL_SLM=1`

## CLI (thin)

Prefer the library API. CLI mirrors it:

```bash
python -m autocausal suite cleanse --csv data.csv --no-slm -o cleanse.md
python -m autocausal suite eda --csv data.csv -o eda.md
python -m autocausal suite mine --csv data.csv --format json -o mine.json
```

## Soft siblings

Patterns inspired by (not hard-required):

- `research/CausalIVSuite` — `auto_cleanse` / `autoeda`
- `research/DataMineLib` — via `autocausal.datamine_adapter`
