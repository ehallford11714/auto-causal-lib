# Examples

Library-first demos using **bundled real datasets** (offline). See also
[DATASETS.md](../DATASETS.md) for licenses and paths.

## Quick imports

```python
from autocausal import AutoCausal, load_dataset, list_datasets
from autocausal.datasets import get_dataset

print([d.id for d in list_datasets()])
df = load_dataset("iris")  # offline CSV from package data

ac = AutoCausal(df)
ac.mine().impute().discover(use_iv=False, min_abs_corr=0.2)
print(ac.report())
for e in ac.result.edges[:10]:
    print(e)
```

```python
from autocausal.insight import run_insight_loop, demo_insight

report = run_insight_loop(load_dataset("iris"), text="exploratory iris associations", use_slm=False)
print(report.to_markdown())

# or
report = demo_insight(dataset="iris", use_slm=False, research_loop=False)
```

## Scripts (repo root)

```bash
pip install -e ".[dev]"

# Iris: mine → discover → optional insight
python examples/iris_causal.py
python examples/iris_causal.py --insight

# Tour all bundled real datasets
python examples/multi_dataset_tour.py
python examples/multi_dataset_tour.py --ids iris,wine,titanic
```

CSV mirrors for browsing: `examples/data/*.csv` (same bytes as package data).

## CLI

```bash
# List / load (offline)
python -m autocausal public list --offline
python -m autocausal public load iris
python -m autocausal public load wine -o wine.csv
python -m autocausal public info iris

# Discover on a loaded CSV
python -m autocausal public load iris -o /tmp/iris.csv
python -m autocausal discover --csv /tmp/iris.csv --no-iv

# Insight demo on a real bundled dataset
python -m autocausal insight demo --dataset iris --no-slm --rounds 1
python -m autocausal insight demo --dataset titanic --no-slm

# Public causal mining still defaults to synthetic join fixtures
python -m autocausal public causal --sources finance_demo,demographics_demo -o report.md
```

## Dataset ids

| id | notes |
|----|--------|
| `iris` | Classic morphology; edges are illustrative only |
| `wine` | UCI chemical features → cultivar |
| `diabetes` | sklearn progression target |
| `titanic` | Survival tabular classic |
| `gapminder_subset` | Small country-year panel |
| `california_housing_sample` | 250-row housing sample |

## Caveats

- Offline-first: no network required.
- Soft optional deps (SLM, sklearn extras) are not needed for these demos.
- Do not treat discovered edges as identified causal effects.
