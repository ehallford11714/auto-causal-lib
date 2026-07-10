# Example datasets (AutoCausalLib)

Bundled **real / public-style** tabular CSVs for offline library and CLI demos.
Synthetic join fixtures (finance/marketing/…) remain under
`src/autocausal/data/public/` — see [docs/PUBLIC_SUITE.md](docs/PUBLIC_SUITE.md).

## Locations

| Path | Role |
|------|------|
| `src/autocausal/data/examples/*.csv` | Packaged offline fixtures (importable) |
| `examples/data/*.csv` | Same files mirrored for browsing / scripts |
| `autocausal.datasets` | Library loader (`load_dataset`, `list_datasets`) |
| `python -m autocausal public load <id>` | CLI load (offline by default for bundled ids) |

## Catalog

| id | rows (approx) | outcome hint | license / attribution |
|----|---------------|--------------|------------------------|
| `iris` | 150 | `species` | Fisher / UCI Iris — public domain / open educational |
| `wine` | 178 | `cultivar` | UCI Wine recognition — open educational |
| `diabetes` | 442 | `disease_progression` | sklearn / Efron et al. — redistributable educational |
| `titanic` | 891 | `Survived` | Public-domain passenger records via open educational mirrors |
| `gapminder_subset` | 36 | `lifeExp` | Gapminder open indicators subset — **cite Gapminder** |
| `california_housing_sample` | 250 | `median_house_value` | sklearn California housing sample — educational |

Aliases: `iris_open` → `iris`, `gapminder` / `gapminder_open` → `gapminder_subset`,
`housing` → `california_housing_sample`.

## Epistemic honesty

Exploratory edges on these tables are **library demos**, not scientific causal
identification. Iris morphological associations do **not** claim that one flower
trait causes another. Titanic / Gapminder / housing runs are similarly
illustrative association discovery.

## Soft network refresh

```python
from autocausal.datasets import load_dataset

df = load_dataset("iris")  # bundled, offline
df = load_dataset("iris", allow_network=True, prefer_network=True)  # soft; falls back
```

Optional mirrors are documented on each `ExampleDataset.url`. Network is never
required for demos or tests.

## Regenerate

```bash
python scripts/generate_example_datasets.py
```

Requires `scikit-learn` for Iris / Wine / Diabetes / housing sample generation.
Titanic prefers an open GitHub CSV mirror when network is available.
