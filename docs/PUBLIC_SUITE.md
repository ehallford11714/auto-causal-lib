# Public database suite

`autocausal.public_suite` ships **joinable** demo / open datasets for enriching user tables before mine → discover → guide → ground.

```bash
python -m autocausal public list
python -m autocausal public info finance_demo
python -m autocausal public load demographics_demo -o demo.csv
python -m autocausal public load iris
```

## Catalog (ids)

| id | access | domain | join keys (suggested) | notes |
|----|--------|--------|------------------------|-------|
| `finance_demo` | bundled | finance | `region`, `date` | Synthetic returns / rates |
| `marketing_demo` | bundled | marketing | `user_id`, `region` | Campaign panel |
| `policy_demo` | bundled | policy | `region`, `year`, `unit_id` | DiD-style panel |
| `demographics_demo` | bundled | demographics | `region` | Region aggregates |
| `vision_stub` | bundled | vision | `region`, `clip_id` | Motion / frame stub |
| `instruments_demo` | bundled | policy | `user_id`, `region` | IV assignment table |
| `climate_demo` | bundled | climate | `region`, `year` | Temp / precip / CO2 stub |
| `health_demo` | bundled | health | `region` | Life expectancy / access stub |
| `iris` | bundled | demo | — | Fisher Iris (real); illustrative edges only |
| `wine` | bundled | chemistry | — | UCI Wine recognition |
| `diabetes` | bundled | health | — | sklearn diabetes progression |
| `titanic` | bundled | demographics | — | Educational passenger table |
| `gapminder_subset` | bundled | demographics | `year` | Small Gapminder country-year panel |
| `california_housing_sample` | bundled | housing | — | 250-row housing sample |
| `iris_open` | download | demo | — | Optional network refresh; offline → `iris` |
| `gapminder_open` | download | demographics | `year` | Optional full Gapminder; offline → subset |
| `public_pg_env` | sql_demo | sql_demo | — | `AUTOCAUSAL_PUBLIC_PG_URL` only |

Synthetic fixtures live under `src/autocausal/data/public/`.  
Real example CSVs live under `src/autocausal/data/examples/` (see [DATASETS.md](../DATASETS.md)).  
`ensure_bundled_public_data()` creates synthetic fixtures on first use if missing.

Prefer the library API for real examples:

```python
from autocausal.datasets import load_dataset, list_datasets
df = load_dataset("iris")
```

## Join into mining / discover

```python
from autocausal import AutoCausal

ac = AutoCausal.from_csv("user.csv")
ac.join_public("demographics_demo")          # auto key: region
# or
ac.join_public(["marketing_demo", "instruments_demo"], on="user_id")

ac.mine().impute().discover()
ac.guide(text="what causes revenue?")
ac.ground()
```

```bash
python -m autocausal mine --csv user.csv --join finance_demo,demographics_demo
python -m autocausal auto --csv user.csv --join demographics_demo --text "what causes outcome?"
python -m autocausal discover --csv user.csv --join policy_demo --guide --ground
```

Join keys are suggested by exact column-name overlap (preferring `id` / `region` / `date` / `year` / `user`), falling back to suite `suggested_join_keys`. Pass `--join-on region` when auto-detect fails.

## Offline vs network

- **Bundled** sources always work offline (synthetic MIT fixtures **and** real example CSVs).
- **Download** (`iris_open`, `gapminder_open`) soft-falls back to bundled `iris` / `gapminder_subset` when offline.
- **SQL demos**: no hardcoded credentials. Set `AUTOCAUSAL_PUBLIC_PG_URL` to your own read-only demo URL if needed; prefer bundled fixtures.

## Licenses

Bundled synthetic tables are **MIT-licensed** with the library.  
Real example CSVs follow upstream open/educational terms — see [DATASETS.md](../DATASETS.md). Cite Gapminder / UCI when publishing.

## Related

- **Examples walkthrough**: [EXAMPLES.md](EXAMPLES.md)
- **Public causal mining** (multi-source join → mine → discover): [PUBLIC_CAUSAL_MINING.md](PUBLIC_CAUSAL_MINING.md)
- Connection matrix: [CONNECTIONS.md](CONNECTIONS.md)
- Ping public targets: `python -m autocausal ping --public --no-network`
