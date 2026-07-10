# Public causal mining

End-to-end path: **catalog → multi-source join → mine → impute → discover → report**.

Exploratory only. Edges are **candidate** relationships from heuristic PC-lite / association scores — **not** identified causal effects. Correlation is not causation.

## Quick start

```bash
python -m autocausal public list --offline
python -m autocausal public mine --sources finance_demo,demographics_demo --discover
python -m autocausal public causal --sources finance_demo,demographics_demo,climate_demo,health_demo -o report.md
```

```python
from autocausal import AutoCausal, PublicCausalMiner, mine_public

# Facade
report = AutoCausal.mine_public(
    ["finance_demo", "demographics_demo", "health_demo"],
    join_on="region",
    discover=True,
    use_iv=True,
)
print(report.to_markdown())

# Or explicit miner
miner = PublicCausalMiner(["marketing_demo", "instruments_demo", "demographics_demo"])
report = miner.run(discover=True, validate=True)
```

## Pipeline

1. **Catalog / load** — `public_suite` bundled CSVs (offline) + optional download/SQL soft-fail.
2. **Auto-join** — `join_public_corpus` aligns sources on suggested keys (`region`, `year`, …).
3. **Mine** — column profiles, associations, KPI-like suggestions.
4. **Impute** — median/mode or knn before discovery.
5. **Discover** — exploratory edges + X/Y/Z/W role candidates; optional soft IV via causaliv / 2SLS lite.
6. **Report** — markdown/JSON with sources, joins, edges, confidence, caveats.

## Bundled sources (offline)

| id | domain | join keys |
|----|--------|-----------|
| `finance_demo` | finance | region, date |
| `marketing_demo` | marketing | user_id, region |
| `policy_demo` | policy | region, year |
| `demographics_demo` | demographics | region |
| `vision_stub` | vision | region, clip_id |
| `instruments_demo` | policy | user_id, region |
| `climate_demo` | climate | region, year |
| `health_demo` | health | region |

Optional network (soft-fail): `iris_open`, `gapminder_open`. SQL: `public_pg_env` via `AUTOCAUSAL_PUBLIC_PG_URL`.

## Limitations (read these)

- **Not causal identification.** No DAG from this pipeline is “proven.”
- **Join bias.** Outer/left joins across domains can create ecological fallacy (region aggregates ≠ individual effects).
- **Synthetic fixtures.** Bundled tables are MIT demo data, not real markets/health/climate series.
- **IV is soft.** `causaliv` is optional; numpy 2SLS lite does not validate exclusion or relevance beyond crude F.
- **Offline-first.** Downloads fail closed when network is disabled.

## Related

- Suite catalog: [PUBLIC_SUITE.md](PUBLIC_SUITE.md)
- Connections / dialects: [CONNECTIONS.md](CONNECTIONS.md)
- API: `autocausal.public_causal`, `AutoCausal.mine_public`, `AutoCausal.join_public`
