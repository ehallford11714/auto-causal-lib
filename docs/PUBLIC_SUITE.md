# Public database suite

`autocausal.public_suite` ships **joinable** demo / open datasets for enriching user tables before mine → discover → guide → ground.

```bash
python -m autocausal public list
python -m autocausal public info finance_demo
python -m autocausal public load demographics_demo -o demo.csv
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
| `iris_open` | download | demo | — | Optional network CSV |
| `public_pg_env` | sql_demo | sql_demo | — | `AUTOCAUSAL_PUBLIC_PG_URL` only |

Bundled files live under `src/autocausal/data/public/` with `manifest.json`.  
`ensure_bundled_public_data()` creates them on first use if missing.

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

- **Bundled** sources always work offline (synthetic MIT fixtures — not real market/PII data).
- **Download** (`iris_open`) requires network; soft-fails with a clear error. Enable with `public load iris_open --allow-network` or `AUTOCAUSAL_PUBLIC_NET=1` in tests.
- **SQL demos**: no hardcoded credentials. Set `AUTOCAUSAL_PUBLIC_PG_URL` to your own read-only demo URL if needed; prefer bundled fixtures.

## Licenses

Bundled tables are **synthetic** and MIT-licensed with the library.  
Open downloads (Iris) follow their upstream open/educational terms — cite upstream when publishing.

## Related

- Connection matrix: [CONNECTIONS.md](CONNECTIONS.md)
- Ping public targets: `python -m autocausal ping --public --no-network`
