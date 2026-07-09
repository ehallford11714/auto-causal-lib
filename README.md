![Logo](assets/logo.svg)

# AutoCausalLib (`autocausal`)

Automatically **impute** missing tabular fields and discover *exploratory* causal relationships from CSV / Parquet and SQL databases.

> Scope is intentionally small: impute → role inference → PC-lite / score edges → optional IV.  
> This is **not** a full AutoML OS and does **not** guarantee causal identification.

## Features

- Load from CSV, Parquet, or SQLAlchemy URLs (Postgres, Vertica, DuckDB, and more via extras)
- Auto-imputation (`median_mode`, `knn`, or `auto`) with strategy reporting
- Role inference (treatment / outcome / instrument / confounder candidates)
- Exploratory discovery: PC-lite + scored edges + optional 2SLS
- Markdown / JSON reports and a simple CLI

## Install

```bash
cd research/AutoCausalLib
pip install -e ".[dev]"

# Optional drivers (install only what you need)
pip install -e ".[postgres]"
pip install -e ".[vertica]"
pip install -e ".[mysql,duckdb,parquet]"
```

Core deps: `numpy`, `pandas`, `sqlalchemy`. See [docs/CONNECTIONS.md](docs/CONNECTIONS.md).

## Quick start

```python
from autocausal import AutoCausal

ac = AutoCausal.from_csv("data.csv")
result = ac.run()          # impute + discover
print(ac.report())         # markdown
print(result.to_json())    # graph + edges + candidates
```

```bash
python -m autocausal discover --csv data.csv
python -m autocausal discover --csv data.csv --format json -o report.json
```

### PostgreSQL

```bash
pip install -e ".[postgres]"
python -m autocausal discover \
  --db "postgresql+psycopg2://user:pass@localhost:5432/mydb" \
  --table events
```

## Docs

- [Connection matrix & pip extras](docs/CONNECTIONS.md)
- [SOTA context (PC / GES / NOTEARS, imputation)](docs/SOTA.md)

## Related suite

| Project | Role |
|---------|------|
| [CausalIVSuite](https://github.com/ehallford11714/causal-iv-suite) | IV / DiD / AutoML causal suite |
| [CausalSearch](https://github.com/ehallford11714/causal-search) | Causal evidence search & DAG infill |
| [CausalBridge](https://github.com/ehallford11714/causal-bridge) | Control plane for the product suite |
| [NextFrameSeq](https://github.com/ehallford11714/next-frame-seq) | Vision / next-frame prediction |

## License

MIT
