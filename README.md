![Logo](assets/logo.svg)

# AutoCausalLib (`autocausal`)

Automatically **impute** missing tabular fields, **mine** associations, discover *exploratory* causal relationships, **guide** next steps (rules or optional HF SLM), and **ground** edges against domain glossaries — from CSV / Parquet, SQL databases, and a **public demo suite** you can join in.

> Scope is intentionally library-sized: connect → (join public) → mine → impute → discover → guide → ground.  
> This is **not** a full AutoML OS and does **not** guarantee causal identification.

## Features

- Unified `connect` / `ping` / `list_tables` / `sample_table` / `profile_table` for any SQLAlchemy URL
- Dialect registry: Postgres, Vertica, MySQL/MariaDB, MSSQL/Synapse, Oracle, SQLite, DuckDB, Snowflake, BigQuery, Redshift, Cockroach, Trino/Presto, ClickHouse, Databricks, Firebird, SAP HANA, …
- Public ping targets (bundled SQLite + optional env-based network demos; soft-fail, never hang)
- Auto data mining (profiles, correlations / Cramér’s V / MI, KPI suggestions)
- Real-world grounding (finance / marketing / policy / vision glossaries + optional soft web)
- Rule guide always; HuggingFace SLM behind `autocausal[slm]` + `AUTOCAUSAL_SLM=1`
- **Public database suite** — join bundled demos into your table before mining
- `AutoCausal.auto(...)` orchestrated pipeline

## Install

```bash
cd research/AutoCausalLib
pip install -e ".[dev]"
# optional shared facade: pip install -e ../DataMineLib  (autocausal.datamine_adapter)

# Optional
pip install -e ".[postgres,mysql,duckdb,parquet]"
pip install -e ".[slm]"    # torch + transformers
pip install -e ".[web]"    # httpx grounding
```

## Quick start

```python
from autocausal import AutoCausal

ac = AutoCausal.from_csv("data.csv")
result = ac.run()          # impute + discover
print(ac.report())
```

### Full auto pipeline

```python
result = AutoCausal.auto(
    "data.csv",
    join="demographics_demo",
    text="what causes revenue?",
    use_slm=False,
)
print(result.to_markdown())
```

```bash
python -m autocausal auto --csv data.csv --join demographics_demo --text "what causes outcome?"
python -m autocausal mine --csv data.csv --join finance_demo,policy_demo
python -m autocausal ping --public --no-network
python -m autocausal guide --csv data.csv --text "what causes Y?"
python -m autocausal public list
```

## Public database suite

Joinable bundled demos (always offline):

| id | domain |
|----|--------|
| `finance_demo` | finance |
| `marketing_demo` | marketing |
| `policy_demo` | policy |
| `demographics_demo` | demographics |
| `vision_stub` | vision |
| `instruments_demo` | IV / policy |

```python
ac = AutoCausal.from_csv("users.csv")
ac.join_public("demographics_demo")  # on region if present
ac.mine().impute().discover()
```

See [docs/PUBLIC_SUITE.md](docs/PUBLIC_SUITE.md).

## Connections & ping

```bash
python -m autocausal ping --url "sqlite:///:memory:"
python -m autocausal ping --public
python -m autocausal dialects
```

```python
from autocausal.db import connect, ping
h = connect("sqlite:///./local.db")
print(h.list_tables())
```

Details: [docs/CONNECTIONS.md](docs/CONNECTIONS.md).

## Mining

```bash
python -m autocausal mine --csv data.csv --format both -o mine.md
python -m autocausal mine --db "sqlite:///demo.db" --table demo_obs
```

## Grounding & SLM guide

```bash
python -m autocausal discover --csv data.csv --guide --ground
python -m autocausal guide --csv data.csv --text "what causes conversion?" --slm
```

| Env | Meaning |
|-----|---------|
| `AUTOCAUSAL_SLM=1` | Prefer HuggingFace SLM backend |
| `AUTOCAUSAL_SLM_MODEL` | Model id (default `sshleifer/tiny-gpt2`; try `Qwen/Qwen2.5-0.5B-Instruct`) |
| `AUTOCAUSAL_SLM_TEST=1` | Enable HF download tests |
| `AUTOCAUSAL_NO_WEB=1` | Disable optional web grounding |
| `AUTOCAUSAL_PUBLIC_PG_URL` | Optional read-only demo Postgres for ping |

Default guide is **RuleGuide** (offline). SLM never blocks import; failures soft-fall back to rules.

## Guide loop (architecture)

```text
load / connect ──► ping? ──► join_public? ──► mine
                                              │
                                              ▼
                                         impute ──► discover ──► guide (Rule|HF)
                                              ▲                      │
                                              └──── second-pass ◄────┘
                                                                     │
                                                                     ▼
                                                                  ground ──► JSON/MD report
```

## Docs

- [Connection matrix & ping](docs/CONNECTIONS.md)
- [Public suite & joins](docs/PUBLIC_SUITE.md)
- [SOTA context](docs/SOTA.md)

## Related suite

| Project | Role |
|---------|------|
| [CausalIVSuite](https://github.com/ehallford11714/causal-iv-suite) | IV / DiD / AutoML causal suite |
| [CausalSearch](https://github.com/ehallford11714/causal-search) | Causal evidence search & DAG infill |
| [CausalBridge](https://github.com/ehallford11714/causal-bridge) | Control plane for the product suite |
| [NextFrameSeq](https://github.com/ehallford11714/next-frame-seq) | Vision / next-frame prediction |

## License

MIT
