# SQLAlchemy connection matrix

`autocausal` loads tables through **SQLAlchemy** `create_engine(url)`.  
Only `sqlalchemy` is a hard dependency. Install dialect drivers as **optional extras**.

```bash
pip install 'autocausal[postgres]'
pip install 'autocausal[vertica]'
python -m autocausal dialects   # dump JSON matrix
python -m autocausal ping --url "sqlite:///:memory:"
python -m autocausal ping --public --no-network
```

## Unified connect / ping / helpers

```python
from autocausal.db import connect, ping, list_tables, sample_table, profile_table

h = connect("sqlite:///./demo.db")
# or kwargs:
# h = connect(dialect="postgresql+psycopg2", user="u", password="p",
#             host="localhost", port=5432, database="db")

print(h.list_tables())
print(sample_table(h.url, "events", n=50).head())
print(profile_table(h.url, "events"))

r = ping("sqlite:///:memory:")
assert r.ok and r.latency_ms >= 0
```

Public / bundled health checks (never hang; network soft-fails):

```bash
python -m autocausal ping --public
python -m autocausal ping --public --no-network
# optional: set AUTOCAUSAL_PUBLIC_PG_URL for a read-only demo Postgres ping
```

## First-class targets

### PostgreSQL

| | |
|--|--|
| **Extra** | `autocausal[postgres]` |
| **Pip** | `psycopg2-binary` (or `psycopg[binary]` for v3) |
| **URL** | `postgresql+psycopg2://USER:PASS@HOST:5432/DB` |
| **Alt** | `postgresql+psycopg://USER:PASS@HOST:5432/DB` |

```python
AutoCausal.from_sqlalchemy(
    "postgresql+psycopg2://user:pass@localhost:5432/analytics",
    table="events",
    schema="public",
    limit=100_000,
)
```

### Vertica

| | |
|--|--|
| **Extra** | `autocausal[vertica]` |
| **Pip** | `vertica-python` and, when available, `sqlalchemy-vertica-python` |
| **URL** | `vertica+vertica_python://USER:PASS@HOST:5433/DB` |

Notes:

- Vertica dialects are **community-maintained**; pin versions in production.
- If the Vertica SQLAlchemy dialect is unavailable, you can still export a CSV/Parquet extract and use `from_csv` / `from_parquet`.
- Prefer `--query` for complex Vertica SQL (projections, `LIMIT`, etc.).

```python
AutoCausal.from_sqlalchemy(
    "vertica+vertica_python://dbadmin:secret@vertica-host:5433/VMart",
    query="SELECT * FROM store.store_sales LIMIT 50000",
)
```

## Exhaustive dialect list (SQLAlchemy ecosystem)

Drivers are **not** all installed by default. Rows marked *bundled extra* ship a `pyproject` optional dependency; others need a manual `pip install`.

| Dialect | Example URL | Extra | Pip packages | Notes |
|---------|-------------|-------|--------------|-------|
| **sqlite** | `sqlite:///:memory:` · `sqlite:///./file.db` | `sqlite` | *(stdlib)* | Default for tests |
| **postgresql** | `postgresql+psycopg2://u:p@h:5432/db` | `postgres` | `psycopg2-binary` | First-class |
| **postgresql** (psycopg3) | `postgresql+psycopg://u:p@h/db` | `postgres` | `psycopg[binary]` | Modern driver |
| **vertica** | `vertica+vertica_python://u:p@h:5433/db` | `vertica` | `vertica-python`, `sqlalchemy-vertica-python` | First-class docs |
| **mysql** | `mysql+pymysql://u:p@h:3306/db` | `mysql` | `pymysql` | Also `mysql+mysqldb` + `mysqlclient` |
| **mariadb** | `mariadb+pymysql://u:p@h:3306/db` | `mariadb` | `pymysql` | Often same as MySQL |
| **mssql** | `mssql+pyodbc://u:p@dsn` | `mssql` | `pyodbc` | Needs host ODBC driver |
| **mssql** (pymssql) | `mssql+pymssql://u:p@h/db` | `mssql` | `pymssql` | Alt driver |
| **oracle** | `oracle+oracledb://u:p@h:1521/?service_name=ORCL` | `oracle` | `oracledb` | Thick/thin modes |
| **duckdb** | `duckdb:///path/to.duckdb` | `duckdb` | `duckdb`, `duckdb-engine` | Local OLAP |
| **snowflake** | `snowflake://u:p@account/db/schema?warehouse=WH` | `snowflake` | `snowflake-sqlalchemy` | Account in host |
| **bigquery** | `bigquery://project/dataset` | `bigquery` | `sqlalchemy-bigquery` | ADC / SA JSON |
| **redshift** | `redshift+psycopg2://u:p@h:5439/db` | `redshift` | `sqlalchemy-redshift`, `psycopg2-binary` | PG wire |
| **cockroachdb** | `cockroachdb+psycopg2://u:p@h:26257/db` | `cockroachdb` | `psycopg2-binary` | PG compatible |
| **trino** | `trino://u@h:8080/catalog/schema` | `trino` | `trino` | Prefer over Presto |
| **presto** | `presto://u@h:8080/catalog/schema` | `presto` | `pyhive[presto]` | Legacy PrestoDB |
| **clickhouse** | `clickhouse+native://u:p@h:9000/db` | `clickhouse` | `clickhouse-sqlalchemy` | Also `+http` |
| **databricks** | `databricks://token:TOKEN@host?http_path=...` | `databricks` | `databricks-sqlalchemy` | External |
| **synapse** | `mssql+pyodbc://...database.windows.net/...` | `synapse` | `pyodbc` | Azure Synapse / SQL |
| **firebird** | `firebird+fdb://u:p@h:3050/db` | `firebird` | `sqlalchemy-firebird` / `fdb` | Optional driver |
| **sap hana** | `hana://u:p@h:30015` | `hana` | `sqlalchemy-hana` | Optional driver |
| **sybase** | `sybase+pyodbc://...` | — | `pyodbc` | Legacy |
| **athena** | `awsathena+rest://:@region/db?s3_staging_dir=s3://...` | — | `pyathena[sqlalchemy]` | External |
| **hive** | `hive://u@h:10000/db` | — | `pyhive[hive]` | External |
| **impala** | `impala://h:21050/db` | — | `impyla` | External |
| **cratedb** | `crate://u:p@h:4200/` | — | `crate[sqlalchemy]` | External |

### Convenience extras

```text
autocausal[parquet]       # pyarrow for Parquet ingest
autocausal[slm]           # torch + transformers for HF guide
autocausal[web]           # httpx for optional web grounding
autocausal[all-drivers]   # attempts to pull bundled dialect extras
autocausal[dev]           # pytest + pyarrow
```

## Helpers

```python
from autocausal.ingest import dialect_from_url, DIALECT_MATRIX, load_sqlalchemy
from autocausal.db import connect, ping, list_tables, sample_table, profile_table

dialect_from_url("postgresql+psycopg2://localhost/db")  # -> "postgresql"
```

`LIMIT` is appended for `table=` loads. Dialects that need `TOP` / `FETCH FIRST` should pass an explicit `query=`.

## Security

- Prefer env vars / secret stores for credentials; do not commit URLs with passwords.
- `table` / `schema` identifiers are restricted to alphanumeric + underscore when auto-quoted.
- Public ping never ships secrets; optional Postgres uses `AUTOCAUSAL_PUBLIC_PG_URL` only.
