"""Tabular ingest: CSV, Parquet, SQLAlchemy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import pandas as pd


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, **kwargs)


def load_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    try:
        return pd.read_parquet(path, **kwargs)
    except ImportError as e:
        raise ImportError(
            "Parquet support requires pyarrow (or fastparquet). "
            "Install with: pip install 'autocausal[parquet]'"
        ) from e


def _quote_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return f'"{name}"'


def load_sqlalchemy(
    url: str,
    *,
    table: Optional[str] = None,
    query: Optional[str] = None,
    schema: Optional[str] = None,
    limit: Optional[int] = None,
    **engine_kwargs: Any,
) -> pd.DataFrame:
    """Load a table or SQL query via SQLAlchemy.

    Drivers are optional — only SQLAlchemy itself is required. Install dialect
    extras as documented in docs/CONNECTIONS.md.
    """
    if not table and not query:
        raise ValueError("Provide table=... or query=...")
    if table and query:
        raise ValueError("Provide only one of table= or query=")

    try:
        from sqlalchemy import create_engine, text
    except ImportError as e:
        raise ImportError("SQLAlchemy is required: pip install sqlalchemy") from e

    engine = create_engine(url, **engine_kwargs)
    try:
        if query:
            sql = query
        else:
            assert table is not None
            if schema:
                fq = f"{_quote_ident(schema)}.{_quote_ident(table)}"
            else:
                fq = _quote_ident(table)
            sql = f"SELECT * FROM {fq}"
            if limit is not None:
                # Portable-ish LIMIT; dialects that need TOP should use query=
                sql = f"{sql} LIMIT {int(limit)}"
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn)
    finally:
        engine.dispose()


def dialect_from_url(url: str) -> str:
    """Return SQLAlchemy dialect name from a URL (e.g. postgresql, vertica).

    Parses manually first: urllib rejects schemes with underscores
    (e.g. ``vertica+vertica_python://...``).
    """
    # sqlalchemy URLs: dialect+driver://...
    if "://" in url:
        scheme = url.split("://", 1)[0]
    else:
        scheme = urlparse(url).scheme or ""
    return scheme.split("+", 1)[0].lower() if scheme else ""


# Exhaustive-ish matrix of known SQLAlchemy dialects / URL schemes.
# Implementation uses create_engine; drivers listed are documentation hints.
DIALECT_MATRIX: list[dict[str, Any]] = [
    {
        "dialect": "sqlite",
        "url": "sqlite:///:memory:  |  sqlite:///path/to.db",
        "extra": "sqlite",
        "pip": "(stdlib / built-in)",
        "notes": "Great for tests; no extra driver.",
    },
    {
        "dialect": "postgresql",
        "url": "postgresql+psycopg2://user:pass@host:5432/db",
        "extra": "postgres",
        "pip": "psycopg2-binary",
        "notes": "Also postgresql+psycopg:// with psycopg3.",
    },
    {
        "dialect": "vertica",
        "url": "vertica+vertica_python://user:pass@host:5433/db",
        "extra": "vertica",
        "pip": "vertica-python ; sqlalchemy-vertica-python (if available)",
        "notes": "Community dialect; pin versions carefully.",
    },
    {
        "dialect": "mysql",
        "url": "mysql+pymysql://user:pass@host:3306/db",
        "extra": "mysql",
        "pip": "pymysql",
        "notes": "mysqlclient also works with mysql+mysqldb://",
    },
    {
        "dialect": "mariadb",
        "url": "mariadb+pymysql://user:pass@host:3306/db",
        "extra": "mariadb",
        "pip": "pymysql",
        "notes": "Often interchangeable with MySQL URLs.",
    },
    {
        "dialect": "mssql",
        "url": "mssql+pyodbc://user:pass@dsn",
        "extra": "mssql",
        "pip": "pyodbc",
        "notes": "Requires ODBC driver on host.",
    },
    {
        "dialect": "oracle",
        "url": "oracle+oracledb://user:pass@host:1521/?service_name=ORCL",
        "extra": "oracle",
        "pip": "oracledb",
        "notes": "Thick/thin mode per oracledb docs.",
    },
    {
        "dialect": "duckdb",
        "url": "duckdb:///path/to.duckdb",
        "extra": "duckdb",
        "pip": "duckdb duckdb-engine",
        "notes": "Local analytical SQL.",
    },
    {
        "dialect": "snowflake",
        "url": "snowflake://user:pass@account/db/schema?warehouse=WH",
        "extra": "snowflake",
        "pip": "snowflake-sqlalchemy",
        "notes": "Account locator in host.",
    },
    {
        "dialect": "bigquery",
        "url": "bigquery://project/dataset",
        "extra": "bigquery",
        "pip": "sqlalchemy-bigquery",
        "notes": "Uses Google ADC / service account.",
    },
    {
        "dialect": "redshift",
        "url": "redshift+psycopg2://user:pass@host:5439/db",
        "extra": "redshift",
        "pip": "sqlalchemy-redshift psycopg2-binary",
        "notes": "Postgres-wire compatible.",
    },
    {
        "dialect": "cockroachdb",
        "url": "cockroachdb+psycopg2://user:pass@host:26257/db",
        "extra": "cockroachdb",
        "pip": "psycopg2-binary",
        "notes": "Postgres-compatible protocol.",
    },
    {
        "dialect": "trino",
        "url": "trino://user@host:8080/catalog/schema",
        "extra": "trino",
        "pip": "trino",
        "notes": "SQLAlchemy dialect via trino package.",
    },
    {
        "dialect": "presto",
        "url": "presto://user@host:8080/catalog/schema",
        "extra": "presto",
        "pip": "pyhive[presto]",
        "notes": "Legacy PrestoDB; prefer Trino when possible.",
    },
    {
        "dialect": "clickhouse",
        "url": "clickhouse+native://user:pass@host:9000/db",
        "extra": "clickhouse",
        "pip": "clickhouse-sqlalchemy",
        "notes": "Also clickhouse+http:// for HTTP interface.",
    },
    {
        "dialect": "mariadb / mysql (generic)",
        "url": "mysql+mysqldb://...",
        "extra": "mysql",
        "pip": "mysqlclient",
        "notes": "Alternative driver.",
    },
    {
        "dialect": "postgresql (psycopg3)",
        "url": "postgresql+psycopg://user:pass@host/db",
        "extra": "postgres",
        "pip": "psycopg[binary]",
        "notes": "Modern psycopg3 driver.",
    },
    {
        "dialect": "mssql+pymssql",
        "url": "mssql+pymssql://user:pass@host/db",
        "extra": "mssql",
        "pip": "pymssql",
        "notes": "Alternative to pyodbc.",
    },
    {
        "dialect": "sybase",
        "url": "sybase+pyodbc://...",
        "extra": None,
        "pip": "pyodbc",
        "notes": "Legacy; community support.",
    },
    {
        "dialect": "databricks",
        "url": "databricks://token:TOKEN@host?http_path=/sql/1.0/warehouses/...",
        "extra": "databricks",
        "pip": "databricks-sqlalchemy / sqlalchemy-databricks",
        "notes": "External dialect package.",
    },
    {
        "dialect": "synapse / mssql azure",
        "url": "mssql+pyodbc://user:pass@server.database.windows.net/db?driver=ODBC+Driver+18+for+SQL+Server",
        "extra": "synapse",
        "pip": "pyodbc",
        "notes": "Azure Synapse / SQL via mssql+pyodbc.",
    },
    {
        "dialect": "athena",
        "url": "awsathena+rest://:@region/db?s3_staging_dir=s3://...",
        "extra": None,
        "pip": "pyathena[sqlalchemy]",
        "notes": "External dialect package.",
    },
    {
        "dialect": "hive",
        "url": "hive://user@host:10000/db",
        "extra": None,
        "pip": "pyhive[hive]",
        "notes": "External dialect package.",
    },
    {
        "dialect": "impala",
        "url": "impala://host:21050/db",
        "extra": None,
        "pip": "impyla",
        "notes": "External dialect package.",
    },
    {
        "dialect": "cratedb",
        "url": "crate://user:pass@host:4200/",
        "extra": None,
        "pip": "crate[sqlalchemy]",
        "notes": "External dialect package.",
    },
    {
        "dialect": "sap hana",
        "url": "hana://user:pass@host:30015",
        "extra": "hana",
        "pip": "sqlalchemy-hana",
        "notes": "External dialect package; driver optional.",
    },
    {
        "dialect": "firebird",
        "url": "firebird+fdb://user:pass@host:3050/db",
        "extra": "firebird",
        "pip": "sqlalchemy-firebird / fdb",
        "notes": "Documented even if driver optional.",
    },
]
