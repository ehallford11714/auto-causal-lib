"""Unified database connection, ping, and table helpers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import pandas as pd

from autocausal.ingest import DIALECT_MATRIX, dialect_from_url, load_sqlalchemy


__all__ = [
    "ConnectionHandle",
    "PingResult",
    "connect",
    "ping",
    "list_tables",
    "sample_table",
    "profile_table",
    "bundled_sample_url",
    "PUBLIC_TARGETS",
    "ping_public",
    "DIALECT_MATRIX",
]


@dataclass
class PingResult:
    ok: bool
    latency_ms: float
    url_safe: str
    dialect: str = ""
    error: Optional[str] = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConnectionHandle:
    """Thin wrapper around a SQLAlchemy engine URL + optional live engine."""

    url: str
    dialect: str
    engine: Any = None
    _owns_engine: bool = True

    def dispose(self) -> None:
        if self._owns_engine and self.engine is not None:
            self.engine.dispose()
            self.engine = None

    def __enter__(self) -> "ConnectionHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.dispose()

    def list_tables(self, schema: Optional[str] = None) -> list[str]:
        return list_tables(self.url, schema=schema, engine=self.engine)

    def sample_table(
        self,
        table: str,
        *,
        n: int = 100,
        schema: Optional[str] = None,
    ) -> pd.DataFrame:
        return sample_table(self.url, table, n=n, schema=schema, engine=self.engine)

    def profile_table(
        self,
        table: str,
        *,
        schema: Optional[str] = None,
        sample_n: int = 5000,
    ) -> dict[str, Any]:
        return profile_table(
            self.url, table, schema=schema, sample_n=sample_n, engine=self.engine
        )

    def load(
        self,
        *,
        table: Optional[str] = None,
        query: Optional[str] = None,
        schema: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        return load_sqlalchemy(
            self.url, table=table, query=query, schema=schema, limit=limit
        )


def _safe_url(url: str) -> str:
    """Redact password from SQLAlchemy URL for logs/reports."""
    try:
        p = urlparse(url)
        if p.password:
            netloc = p.netloc.replace(f":{p.password}", ":****", 1)
            return p._replace(netloc=netloc).geturl()
        return url
    except Exception:
        return "<unparseable>"


def _build_url_from_kwargs(**kwargs: Any) -> str:
    """Build a SQLAlchemy URL from keyword args.

    Accepted keys: dialect/driver, user/username, password, host, port,
    database/db, query (dict or str for query string).
    """
    dialect = kwargs.pop("dialect", None) or kwargs.pop("driver", None)
    if not dialect:
        raise ValueError("kwargs connect requires dialect=...")
    user = kwargs.pop("user", None) or kwargs.pop("username", None)
    password = kwargs.pop("password", None)
    host = kwargs.pop("host", "localhost")
    port = kwargs.pop("port", None)
    database = kwargs.pop("database", None) or kwargs.pop("db", None)
    query = kwargs.pop("query", None)
    if kwargs:
        raise TypeError(f"Unexpected connect kwargs: {sorted(kwargs)}")

    auth = ""
    if user is not None:
        if password is not None:
            auth = f"{user}:{password}@"
        else:
            auth = f"{user}@"
    hostpart = host
    if port is not None:
        hostpart = f"{host}:{port}"
    path = f"/{database}" if database else ""
    url = f"{dialect}://{auth}{hostpart}{path}"
    if query:
        if isinstance(query, dict):
            from urllib.parse import urlencode

            url = f"{url}?{urlencode(query)}"
        else:
            url = f"{url}?{query}"
    return url


def connect(
    url: Optional[str] = None,
    /,
    **kwargs: Any,
) -> ConnectionHandle:
    """Unified connect accepting a SQLAlchemy URL or kwargs.

    Examples:
        connect("sqlite:///./demo.db")
        connect(dialect="postgresql+psycopg2", user="u", password="p",
                host="localhost", port=5432, database="db")
    """
    if url is None:
        if not kwargs:
            raise ValueError("connect() requires a URL or dialect kwargs")
        url = _build_url_from_kwargs(**kwargs)
    elif kwargs:
        # allow engine kwargs like pool_pre_ping alongside URL
        pass

    try:
        from sqlalchemy import create_engine
    except ImportError as e:
        raise ImportError("SQLAlchemy is required: pip install sqlalchemy") from e

    engine_kwargs = {k: v for k, v in kwargs.items() if k not in {
        "dialect", "driver", "user", "username", "password", "host", "port",
        "database", "db", "query",
    }}
    # If URL was built from kwargs, engine_kwargs should be empty; if URL given,
    # remaining kwargs go to create_engine.
    if url and not any(k in kwargs for k in ("dialect", "driver")):
        engine_kwargs = dict(kwargs)

    engine = create_engine(url, **engine_kwargs)
    return ConnectionHandle(url=url, dialect=dialect_from_url(url), engine=engine)


def ping(url: str, *, timeout: float = 5.0) -> PingResult:
    """Test connectivity and measure latency without running discovery."""
    safe = _safe_url(url)
    dialect = dialect_from_url(url)
    t0 = time.perf_counter()
    try:
        from sqlalchemy import create_engine, text
    except ImportError as e:
        return PingResult(
            ok=False,
            latency_ms=0.0,
            url_safe=safe,
            dialect=dialect,
            error=str(e),
        )

    engine = None
    try:
        # connect_args timeout is dialect-specific; use execution timeout via connect
        connect_args: dict[str, Any] = {}
        if dialect == "postgresql":
            connect_args["connect_timeout"] = max(1, int(timeout))
        elif dialect in ("mysql", "mariadb"):
            connect_args["connect_timeout"] = max(1, int(timeout))
        engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        ms = (time.perf_counter() - t0) * 1000.0
        return PingResult(
            ok=True,
            latency_ms=round(ms, 2),
            url_safe=safe,
            dialect=dialect,
            detail="SELECT 1 ok",
        )
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        return PingResult(
            ok=False,
            latency_ms=round(ms, 2),
            url_safe=safe,
            dialect=dialect,
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        if engine is not None:
            engine.dispose()


def list_tables(
    url: str,
    *,
    schema: Optional[str] = None,
    engine: Any = None,
) -> list[str]:
    from sqlalchemy import create_engine, inspect

    owns = engine is None
    eng = engine or create_engine(url)
    try:
        insp = inspect(eng)
        return sorted(insp.get_table_names(schema=schema))
    finally:
        if owns:
            eng.dispose()


def sample_table(
    url: str,
    table: str,
    *,
    n: int = 100,
    schema: Optional[str] = None,
    engine: Any = None,
) -> pd.DataFrame:
    return load_sqlalchemy(url, table=table, schema=schema, limit=int(n))


def profile_table(
    url: str,
    table: str,
    *,
    schema: Optional[str] = None,
    sample_n: int = 5000,
    engine: Any = None,
) -> dict[str, Any]:
    """Lightweight table profile (row sample + column dtypes / null%)."""
    from autocausal.mining import profile_dataframe

    df = sample_table(url, table, n=sample_n, schema=schema, engine=engine)
    profile = profile_dataframe(df)
    return {
        "table": table,
        "schema": schema,
        "sample_rows": len(df),
        "columns": profile["columns"],
    }


def bundled_sample_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "demo.db"


def bundled_sample_url() -> str:
    path = bundled_sample_path()
    return f"sqlite:///{path.as_posix()}"


def ensure_bundled_sample() -> Path:
    """Create a tiny demo SQLite DB if missing (idempotent)."""
    path = bundled_sample_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    from sqlalchemy import create_engine

    import numpy as np

    rng = np.random.default_rng(42)
    n = 80
    z = rng.normal(size=n)
    treatment = (z + rng.normal(scale=0.4, size=n) > 0).astype(float)
    revenue = 1.8 * treatment + 0.3 * z + rng.normal(scale=0.5, size=n)
    segment = rng.choice(["A", "B", "C"], size=n)
    df = pd.DataFrame(
        {
            "instrument_z": z,
            "treatment": treatment,
            "revenue": revenue,
            "segment": segment,
            "age": rng.normal(40, 12, size=n),
        }
    )
    url = f"sqlite:///{path.as_posix()}"
    engine = create_engine(url)
    try:
        df.to_sql("demo_obs", engine, index=False, if_exists="replace")
    finally:
        engine.dispose()
    return path


# Safe public / demo targets. Network ones soft-fail; no secrets.
PUBLIC_TARGETS: list[dict[str, Any]] = [
    {
        "name": "bundled_sqlite",
        "kind": "bundled",
        "url": None,  # resolved at runtime
        "table": "demo_obs",
        "timeout": 3.0,
        "notes": "Package-local SQLite fixture (always offline-safe).",
    },
    {
        "name": "sqlite_memory",
        "kind": "local",
        "url": "sqlite:///:memory:",
        "timeout": 2.0,
        "notes": "In-memory SQLite smoke ping.",
    },
    {
        "name": "elephantsql_demo_doc",
        "kind": "optional_network",
        "url": None,
        "timeout": 3.0,
        "notes": (
            "No hardcoded public Postgres credentials. "
            "Set AUTOCAUSAL_PUBLIC_PG_URL to an optional read-only demo URL."
        ),
        "env_url": "AUTOCAUSAL_PUBLIC_PG_URL",
    },
]


def ping_public(
    *,
    include_network: bool = True,
    timeout: float = 3.0,
) -> list[PingResult]:
    """Health-check bundled + optional public endpoints; never hang."""
    import os

    ensure_bundled_sample()
    results: list[PingResult] = []
    for target in PUBLIC_TARGETS:
        name = target["name"]
        kind = target["kind"]
        t_out = float(target.get("timeout", timeout))

        if kind == "bundled":
            url = bundled_sample_url()
            r = ping(url, timeout=t_out)
            r.detail = f"{name}: {r.detail or ''}".strip()
            results.append(r)
            continue

        if kind == "local":
            url = target["url"]
            r = ping(url, timeout=t_out)
            r.detail = f"{name}: {r.detail or ''}".strip()
            results.append(r)
            continue

        if kind == "optional_network":
            if not include_network:
                results.append(
                    PingResult(
                        ok=False,
                        latency_ms=0.0,
                        url_safe=f"env:{target.get('env_url', '')}",
                        dialect="",
                        error="skipped (offline / --no-network)",
                        detail=name,
                    )
                )
                continue
            env_key = target.get("env_url")
            url = os.environ.get(env_key, "") if env_key else ""
            if not url:
                results.append(
                    PingResult(
                        ok=False,
                        latency_ms=0.0,
                        url_safe=f"env:{env_key}",
                        dialect="",
                        error="not configured (soft-fail)",
                        detail=f"{name}: set {env_key} for optional network ping",
                    )
                )
                continue
            r = ping(url, timeout=t_out)
            r.detail = f"{name}: {r.detail or ''}".strip()
            results.append(r)
    return results
