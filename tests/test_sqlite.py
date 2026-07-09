"""SQLite SQLAlchemy path (no network)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from autocausal import AutoCausal


@pytest.fixture
def sqlite_url(tmp_path):
    """Seed via sqlite3 DB-API (avoids pandas/SQLAlchemy version quirks on to_sql)."""
    db = tmp_path / "test.db"
    url = f"sqlite:///{db.as_posix()}"
    rng = np.random.default_rng(1)
    n = 120
    z = rng.normal(size=n)
    treatment = (z + rng.normal(scale=0.5, size=n) > 0).astype(float)
    outcome = 2.0 * treatment + rng.normal(size=n)
    df = pd.DataFrame(
        {"z": z, "treatment": treatment, "outcome": outcome, "u": rng.normal(size=n)}
    )
    with sqlite3.connect(db.as_posix()) as conn:
        df.to_sql("obs", conn, index=False, if_exists="replace")
    return url


def test_from_sqlalchemy_table(sqlite_url):
    ac = AutoCausal.from_sqlalchemy(sqlite_url, table="obs")
    assert len(ac.df) == 120
    result = ac.run(use_iv=True, min_abs_corr=0.05)
    assert result.edges
    involved = {c for e in result.edges for c in (e["source"], e["target"])}
    assert "outcome" in involved or "treatment" in involved


def test_from_sqlalchemy_query(sqlite_url):
    ac = AutoCausal.from_sqlalchemy(
        sqlite_url, query="SELECT z, treatment, outcome FROM obs LIMIT 50"
    )
    assert len(ac.df) == 50
    ac.impute().discover(use_iv=False)


@pytest.mark.skip(reason="Live Postgres not required; use offline SQLite tests")
def test_live_postgres_skipped():
    AutoCausal.from_sqlalchemy("postgresql+psycopg2://localhost/db", table="t")


@pytest.mark.skip(reason="Live Vertica not required; use offline SQLite tests")
def test_live_vertica_skipped():
    AutoCausal.from_sqlalchemy("vertica+vertica_python://localhost/db", table="t")
