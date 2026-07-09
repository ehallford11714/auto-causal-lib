"""SQLite ping + public ping offline."""

from __future__ import annotations

from autocausal.db import (
    bundled_sample_url,
    connect,
    ensure_bundled_sample,
    ping,
    ping_public,
)


def test_sqlite_ping_memory():
    r = ping("sqlite:///:memory:", timeout=2.0)
    assert r.ok
    assert r.latency_ms >= 0
    assert r.dialect == "sqlite"


def test_bundled_sample_ping():
    ensure_bundled_sample()
    r = ping(bundled_sample_url(), timeout=3.0)
    assert r.ok
    h = connect(bundled_sample_url())
    try:
        tables = h.list_tables()
        assert "demo_obs" in tables
        sample = h.sample_table("demo_obs", n=10)
        assert len(sample) == 10
        prof = h.profile_table("demo_obs", sample_n=50)
        assert "columns" in prof
    finally:
        h.dispose()


def test_public_ping_offline():
    results = ping_public(include_network=False, timeout=2.0)
    assert results
    # bundled + memory should succeed; env network skipped
    by_detail = " ".join((r.detail or "") + (r.error or "") for r in results)
    assert any(r.ok for r in results)
    assert "skipped" in by_detail or "not configured" in by_detail
