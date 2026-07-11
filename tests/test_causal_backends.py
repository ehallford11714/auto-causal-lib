"""Offline tests for 0.11 causal backends + engines connectivity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autocausal import AutoCausal, __version__
from autocausal.backends import backend_status
from autocausal.engines import (
    connectivity_map,
    discover_with,
    engine_status,
    estimate,
    list_engines,
)
from autocausal.suite_tools import get_tool, list_tools, refute


def _toy_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = 0.7 * z + rng.normal(scale=0.4, size=n)
    y = 0.6 * x + 0.2 * z + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"z": z, "x": x, "y": y, "noise": rng.normal(size=n)})


def test_version_0_13():
    assert __version__.startswith("0.14")


def test_backend_status_catalog():
    st = backend_status()
    assert "discovery" in st and "estimate" in st and "refute" in st
    assert st["discovery"]["score_pc_lite"]["available"] is True
    # soft backends may or may not be installed
    assert "causal_learn_pc" in st["discovery"]
    assert "doubleml" in st["estimate"]
    assert "dowhy" in st["refute"]


def test_list_engines_and_connectivity():
    engines = list_engines()
    ids = {e.id for e in engines}
    assert "score_pc_lite" in ids
    assert "insight" in ids
    assert "mcp" in ids
    assert "skilling" in ids
    assert "cli" in ids
    full = engine_status()
    assert full["schema"] == "AutoCausalEngineStatus.v1"
    cmap = connectivity_map()
    assert "mcp_tools" in cmap
    assert "autocausal_list_engines" in cmap["mcp_tools"]


def test_discover_with_soft_skip_wiring():
    df = _toy_df()
    # Always works for builtin
    raw = discover_with(df, method="score_pc_lite")
    assert raw["ok"]
    assert isinstance(raw.get("edges"), list)
    # Soft path always returns a dict (skip or edges)
    for m in ("causal_learn_pc", "lingam", "gcastle_notears"):
        out = discover_with(df, method=m)
        assert out["ok"] or out.get("error")
        assert "notes" in out
        assert "edges" in out


def test_estimate_builtin_ols():
    df = _toy_df(100)
    res = estimate(df, backend="builtin_ols", y="y", d="x", x=["z"])
    assert res.ok
    assert res.estimate is not None
    assert "ate" in res.estimate
    # Soft skip when missing
    soft = estimate(df, backend="doubleml", y="y", d="x")
    assert soft.ok
    assert soft.soft_skip or soft.estimate is not None


def test_ac_estimate_and_refute():
    ac = AutoCausal.from_dataframe(_toy_df(90))
    ac.mine()
    ac.discover(qc="off", use_iv=False)
    est = ac.estimate(backend="builtin_ols")
    assert est.ok
    r = ac.refute(method="placebo")
    assert r.ok
    soft = ac.refute(method="dowhy")
    assert soft.ok
    # soft_skip when missing; real run when installed
    assert soft.soft_skip or soft.backend.startswith("dowhy")


def test_ensemble_accepts_external_method_names():
    df = _toy_df(60)
    ac = AutoCausal.from_dataframe(df)
    # Explicit methods including soft name — must not crash
    res = ac.discover(
        methods=["score_pc_lite", "causal_learn_pc"],
        qc="off",
        use_iv=False,
        min_methods=1,
    )
    assert res.method == "consensus"
    assert "score_pc_lite" in (res.ensemble_methods or [])


def test_suite_tools_register_new_adapters():
    ids = {t.id for t in list_tools(category="causal")}
    assert "doubleml" in ids
    assert "causal_learn" in ids
    assert "dowhy" in ids
    assert "econml" in ids
    # invoke without df → available or missing, never crash
    from autocausal.suite_tools import invoke_tool

    for tid in ("doubleml", "econml", "dowhy", "causal_learn"):
        r = invoke_tool(tid)
        assert r.tool_id == tid


def test_mcp_registry_has_engine_tools():
    from autocausal.mcp.registry import build_default_registry

    reg = build_default_registry()
    names = set(reg.list_names())
    assert "autocausal_list_engines" in names
    assert "autocausal_estimate" in names
    assert "autocausal_refute" in names
    assert "autocausal_discover" in names
    assert "autocausal_insight_loop" in names
    assert "autocausal_skilling_list" in names
    out = reg.invoke("autocausal_list_engines", {})
    assert out.get("ok") is True


def test_package_modules_importable():
    import autocausal.insight  # noqa: F401
    import autocausal.mcp  # noqa: F401
    import autocausal.skilling  # noqa: F401
    import autocausal.backends  # noqa: F401
    import autocausal.engines  # noqa: F401
    import autocausal.agentic  # noqa: F401
    import autocausal.grail  # noqa: F401
    import autocausal.suites  # noqa: F401
    import autocausal.connective  # noqa: F401


def test_refute_placebo_still_builtin():
    df = _toy_df(50)
    edge = {"source": "x", "target": "y"}
    r = refute(edge, method="placebo", df=df, seed=0)
    assert r.backend == "builtin"
    assert "placebo_corr" in r.data
