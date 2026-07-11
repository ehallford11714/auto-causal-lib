"""Ergonomic DiscoveryResult.report() + autocausal.engines import surface."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autocausal import AutoCausal, DiscoveryResult, __version__, engine_status, list_engines
from autocausal.engines import (
    connectivity_map,
    discover_with,
    engine_status as eng_status,
    estimate,
    list_engines as eng_list,
    refute,
)
from autocausal.results import AutoResult


def _toy_df(n: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = 0.7 * z + rng.normal(scale=0.4, size=n)
    y = 0.6 * x + 0.2 * z + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"z": z, "x": x, "y": y})


def test_version_0_11_2():
    assert __version__ == "0.11.2"


def test_discovery_result_report_alias():
    ac = AutoCausal.from_dataframe(_toy_df())
    result = ac.discover()
    assert isinstance(result, DiscoveryResult)
    md = result.report()
    assert isinstance(md, str)
    assert len(md) > 20
    assert md == result.to_markdown()
    js = result.report(as_markdown=False)
    assert js.strip().startswith("{")
    assert js == result.to_json()


def test_auto_causal_report_after_discover():
    ac = AutoCausal.from_dataframe(_toy_df())
    result = ac.discover()
    assert ac.report() == result.report()
    assert ac.report() == result.to_markdown()


def test_auto_result_report_alias():
    ac = AutoCausal.from_dataframe(_toy_df())
    discovery = ac.discover()
    auto = AutoResult(discovery=discovery, source="test")
    assert auto.report() == auto.to_markdown()
    assert auto.report(as_markdown=False) == auto.to_json()


def test_import_autocausal_engines():
    import autocausal.engines as engines

    assert hasattr(engines, "list_engines")
    assert hasattr(engines, "engine_status")
    assert hasattr(engines, "estimate")
    assert hasattr(engines, "refute")
    assert hasattr(engines, "discover_with")
    assert hasattr(engines, "connectivity_map")
    engines_list = eng_list()
    assert len(engines_list) >= 5
    status = eng_status()
    assert status["schema"] == "AutoCausalEngineStatus.v1"
    # Top-level lazy re-exports
    assert list_engines is not None
    assert engine_status is not None
    assert len(list_engines()) == len(engines_list)
    assert connectivity_map()["library"]["status"]
    # Smoke: estimate / refute / discover_with reachable from engines
    df = _toy_df()
    assert discover_with(df, method="score_pc_lite")["ok"]
    assert estimate(df, backend="builtin_ols", y="y", d="x").ok
    assert refute({"source": "x", "target": "y"}, method="placebo", df=df) is not None
