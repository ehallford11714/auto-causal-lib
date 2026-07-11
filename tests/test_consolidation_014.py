"""0.14 consolidation: engines→inference routing and public expansion exports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _synthetic_iv_frame(n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    treatment = (0.9 * z + 0.25 * rng.normal(size=n) > 0).astype(float)
    outcome = 1.4 * treatment + 0.35 * rng.normal(size=n)
    return pd.DataFrame(
        {
            "z": z,
            "treatment": treatment,
            "outcome": outcome,
            "confounder": rng.normal(size=n),
        }
    )


def test_engines_estimate_routes_through_inference():
    from autocausal.engines import estimate

    df = _synthetic_iv_frame()
    result = estimate(
        df,
        backend="builtin_ols",
        y="outcome",
        d="treatment",
        x=["confounder"],
        random_state=7,
    )
    assert result.ok
    assert result.backend == "autocausal.inference"
    assert result.estimate is not None
    assert "ate" in result.estimate
    assert any("unified runtime" in note.lower() for note in result.notes)


def test_engines_iv_estimate_routes_through_inference():
    from autocausal.engines import estimate

    df = _synthetic_iv_frame()
    result = estimate(
        df,
        backend="builtin_2sls",
        y="outcome",
        d="treatment",
        z="z",
        x=["confounder"],
        random_state=7,
    )
    assert result.ok
    assert result.method in ("iv_2sls", "builtin_2sls")
    assert result.backend == "autocausal.inference"
    assert result.estimate is not None
    assert result.estimate.get("ate") is not None


def test_top_level_expansion_exports():
    import autocausal as ac

    assert ac.__version__ == "0.14.0"
    assert ac.DeepResearchSuite is not None
    assert ac.ReportEngine is not None
    assert ac.AutoTabularML is not None
    assert ac.AutoVizSuite is not None
    assert ac.list_integrations is not None
    assert ac.CapabilityRouter is not None


def test_autocausal_tabular_ml_and_autoviz():
    from autocausal import AutoCausal

    df = _synthetic_iv_frame()
    ac = AutoCausal.from_dataframe(df)
    ml_report = ac.tabular_ml(target="outcome", features=["treatment", "confounder", "z"])
    assert getattr(ml_report, "selected_name", None)
    viz_report = ac.autoviz(use_slm=False)
    assert viz_report is not None
    assert hasattr(viz_report, "plan") or hasattr(viz_report, "to_dict")


def test_public_estimate_uses_inference_backend():
    from autocausal import AutoCausal

    df = _synthetic_iv_frame()
    ac = AutoCausal.from_dataframe(df)
    result = ac.estimate(
        backend="builtin_ols",
        y="outcome",
        d="treatment",
        x=["confounder"],
    )
    assert result.ok
    assert result.backend == "autocausal.inference"


def test_resolve_roles_respects_explicit_empty_controls():
    from autocausal.backends._common import resolve_roles

    df = _synthetic_iv_frame()
    filled = resolve_roles(df, y="outcome", d="treatment", x=None)
    empty = resolve_roles(df, y="outcome", d="treatment", x=[])
    assert "confounder" in filled["x"]
    assert empty["x"] == []


def test_chartspec_bridges_to_report_layer():
    from autocausal.autochart import AccessibilitySpec, ChartSpec

    chart = ChartSpec(
        id="corr-1",
        type="correlation",
        title="Association heatmap",
        accessibility=AccessibilitySpec(alt_text="Matrix of pairwise associations"),
    )
    report_chart = chart.to_report_chart_spec(priority=70)
    assert report_chart.chart_type == "correlation"
    assert report_chart.alt_text
    assert report_chart.spec["schema"] == "AutoCausalChartSpec.v1"
    assert report_chart.priority == 70


def test_mcp_registers_consolidation_tools():
    from autocausal.mcp.registry import build_default_registry

    names = build_default_registry().list_names()
    assert "autocausal_correlate" in names
    assert "autocausal_tabular_ml" in names
    assert "autocausal_autoviz" in names


def test_cli_exposes_consolidation_commands():
    from autocausal.cli import _build_parser

    parser = _build_parser()
    choices = set()
    for action in parser._actions:
        if getattr(action, "choices", None) and isinstance(action.choices, dict):
            choices.update(action.choices.keys())
    assert {"correlate", "tabular-ml", "autoviz", "report-artifact"} <= choices
