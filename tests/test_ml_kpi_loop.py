"""Offline KPI ML loop tests (torch gated)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autocausal.ml import (
    KPIMinedCausalLoop,
    ModelConstructPlan,
    construct_model_plan,
    torch_available,
)
from autocausal.ml.imputers import apply_imputer
from autocausal.cli import main


def _demo_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spend = rng.normal(100, 20, n)
    noise = rng.normal(0, 5, n)
    revenue = 2.0 * spend + noise
    df = pd.DataFrame(
        {
            "spend": spend,
            "revenue": revenue,
            "region": rng.choice(["N", "S", "E", "W"], size=n),
            "z_assign": rng.integers(0, 2, n),
        }
    )
    # inject missingness
    miss = rng.random(n) < 0.15
    df.loc[miss, "spend"] = np.nan
    miss2 = rng.random(n) < 0.1
    df.loc[miss2, "revenue"] = np.nan
    return df


def test_construct_plan_rule_median_default():
    df = _demo_df()
    ctx = {
        "text": "what drives revenue?",
        "columns": [{"name": c} for c in df.columns],
        "kpis": ["revenue", "spend"],
        "associations": [{"a": "spend", "b": "revenue", "score": 0.8, "metric": "pearson"}],
    }
    plan = construct_model_plan(ctx, use_slm=False, use_torch=False, guides=["rule"])
    assert isinstance(plan, ModelConstructPlan)
    assert plan.imputer in ("median", "iterative", "sklearn")
    assert plan.imputer != "torch_mlp"
    assert "revenue" in plan.kpi_focus or plan.outcome == "revenue"


def test_kpi_loop_offline_median(tmp_path: Path):
    df = _demo_df()
    path = tmp_path / "demo.csv"
    df.to_csv(path, index=False)
    result = KPIMinedCausalLoop.from_csv(path).run(
        text="what drives revenue?",
        use_slm=False,
        use_torch=False,
        guides=["rule"],
        horizon=3,
        physics=True,
        epochs=5,
    )
    assert result.plan.imputer != "torch_mlp" or not torch_available()
    assert result.fit.schema == "FitReport.v1"
    assert isinstance(result.edges, list)
    assert result.mining is not None
    md = result.to_markdown()
    assert "Model construct plan" in md or "KPI-mined" in md


def test_fit_imputer_median():
    df = _demo_df()
    out, meta, fit = apply_imputer(df, "median")
    assert out.isna().sum().sum() == 0 or meta.get("method") == "median_mode"
    assert fit.imputer
    assert fit.torch_used is False


def test_cli_ml_loop(tmp_path: Path):
    df = _demo_df()
    path = tmp_path / "demo.csv"
    df.to_csv(path, index=False)
    code = main(["ml", "loop", "--csv", str(path), "--text", "what drives revenue?", "--horizon", "2", "--no-physics"])
    assert code == 0


def test_cli_ml_fit_imputer(tmp_path: Path):
    df = _demo_df()
    path = tmp_path / "demo.csv"
    df.to_csv(path, index=False)
    code = main(["ml", "fit-imputer", "--csv", str(path), "--backend", "median"])
    assert code == 0


def test_autocausal_ml_loop():
    from autocausal import AutoCausal

    ac = AutoCausal.from_dataframe(_demo_df())
    result = ac.ml_loop(text="revenue", use_torch=False, horizon=2, physics=False)
    assert result.fit is not None
    assert result.plan.backend


@pytest.mark.skipif(
    not torch_available() or os.environ.get("AUTOCAUSAL_TORCH_TEST", "").strip() not in ("1", "true", "yes"),
    reason="Requires torch + AUTOCAUSAL_TORCH_TEST=1",
)
def test_torch_imputer_gated():
    df = _demo_df(n=60)
    out, meta, fit = apply_imputer(df, "torch_mlp", columns=["spend", "revenue"], epochs=15)
    assert meta.get("method") == "torch_mlp"
    assert fit.torch_used is True
    assert out[["spend", "revenue"]].isna().sum().sum() == 0


@pytest.mark.skipif(
    not torch_available() or os.environ.get("AUTOCAUSAL_TORCH_TEST", "").strip() not in ("1", "true", "yes"),
    reason="Requires torch + AUTOCAUSAL_TORCH_TEST=1",
)
def test_kpi_loop_torch_gated(tmp_path: Path):
    df = _demo_df(n=60)
    path = tmp_path / "demo.csv"
    df.to_csv(path, index=False)
    result = KPIMinedCausalLoop.from_csv(path).run(
        text="what drives revenue?",
        use_torch=True,
        guides=["rule"],
        horizon=2,
        physics=False,
        epochs=15,
    )
    assert result.plan.imputer == "torch_mlp"
    assert result.fit.torch_used is True
