"""Synthetic CSV → impute → discover (offline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autocausal import AutoCausal, __version__


def _synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    age = rng.normal(40, 10, size=n)
    treatment = (0.8 * z + 0.2 * rng.normal(size=n) > 0).astype(int)
    outcome = 1.5 * treatment + 0.4 * age + 0.3 * rng.normal(size=n)
    noise = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "z": z,
            "age": age,
            "treatment": treatment,
            "outcome": outcome,
            "noise": noise,
        }
    )
    # punch missing values
    miss = rng.choice(n, size=15, replace=False)
    df.loc[miss, "age"] = np.nan
    miss2 = rng.choice(n, size=10, replace=False)
    df.loc[miss2, "outcome"] = np.nan
    return df


def test_version():
    assert __version__


def test_csv_impute_discover(tmp_path: Path):
    path = tmp_path / "synth.csv"
    _synthetic_df().to_csv(path, index=False)

    ac = AutoCausal.from_csv(path)
    assert ac.df["age"].isna().any()
    ac.impute(method="median_mode")
    assert ac.imputation is not None
    assert ac.imputation.total_missing_before > 0
    assert "age" in ac.imputation.imputed_columns

    result = ac.discover(use_iv=True, min_abs_corr=0.1)
    assert result.edges, "expected at least one edge on synthetic causal data"
    assert "treatment" in result.roles
    assert result.graph["edges"]
    md = result.to_markdown()
    assert "AutoCausal discovery report" in md
    payload = result.to_dict()
    assert "candidates" in payload
    assert "edges" in payload


def test_knn_impute():
    df = _synthetic_df(n=80)
    ac = AutoCausal.from_dataframe(df)
    ac.impute(method="knn", knn_k=3)
    assert ac.imputation is not None
    assert ac.imputation.method.startswith("knn") or ac.imputation.method == "knn"


def test_run_pipeline(tmp_path: Path):
    path = tmp_path / "s.csv"
    _synthetic_df().to_csv(path, index=False)
    result = AutoCausal.from_csv(path).run(impute_method="auto", use_iv=False)
    assert result.method == "score_pc_lite"
    assert isinstance(result.notes, list)
