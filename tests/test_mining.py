"""Mining on synthetic CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from autocausal import AutoCausal
from autocausal.mining import mine


def _df(n: int = 150, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, size=n)
    revenue = 2.0 * treatment + rng.normal(size=n)
    segment = rng.choice(["A", "B", "C"], size=n)
    return pd.DataFrame(
        {
            "treatment": treatment,
            "revenue": revenue,
            "age": rng.normal(40, 10, size=n),
            "segment": segment,
            "noise": rng.normal(size=n),
        }
    )


def test_mine_synthetic(tmp_path: Path):
    path = tmp_path / "m.csv"
    _df().to_csv(path, index=False)
    ac = AutoCausal.from_csv(path)
    ac.mine(min_score=0.05)
    report = ac.mining
    assert report is not None
    assert report.columns
    assert report.associations or report.suggestions
    assert "revenue" in report.kpis or report.kpis
    md = report.to_markdown()
    assert "mining report" in md.lower()
    assert report.to_json()


def test_mine_then_discover():
    ac = AutoCausal.from_dataframe(_df())
    ac.mine().impute()
    result = ac.discover(use_iv=False, min_abs_corr=0.05)
    assert result.edges
    assert result.mining is not None
