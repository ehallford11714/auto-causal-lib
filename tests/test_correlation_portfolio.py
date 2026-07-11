from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autocausal.correlation import (
    ASSOCIATION_NOTICE,
    CorrelationSuite,
    correlation,
    correlation_matrix,
)


def test_continuous_correlations_partial_and_bootstrap_are_deterministic() -> None:
    rng = np.random.default_rng(12)
    n = 350
    control = rng.normal(size=n)
    x = 0.8 * control + rng.normal(scale=0.5, size=n)
    y = 0.7 * control + 0.5 * x + rng.normal(scale=0.6, size=n)
    frame = pd.DataFrame({"x": x, "y": y, "control": control})

    pearson = correlation("x", "y", data=frame, method="pearson")
    spearman = correlation("x", "y", data=frame, method="spearman")
    partial_a = correlation(
        "x",
        "y",
        data=frame,
        method="partial",
        controls=["control"],
        bootstrap_n=80,
        random_state=9,
    )
    partial_b = correlation(
        "x",
        "y",
        data=frame,
        method="partial",
        controls=["control"],
        bootstrap_n=80,
        random_state=9,
    )

    assert pearson.coefficient is not None and pearson.coefficient > 0.7
    assert spearman.coefficient is not None and spearman.coefficient > 0.65
    assert partial_a.coefficient is not None and partial_a.coefficient > 0.25
    assert partial_a.ci_low == partial_b.ci_low
    assert partial_a.ci_high == partial_b.ci_high
    assert partial_a.epistemic_notice == ASSOCIATION_NOTICE


def test_mixed_type_measures_are_typed() -> None:
    rng = np.random.default_rng(3)
    n = 300
    binary = rng.integers(0, 2, size=n)
    continuous = 1.2 * binary + rng.normal(size=n)
    category = np.where(continuous > 0.5, "high", "low")
    category_three = np.where(
        continuous < -0.5, "a", np.where(continuous > 0.8, "c", "b")
    )
    frame = pd.DataFrame(
        {
            "binary": binary,
            "continuous": continuous,
            "category": category,
            "category_three": category_three,
        }
    )

    point = correlation("binary", "continuous", data=frame, method="auto")
    phi = correlation("binary", "category", data=frame, method="phi")
    cramer = correlation(
        "category", "category_three", data=frame, method="cramers_v"
    )
    eta = correlation(
        "category_three", "continuous", data=frame, method="correlation_ratio"
    )
    distance = correlation(
        "continuous",
        pd.Series(continuous**2, name="squared"),
        data=frame,
        method="distance_correlation",
        permutation_n=20,
        random_state=4,
    )

    assert point.measure == "point_biserial"
    assert phi.measure == "phi"
    assert cramer.measure == "cramers_v_bias_corrected"
    assert eta.measure == "correlation_ratio_eta"
    assert eta.metadata["eta_squared"] > 0
    assert distance.coefficient is not None and distance.coefficient > 0.4
    assert distance.p_value is not None


def test_matrix_scan_attaches_bh_fdr_q_values() -> None:
    rng = np.random.default_rng(21)
    n = 400
    a = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "a": a,
            "b": a + rng.normal(scale=0.1, size=n),
            "null_1": rng.normal(size=n),
            "null_2": rng.normal(size=n),
        }
    )
    result = correlation_matrix(frame, random_state=8)
    strong = next(
        value
        for value in result.results
        if {value.x, value.y} == {"a", "b"}
    )
    assert strong.q_value is not None and strong.q_value < 0.01
    assert strong.fdr_reject_null is True
    assert result.coefficients().shape == (4, 4)
    assert result.to_dict()["epistemic_notice"] == ASSOCIATION_NOTICE


def test_weighted_and_cluster_bootstrap_metadata() -> None:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=120),
            "y": rng.normal(size=120),
            "weight": rng.uniform(0.5, 2.0, size=120),
            "cluster": np.repeat(np.arange(30), 4),
        }
    )
    result = CorrelationSuite(frame, bootstrap_n=40, random_state=2).correlation(
        "x",
        "y",
        method="pearson",
        weights="weight",
        cluster="cluster",
    )
    assert result.weights == "weight"
    assert result.cluster == "cluster"
    assert result.p_value is None
    assert result.ci_low is not None
    assert any("Clustered" in value for value in result.warnings)


def test_invalid_partial_and_weighted_rank_are_explicit() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 5], "w": [1, 1, 1]})
    with pytest.raises(ValueError, match="controls"):
        correlation("x", "y", data=frame, method="partial")
    with pytest.raises(ValueError, match="Weighted Spearman"):
        correlation("x", "y", data=frame, method="spearman", weights="w")
