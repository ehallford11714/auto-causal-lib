from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autocausal.inference import (
    AutoInference,
    AutoInferencePlanner,
    CausalSpec,
    method_support_matrix,
)
from autocausal.production import (
    CausalEvidencePolicy,
    EvidenceGateError,
    ProductionGateError,
    ProductionPolicy,
    StatisticalValidityPolicy,
)


def _confounded_data(seed: int = 4, n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    propensity = 1 / (1 + np.exp(-0.7 * x))
    treatment = (rng.random(n) < propensity).astype(int)
    outcome = 2.0 * treatment + 0.8 * x + rng.normal(size=n)
    return pd.DataFrame({"x": x, "treatment": treatment, "outcome": outcome})


@pytest.mark.parametrize("method", ["regression", "iptw", "aipw"])
def test_adjustment_estimators_recover_plausible_known_ate(method: str) -> None:
    frame = _confounded_data()
    result = AutoInference(
        CausalSpec(
            treatment="treatment",
            outcome="outcome",
            confounders=["x"],
        ),
        random_state=15,
    ).fit(frame, method=method)
    assert result.ok
    assert result.estimate == pytest.approx(2.0, abs=0.35)
    assert result.standard_error is not None
    assert result.ci_low < result.estimate < result.ci_high
    assert result.provenance["correlation_used_as_identification"] is False


def test_aipw_repeatability_and_crossfit_diagnostics() -> None:
    frame = _confounded_data(seed=11)
    spec = CausalSpec(
        treatment="treatment",
        outcome="outcome",
        confounders=["x"],
    )
    first = AutoInference(spec, random_state=31).fit(frame, method="aipw")
    second = AutoInference(spec, random_state=31).fit(frame, method="aipw")
    assert first.estimate == second.estimate
    assert first.standard_error == second.standard_error
    assert first.diagnostics["fold_local_nuisance_fits"] is True
    assert first.diagnostics["crossfit_folds"] >= 2


def test_matching_is_explicit_att_with_balance_diagnostics() -> None:
    result = AutoInference(
        CausalSpec(
            treatment="treatment",
            outcome="outcome",
            confounders=["x"],
        ),
        random_state=2,
    ).fit(_confounded_data(), method="matching")
    assert result.estimand == "ATT"
    assert result.diagnostics["n_matched_treated"] > 20
    assert "max_abs_smd_after" in result.diagnostics
    assert any("replacement" in value for value in result.warnings)


def test_strong_iv_recovers_effect_and_weak_iv_fails_closed() -> None:
    rng = np.random.default_rng(7)
    n = 900
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    d = 1.4 * z + 0.5 * x + rng.normal(size=n)
    y = 1.7 * d + 0.8 * x + rng.normal(size=n)
    strong = pd.DataFrame({"x": x, "z": z, "d": d, "y": y})
    spec = CausalSpec(
        treatment="d",
        outcome="y",
        confounders=["x"],
        instrument="z",
        instrument_provenance="observed",
    )
    result = AutoInference(spec, random_state=3).fit(strong, method="iv_2sls")
    assert result.estimate == pytest.approx(1.7, abs=0.2)
    assert result.diagnostics["first_stage_f"] > 10
    assert result.diagnostics["exclusion_restriction"] == "unverified"

    weak = strong.copy()
    weak["z"] = rng.normal(size=n)
    with pytest.raises(EvidenceGateError, match="Statistical/design assumptions"):
        AutoInference(
            spec,
            mode="production",
            policy=ProductionPolicy.strict(),
        ).fit(weak, method="iv_2sls")


def test_synthetic_instrument_is_forbidden_in_production() -> None:
    frame = _confounded_data()
    frame["auto_instrument_z"] = np.arange(len(frame))
    spec = CausalSpec(
        treatment="treatment",
        outcome="outcome",
        confounders=["x"],
        instrument="auto_instrument_z",
        instrument_provenance="synthetic",
    )
    with pytest.raises(ProductionGateError, match="design metadata"):
        AutoInference(
            spec,
            mode="production",
            policy=ProductionPolicy.strict(),
        ).fit(frame, method="iv_2sls")


def test_did_panel_rdd_and_its_native_designs() -> None:
    rng = np.random.default_rng(9)
    units = np.repeat(np.arange(80), 6)
    time = np.tile(np.arange(6), 80)
    treated_unit = (units < 40).astype(int)
    post = (time >= 3).astype(int)
    outcome = (
        0.4 * time
        + rng.normal(size=80)[units]
        + 1.3 * treated_unit * post
        + rng.normal(scale=0.4, size=len(units))
    )
    panel = pd.DataFrame(
        {
            "unit": units,
            "time": time,
            "treated": treated_unit,
            "post": post,
            "outcome": outcome,
        }
    )
    did = AutoInference(
        CausalSpec(
            treatment="treated",
            outcome="outcome",
            unit="unit",
            time="time",
            post="post",
        )
    ).fit(panel, method="did")
    assert did.estimate == pytest.approx(1.3, abs=0.25)
    assert did.diagnostics["pre_periods"] == 3

    fixed = AutoInference(
        CausalSpec(
            treatment="post",
            outcome="outcome",
            unit="unit",
            time="time",
        )
    ).fit(panel, method="panel_fixed_effects")
    assert fixed.diagnostics["n_units"] == 80

    running = rng.uniform(-2, 2, size=1000)
    treatment = (running >= 0).astype(int)
    rdd_frame = pd.DataFrame(
        {
            "running": running,
            "treatment": treatment,
            "outcome": 2.2 * treatment
            + 0.5 * running
            + rng.normal(scale=0.4, size=1000),
        }
    )
    rdd = AutoInference(
        CausalSpec(
            treatment="treatment",
            outcome="outcome",
            running="running",
            cutoff=0.0,
            bandwidth=1.0,
        )
    ).fit(rdd_frame, method="rdd")
    assert rdd.estimate == pytest.approx(2.2, abs=0.25)
    assert rdd.diagnostics["local_linear"] is True

    series_time = np.arange(120)
    series_post = (series_time >= 60).astype(int)
    its_frame = pd.DataFrame(
        {
            "time": series_time,
            "post": series_post,
            "outcome": 0.05 * series_time
            + 1.8 * series_post
            + rng.normal(scale=0.25, size=120),
        }
    )
    its = AutoInference(
        CausalSpec(
            treatment="post",
            outcome="outcome",
            time="time",
            post="post",
        )
    ).fit(its_frame, method="its")
    assert its.estimate == pytest.approx(1.8, abs=0.35)
    assert its.diagnostics["covariance"].startswith("HAC")


@pytest.mark.parametrize(
    ("method", "spec", "message"),
    [
        (
            "did",
            CausalSpec(treatment="d", outcome="y"),
            "unit=, time=, and post=",
        ),
        (
            "rdd",
            CausalSpec(treatment="d", outcome="y"),
            "running= and cutoff=",
        ),
        (
            "its",
            CausalSpec(treatment="d", outcome="y"),
            "time=",
        ),
    ],
)
def test_design_specific_methods_require_explicit_fields(
    method: str, spec: CausalSpec, message: str
) -> None:
    frame = pd.DataFrame({"d": [0, 1] * 30, "y": np.arange(60)})
    with pytest.raises(ValueError, match=message):
        AutoInference(spec).fit(frame, method=method)


def test_production_auto_method_selection_is_forbidden() -> None:
    frame = _confounded_data()
    spec = CausalSpec(
        treatment="treatment",
        outcome="outcome",
        confounders=["x"],
    )
    recommendations = AutoInferencePlanner(
        spec,
        mode="production",
        policy=ProductionPolicy.strict(),
    ).recommend(frame)
    assert recommendations
    with pytest.raises(
        ProductionGateError, match="requires an explicit method"
    ):
        AutoInference(
            spec,
            mode="production",
            policy=ProductionPolicy.strict(),
        ).fit(frame, method="auto")


def test_optional_adapter_soft_skips_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    from autocausal.backends import doubleml_backend

    monkeypatch.setattr(doubleml_backend, "available", lambda: False)
    result = AutoInference(
        CausalSpec(
            treatment="treatment",
            outcome="outcome",
            confounders=["x"],
        )
    ).fit(_confounded_data(), method="doubleml")
    assert result.soft_skip is True
    assert result.evidence_grade == "insufficient"
    assert any(gate.id == "optional_adapter:doubleml" for gate in result.gates.results)


def test_support_matrix_is_honest_about_deferred_methods() -> None:
    support = {item["method"]: item["status"] for item in method_support_matrix()}
    assert support["aipw"] == "native"
    assert support["doubleml"] == "optional_adapter"
    assert support["tmle"] == "planned_deferred"
    assert support["causalml_uplift"] == "planned_deferred"
