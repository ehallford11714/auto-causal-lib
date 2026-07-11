from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autocausal import AutoCausal
from autocausal.ml import AutoML
from autocausal.production import (
    GateReport,
    GateResult,
    ProductionGateError,
    ProductionPolicy,
    apply_mode_defaults,
    resolve_policy,
    run_production_pipeline,
)
from autocausal.suites.autocleanse import AutoCleanseSuite
from autocausal.suites.autoeda import AutoEDASuite


def _frame(seed: int = 6, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = (rng.random(n) < 1 / (1 + np.exp(-0.3 * x))).astype(int)
    y = 1.4 * d + 0.6 * x + rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "d": d,
            "y": y,
            "group": np.repeat(np.arange(n // 4), 4),
        }
    )


def test_review_profile_escalates_and_policy_profiles_stay_aligned() -> None:
    policy = resolve_policy("review")
    assert policy.profile == "review"
    settings = apply_mode_defaults(
        mode="review",
        policy=policy,
        auto_instrument=True,
        allow_iv_fallback=True,
    )
    assert settings.mode == "review"
    assert settings.auto_instrument is False
    assert settings.allow_iv_fallback is False
    assert any(note.startswith("REVIEW:") for note in settings.notes)

    gates = GateReport(profile="review")
    gates.add(
        GateResult(
            id="domain_review",
            ok=False,
            status="escalate",
            detail="Identification assumption requires domain review.",
        )
    )
    assert gates.ok is False
    assert gates.failed[0].status == "escalate"

    session = AutoCausal.from_dataframe(_frame(n=100), mode="review")
    assert session.mode == "review"
    assert session.policy.profile == "review"
    with pytest.raises(ProductionGateError) as exc:
        AutoCausal.from_dataframe(
            _frame(n=100),
            mode="production",
            policy=ProductionPolicy.exploratory(),
        )
    assert exc.value.code == "policy_profile_mismatch"


def test_cleanse_dry_run_fingerprints_and_rollback() -> None:
    frame = _frame(n=200)
    frame.loc[[0, 1, 2], "x"] = np.nan
    frame = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
    suite = AutoCleanseSuite(
        frame,
        mode="exploratory",
        dry_run=True,
        use_slm=False,
    ).run()
    assert suite.frame is not None and suite.report is not None
    pd.testing.assert_frame_equal(suite.frame, frame)
    assert suite.report.dry_run is True
    assert suite.report.before_fingerprint
    assert suite.report.after_fingerprint
    assert suite.report.reversible is True
    rolled_back = suite.rollback()
    pd.testing.assert_frame_equal(rolled_back, frame)


def test_train_test_imputation_is_fit_on_train_and_redacted() -> None:
    train = pd.DataFrame(
        {"x": [1.0, 2.0, np.nan], "category": ["a", None, "a"]}
    )
    test = pd.DataFrame(
        {"x": [100.0, np.nan], "category": ["b", None]}
    )
    train_out, test_out, ledger = AutoCleanseSuite.transform_train_test(
        train, test
    )
    assert test_out.loc[1, "x"] == 1.5
    assert test_out.loc[1, "category"] == "a"
    assert all(
        item["fit_scope"] == "train_only"
        for item in ledger["transformations"]
    )
    assert all(
        item["fill_value"] == "<redacted>"
        for item in ledger["transformations"]
    )


def test_eda_gate_inputs_include_typed_associations_without_raw_values() -> None:
    report = AutoEDASuite(
        _frame(),
        use_slm=False,
        treatment="d",
        outcome="y",
        confounders=["x"],
        unit="group",
    ).run().report
    assert report is not None
    inputs = report.to_gate_inputs()
    assert inputs["raw_values_included"] is False
    assert inputs["roles"]["treatment"] == "d"
    assert inputs["roles"]["outcome"] == "y"
    assert inputs["association_tests"]
    assert all(
        item["identification_evidence"] is False
        for item in inputs["association_tests"]
    )
    assert report.descriptive_findings
    assert report.causal_readiness_findings


def test_automl_uses_fold_local_preprocessing_and_is_deterministic() -> None:
    frame = _frame()
    first = AutoML(random_state=13).fit(
        frame,
        target="d",
        features=["x"],
        candidates=["dummy", "logistic"],
    )
    second = AutoML(random_state=13).fit(
        frame,
        target="d",
        features=["x"],
        candidates=["dummy", "logistic"],
    )
    assert first.cv_plan["preprocessing_fit_scope"] == "training_fold_only"
    assert first.raw_predictions_included is False
    assert first.selected_model == second.selected_model
    assert [value.fold_scores for value in first.candidates] == [
        value.fold_scores for value in second.candidates
    ]
    assert "do not identify causal effects" in first.epistemic_notice


def test_automl_production_rejects_direct_target_leakage() -> None:
    frame = _frame()
    frame["ground_truth_copy"] = frame["d"]
    with pytest.raises(Exception, match="AutoML blocked"):
        AutoML(
            mode="production",
            policy=ProductionPolicy.strict(),
        ).fit(
            frame,
            target="d",
            features=["x", "ground_truth_copy"],
        )


def test_pipeline_stops_for_method_review_or_runs_explicit_method() -> None:
    frame = _frame(n=500)
    review = run_production_pipeline(
        frame,
        treatment="d",
        outcome="y",
        confounders=["x"],
        method=None,
        random_state=8,
    )
    assert review.status == "review_required"
    assert review.inference_result is None
    assert any(
        gate.id == "inference_method_review" for gate in review.gates.results
    )
    assert review.to_dict()["frame_included"] is False

    fitted = run_production_pipeline(
        frame,
        treatment="d",
        outcome="y",
        confounders=["x"],
        method="aipw",
        random_state=8,
    )
    assert fitted.status == "ok"
    assert fitted.inference_result.estimate == pytest.approx(1.4, abs=0.35)
    assert fitted.inference_result.evidence_grade == "supported"


def test_autocausal_correlation_inference_and_production_wrappers() -> None:
    session = AutoCausal.from_dataframe(_frame(n=500), random_state=10)
    association = session.correlate("x", "y", method="spearman")
    assert association.measure == "spearman"

    inference = session.infer(
        spec={
            "treatment": "d",
            "outcome": "y",
            "confounders": ["x"],
        },
        method="aipw",
    )
    assert inference.method == "aipw"
    bundle = session.to_fabric_bundle()
    assert bundle["payload"]["meta"]["inference"][0]["estimand"] == "ATE"

    check = session.production_check(
        treatment="d",
        outcome="y",
        confounders=["x"],
    )
    assert check.status == "review_required"
