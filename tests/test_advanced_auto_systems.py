"""Offline tests for AutoViz, AutoChart, AutoTabularML, and AutoNLP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autocausal.autochart import AutoChart, ChartSpec, ChartSpecError
from autocausal.automl import (
    AutoTabularML,
    infer_task,
    load_trusted_model,
    make_splits,
)
from autocausal.autonlp import (
    AutoNLPSuite,
    FoldSafeTextVectorizer,
    TextRetriever,
    aggregate_text_features,
    extract_causal_claims,
)
from autocausal.autoviz import AutoVizSuite


def _causal_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    treatment = rng.integers(0, 2, n)
    age = rng.normal(45, 12, n)
    instrument = rng.integers(0, 2, n)
    outcome = 0.8 * treatment + 0.03 * age + rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=n, freq="D"),
            "group": np.repeat(np.arange(n // 4), 4),
            "instrument": instrument,
            "treatment": treatment,
            "age": age,
            "outcome": outcome,
            "segment": rng.choice(["north", "south", "east"], n),
        }
    )
    frame.loc[::9, "age"] = np.nan
    return frame


def _titanic_like(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    sex = rng.choice(["female", "male"], n)
    passenger_class = rng.choice(["first", "second", "third"], n, p=[0.2, 0.25, 0.55])
    age = np.clip(rng.normal(32, 14, n), 0.5, 85)
    fare = np.exp(rng.normal(3.0, 0.8, n))
    logit = (
        1.5 * (sex == "female")
        + 0.8 * (passenger_class == "first")
        - 0.5 * (passenger_class == "third")
        - 0.015 * age
    )
    probability = 1 / (1 + np.exp(-logit))
    survived = rng.binomial(1, probability)
    frame = pd.DataFrame(
        {
            "survived": survived,
            "sex": sex,
            "passenger_class": passenger_class,
            "age": age,
            "fare": fare,
            "embarked": rng.choice(["S", "C", "Q"], n),
            "family_id": np.repeat(np.arange(n // 4), 4),
        }
    )
    frame.loc[::11, "age"] = np.nan
    frame.loc[::17, "embarked"] = None
    return frame


def test_autoviz_analysis_aware_plan_and_safe_enrichment():
    frame = _causal_frame()
    suite = AutoVizSuite(
        frame,
        mode="production",
        panel={"entity": "group", "time": "time"},
        candidates={
            "treatment": ["treatment"],
            "outcome": ["outcome"],
            "instrument": ["instrument"],
            "confounder": ["age"],
        },
        edges=[
            {
                "source": "treatment",
                "target": "outcome",
                "stability": 0.78,
                "evidence_grade": "exploratory",
            }
        ],
        model_metrics={"task": "binary_classification", "brier": 0.18},
        gate_results=[{"id": "sample", "ok": False, "status": "fail"}],
    )

    def enrich(payload):
        assert payload["contains_raw_values"] is False
        return {
            "recommendations": [
                {
                    "id": "custom:valid",
                    "chart_type": "distribution",
                    "title": "Outcome detail",
                    "priority": 55,
                    "rationale": "Review outcome support.",
                    "required_columns": ["outcome"],
                },
                {
                    "id": "custom:invalid",
                    "chart_type": "made_up",
                    "title": "Invalid",
                    "priority": 10,
                    "rationale": "Must be rejected.",
                },
            ]
        }

    report = suite.run(use_slm=True, slm_enricher=enrich)
    chart_types = {item.chart_type for item in report.plan.recommendations}
    assert {
        "missingness",
        "iv_first_stage",
        "covariate_balance",
        "overlap",
        "edge_stability",
        "dag",
        "panel_trend",
        "gate_dashboard",
        "calibration",
    }.issubset(chart_types)
    assert any(item.id == "custom:valid" for item in report.plan.recommendations)
    assert not any(item.id == "custom:invalid" for item in report.plan.recommendations)
    payload = report.to_dict()
    assert payload["contains_raw_values"] is False
    assert payload["plan"]["frame_summary"]["redacted"] is True
    assert "not establish" in report.to_markdown().lower()
    assert report.report() == report.to_markdown()
    assert report.plan.report() == report.plan.to_markdown()


def test_autochart_validation_determinism_and_production_redaction(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "category": ["secret-a", "secret-b", "secret-a", "secret-c"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(ChartSpecError):
        ChartSpec("bad", "scatter", "Bad", x="missing", y="value").validate(frame)

    spec = ChartSpec(
        "distribution",
        "distribution",
        "Private category distribution",
        x="category",
        random_state=9,
    )
    first = AutoChart(spec, backend="data", production=True).render(frame)
    second = AutoChart(spec, backend="data", production=True).render(frame)
    assert first.data_payload == second.data_payload
    serialized = first.to_json()
    assert "secret-a" not in serialized
    assert first.provenance["contains_raw_values"] is False
    assert first.spec.accessibility.alt_text
    output = first.write(tmp_path / "chart.json")
    assert json.loads(output.read_text(encoding="utf-8"))["backend"] == "data"


def test_autochart_causal_renderers_data_only():
    frame = _causal_frame(40)
    edges = [
        {
            "source": "treatment",
            "target": "outcome",
            "stability": 0.8,
            "evidence_grade": "exploratory",
        }
    ]
    dag = AutoChart(
        ChartSpec("dag", "dag", "Hypothesis graph"), backend="data"
    ).render(frame, context={"edges": edges})
    assert dag.data_payload["records"][0]["source"] == "treatment"
    assert dag.provenance["causal_interpretation"] == "not_established"
    gates = AutoChart(
        ChartSpec("gates", "gate_dashboard", "Gates"), backend="data"
    ).render(
        frame,
        context={"gates": [{"id": "overlap", "status": "warn", "ok": True}]},
    )
    assert gates.data_payload["records"][0]["status"] == "warn"
    first_stage = AutoChart(
        ChartSpec(
            "first-stage",
            "iv_first_stage",
            "First stage",
            x="instrument",
            y="treatment",
        ),
        backend="data",
        production=True,
    ).render(frame)
    assert first_stage.data_payload["aggregated"] is True


def test_autochart_auto_backend_dependency_fallback(monkeypatch):
    import autocausal.autochart.renderer as renderer

    monkeypatch.setattr(renderer, "_plotly_available", lambda: False)
    monkeypatch.setattr(renderer, "_matplotlib_available", lambda: False)
    frame = pd.DataFrame({"x": [1, 2, 3]})
    rendered = AutoChart(
        ChartSpec("fallback", "distribution", "Fallback", x="x"),
        backend="auto",
    ).render(frame)
    assert rendered.backend == "data"
    assert any("falling back" in warning for warning in rendered.warnings)


def test_autochart_matplotlib_is_headless(tmp_path: Path):
    pytest.importorskip("matplotlib")
    frame = pd.DataFrame({"x": np.arange(20)})
    rendered = AutoChart(
        ChartSpec("hist", "distribution", "Distribution", x="x"),
        backend="matplotlib",
    ).render(frame)
    assert rendered.backend == "matplotlib"
    assert rendered.save(tmp_path / "chart.png").exists()
    rendered.close()


def test_iris_autoviz_and_automl_are_deterministic():
    sklearn_datasets = pytest.importorskip("sklearn.datasets")
    frame = sklearn_datasets.load_iris(as_frame=True).frame
    viz = AutoVizSuite(frame).run()
    assert viz.plan.recommendations
    first = AutoTabularML(random_state=7, cv=3, max_candidates=2).run(
        frame,
        target="target",
        compute_importance=False,
    )
    second = AutoTabularML(random_state=7, cv=3, max_candidates=2).run(
        frame,
        target="target",
        compute_importance=False,
    )
    assert first.task.task_type == "multiclass_classification"
    assert first.selected_name == second.selected_name
    first_metric = first.selected.metrics["balanced_accuracy"]
    second_metric = second.selected.metrics["balanced_accuracy"]
    assert first_metric.values == second_metric.values
    assert all(
        fold["preprocessing_fit_scope"] == "fold_train_only"
        for fold in first.split_plan.to_dict()["folds"]
    )
    assert "target" not in first.feature_schema.features
    assert first.report() == first.to_markdown()
    assert first.to_dict()["model_selection_ledger"]


def test_titanic_like_group_aware_automl_and_ledger():
    frame = _titanic_like()
    report = AutoTabularML(random_state=2, cv=3, max_candidates=2).run(
        frame,
        target="survived",
        group_column="family_id",
        candidates=["dummy", "logistic"],
        subgroup_columns=["sex"],
        compute_importance=True,
    )
    assert report.task.task_type == "binary_classification"
    assert report.split_plan.strategy == "group"
    assert report.selected_name in {"dummy", "logistic"}
    assert report.selected_pipeline is not None
    assert report.feature_importance
    assert "sex" in report.subgroup_performance
    assert any(gate["id"] == "automl_calibration" for gate in report.gates)
    assert "predictive" in report.to_markdown().lower()
    viz = AutoVizSuite(frame, model_metrics=report).run()
    chart_types = {item.chart_type for item in viz.plan.recommendations}
    assert "calibration" in chart_types
    assert "feature_importance" in chart_types
    assert "gate_dashboard" in chart_types


def test_automl_production_target_and_safe_persistence(tmp_path: Path):
    frame = _titanic_like(80)
    with pytest.raises(ValueError, match="explicit target"):
        AutoTabularML(mode="production", cv=2, max_candidates=1).run(
            frame,
            compute_importance=False,
        )
    report = AutoTabularML(
        mode="production", random_state=3, cv=2, max_candidates=1
    ).run(
        frame,
        target="survived",
        candidates=["logistic"],
        compute_importance=False,
    )
    artifact, manifest = report.save_model(tmp_path / "model.joblib")
    assert artifact.exists() and manifest.exists()
    with pytest.raises(ValueError, match="untrusted"):
        load_trusted_model(artifact)
    loaded = load_trusted_model(artifact, trusted=True)
    assert hasattr(loaded, "predict")


def test_tabular_automl_accepts_shared_policy_without_hard_dependency():
    from autocausal.production import AutoMLRiskPolicy, ProductionPolicy

    policy = ProductionPolicy.exploratory(
        automl_risk=AutoMLRiskPolicy(
            cv_folds=2,
            allow_feature_importance=False,
        )
    )
    report = AutoTabularML(policy=policy, max_candidates=1).run(
        _titanic_like(80),
        target="survived",
        candidates=["logistic"],
        gate_results=[{"id": "external_review", "ok": True}],
    )
    assert len(report.split_plan.splits) == 2
    assert report.feature_importance == []
    assert any(gate["id"] == "external_review" for gate in report.gates)


def test_time_split_is_forward_chaining():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=30, freq="D"),
            "x": np.arange(30),
            "target": np.arange(30) * 0.5,
        }
    )
    task = infer_task(frame, target="target", task="regression")
    plan = make_splits(
        frame,
        task,
        y=frame.target,
        strategy="time",
        time_column="time",
        n_splits=3,
    )
    assert plan.strategy == "time"
    for train, validation in plan.splits:
        assert frame.time.iloc[train].max() < frame.time.iloc[validation].min()


def test_autonlp_profile_claims_privacy_and_report_alias(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "notes": [
                "Treatment for 555-123-4567 leads to improved recovery.",
                "The policy may not cause retention; token=supersecret123.",
                "Baseline age is associated with the outcome.",
            ],
            "label": [1, 0, 1],
        }
    )
    report = AutoNLPSuite(
        frame,
        mode="production",
        text_columns=["notes"],
        target="label",
    ).run()
    assert report.profile.privacy_risk == "high"
    assert report.claims
    assert any(claim.negated for claim in report.claims)
    serialized = report.to_json()
    assert "555-123-4567" not in serialized
    assert "supersecret123" not in serialized
    assert report.to_markdown() == report.report()
    output = report.write(tmp_path / "nlp.json")
    assert json.loads(output.read_text(encoding="utf-8"))["schema"].endswith(
        "Report.v1"
    )


def test_causal_claim_offsets_and_hypothesis_flags():
    text = "Baseline differs. A randomized treatment might lead to higher sales."
    claims = extract_causal_claims(text)
    assert len(claims) == 1
    claim = claims[0]
    assert text[claim.start : claim.end].strip() == claim.evidence_span
    assert claim.uncertain is True
    assert claim.hypothesis is True


def test_fold_safe_text_vectorizer_and_local_retrieval():
    training = ["policy improves health", "treatment may affect sales"]
    vectorizer = FoldSafeTextVectorizer(max_features=50).fit(training)
    names = set(vectorizer.get_feature_names_out())
    assert not any("heldoutsecret" in name for name in names)
    transformed = vectorizer.transform(["heldoutsecret appears only now"])
    assert transformed.shape[0] == 1
    assert vectorizer.audit()["fit_document_count"] == 2
    results = TextRetriever().fit(training).search("health policy", top_k=1)
    assert results[0].index == 0
    assert results[0].document is None


def test_text_feature_aggregation_and_external_consent():
    frame = pd.DataFrame(
        {
            "subject": ["a", "a", "b"],
            "time": pd.date_range("2025-01-01", periods=3, freq="D"),
            "notes": ["good outcome", "policy may help", "no effect"],
        }
    )
    aggregated = aggregate_text_features(
        frame,
        text_column="notes",
        group_columns=["subject"],
    )
    assert "document_count" in aggregated
    suite = AutoNLPSuite(frame, text_columns=["notes"])
    with pytest.raises(ValueError, match="allow_external_text"):
        suite.run(external_enricher=lambda payload: {})
