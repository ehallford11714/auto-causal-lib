"""Production, privacy, provenance, and renderer tests for reporting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from autocausal import AutoCausal
from autocausal.reporting import (
    ReportEngine,
    ReportPolicy,
    ReportSafetyError,
    ReportValidationError,
    SLMReportDirector,
    deterministic_report_plan,
    normalize_report_sources,
    report_tool_surface,
    validate_report_bundle,
    validate_report_plan,
    validate_slm_proposal,
)
from autocausal.results import AutoResult, DiscoveryResult


def _artifact(class_name: str, payload: dict) -> object:
    """Build a duck-typed concurrent-module artifact without importing it."""

    artifact_type = type(
        class_name,
        (),
        {"to_dict": lambda self, value=payload: dict(value)},
    )
    return artifact_type()


@pytest.mark.parametrize(
    ("class_name", "family", "payload"),
    [
        ("DiscoveryResult", "discovery", {"edges": [], "method": "pc"}),
        ("AutoResult", "auto_result", {"source": "test"}),
        ("MiningReport", "mining", {"associations": []}),
        ("MineReport", "mine", {"associations": []}),
        ("QCReport", "qc", {"ok": True, "checks": []}),
        ("CleanseReport", "cleanse", {"n_rows_in": 10, "n_rows_out": 10}),
        ("EDAReport", "eda", {"n_rows": 10, "n_cols": 3}),
        ("ImputationReport", "cleanse", {"imputed_cells": 2}),
        (
            "EstimateResult",
            "estimate",
            {"treatment": "x", "outcome": "y", "estimate": 0.4},
        ),
        ("RefuteResult", "refute", {"method": "placebo", "passed": True}),
        ("SensitivityReport", "sensitivity", {"robustness_score": 0.8}),
        ("GateReport", "gate", {"passed": True, "results": []}),
        ("RunManifest", "manifest", {"run_id": "run-1"}),
        ("InsightReport", "insight", {"insights": []}),
        ("AgenticLoopReport", "agentic", {"iterations": []}),
        ("GrailReport", "grail", {"status": "ok"}),
        (
            "FitReport",
            "automl",
            {"selected_model": "ridge", "metric": "rmse"},
        ),
        ("TextCausalHints", "nlp", {"claims": []}),
        ("BehavioralReport", "nlp_behavioral", {"edges": []}),
        ("AutoVizReport", "autoviz", {"recommendations": []}),
        ("AutoChartReport", "autochart", {"charts": []}),
        ("ResearchReport", "deep_research", {"sources": [], "claims": []}),
        ("ValidationReport", "validation", {"ok": True, "checks": []}),
        (
            "PublicCausalReport",
            "public_causal",
            {"sources": ["public"], "associations": []},
        ),
    ],
)
def test_normalizes_supported_report_families(
    class_name: str,
    family: str,
    payload: dict,
) -> None:
    sources, warnings = normalize_report_sources(
        _artifact(class_name, payload),
        policy=ReportPolicy.production(),
    )

    assert len(sources) == 1
    assert sources[0].family == family
    assert not sources[0].contains_raw_data
    assert isinstance(warnings, list)
    assert sources[0].to_json().startswith("{")
    assert sources[0].report() == sources[0].to_markdown()


def _estimate_source() -> object:
    return _artifact(
        "EstimateResult",
        {
            "run_id": "run-report-test",
            "treatment": "exposure",
            "outcome": "outcome",
            "method": "builtin_ols",
            "estimate": 0.42,
            "std_error": 0.08,
            "ci_low": 0.26,
            "ci_high": 0.58,
            "n_obs": 120,
            "assumptions": ["conditional exchangeability"],
        },
    )


def test_deterministic_plan_is_stable_and_omits_missing_sections() -> None:
    policy = ReportPolicy.production()
    sources, _ = normalize_report_sources(_estimate_source(), policy=policy)

    first = deterministic_report_plan(sources, policy)
    second = deterministic_report_plan(sources, policy)

    assert first.to_dict() == second.to_dict()
    assert first.director_backend == "rule"
    assert "causal_estimates" in first.section_order
    assert "automl" not in first.section_order
    assert "deep_research" not in first.section_order

    no_appendix = ReportPolicy.production(include_appendix=False)
    no_appendix_sources, _ = normalize_report_sources(
        _estimate_source(), policy=no_appendix
    )
    assert "technical_appendix" not in deterministic_report_plan(
        no_appendix_sources, no_appendix
    ).section_order


@pytest.mark.parametrize(
    "proposal",
    [
        {"section_order": ["cover", "invented_results"]},
        {
            "section_order": [],
            "claims": [
                {
                    "section_id": "causal_estimates",
                    "text": "Grounded statement.",
                    "fact_ids": ["invented-fact"],
                }
            ],
        },
    ],
)
def test_slm_plan_rejects_invented_sections_or_facts(proposal: dict) -> None:
    policy = ReportPolicy.production()
    sources, _ = normalize_report_sources(_estimate_source(), policy=policy)
    baseline = deterministic_report_plan(sources, policy)
    if not proposal["section_order"]:
        proposal["section_order"] = list(baseline.section_order)

    with pytest.raises(ReportValidationError):
        validate_slm_proposal(
            proposal,
            baseline=baseline,
            sources=sources,
            policy=policy,
        )


def test_slm_plan_rejects_unsupported_citation_and_falls_back() -> None:
    policy = ReportPolicy.production()
    sources, _ = normalize_report_sources(_estimate_source(), policy=policy)
    baseline = deterministic_report_plan(sources, policy)
    fact = next(
        item
        for source in sources
        for item in source.facts
        if item.category == "causal_estimates"
    )
    proposal = {
        "section_order": list(baseline.section_order),
        "claims": [
            {
                "section_id": "causal_estimates",
                "text": "The normalized estimate is available.",
                "fact_ids": [fact.id],
                "citation_ids": ["invented-citation"],
            }
        ],
    }
    with pytest.raises(ReportValidationError, match="unsupported citation"):
        validate_slm_proposal(
            proposal,
            baseline=baseline,
            sources=sources,
            policy=policy,
        )

    class UnsafeBackend:
        name = "unsafe-test-backend"

        def generate_report_plan(self, inventory: dict) -> dict:
            return {"section_order": ["invented_results"]}

    director = SLMReportDirector(
        use_slm=True,
        policy=policy,
        backend=UnsafeBackend(),
    )
    plan = director.plan(sources)
    assert plan.director_backend.startswith("rule-fallback:")
    assert director.last_error
    assert any("discarded" in warning for warning in plan.warnings)


def test_slm_plan_enforces_policy_permissions() -> None:
    policy = ReportPolicy.production(slm_permissions=())
    sources, _ = normalize_report_sources(_estimate_source(), policy=policy)
    baseline = deterministic_report_plan(sources, policy)

    with pytest.raises(ReportValidationError, match="does not permit section"):
        validate_slm_proposal(
            {"section_order": list(baseline.section_order)},
            baseline=baseline,
            sources=sources,
            policy=policy,
        )


def test_synthetic_iv_and_failed_gates_cannot_be_suppressed() -> None:
    discovery = _artifact(
        "DiscoveryResult",
        {
            "edges": [
                {
                    "source": "exposure",
                    "target": "outcome",
                    "type": "iv",
                    "instrument": "synthetic_iv_proxy",
                    "synthetic": True,
                    "estimate": 0.7,
                }
            ]
        },
    )
    gates = _artifact(
        "GateReport",
        {
            "ok": False,
            "results": [
                {
                    "id": "stability",
                    "status": "fail",
                    "detail": "Insufficient bootstrap stability.",
                }
            ],
        },
    )
    engine = ReportEngine(use_slm=False)
    bundle = engine.build_bundle([discovery, gates])

    assert "iv_evidence" in bundle.plan.section_order
    assert "refutations_sensitivity" in bundle.plan.section_order
    synthetic = [
        fact
        for fact in bundle.facts
        if fact.attributes.get("synthetic_iv")
    ]
    assert synthetic
    assert all(not fact.evidence_eligible for fact in synthetic)

    suppressed = deterministic_report_plan(bundle.sources, bundle.policy)
    suppressed.section_order.remove("iv_evidence")
    with pytest.raises(ReportValidationError, match="required"):
        validate_report_plan(suppressed, bundle.sources, bundle.policy)


def test_verified_source_record_citations_are_preserved() -> None:
    report = _artifact(
        "ResearchReport",
        {
            "handoff_run_id": "research-1",
            "status": "complete",
            "agenda": [{"id": "q1", "question": "Does X affect Y?"}],
            "sources": [
                {
                    "provider": "crossref",
                    "stable_id": "10.1000/example",
                    "source_id": "crossref:10.1000/example",
                    "title": "A retrieved study",
                    "authors": ["Researcher"],
                    "date": "2025",
                    "url": "https://doi.org/10.1000/example",
                }
            ],
            "claims": [
                {
                    "normalized_claim": "The supplied literature contextualizes X and Y.",
                    "evidence_spans": [
                        {
                            "source_id": "crossref:10.1000/example",
                            "claim_relation": "contextualizes",
                        }
                    ],
                }
            ],
        },
    )

    bundle = ReportEngine(use_slm=False).build_bundle(report)
    assert [item.id for item in bundle.citations] == [
        "crossref:10.1000/example"
    ]
    assert "deep_research" in bundle.plan.section_order
    assert bundle.validation["citation_count"] == 1


def test_unsupported_deep_research_citation_fails_closed() -> None:
    report = _artifact(
        "ResearchReport",
        {
            "sources": [],
            "claims": [
                {
                    "normalized_claim": "Unsupported literature claim.",
                    "evidence_spans": [{"source_id": "missing:record"}],
                }
            ],
        },
    )

    with pytest.raises(ReportValidationError, match="unsupported citation"):
        ReportEngine(use_slm=False).plan(report)


def test_privacy_redaction_raw_frame_refusal_and_bundle_scan() -> None:
    source = _artifact(
        "GroundingReport",
        {
            "summary": "Contact analyst@example.com using sk_abcdefghijklmnop.",
            "recommendations": [],
        },
    )
    normalized = ReportEngine(use_slm=False).normalize(source)[0]
    serialized = normalized.to_json()
    assert "analyst@example.com" not in serialized
    assert "sk_abcdefghijklmnop" not in serialized
    assert "[REDACTED_EMAIL]" in serialized
    assert "[REDACTED_SECRET]" in serialized

    with pytest.raises(ReportSafetyError, match="Raw DataFrame"):
        ReportEngine(use_slm=False).normalize(pd.DataFrame({"secret": [1]}))

    bundle = ReportEngine(use_slm=False).build_bundle(_estimate_source())
    bundle.audit_notes.append("Unredacted owner@example.com")
    with pytest.raises(ReportSafetyError, match="sensitive string"):
        validate_report_bundle(bundle)


def test_headless_pdf_hash_metadata_siblings_and_missing_chart(
    tmp_path: Path,
) -> None:
    chart_report = _artifact(
        "AutoChartReport",
        {
            "charts": [
                {
                    "id": "estimate-chart",
                    "type": "forest",
                    "title": "Estimate interval",
                    "alt_text": "Forest plot of the estimate and interval.",
                    "path": str(tmp_path / "missing-chart.png"),
                    "spec": {"estimate": 0.42, "ci": [0.26, 0.58]},
                }
            ]
        },
    )
    output = tmp_path / "autocausal-report.pdf"
    artifact = ReportEngine(use_slm=False).generate(
        source=[_estimate_source(), chart_report],
        output=output,
        siblings=("markdown", "html", "json"),
    )

    data = output.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 2_000
    assert b"/Title" in data
    assert artifact.size_bytes == len(data)
    assert artifact.sha256 == hashlib.sha256(data).hexdigest()
    assert len(artifact.sha256) == 64
    assert artifact.page_count and artifact.page_count >= 1
    assert artifact.provenance["validation"]["ok"] is True
    assert set(artifact.siblings) == {"markdown", "html", "json"}
    assert all(Path(path).exists() for path in artifact.siblings.values())
    assert any("does not exist" in warning for warning in artifact.warnings)


def test_model_aliases_integrations_and_lazy_tool_registration(
    tmp_path: Path,
) -> None:
    source = ReportEngine(use_slm=False).normalize(_estimate_source())[0]
    assert source.report() == source.to_markdown()
    metadata_path = source.write(tmp_path / "source.json")
    assert metadata_path.exists()
    assert source.report(as_markdown=False) == source.to_json()

    assert hasattr(AutoCausal, "generate_report")
    assert hasattr(DiscoveryResult, "generate_report")
    assert hasattr(AutoResult, "generate_report")

    assert {
        "report.plan",
        "report.generate",
        "report.validate",
    } <= set(report_tool_surface().list_names())

    from autocausal.mcp.registry import build_default_registry

    assert {
        "autocausal_report_plan",
        "autocausal_generate_report",
        "autocausal_report_status",
    } <= set(build_default_registry().list_names())
