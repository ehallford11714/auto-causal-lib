"""Offline tests for citation-grounded AutoCausal deep research."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from autocausal.research import (
    AgendaPlanner,
    CrossMatchEngine,
    DeepResearchSuite,
    LocalDocumentProvider,
    PrivacyGateError,
    ResearchApprovalRequired,
    ResearchBudget,
    ResearchHandoff,
    ResearchPolicy,
    ResearchReport,
    SearchIntensity,
    SourceRecord,
    deduplicate_sources,
    source_independence_groups,
    to_research_handoff,
)
from autocausal.mcp.registry import build_default_registry
from autocausal.production import ProductionPolicy
from autocausal.research.providers import ProviderQuery
from autocausal.results import DiscoveryResult
from autocausal.roles import ColumnRole


def _handoff(*, mode: str = "exploratory", domain: str = "general") -> ResearchHandoff:
    return ResearchHandoff(
        run_id="run-research-test",
        findings=[
            {
                "id": "edge-1",
                "kind": "edge",
                "summary": "exercise -> blood pressure",
            }
        ],
        edges=[
            {
                "finding_id": "edge-1",
                "source": "exercise",
                "target": "blood pressure",
                "stability": 0.31,
                "agreement": 0.5,
                "coefficient": -0.2,
                "evidence_grade": "exploratory",
                "orientation": "score_r2",
                "method": "score_pc_lite",
            }
        ],
        evidence_grades={"edge-1": "exploratory"},
        uncertainty=[
            {
                "kind": "low_bootstrap_stability",
                "finding_id": "edge-1",
                "severity": "high",
                "detail": "Bootstrap stability is 0.31.",
            },
            {
                "kind": "unsupported_orientation",
                "finding_id": "edge-1",
                "severity": "medium",
                "detail": "Direction is heuristic.",
            },
        ],
        candidate_roles={
            "treatment": ["exercise"],
            "outcome": ["blood pressure"],
            "confounder": ["age"],
        },
        domain=domain,
        context={
            "population": "adults",
            "setting": "community",
            "time_period": "2020 2025",
        },
        variable_labels={
            "exercise": "exercise",
            "blood pressure": "blood pressure",
        },
        aliases={
            "exercise": ["exercise", "physical activity"],
            "blood pressure": ["blood pressure", "hypertension"],
        },
        mode=mode,
        source_type="DiscoveryResult",
    )


def _records() -> list[SourceRecord]:
    return [
        SourceRecord(
            provider="local",
            stable_id="study-a",
            doi="10.1000/studya",
            title="Exercise and blood pressure in adults",
            authors=["A. One"],
            date="2022",
            abstract=(
                "In community adults, increased exercise reduced blood pressure "
                "during longitudinal follow-up."
            ),
            availability="abstract",
            metadata={"population": "adults", "context": "community", "study_id": "A"},
        ),
        SourceRecord(
            provider="local",
            stable_id="study-b",
            doi="10.1000/studyb",
            title="Physical activity intervention for hypertension",
            authors=["B. Two"],
            date="2023",
            abstract=(
                "Physical activity and exercise were associated with lower blood "
                "pressure among adults in community clinics."
            ),
            availability="abstract",
            metadata={"population": "adults", "context": "community", "study_id": "B"},
        ),
        SourceRecord(
            provider="local",
            stable_id="study-c",
            doi="10.1000/studyc",
            title="Exercise did not lower blood pressure",
            authors=["C. Three"],
            date="2024",
            abstract=(
                "In adults, exercise did not reduce blood pressure after adjustment "
                "for baseline health."
            ),
            availability="abstract",
            metadata={"population": "adults", "context": "community", "study_id": "C"},
        ),
        SourceRecord(
            provider="local",
            stable_id="animal-study",
            doi="10.1000/mice",
            title="Exercise and arterial pressure in laboratory mice",
            authors=["D. Four"],
            date="2018",
            abstract=("Exercise reduced arterial blood pressure in laboratory mice."),
            availability="abstract",
            metadata={"population": "mice", "context": "laboratory", "study_id": "M"},
        ),
    ]


def _policy(**overrides: Any) -> ResearchPolicy:
    values: dict[str, Any] = {
        "allowed_providers": ("local",),
        "minimum_independent_sources": 2,
        "approval_granted": True,
    }
    values.update(overrides)
    return ResearchPolicy(**values)


def test_search_intensity_budgets_are_predictable_and_capped():
    quick = ResearchBudget.for_intensity("quick")
    standard = ResearchBudget.for_intensity("standard")
    deep = ResearchBudget.for_intensity("deep")
    exhaustive = ResearchBudget.for_intensity("exhaustive")
    assert (
        quick.max_sources
        < standard.max_sources
        < deep.max_sources
        < exhaustive.max_sources
    )
    assert quick.max_rounds < deep.max_rounds < exhaustive.max_rounds

    policy = _policy(maximum_budget=standard)
    requested = policy.budget_for("exhaustive", {"max_sources": 999, "max_rounds": 99})
    assert requested.max_sources == standard.max_sources
    assert requested.max_rounds == standard.max_rounds
    multilingual = _policy().budget_for("standard", {"languages": ["en", "fr"]})
    assert multilingual.languages == ["en", "fr"]
    aliased = _policy().budget_for(
        "standard",
        {
            "max_queries_per_question": 2,
            "max_sources_per_provider": 5,
            "max_wall_time_seconds": 20,
        },
    )
    assert aliased.queries_per_question == 2
    assert aliased.sources_per_provider == 5
    assert aliased.wall_time_seconds == 20

    inherited = ResearchPolicy.from_production_policy(
        ProductionPolicy(max_rounds=2, max_seconds=45.0),
        allowed_providers=("local",),
    )
    assert inherited.production_mode is True
    assert inherited.maximum_budget.max_rounds == 2
    assert inherited.maximum_budget.wall_time_seconds == 45.0


def test_handoff_low_stability_weak_iv_and_privacy_redaction():
    result = DiscoveryResult(
        edges=[
            {
                "source": "patient_email",
                "target": "outcome",
                "stability": 0.2,
                "type": "iv_2sls",
                "instrument": "weak_z",
                "first_stage_f": 3.2,
                "orientation": "score_r2",
            }
        ],
        graph={},
        roles={
            "patient_email": ColumnRole.CATEGORICAL,
            "outcome": ColumnRole.NUMERIC,
        },
        candidates={
            "treatment": ["patient_email"],
            "outcome": ["outcome"],
            "instrument": ["weak_z"],
            "confounder": [],
        },
        run_id="privacy-run",
    )
    handoff = to_research_handoff(result)
    serialized = handoff.to_json()
    assert "patient_email" not in serialized
    kinds = {item["kind"] for item in handoff.uncertainty}
    assert "low_bootstrap_stability" in kinds
    assert "weak_instrument" in kinds
    assert handoff.provenance["contains_raw_frame"] is False

    unsafe = _handoff()
    unsafe.edges[0]["source"] = "patient_email"
    with pytest.raises(PrivacyGateError):
        DeepResearchSuite(
            policy=_policy(),
            providers=[LocalDocumentProvider(_records())],
        ).run(unsafe, intensity="quick")


def test_rule_agenda_is_deterministic_and_peco_structured():
    handoff = _handoff()
    planner = AgendaPlanner()
    budget = ResearchBudget.for_intensity("standard")
    first, trace = planner.plan(handoff, budget=budget)
    second, _ = planner.plan(handoff, budget=budget)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert first[0].population == "adults"
    assert first[0].exposure
    assert first[0].outcome
    assert first[0].query_variants
    assert trace[0]["deterministic"] is True


def test_offline_workflow_dedup_contradiction_and_citation_integrity():
    records = _records()
    duplicate = SourceRecord.from_dict(
        {
            **records[0].to_dict(),
            "provider": "crossref",
            "stable_id": records[0].doi,
            "source_id": "crossref:10.1000/studya",
        }
    )
    suite = DeepResearchSuite(
        policy=_policy(),
        use_slm=False,
        providers=[LocalDocumentProvider([*records, duplicate])],
    )
    report = suite.run(_handoff(), intensity="standard")
    assert report.sources
    assert len({source.doi for source in report.sources if source.doi}) == len(
        [source for source in report.sources if source.doi]
    )
    assert report.claims
    assert any(
        span.claim_relation == "contradicts"
        for claim in report.claims
        for span in claim.evidence_spans
    )
    assert report.contradictions
    assert report.validate_citations(strict=True) == []
    assert "10.9999/fabricated" not in report.to_markdown()
    assert report.claims[0].literature_label in (
        "mixed",
        "supported_literature_context",
    )


def test_dedup_and_source_independence_groups():
    records = _records()
    duplicate = SourceRecord.from_dict(
        {
            **records[0].to_dict(),
            "provider": "crossref",
            "stable_id": records[0].doi,
            "source_id": "crossref:10.1000/studya",
        }
    )
    unique, log, aliases = deduplicate_sources([records[0], duplicate, records[1]])
    assert len(unique) == 2
    assert log and aliases[duplicate.source_id] in {item.source_id for item in unique}

    companion = SourceRecord(
        provider="local",
        stable_id="companion",
        title="Companion analysis of study A",
        abstract="Exercise and blood pressure companion analysis.",
        availability="abstract",
        metadata={"study_id": "A"},
    )
    groups = source_independence_groups([records[0], companion, records[1]])
    assert len(groups["study:a"]) == 2
    assert len(groups) == 2


def test_cross_match_alias_direction_and_population_context_components():
    handoff = _handoff()
    engine = CrossMatchEngine()
    matches = engine.match(handoff, [_records()[0], _records()[3]])
    by_source = {match.source_id: match for match in matches}
    adult = by_source[_records()[0].source_id]
    animal = by_source[_records()[3].source_id]
    assert adult.comparability.lexical_alias > 0
    assert (
        adult.comparability.population_overlap > animal.comparability.population_overlap
    )
    assert adult.comparability.overall > animal.comparability.overall
    assert any("population" in warning for warning in animal.comparability.warnings)
    assert adult.reasons and all(reason.component for reason in adult.reasons)


class CountingProvider:
    name = "local"
    network = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(
        self, request: ProviderQuery, *, policy: ResearchPolicy
    ) -> list[SourceRecord]:
        self.calls.append(request.query)
        index = len(set(self.calls))
        return [
            SourceRecord(
                provider="local",
                stable_id=f"query-{index}",
                title=f"Exercise blood pressure evidence {index}",
                abstract=(
                    "Adults receiving exercise had lower blood pressure in a "
                    f"community study number {index}."
                ),
                availability="abstract",
                metadata={"study_id": f"study-{index}", "population": "adults"},
            )
        ]


def test_deepen_resumes_cache_and_skips_completed_queries():
    provider = CountingProvider()
    suite = DeepResearchSuite(
        policy=_policy(minimum_independent_sources=5),
        providers=[provider],
    )
    first = suite.run(_handoff(), intensity="quick")
    first_calls = list(provider.calls)
    assert first_calls
    second = first.deepen(intensity="deep")
    counts = Counter(provider.calls)
    assert all(counts[query] == 1 for query in first_calls)
    assert second.provenance["resumed"] is True
    assert second.budget_used.cache_hits >= first.budget_used.cache_hits
    assert len(second.provenance["completed_queries"]) >= len(
        first.provenance["completed_queries"]
    )
    assert len(second.sources) >= len(first.sources)


class ConstantProvider:
    name = "local"
    network = False

    def search(
        self, request: ProviderQuery, *, policy: ResearchPolicy
    ) -> list[SourceRecord]:
        return [_records()[0]]


def test_saturation_and_policy_source_stop():
    maximum = ResearchBudget.for_intensity("deep").with_overrides(max_sources=2)
    suite = DeepResearchSuite(
        policy=_policy(
            minimum_independent_sources=3,
            maximum_budget=maximum,
        ),
        providers=[ConstantProvider()],
    )
    report = suite.run(_handoff(), intensity="deep")
    assert len(report.sources) <= 2
    assert any(row["new_unique_sources"] == 0 for row in report.saturation_curve[1:])
    assert "saturation" in report.stop_reason.lower() or report.status == "policy_limit"


def test_exhaustive_requires_explicit_approval_in_production():
    exploratory = DeepResearchSuite(
        policy=ResearchPolicy(
            allowed_providers=("local",),
            approval_granted=False,
        ),
        providers=[LocalDocumentProvider(_records())],
    ).run(
        _handoff(),
        intensity="exhaustive",
        budget_overrides={"max_rounds": 1, "max_sources": 4},
    )
    assert exploratory.selected_intensity is SearchIntensity.EXHAUSTIVE

    policy = ResearchPolicy(
        allowed_providers=("local",),
        production_mode=True,
        approval_granted=False,
    )
    suite = DeepResearchSuite(
        policy=policy,
        providers=[LocalDocumentProvider(_records())],
    )
    with pytest.raises(ResearchApprovalRequired):
        suite.run(_handoff(mode="production"), intensity="exhaustive")

    approved = suite.run(
        _handoff(mode="production"),
        intensity="exhaustive",
        approval_granted=True,
        budget_overrides={"max_rounds": 1, "max_sources": 4},
    )
    assert approved.selected_intensity is SearchIntensity.EXHAUSTIVE


class UnsafeMockSLM:
    tokens_used = 17

    def plan_questions(self, payload):
        return {
            "questions": [
                {
                    "question": "Does exercise lower blood pressure across adults?",
                    "rationale": "Structured enrichment of the rule agenda.",
                    "query_variants": ["exercise blood pressure adult replication"],
                    "finding_ids": ["edge-1"],
                }
            ]
        }

    def screen_sources(self, payload):
        return {"keep_source_ids": [payload["sources"][0]["source_id"], "fake:source"]}

    def adjudicate_match(self, payload):
        return {"relation": "supports", "semantic_relevance": 0.8}

    def extract_evidence(self, payload):
        return {
            "spans": [
                {
                    "source_id": "fabricated:paper",
                    "exact_text": "Fabricated quotation.",
                    "claim_relation": "supports",
                    "confidence": 1.0,
                }
            ]
        }

    def synthesize(self, payload):
        return {
            "narrative": "A fabricated source proves causality [doi:10.9999/fabricated]."
        }

    def expand_queries(self, payload):
        return {"queries": ["safe exercise replication query"]}


def test_mocked_qwen_output_cannot_create_citations():
    suite = DeepResearchSuite(
        policy=_policy(),
        use_slm=True,
        slm_backend=UnsafeMockSLM(),
        providers=[LocalDocumentProvider(_records())],
    )
    report = suite.run(_handoff(), intensity="standard")
    assert all(
        span.source_id != "fabricated:paper"
        for claim in report.claims
        for span in claim.evidence_spans
    )
    assert "10.9999/fabricated" not in report.provenance["slm_synthesis"]
    assert any(
        trace.get("reason") == "unsupported citation/reference blocked"
        for trace in report.provenance["tool_traces"]
    )
    assert report.validate_citations(strict=True) == []


def test_report_round_trip_and_write(tmp_path: Path):
    report = DeepResearchSuite(
        policy=_policy(),
        providers=[LocalDocumentProvider(_records())],
    ).run(_handoff(), intensity="quick")
    restored = ResearchReport.from_json(report.to_json())
    assert restored.to_dict() == report.to_dict()
    path = restored.write(tmp_path / "research.json")
    assert path.exists()
    markdown = restored.write(tmp_path / "research.md")
    assert "Retrieved literature evidence" in markdown.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        restored.deepen()


def test_result_adapter_methods_are_installed_without_shared_model_coupling():
    result = DiscoveryResult(
        edges=[
            {
                "source": "exercise",
                "target": "blood_pressure",
                "stability": 0.4,
            }
        ],
        graph={},
        roles={
            "exercise": ColumnRole.NUMERIC,
            "blood_pressure": ColumnRole.NUMERIC,
        },
        candidates={
            "treatment": ["exercise"],
            "outcome": ["blood_pressure"],
            "instrument": [],
            "confounder": [],
        },
        run_id="adapter-run",
    )
    handoff = result.to_research_handoff(context={"population": "adults"})
    assert handoff.run_id == "adapter-run"
    report = result.deep_research(
        intensity="quick",
        policy=_policy(),
        providers=[LocalDocumentProvider(_records())],
        context={"population": "adults"},
    )
    assert isinstance(report, ResearchReport)
    assert report.to_agentic_handback()["mutates_discovered_edges"] is False


def test_mcp_research_plan_run_status_and_report_are_registered():
    registry = build_default_registry()
    expected = {
        "autocausal_research_plan",
        "autocausal_deep_research",
        "autocausal_research_status",
        "autocausal_research_report",
    }
    assert expected <= set(registry.list_names())
    common = {
        "research_id": "mcp-offline-run",
        "handoff": _handoff().to_dict(),
        "providers": ["local"],
        "sources": [item.to_dict() for item in _records()],
        "intensity": "quick",
    }
    plan = registry.invoke("autocausal_research_plan", common)
    assert plan["ok"] is True
    assert plan["plan"]["contains_raw_values"] is False
    run = registry.invoke("autocausal_deep_research", common)
    assert run["ok"] is True
    status = registry.invoke(
        "autocausal_research_status",
        {"research_id": "mcp-offline-run"},
    )
    assert status["found"] is True
    rendered = registry.invoke(
        "autocausal_research_report",
        {"research_id": "mcp-offline-run", "format": "markdown"},
    )
    assert rendered["ok"] is True
    assert "Retrieved literature evidence" in rendered["markdown"]


def test_related_work_identifier_expansion_prefers_doi_and_arxiv():
    from autocausal.research import (
        expand_related_work_queries,
        extract_reference_identifiers,
    )

    ids = extract_reference_identifiers(
        [
            "doi:10.1234/abc.def",
            "arXiv:2301.12345v2",
            "Some unrelated short",
            "A longer bibliographic title fingerprint for related work",
        ]
    )
    kinds = [item["kind"] for item in ids]
    assert "doi" in kinds
    assert "arxiv" in kinds
    queries = expand_related_work_queries(
        [
            SourceRecord(
                provider="local",
                stable_id="seed",
                title="Seed",
                authors=["A"],
                date="2024",
                abstract="Seed abstract about treatment and outcome.",
                references=[
                    "10.5555/xyz.123",
                    "arXiv:2401.00001",
                    "Another related paper title for expansion",
                ],
            )
        ],
        limit=5,
    )
    joined = " | ".join(query for _, query in queries)
    assert "doi:10.5555/xyz.123" in joined
    assert "arXiv:2401.00001" in joined


def test_match_prior_sources_tags_prior_corpus_without_breaking_dedupe():
    from autocausal.research import match_prior_sources

    handoff = _handoff()
    retrieved = SourceRecord(
        provider="local",
        stable_id="ret-1",
        title="treatment outcome adults community instrument",
        authors=["R"],
        date="2024",
        abstract="treatment increases outcome among adults in community",
        snippet="treatment outcome adults community",
        metadata={"population": "adults", "context": "community"},
    )
    prior = SourceRecord(
        provider="local",
        stable_id="prior-1",
        title="Prior treatment outcome adults community instrument",
        authors=["P"],
        date="2023",
        abstract="Prior notes treatment increases outcome for adults community",
        snippet="treatment outcome adults community",
        metadata={"population": "adults", "context": "community"},
    )
    matches = match_prior_sources(handoff, [retrieved], prior_sources=[prior])
    assert matches
    assert any(
        any(reason.component == "prior_corpus" for reason in match.reasons)
        for match in matches
    )


def test_deep_intensity_expands_related_work_and_cross_matches_prior_corpus():
    prior = SourceRecord(
        provider="local",
        stable_id="prior-1",
        title="Prior episode on treatment outcome adults community",
        authors=["Prior"],
        date="2023",
        abstract=(
            "Prior AutoCausal episode notes treatment increases outcome for adults "
            "in community settings with instrument assignment."
        ),
        snippet="treatment outcome adults community",
        metadata={
            "population": "adults",
            "context": "community",
            "study_id": "prior-episode",
        },
    )
    seed = SourceRecord(
        provider="local",
        stable_id="seed-related",
        title="Seed paper with references",
        authors=["Seed"],
        date="2024",
        abstract="treatment outcome adults community instrument",
        snippet="treatment causes outcome",
        references=["10.9999/related.work", "arXiv:2501.99999"],
        metadata={"population": "adults", "context": "community", "study_id": "S"},
    )
    related = SourceRecord(
        provider="local",
        stable_id="related-hit",
        title="doi:10.9999/related.work treatment outcome adults community",
        authors=["Rel"],
        date="2022",
        abstract=(
            "This related work finds treatment increases outcome among adults in "
            "community trials."
        ),
        snippet="doi:10.9999/related.work treatment outcome",
        doi="10.9999/related.work",
        metadata={"population": "adults", "context": "community", "study_id": "R"},
    )
    suite = DeepResearchSuite(
        policy=_policy(),
        providers=[LocalDocumentProvider([seed, related])],
        prior_sources=[prior],
    )
    report = suite.run(_handoff(), intensity="deep")
    assert report.selected_intensity is SearchIntensity.DEEP
    assert any(
        "related_work_identifiers" == item.get("tool")
        for item in report.provenance.get("tool_traces") or []
    ) or any(
        "related" in " ".join(item.get("matched_concepts") or []).lower()
        or item.source_id in {"prior-1", "related-hit", "seed-related"}
        for item in report.cross_matches
    )
    assert any(
        any(reason.component == "prior_corpus" for reason in match.reasons)
        for match in report.cross_matches
    )
    # Semantic similarity / prior corpus never upgrades identification status.
    assert all(
        "identification" not in (claim.literature_label or "").lower()
        for claim in report.claims
    )
    caveats = " ".join(report.caveats or [])
    assert "identification" in caveats.lower()


def test_intensity_router_recommends_deepen_on_contradiction_and_gaps():
    from autocausal.research.planning import IntensityRouter

    handoff = _handoff()
    suite = DeepResearchSuite(
        policy=_policy(minimum_independent_sources=3),
        providers=[LocalDocumentProvider(_records())],
    )
    report = suite.run(handoff, intensity="quick")
    report.contradictions = [
        {
            "finding_id": "edge:0",
            "claim_ids": ["c1", "c2"],
            "detail": "supporting and contradicting spans",
        }
    ]
    report.context_transfer_warnings = ["population mismatch: adults vs mice"]
    route = IntensityRouter().recommend(
        handoff,
        selected="quick",
        policy=_policy(minimum_independent_sources=3),
        report=report,
        slm_request="please search more",
    )
    assert route.recommended.rank >= SearchIntensity.STANDARD.rank
    assert any("did not affect intensity routing" in reason for reason in route.reasons)
    assert route.decision in {"deepen", "stop", "proceed", "human"}