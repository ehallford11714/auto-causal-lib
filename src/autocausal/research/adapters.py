"""Conflict-light result/AutoCausal/agentic adapters for deep research.

Importing :mod:`autocausal.research` installs these methods only when the host
class does not already define them.  This keeps the feature in its own package
while concurrent production and visualization work can evolve shared modules.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from autocausal.research.handoff import handoff_from_gate_report, to_research_handoff
from autocausal.research.models import (
    ResearchHandoff,
    ResearchPolicy,
    ResearchReport,
    SearchIntensity,
)
from autocausal.research.suite import DeepResearchSuite


def research_handback_context(report: ResearchReport) -> dict[str, Any]:
    """Return safe loop inputs without mutating discovered edges."""

    return {
        "schema": "AutoCausalResearchHandback.v1",
        "handoff_run_id": report.handoff_run_id,
        "research_questions": [
            {
                "id": question.id,
                "question": question.question,
                "query_variants": list(question.query_variants),
                "finding_ids": list(question.finding_ids),
            }
            for question in report.agenda
        ],
        "experiment_recommendations": list(report.experiment_recommendations),
        "actions": list(report.handback_recommendations),
        "unresolved_evidence_gaps": list(report.unresolved_evidence_gaps),
        "context_transfer_warnings": list(report.context_transfer_warnings),
        "source_ids": [source.source_id for source in report.sources],
        "mutates_discovered_edges": False,
        "literature_changes_identification_grade": False,
    }


def _deep_research(
    self: Any,
    *,
    intensity: SearchIntensity | str = SearchIntensity.STANDARD,
    policy: Optional[ResearchPolicy | Mapping[str, Any]] = None,
    use_slm: bool = False,
    suite: Optional[DeepResearchSuite] = None,
    domain: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
    budget_overrides: Optional[Mapping[str, Any]] = None,
    approval_granted: Optional[bool] = None,
    providers: Any = None,
    local_records: Any = None,
    slm_backend: Any = None,
    model_name: Optional[str] = None,
) -> ResearchReport:
    handoff_method = getattr(self, "to_research_handoff", None)
    if callable(handoff_method):
        handoff = handoff_method(
            domain=domain,
            context=context,
            policy=(
                policy
                if isinstance(policy, ResearchPolicy)
                else ResearchPolicy.from_dict(policy)
                if isinstance(policy, Mapping)
                else None
            ),
        )
    else:
        handoff = to_research_handoff(
            self,
            domain=domain,
            context=context,
            policy=(
                policy
                if isinstance(policy, ResearchPolicy)
                else ResearchPolicy.from_dict(policy)
                if isinstance(policy, Mapping)
                else None
            ),
        )
    runner = suite or DeepResearchSuite(
        policy=policy,
        use_slm=use_slm,
        providers=providers,
        local_records=local_records,
        slm_backend=slm_backend,
        model_name=model_name,
    )
    report = runner.run(
        handoff,
        intensity=intensity,
        budget_overrides=budget_overrides,
        approval_granted=approval_granted,
    )
    try:
        setattr(self, "research_report", report)
    except Exception:
        pass
    return report


def _generic_handoff(
    self: Any,
    *,
    domain: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
    policy: Optional[ResearchPolicy] = None,
) -> ResearchHandoff:
    return to_research_handoff(self, domain=domain, context=context, policy=policy)


def _autocausal_handoff(
    self: Any,
    *,
    domain: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
    policy: Optional[ResearchPolicy] = None,
) -> ResearchHandoff:
    result = getattr(self, "result", None)
    if result is None:
        raise RuntimeError(
            "AutoCausal.to_research_handoff() requires a discovery result. "
            "Run discover() first; deep research never silently discovers or mutates edges."
        )
    manifest = getattr(self, "run_manifest", None)
    payload = {
        "run_id": str(getattr(self, "run_id", "") or ""),
        "mode": str(getattr(self, "mode", "exploratory") or "exploratory"),
        "discovery": (result.to_dict() if hasattr(result, "to_dict") else result),
        "manifest": (
            manifest.to_dict()
            if manifest is not None and hasattr(manifest, "to_dict")
            else None
        ),
        "sensitivity_report": (
            getattr(self, "sensitivity_report", None).to_dict()
            if getattr(self, "sensitivity_report", None) is not None
            and hasattr(getattr(self, "sensitivity_report"), "to_dict")
            else getattr(self, "sensitivity_report", None)
        ),
        "experiments": [],
    }
    return to_research_handoff(payload, domain=domain, context=context, policy=policy)


def _gate_handoff(
    self: Any,
    *,
    domain: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
    policy: Optional[ResearchPolicy] = None,
) -> ResearchHandoff:
    return handoff_from_gate_report(
        self,
        domain=domain or "general",
        context=context,
        policy=policy,
    )


def _handoff_handback(self: ResearchReport) -> dict[str, Any]:
    return research_handback_context(self)


def research_escalation_node(
    state: Any,
    *,
    suite: DeepResearchSuite,
    policy_triggers: bool = True,
    intensity: SearchIntensity | str = SearchIntensity.DEEP,
    approval_granted: Optional[bool] = None,
) -> Any:
    """LangGraph/FSM-compatible escalation node.

    It returns a copied mapping where possible and never changes ``edges``.
    External/high-cost work still passes through the suite's policy router.
    """

    if not policy_triggers:
        return state
    payload = (
        dict(state)
        if isinstance(state, Mapping)
        else state.to_dict()
        if hasattr(state, "to_dict")
        else {"state": str(state)}
    )
    source = payload.get("research_handoff") or payload.get("report") or payload
    handoff = (
        source
        if isinstance(source, ResearchHandoff)
        else to_research_handoff(source, policy=suite.policy)
    )
    report = suite.run(
        handoff,
        intensity=intensity,
        approval_granted=approval_granted,
    )
    out = dict(payload)
    original_edges = list(out.get("edges") or [])
    out["research_report"] = report.to_dict()
    out["research_handback"] = research_handback_context(report)
    out["edges"] = original_edges
    return out


def install_research_adapters() -> dict[str, list[str]]:
    """Install ergonomic methods idempotently and return an audit summary."""

    installed: dict[str, list[str]] = {}

    def attach(cls: Any, name: str, value: Any) -> None:
        if cls is None or hasattr(cls, name):
            return
        setattr(cls, name, value)
        installed.setdefault(f"{cls.__module__}.{cls.__name__}", []).append(name)

    from autocausal.api import AutoCausal
    from autocausal.results import AutoResult, DiscoveryResult

    attach(AutoCausal, "to_research_handoff", _autocausal_handoff)
    attach(AutoCausal, "deep_research", _deep_research)
    for cls in (DiscoveryResult, AutoResult):
        attach(cls, "to_research_handoff", _generic_handoff)
        attach(cls, "deep_research", _deep_research)

    try:
        from autocausal.insight.report import InsightReport

        attach(InsightReport, "to_research_handoff", _generic_handoff)
        attach(InsightReport, "deep_research", _deep_research)
    except Exception:
        pass
    try:
        from autocausal.agentic.report import AgenticLoopReport

        attach(AgenticLoopReport, "to_research_handoff", _generic_handoff)
        attach(AgenticLoopReport, "deep_research", _deep_research)
    except Exception:
        pass
    try:
        from autocausal.production import GateReport

        attach(GateReport, "to_research_handoff", _gate_handoff)
        attach(GateReport, "deep_research", _deep_research)
    except Exception:
        pass
    attach(ResearchReport, "to_agentic_handback", _handoff_handback)
    return installed


__all__ = [
    "install_research_adapters",
    "research_escalation_node",
    "research_handback_context",
]
