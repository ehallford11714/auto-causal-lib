"""Deterministic agenda generation and auditable search-intensity routing."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Optional, Protocol, Sequence

from autocausal.research.models import (
    IntensityRecommendation,
    ResearchBudget,
    ResearchHandoff,
    ResearchPolicy,
    ResearchQuestion,
    ResearchReport,
    SearchIntensity,
)


def _hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _terms(value: str) -> str:
    text = re.sub(r"[_\-]+", " ", str(value))
    return " ".join(text.split())


class QuestionEnricher(Protocol):
    """Optional structured SLM interface used after rule planning."""

    def plan_questions(self, payload: Mapping[str, Any]) -> Any: ...


class AgendaPlanner:
    """Produce rule-first PECO/PICO questions from uncertainty signals."""

    def __init__(self, *, slm_backend: Optional[QuestionEnricher] = None) -> None:
        self.slm_backend = slm_backend

    @staticmethod
    def _edge_context(handoff: ResearchHandoff, finding_id: str) -> dict[str, Any]:
        for edge in handoff.edges:
            if str(edge.get("finding_id")) == str(finding_id):
                return edge
        return {}

    @staticmethod
    def _roles(handoff: ResearchHandoff) -> tuple[str, str, str, list[str]]:
        roles = handoff.candidate_roles
        exposure = str((roles.get("treatment") or [""])[0])
        outcome = str((roles.get("outcome") or [""])[0])
        instrument = str((roles.get("instrument") or [""])[0])
        confounders = [str(item) for item in roles.get("confounder") or []]
        return exposure, outcome, instrument, confounders

    def _make_question(
        self,
        *,
        kind: str,
        finding_ids: Sequence[str],
        question: str,
        rationale: str,
        priority: int,
        population: Optional[str],
        exposure: Optional[str],
        comparator: Optional[str],
        outcome: Optional[str],
        context: Optional[str],
        inclusion: Sequence[str],
        exclusion: Sequence[str],
        variants: Sequence[str],
        source: str = "rule",
    ) -> ResearchQuestion:
        normalized = " ".join(question.split())
        return ResearchQuestion(
            id=f"rq:{kind}:{_hash(normalized)}",
            priority=priority,
            finding_ids=list(finding_ids),
            question=normalized,
            rationale=" ".join(rationale.split()),
            inclusion_criteria=list(inclusion),
            exclusion_criteria=list(exclusion),
            query_variants=list(
                dict.fromkeys(" ".join(item.split()) for item in variants)
            ),
            population=population,
            exposure=exposure,
            comparator=comparator,
            outcome=outcome,
            context=context,
            source=source,
        )

    def plan(
        self,
        handoff: ResearchHandoff,
        *,
        budget: ResearchBudget,
        use_slm: bool = False,
    ) -> tuple[list[ResearchQuestion], list[dict[str, Any]]]:
        exposure, outcome, instrument, confounders = self._roles(handoff)
        population = str(
            handoff.context.get("population") or handoff.context.get("cohort") or ""
        )
        comparator = str(handoff.context.get("comparator") or "")
        context = str(
            handoff.context.get("setting")
            or handoff.context.get("context")
            or handoff.domain
        )
        questions: list[ResearchQuestion] = []
        trace: list[dict[str, Any]] = []

        for item in handoff.uncertainty:
            kind = str(item.get("kind") or "uncertainty")
            finding_id = str(item.get("finding_id") or "")
            edge = self._edge_context(handoff, finding_id)
            source = _terms(edge.get("source") or exposure or "the exposure")
            target = _terms(edge.get("target") or outcome or "the outcome")
            finding_ids = [finding_id] if finding_id else []
            detail = str(item.get("detail") or "An assumption remains unresolved.")
            severity = str(item.get("severity") or "medium")
            base_priority = 90 if severity == "high" else 72
            common_inclusion = [
                "Reports a defined population, exposure/intervention, and outcome",
                "Provides study design and publication date metadata",
            ]
            common_exclusion = [
                "No stable source identifier or retrievable metadata",
                "Pure opinion without a described empirical or methodological basis",
            ]

            if kind in (
                "low_bootstrap_stability",
                "engine_disagreement",
                "sensitivity_instability",
            ):
                question = (
                    f"Across independent studies and methods, how stable and directionally "
                    f"consistent is the relationship between {source} and {target}?"
                )
                variants = [
                    f"{source} {target} replication stability",
                    f"{source} {target} causal direction sensitivity analysis",
                    f"{source} {target} method comparison robustness",
                ]
                rationale = f"{detail} External studies may contextualize, not repair, instability."
            elif kind in (
                "weak_instrument",
                "invalid_iv_assumption",
                "unverified_iv_assumptions",
                "weak_iv_assumption",
            ):
                z = _terms(
                    edge.get("instrument") or instrument or "the proposed instrument"
                )
                question = (
                    f"Is {z} a relevant, independent, and exclusion-valid instrument for "
                    f"estimating the effect of {source} on {target} in comparable contexts?"
                )
                variants = [
                    f"{z} instrument {source} {target} relevance exclusion restriction",
                    f"{z} instrumental variable first stage {source}",
                    f"{z} invalid instrument direct effect {target}",
                ]
                rationale = f"{detail} Literature can expose assumption risks but cannot validate this dataset's IV design."
                common_inclusion.append(
                    "Explicitly discusses instrument relevance or exclusion"
                )
            elif kind == "low_overlap_positivity":
                question = (
                    f"For populations comparable to {population or 'the study population'}, "
                    f"where is there adequate treatment/exposure overlap for {source} when "
                    f"evaluating {target}?"
                )
                variants = [
                    f"{source} {target} positivity overlap population",
                    f"{source} propensity score common support {target}",
                    f"{source} {target} transportability population overlap",
                ]
                rationale = f"{detail} Population and treatment-assignment context must be cross-matched."
                common_inclusion.append(
                    "Reports population eligibility or treatment assignment"
                )
            elif kind in ("confounder_uncertainty", "collider_selection_bias"):
                bias = (
                    "collider or selection mechanisms"
                    if kind == "collider_selection_bias"
                    else "plausible common causes and adjustment sets"
                )
                question = (
                    f"What {bias} are reported for the relationship between {source} and "
                    f"{target} in {context or 'comparable contexts'}?"
                )
                variants = [
                    f"{source} {target} confounders adjustment set DAG",
                    f"{source} {target} collider selection bias",
                    f"{source} {target} causal diagram common causes",
                ]
                rationale = detail
                common_inclusion.append(
                    "Describes measured covariates, DAG, or selection mechanism"
                )
            elif kind in ("unsupported_orientation", "generative_only_claim"):
                question = (
                    f"What temporal, mechanistic, experimental, or quasi-experimental evidence "
                    f"distinguishes {source} causing {target} from reverse causation?"
                )
                variants = [
                    f"{source} causes {target} longitudinal",
                    f"{target} causes {source} reverse causality",
                    f"{source} {target} randomized natural experiment mechanism",
                ]
                rationale = (
                    f"{detail} SLM/GRAIL/NLP similarity is hypothesis generation only."
                )
                common_inclusion.append(
                    "Contains temporal ordering or an explicit identification design"
                )
            elif kind == "surprising_subgroup_effect":
                question = (
                    f"How does the relationship between {source} and {target} vary across "
                    f"pre-specified populations, settings, and time periods?"
                )
                variants = [
                    f"{source} {target} subgroup heterogeneity effect modification",
                    f"{source} {target} population interaction",
                    f"{source} {target} external validity context",
                ]
                rationale = detail
                common_inclusion.append(
                    "Pre-specifies subgroup or effect-modifier analysis"
                )
                common_exclusion.append(
                    "Post-hoc subgroup claim without uncertainty reporting"
                )
            elif kind in ("refuted_estimate", "production_gate_failure"):
                question = (
                    f"Which assumptions or design differences could explain why the estimated "
                    f"{source}–{target} relationship fails validation or conflicts across analyses?"
                )
                variants = [
                    f"{source} {target} placebo refutation sensitivity",
                    f"{source} {target} contradictory findings study design",
                    f"{source} {target} negative control robustness",
                ]
                rationale = detail
                common_inclusion.append(
                    "Reports robustness, falsification, or sensitivity analysis"
                )
            else:
                question = (
                    f"What credible evidence and unresolved assumptions concern the relationship "
                    f"between {source} and {target}?"
                )
                variants = [
                    f"{source} {target} causal evidence",
                    f"{source} {target} systematic review",
                ]
                rationale = detail

            questions.append(
                self._make_question(
                    kind=kind,
                    finding_ids=finding_ids,
                    question=question,
                    rationale=rationale,
                    priority=base_priority,
                    population=population or None,
                    exposure=source or None,
                    comparator=comparator or None,
                    outcome=target or None,
                    context=context or None,
                    inclusion=common_inclusion,
                    exclusion=common_exclusion,
                    variants=variants,
                )
            )

        if not questions:
            for index, edge in enumerate(handoff.edges[: budget.max_questions]):
                finding_id = str(edge.get("finding_id") or f"edge:{index}")
                source = _terms(edge.get("source") or exposure or "exposure")
                target = _terms(edge.get("target") or outcome or "outcome")
                questions.append(
                    self._make_question(
                        kind="edge_validation",
                        finding_ids=[finding_id],
                        question=(
                            f"What independent evidence supports, contradicts, or contextualizes "
                            f"the relationship between {source} and {target}?"
                        ),
                        rationale=(
                            "Every empirical edge needs external context while preserving its "
                            "original identification grade."
                        ),
                        priority=65,
                        population=population or None,
                        exposure=source,
                        comparator=comparator or None,
                        outcome=target,
                        context=context or None,
                        inclusion=[
                            "Stable source identifier",
                            "Comparable exposure and outcome definitions",
                        ],
                        exclusion=["No retrievable metadata"],
                        variants=[
                            f"{source} {target} causal evidence",
                            f"{source} {target} systematic review",
                        ],
                    )
                )

        # Rule questions always retain precedence.
        deduped: dict[str, ResearchQuestion] = {}
        for question in sorted(questions, key=lambda item: (-item.priority, item.id)):
            fingerprint = re.sub(r"\W+", " ", question.question.lower()).strip()
            deduped.setdefault(fingerprint, question)
        questions = list(deduped.values())
        trace.append(
            {
                "stage": "rule_plan",
                "ok": True,
                "questions": len(questions),
                "deterministic": True,
            }
        )

        if use_slm and self.slm_backend is not None:
            try:
                raw = self.slm_backend.plan_questions(
                    {
                        "handoff": handoff.to_dict(),
                        "rule_questions": [
                            item.to_dict() for item in questions[: budget.max_questions]
                        ],
                        "schema": {
                            "questions": [
                                {
                                    "question": "string",
                                    "rationale": "string",
                                    "query_variants": ["string"],
                                    "finding_ids": ["string"],
                                }
                            ]
                        },
                    }
                )
                payload = raw if isinstance(raw, Mapping) else {}
                enriched = payload.get("questions") or []
                valid = 0
                known_findings = {str(item.get("id")) for item in handoff.findings} | {
                    str(item.get("finding_id")) for item in handoff.edges
                }
                for index, item in enumerate(enriched[: budget.max_questions]):
                    if not isinstance(item, Mapping):
                        continue
                    question = str(item.get("question") or "").strip()
                    rationale = str(item.get("rationale") or "").strip()
                    finding_ids = [
                        str(value)
                        for value in item.get("finding_ids") or []
                        if str(value) in known_findings
                    ]
                    variants = [
                        str(value).strip()
                        for value in item.get("query_variants") or []
                        if str(value).strip()
                    ]
                    if len(question) < 12 or len(rationale) < 8 or not variants:
                        continue
                    candidate = self._make_question(
                        kind="slm_enrichment",
                        finding_ids=finding_ids,
                        question=question,
                        rationale=rationale,
                        priority=min(70, int(item.get("priority") or 55)),
                        population=str(item.get("population") or population or "")
                        or None,
                        exposure=str(item.get("exposure") or exposure or "") or None,
                        comparator=str(item.get("comparator") or comparator or "")
                        or None,
                        outcome=str(item.get("outcome") or outcome or "") or None,
                        context=str(item.get("context") or context or "") or None,
                        inclusion=[
                            str(value) for value in item.get("inclusion_criteria") or []
                        ],
                        exclusion=[
                            str(value) for value in item.get("exclusion_criteria") or []
                        ],
                        variants=variants,
                        source="slm_validated",
                    )
                    fingerprint = re.sub(
                        r"\W+", " ", candidate.question.lower()
                    ).strip()
                    if fingerprint not in deduped:
                        deduped[fingerprint] = candidate
                        questions.append(candidate)
                        valid += 1
                trace.append(
                    {
                        "stage": "slm_plan_enrichment",
                        "ok": True,
                        "accepted": valid,
                        "validated_structured_output": True,
                    }
                )
            except Exception as exc:
                trace.append(
                    {
                        "stage": "slm_plan_enrichment",
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "fallback": "rule",
                    }
                )

        questions.sort(key=lambda item: (-item.priority, item.id))
        return questions[: budget.max_questions], trace


class IntensityRouter:
    """Recommend depth from deterministic risk/coverage signals."""

    def recommend(
        self,
        handoff: ResearchHandoff,
        *,
        selected: SearchIntensity | str,
        policy: ResearchPolicy,
        report: Optional[ResearchReport] = None,
        external: bool = False,
        slm_request: Optional[str] = None,
    ) -> IntensityRecommendation:
        chosen = SearchIntensity.parse(selected)
        score = 0
        reasons: list[str] = []
        metrics: dict[str, Any] = {}

        high_uncertainty = [
            item
            for item in handoff.uncertainty
            if str(item.get("severity") or "") == "high"
        ]
        score += min(3, len(high_uncertainty))
        if high_uncertainty:
            reasons.append(
                f"{len(high_uncertainty)} high-severity uncertainty signal(s)"
            )
        if handoff.gate_failures:
            score += 2
            reasons.append(f"{len(handoff.gate_failures)} failed/escalated gate(s)")
        if handoff.mode == "production" or policy.production_mode:
            score += 1
            reasons.append("production-mode findings require stronger audit coverage")
        high_impact = any(
            token in handoff.domain.lower() for token in policy.high_impact_domains
        )
        if high_impact:
            score += 2
            reasons.append(f"high-impact domain: {handoff.domain}")

        unresolved_contradictions = 0
        independent_min = 0
        mismatch_count = 0
        saturation = None
        budget_fraction = 0.0
        if report is not None:
            unresolved_contradictions = len(report.contradictions)
            independent_min = min(
                (claim.independent_source_count for claim in report.claims),
                default=0,
            )
            mismatch_count = len(report.context_transfer_warnings)
            if report.saturation_curve:
                saturation = report.saturation_curve[-1].get("new_unique_sources")
            if report.budget_planned:
                budget_fraction = report.budget_used.sources_retained / max(
                    1, report.budget_planned.max_sources
                )
            if unresolved_contradictions:
                score += 2
                reasons.append(
                    f"{unresolved_contradictions} unresolved contradiction(s)"
                )
            if independent_min < policy.minimum_independent_sources:
                score += 2
                reasons.append("minimum independent-source coverage has not been met")
            if mismatch_count:
                score += 1
                reasons.append(
                    f"{mismatch_count} context/population mismatch warning(s)"
                )
            if saturation == 0:
                score -= 2
                reasons.append(
                    "latest round added no unique sources (diminishing returns)"
                )
            if budget_fraction >= 0.9:
                score -= 1
                reasons.append("planned source budget is at least 90% consumed")

        if score <= 1:
            recommended = SearchIntensity.QUICK
        elif score <= 3:
            recommended = SearchIntensity.STANDARD
        elif score <= 6:
            recommended = SearchIntensity.DEEP
        else:
            recommended = SearchIntensity.EXHAUSTIVE

        # An SLM request is auditable context only and cannot trigger escalation.
        if slm_request:
            metrics["slm_requested_more_context"] = True
            reasons.append(
                "SLM requested more context; this did not affect intensity routing"
            )

        approval_reasons = policy.approval_reasons(
            chosen, domain=handoff.domain, external=external
        )
        approval_required = bool(approval_reasons and not policy.approval_granted)
        reasons.extend(approval_reasons)
        decision = "human" if approval_required else "proceed"
        if report is not None and (saturation == 0 or budget_fraction >= 1.0):
            decision = "stop" if not approval_required else "human"
        elif report is not None and recommended.rank > chosen.rank:
            decision = "deepen" if not approval_required else "human"

        metrics.update(
            {
                "risk_score": score,
                "high_uncertainty": len(high_uncertainty),
                "gate_failures": len(handoff.gate_failures),
                "unresolved_contradictions": unresolved_contradictions,
                "minimum_independent_sources_observed": independent_min,
                "context_mismatch_warnings": mismatch_count,
                "last_round_new_sources": saturation,
                "source_budget_fraction": round(budget_fraction, 4),
            }
        )
        return IntensityRecommendation(
            selected=chosen,
            recommended=recommended,
            reasons=reasons or ["default standard evidence coverage"],
            approval_required=approval_required,
            decision=decision,
            metrics=metrics,
        )


__all__ = ["AgendaPlanner", "IntensityRouter", "QuestionEnricher"]
