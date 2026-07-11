"""Constrained SLM director and deterministic report-plan fallback."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

from .models import (
    ChartSpec,
    DEFAULT_SECTION_ORDER,
    REPORT_SECTION_CATALOG,
    ReportCitation,
    ReportClaim,
    ReportFact,
    ReportPlan,
    ReportPolicy,
    ReportSource,
    ReportValidationError,
    ensure_unique_ids,
)


_ALLOWED_DIRECTOR_CHARTS = frozenset(
    {
        "bar",
        "line",
        "scatter",
        "forest",
        "heatmap",
        "network",
        "dag",
        "missingness",
        "association",
        "edge_stability",
        "gate_dashboard",
        "evidence_matrix",
        "calibration",
        "roc",
        "pr",
        "feature_importance",
        "table",
    }
)
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)%?")
_ARROW_RE = re.compile(r"`?([^`\n]{1,80}?)`?\s*(?:→|->)\s*`?([^`\n]{1,80}?)`?")


def _fact_index(sources: Sequence[ReportSource]) -> dict[str, ReportFact]:
    return {fact.id: fact for source in sources for fact in source.facts}


def _citation_index(sources: Sequence[ReportSource]) -> dict[str, ReportCitation]:
    return {
        citation.id: citation
        for source in sources
        for citation in source.citations
    }


def _table_ids(sources: Sequence[ReportSource]) -> set[str]:
    return {table.id for source in sources for table in source.tables}


def _available_sections(sources: Sequence[ReportSource]) -> set[str]:
    available = {
        fact.category for source in sources for fact in source.facts if fact.category
    }
    available.update(
        table.category for source in sources for table in source.tables if table.category
    )
    available.update(
        chart.category for source in sources for chart in source.charts if chart.category
    )
    available.update(
        {
            "cover",
            "executive_summary",
            "scope_provenance",
            "limitations",
            "technical_appendix",
        }
    )
    return available


def _dynamic_required_sections(
    sources: Sequence[ReportSource], policy: ReportPolicy
) -> set[str]:
    required = set(policy.required_sections)
    if not policy.include_appendix:
        required.discard("technical_appendix")
    facts = [fact for source in sources for fact in source.facts]
    if any(
        fact.attributes.get("synthetic_iv")
        or "synthetic instrument" in fact.caveat.lower()
        for fact in facts
    ):
        required.add("iv_evidence")
    if any(
        (
            fact.category == "refutations_sensitivity"
            and (
                "failed" in fact.label.lower()
                or "escalat" in fact.label.lower()
                or "rejected" in fact.label.lower()
                or fact.attributes.get("rejected")
            )
            and fact.value not in (0, False, None, "", [])
        )
        for fact in facts
    ):
        required.add("refutations_sensitivity")
    if any(fact.attributes.get("contradiction") for fact in facts):
        required.add("deep_research")
    return required


def _short_value(value: Any, *, limit: int = 180) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _fallback_claim(fact: ReportFact, *, generated_by: str = "rule") -> ReportClaim:
    eligibility = (
        " This item is retained for audit only and excluded from evidence."
        if not fact.evidence_eligible
        else ""
    )
    return ReportClaim(
        text=f"{fact.label}: {_short_value(fact.value)}.{eligibility}",
        fact_ids=[fact.id],
        provenance_ids=[fact.provenance_id],
        citation_ids=list(fact.citation_ids),
        generated_by=generated_by,
    )


def deterministic_report_plan(
    sources: Sequence[ReportSource],
    policy: ReportPolicy,
    *,
    title: str = "AutoCausal Analysis Report",
    audience: str = "technical and decision stakeholders",
    purpose: str = (
        "Summarize normalized AutoCausal evidence, uncertainty, provenance, "
        "failed gates, and recommended follow-up."
    ),
) -> ReportPlan:
    """Build the stable rule/template plan used by default and on SLM failure."""
    source_ids = [source.id for source in sources]
    available = _available_sections(sources)
    required = _dynamic_required_sections(sources, policy)
    ordered = [
        section_id
        for section_id in DEFAULT_SECTION_ORDER
        if section_id in policy.allowed_sections
        and (section_id != "technical_appendix" or policy.include_appendix)
        and (section_id in available or section_id in required)
    ]
    for section_id in required:
        if section_id not in ordered and section_id in policy.allowed_sections:
            insertion = list(DEFAULT_SECTION_ORDER).index(section_id)
            position = len(ordered)
            for index, present in enumerate(ordered):
                if list(DEFAULT_SECTION_ORDER).index(present) > insertion:
                    position = index
                    break
            ordered.insert(position, section_id)

    fact_map = _fact_index(sources)
    summaries: dict[str, list[ReportClaim]] = {}
    for section_id in ordered:
        if section_id in {"cover", "technical_appendix"}:
            continue
        candidates = sorted(
            (
                fact
                for fact in fact_map.values()
                if fact.category == section_id
            ),
            key=lambda fact: (-fact.priority, fact.id),
        )
        if section_id == "executive_summary":
            candidates = sorted(
                (
                    fact
                    for fact in fact_map.values()
                    if fact.category
                    not in {"technical_appendix", "scope_provenance"}
                ),
                key=lambda fact: (-fact.priority, fact.id),
            )
        if section_id == "limitations":
            candidates = sorted(
                (
                    fact
                    for fact in fact_map.values()
                    if fact.caveat
                    or not fact.evidence_eligible
                    or fact.attributes.get("contradiction")
                    or fact.attributes.get("rejected")
                ),
                key=lambda fact: (-fact.priority, fact.id),
            )
        if candidates:
            summaries[section_id] = [
                _fallback_claim(fact) for fact in candidates[:3]
            ]

    charts = sorted(
        (chart for source in sources for chart in source.charts),
        key=lambda chart: (-chart.priority, chart.id),
    )[: policy.max_charts]
    actions = [
        {
            "action": "select_sections",
            "accepted": True,
            "detail": f"Selected {len(ordered)} sections from normalized evidence.",
            "source_ids": source_ids,
        },
        {
            "action": "preserve_required_caveats",
            "accepted": True,
            "detail": "Required gate, synthetic-IV, contradiction, and limitation sections preserved.",
            "required_sections": sorted(required),
        },
    ]
    return ReportPlan(
        title=title,
        audience=audience,
        purpose=purpose,
        section_order=ordered,
        included_artifacts=source_ids,
        excluded_artifacts=[],
        chart_specs=charts,
        appendix_policy={
            "include": policy.include_appendix,
            "manifest": True,
            "engine_versions": True,
            "policy_thresholds": True,
            "trace_summaries": True,
        },
        citation_policy={
            "integrity": policy.citation_integrity,
            "require_verified": policy.require_verified_citations,
            "source_record_ids_only": True,
        },
        redaction_policy={
            "raw_data_prohibited": policy.raw_data_prohibited,
            "redact_pii": policy.redact_pii,
            "redact_secrets": policy.redact_secrets,
        },
        section_summaries=summaries,
        director_backend="rule",
        director_actions=actions,
    )


def _extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for token in _NUMBER_RE.findall(text):
        percent = token.endswith("%")
        try:
            value = float(token.rstrip("%"))
        except ValueError:
            continue
        values.append(value)
        if percent:
            values.append(value / 100.0)
    return values


def _fact_numbers(facts: Sequence[ReportFact]) -> list[float]:
    values: list[float] = []

    def visit(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                number = float(value)
                values.extend((number, number * 100.0))
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            values.extend(_extract_numbers(value))

    for fact in facts:
        visit(fact.value)
    return values


def _number_supported(number: float, candidates: Sequence[float]) -> bool:
    return any(
        math.isclose(number, candidate, rel_tol=1e-5, abs_tol=1e-8)
        for candidate in candidates
    )


def _validate_claim(
    claim: ReportClaim,
    *,
    fact_map: Mapping[str, ReportFact],
    citations: Mapping[str, ReportCitation],
    section_id: str,
    max_chars: int,
) -> None:
    if not claim.fact_ids:
        raise ReportValidationError(
            f"Narrative claim in `{section_id}` has no normalized fact ids"
        )
    unknown_facts = set(claim.fact_ids) - set(fact_map)
    if unknown_facts:
        raise ReportValidationError(
            f"Narrative claim in `{section_id}` invented/unknown fact ids: "
            f"{sorted(unknown_facts)}"
        )
    if len(claim.text) > max_chars:
        raise ReportValidationError(
            f"Narrative claim in `{section_id}` exceeds {max_chars} characters"
        )
    referenced = [fact_map[fact_id] for fact_id in claim.fact_ids]
    unsupported_citations = set(claim.citation_ids) - set(citations)
    if unsupported_citations:
        raise ReportValidationError(
            f"Narrative claim in `{section_id}` uses unsupported citation ids: "
            f"{sorted(unsupported_citations)}"
        )
    allowed_for_facts = {
        citation_id for fact in referenced for citation_id in fact.citation_ids
    }
    detached_citations = set(claim.citation_ids) - allowed_for_facts
    if detached_citations:
        raise ReportValidationError(
            f"Narrative claim in `{section_id}` cites records not mapped by its "
            f"facts: {sorted(detached_citations)}"
        )
    # Adapter labels can contain grounded ordinal identifiers such as
    # "Literature finding 1" or "Discovery edge 2". Remove exact referenced
    # labels before applying the invented-metric check; values in the claim
    # still must occur in the referenced fact payloads.
    metric_text = claim.text
    for fact in referenced:
        metric_text = metric_text.replace(fact.label, "")
    claim_numbers = _extract_numbers(metric_text)
    supported_numbers = _fact_numbers(referenced)
    unsupported_numbers = [
        number
        for number in claim_numbers
        if not _number_supported(number, supported_numbers)
    ]
    if unsupported_numbers:
        raise ReportValidationError(
            f"Narrative claim in `{section_id}` contains metric(s) absent from "
            f"its referenced facts: {unsupported_numbers}"
        )
    if "→" in claim.text or "->" in claim.text:
        allowed_pairs = {
            tuple(str(item) for item in fact.attributes.get("edge_pair", []))
            for fact in referenced
            if len(fact.attributes.get("edge_pair", [])) == 2
        }
        normalized_text = claim.text.replace("`", "").lower()
        if not any(
            str(source).lower() in normalized_text
            and str(target).lower() in normalized_text
            for source, target in allowed_pairs
        ):
            raise ReportValidationError(
                f"Narrative claim in `{section_id}` contains an edge not mapped "
                "by its referenced facts"
            )


def validate_report_plan(
    plan: ReportPlan,
    sources: Sequence[ReportSource],
    policy: ReportPolicy,
) -> dict[str, Any]:
    """Validate a complete plan against source ids, facts, citations, and policy."""
    source_map = {source.id: source for source in sources}
    fact_map = _fact_index(sources)
    citations = _citation_index(sources)
    tables = _table_ids(sources)
    available = _available_sections(sources)
    dynamic_required = _dynamic_required_sections(sources, policy)

    unknown_sections = set(plan.section_order) - set(REPORT_SECTION_CATALOG)
    if unknown_sections:
        raise ReportValidationError(
            f"Report plan contains invented sections: {sorted(unknown_sections)}"
        )
    disallowed = set(plan.section_order) - set(policy.allowed_sections)
    if disallowed:
        raise ReportValidationError(
            f"Report plan contains policy-disallowed sections: {sorted(disallowed)}"
        )
    if not policy.include_appendix and "technical_appendix" in plan.section_order:
        raise ReportValidationError(
            "Report plan includes the technical appendix while policy disables it"
        )
    unavailable = set(plan.section_order) - available
    if unavailable:
        raise ReportValidationError(
            f"Report plan contains sections without evidence: {sorted(unavailable)}"
        )
    missing_required = dynamic_required - set(plan.section_order)
    if missing_required:
        raise ReportValidationError(
            "Report plan suppresses required caveat/evidence sections: "
            f"{sorted(missing_required)}"
        )
    if len(plan.section_order) != len(set(plan.section_order)):
        raise ReportValidationError("Report plan section ids must be unique")

    unknown_included = set(plan.included_artifacts) - set(source_map)
    unknown_excluded = set(plan.excluded_artifacts) - set(source_map)
    if unknown_included or unknown_excluded:
        raise ReportValidationError(
            "Report plan contains unknown artifact ids: "
            f"{sorted(unknown_included | unknown_excluded)}"
        )
    overlap = set(plan.included_artifacts) & set(plan.excluded_artifacts)
    if overlap:
        raise ReportValidationError(
            f"Artifacts cannot be both included and excluded: {sorted(overlap)}"
        )
    if not plan.included_artifacts:
        raise ReportValidationError("Report plan must include at least one source artifact")

    protected_sources = {
        fact.source_id
        for fact in fact_map.values()
        if fact.attributes.get("synthetic_iv")
        or fact.attributes.get("contradiction")
        or fact.attributes.get("rejected")
        or (
            fact.category == "refutations_sensitivity"
            and (
                "failed" in fact.label.lower()
                or "escalat" in fact.label.lower()
            )
            and fact.value not in (0, False, None, "", [])
        )
    }
    suppressed_protected = protected_sources & set(plan.excluded_artifacts)
    if suppressed_protected:
        raise ReportValidationError(
            "Report plan excludes artifacts carrying failed gates, synthetic-IV "
            f"labels, or contradictions: {sorted(suppressed_protected)}"
        )

    for section_id, claims in plan.section_summaries.items():
        if section_id not in plan.section_order:
            raise ReportValidationError(
                f"Narrative was proposed for unselected section `{section_id}`"
            )
        for claim in claims:
            _validate_claim(
                claim,
                fact_map=fact_map,
                citations=citations,
                section_id=section_id,
                max_chars=policy.max_narrative_chars,
            )
    for section_id, claim in plan.transitions.items():
        if section_id not in plan.section_order:
            raise ReportValidationError(
                f"Transition targets unselected section `{section_id}`"
            )
        _validate_claim(
            claim,
            fact_map=fact_map,
            citations=citations,
            section_id=section_id,
            max_chars=policy.max_narrative_chars,
        )

    if len(plan.chart_specs) > policy.max_charts:
        raise ReportValidationError(
            f"Report plan requests {len(plan.chart_specs)} charts; "
            f"policy maximum is {policy.max_charts}"
        )
    ensure_unique_ids(plan.chart_specs, kind="chart")
    for chart in plan.chart_specs:
        unknown_fact_refs = set(chart.source_fact_ids) - set(fact_map)
        if unknown_fact_refs:
            raise ReportValidationError(
                f"Chart `{chart.id}` references invented facts: "
                f"{sorted(unknown_fact_refs)}"
            )
        if chart.source_table_id and chart.source_table_id not in tables:
            raise ReportValidationError(
                f"Chart `{chart.id}` references unknown table "
                f"`{chart.source_table_id}`"
            )
        if (
            not chart.source_fact_ids
            and not chart.source_table_id
            and not chart.image_path
            and not chart.provenance_ids
        ):
            raise ReportValidationError(
                f"Chart `{chart.id}` has no evidence/provenance mapping"
            )

    for source in sources:
        if source.contains_raw_data:
            raise ReportValidationError(
                f"Source `{source.id}` retained raw data, which reporting forbids"
            )
        unsupported = source.metadata.get("unsupported_citations")
        unsupported_ids = source.metadata.get("unsupported_citation_ids")
        if policy.production_mode and policy.citation_integrity and (
            unsupported or unsupported_ids
        ):
            raise ReportValidationError(
                f"Source `{source.id}` contains unsupported citations: "
                f"{unsupported_ids or unsupported}"
            )
    used_citations = {
        citation_id
        for fact in fact_map.values()
        for citation_id in fact.citation_ids
    }
    unknown_fact_citations = used_citations - set(citations)
    if policy.citation_integrity and unknown_fact_citations:
        raise ReportValidationError(
            f"Facts reference unsupported citation ids: {sorted(unknown_fact_citations)}"
        )
    if policy.require_verified_citations:
        unverified = {
            citation_id
            for citation_id in used_citations
            if not citations[citation_id].verified
        }
        if unverified:
            raise ReportValidationError(
                "Production citation policy requires verification for cited "
                f"records: {sorted(unverified)}"
            )

    synthetic_eligible = [
        fact.id
        for fact in fact_map.values()
        if fact.attributes.get("synthetic_iv") and fact.evidence_eligible
    ]
    if policy.production_mode and synthetic_eligible:
        raise ReportValidationError(
            "Synthetic-IV facts cannot be production evidence: "
            f"{synthetic_eligible}"
        )
    return {
        "ok": True,
        "source_count": len(sources),
        "fact_count": len(fact_map),
        "citation_count": len(citations),
        "section_count": len(plan.section_order),
        "chart_count": len(plan.chart_specs),
        "required_sections": sorted(dynamic_required),
        "director_backend": plan.director_backend,
    }


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ReportValidationError("SLM director did not return a JSON object")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ReportValidationError(
            f"SLM director returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ReportValidationError("SLM director output must be a JSON object")
    return payload


def validate_slm_proposal(
    proposal: Mapping[str, Any],
    *,
    baseline: ReportPlan,
    sources: Sequence[ReportSource],
    policy: ReportPolicy,
    backend_label: str = "slm",
) -> ReportPlan:
    """Convert and validate a structured SLM proposal against strict allowlists."""
    payload = dict(proposal)
    fact_map = _fact_index(sources)
    citations = _citation_index(sources)
    available = _available_sections(sources)
    required = _dynamic_required_sections(sources, policy)
    allowed_for_run = set(baseline.section_order) | required
    permissions = set(policy.slm_permissions)

    if not permissions.intersection({"select_sections", "prioritize_sections"}):
        raise ReportValidationError(
            "SLM policy does not permit section selection or prioritization"
        )
    if (
        payload.get("claims") or payload.get("section_summaries")
    ) and "summarize_facts" not in permissions:
        raise ReportValidationError(
            "SLM policy does not permit normalized-fact summaries"
        )
    if payload.get("transitions") and "propose_transitions" not in permissions:
        raise ReportValidationError("SLM policy does not permit transitions")
    if (
        payload.get("chart_ids") or payload.get("chart_recommendations")
    ) and "recommend_charts" not in permissions:
        raise ReportValidationError(
            "SLM policy does not permit chart recommendations"
        )
    if payload.get("followups") and "recommend_followups" not in permissions:
        raise ReportValidationError(
            "SLM policy does not permit follow-up recommendations"
        )

    section_order = [
        str(section_id) for section_id in payload.get("section_order") or []
    ]
    if not section_order:
        raise ReportValidationError("SLM proposal omitted section_order")
    invented_sections = set(section_order) - set(REPORT_SECTION_CATALOG)
    if invented_sections:
        raise ReportValidationError(
            f"SLM proposed invented sections: {sorted(invented_sections)}"
        )
    unavailable = set(section_order) - available
    if unavailable:
        raise ReportValidationError(
            f"SLM proposed sections without evidence: {sorted(unavailable)}"
        )
    outside_allowlist = set(section_order) - allowed_for_run
    if outside_allowlist:
        raise ReportValidationError(
            f"SLM proposed sections outside this run's allowlist: "
            f"{sorted(outside_allowlist)}"
        )
    if missing := required - set(section_order):
        raise ReportValidationError(
            f"SLM suppressed required sections: {sorted(missing)}"
        )

    summaries: dict[str, list[ReportClaim]] = {}
    actions: list[dict[str, Any]] = []
    for item in payload.get("claims") or payload.get("section_summaries") or []:
        raw = dict(item) if isinstance(item, Mapping) else {}
        section_id = str(raw.get("section_id") or raw.get("section") or "")
        claim = ReportClaim(
            text=str(raw.get("text") or raw.get("summary") or ""),
            fact_ids=[str(item) for item in _as_sequence(raw.get("fact_ids"))],
            citation_ids=[
                str(item) for item in _as_sequence(raw.get("citation_ids"))
            ],
            provenance_ids=[
                fact_map[str(fact_id)].provenance_id
                for fact_id in _as_sequence(raw.get("fact_ids"))
                if str(fact_id) in fact_map
            ],
            generated_by=backend_label,
            label="SLM-directed",
        )
        if section_id not in section_order:
            raise ReportValidationError(
                f"SLM claim targets unselected section `{section_id}`"
            )
        _validate_claim(
            claim,
            fact_map=fact_map,
            citations=citations,
            section_id=section_id,
            max_chars=policy.max_narrative_chars,
        )
        summaries.setdefault(section_id, []).append(claim)
        actions.append(
            {
                "action": "summarize_facts",
                "accepted": True,
                "section_id": section_id,
                "fact_ids": list(claim.fact_ids),
            }
        )

    transitions: dict[str, ReportClaim] = {}
    for item in payload.get("transitions") or []:
        raw = dict(item) if isinstance(item, Mapping) else {}
        section_id = str(raw.get("section_id") or raw.get("to_section") or "")
        claim = ReportClaim(
            text=str(raw.get("text") or ""),
            fact_ids=[str(item) for item in _as_sequence(raw.get("fact_ids"))],
            citation_ids=[
                str(item) for item in _as_sequence(raw.get("citation_ids"))
            ],
            provenance_ids=[
                fact_map[str(fact_id)].provenance_id
                for fact_id in _as_sequence(raw.get("fact_ids"))
                if str(fact_id) in fact_map
            ],
            generated_by=backend_label,
            label="Transition",
        )
        if section_id not in section_order:
            raise ReportValidationError(
                f"SLM transition targets unselected section `{section_id}`"
            )
        _validate_claim(
            claim,
            fact_map=fact_map,
            citations=citations,
            section_id=section_id,
            max_chars=policy.max_narrative_chars,
        )
        transitions[section_id] = claim
        actions.append(
            {
                "action": "propose_transitions",
                "accepted": True,
                "section_id": section_id,
                "fact_ids": list(claim.fact_ids),
            }
        )

    baseline_charts = {chart.id: chart for chart in baseline.chart_specs}
    chart_specs: list[ChartSpec] = []
    for chart_id in payload.get("chart_ids") or []:
        chart_key = str(chart_id)
        if chart_key not in baseline_charts:
            raise ReportValidationError(
                f"SLM selected invented chart id `{chart_key}`"
            )
        chart_specs.append(baseline_charts[chart_key])
        actions.append(
            {
                "action": "recommend_charts",
                "accepted": True,
                "chart_id": chart_key,
            }
        )
    for index, item in enumerate(payload.get("chart_recommendations") or []):
        raw = dict(item) if isinstance(item, Mapping) else {}
        chart_type = str(raw.get("chart_type") or "").lower()
        if chart_type not in _ALLOWED_DIRECTOR_CHARTS:
            raise ReportValidationError(
                f"SLM proposed unsupported chart type `{chart_type}`"
            )
        fact_ids = [str(item) for item in _as_sequence(raw.get("source_fact_ids"))]
        unknown = set(fact_ids) - set(fact_map)
        if not fact_ids or unknown:
            raise ReportValidationError(
                "SLM chart recommendations require existing source_fact_ids; "
                f"unknown={sorted(unknown)}"
            )
        chart_specs.append(
            ChartSpec(
                id=f"director-chart-{index + 1}",
                chart_type=chart_type,
                title=str(raw.get("title") or f"Recommended {chart_type}"),
                alt_text=str(
                    raw.get("alt_text")
                    or f"{chart_type} chart over referenced normalized facts."
                ),
                source_fact_ids=fact_ids,
                spec={
                    "encoding": raw.get("encoding") or {},
                    "reason": raw.get("reason") or "",
                    "slm_recommended": True,
                },
                provenance_ids=[
                    fact_map[fact_id].provenance_id for fact_id in fact_ids
                ],
                priority=int(raw.get("priority") or 50),
            )
        )
        actions.append(
            {
                "action": "recommend_charts",
                "accepted": True,
                "chart_id": f"director-chart-{index + 1}",
                "fact_ids": fact_ids,
            }
        )
    if not chart_specs:
        chart_specs = list(baseline.chart_specs)

    title = str(payload.get("title") or baseline.title).strip()
    if not title or len(title) > 180:
        raise ReportValidationError("SLM title is empty or exceeds 180 characters")
    included = [
        str(item)
        for item in payload.get("included_artifacts")
        or baseline.included_artifacts
    ]
    excluded = [
        str(item)
        for item in payload.get("excluded_artifacts")
        or baseline.excluded_artifacts
    ]
    actions.insert(
        0,
        {
            "action": "prioritize_sections",
            "accepted": True,
            "section_order": section_order,
        },
    )
    for followup in payload.get("followups") or []:
        raw = dict(followup) if isinstance(followup, Mapping) else {}
        fact_ids = [str(item) for item in _as_sequence(raw.get("fact_ids"))]
        if not fact_ids or set(fact_ids) - set(fact_map):
            raise ReportValidationError(
                "SLM follow-up recommendations must reference existing fact ids"
            )
        actions.append(
            {
                "action": "recommend_followups",
                "accepted": True,
                "detail": str(raw.get("text") or ""),
                "fact_ids": fact_ids,
            }
        )

    plan = ReportPlan(
        title=title,
        audience=baseline.audience,
        purpose=baseline.purpose,
        section_order=section_order,
        included_artifacts=included,
        excluded_artifacts=excluded,
        chart_specs=chart_specs[: policy.max_charts],
        appendix_policy=dict(baseline.appendix_policy),
        citation_policy=dict(baseline.citation_policy),
        redaction_policy=dict(baseline.redaction_policy),
        section_summaries=summaries or dict(baseline.section_summaries),
        transitions=transitions,
        director_backend=backend_label,
        director_actions=actions,
    )
    validate_report_plan(plan, sources, policy)
    return plan


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    return [value]


class SLMReportDirector:
    """Direct report composition using only a validated normalized-fact inventory."""

    def __init__(
        self,
        *,
        use_slm: bool = True,
        policy: ReportPolicy | None = None,
        model_name: str | None = None,
        backend: Any = None,
    ) -> None:
        self.use_slm = bool(use_slm)
        self.policy = policy or ReportPolicy.production()
        self.model_name = model_name
        self.backend = backend
        self.last_error: str | None = None
        self.last_backend: str = "rule"

    def _backend(self) -> Any:
        if self.backend is not None:
            return self.backend
        from autocausal.slm import get_backend

        self.backend = get_backend(use_slm=True, model_name=self.model_name)
        return self.backend

    def _backend_label(self, backend: Any) -> str:
        name = str(getattr(backend, "name", type(backend).__name__))
        model = str(
            getattr(backend, "model_name", None)
            or getattr(backend, "model_id", None)
            or self.model_name
            or ""
        )
        return f"{name}:{model}" if model else name

    def _inventory(
        self, sources: Sequence[ReportSource], baseline: ReportPlan
    ) -> dict[str, Any]:
        facts = []
        for fact in sorted(
            (fact for source in sources for fact in source.facts),
            key=lambda item: (-item.priority, item.id),
        ):
            item: dict[str, Any] = {
                "id": fact.id,
                "source_id": fact.source_id,
                "provenance_id": fact.provenance_id,
                "label": fact.label,
                "category": fact.category,
                "citation_ids": list(fact.citation_ids),
                "caveat": fact.caveat,
                "evidence_eligible": fact.evidence_eligible,
                "attributes": {
                    key: value
                    for key, value in fact.attributes.items()
                    if key
                    in {
                        "synthetic_iv",
                        "rejected",
                        "contradiction",
                        "edge_pair",
                        "predictive_only",
                        "metric",
                    }
                },
            }
            if self.policy.send_fact_values_to_slm:
                item["value"] = fact.value
            facts.append(item)
        return {
            "schema": "AutoCausalReportDirectorInput.v1",
            "allowed_sections": list(baseline.section_order),
            "required_sections": sorted(
                _dynamic_required_sections(sources, self.policy)
            ),
            "source_ids": [source.id for source in sources],
            "facts": facts[:300],
            "existing_charts": [
                {
                    "id": chart.id,
                    "type": chart.chart_type,
                    "title": chart.title,
                    "source_fact_ids": chart.source_fact_ids,
                    "source_table_id": chart.source_table_id,
                }
                for chart in baseline.chart_specs
            ],
            "allowed_chart_types": sorted(_ALLOWED_DIRECTOR_CHARTS),
            "rules": {
                "no_new_fact_ids": True,
                "no_new_source_ids": True,
                "no_new_citation_ids": True,
                "every_claim_requires_fact_ids": True,
                "preserve_required_sections": True,
                "synthetic_iv_is_audit_only": True,
                "raw_data_absent": True,
            },
        }

    def _prompt(self, inventory: Mapping[str, Any]) -> str:
        schema = {
            "title": "concise title",
            "section_order": ["only ids from allowed_sections"],
            "claims": [
                {
                    "section_id": "selected section id",
                    "text": "concise claim; every number must occur in referenced facts",
                    "fact_ids": ["existing fact id"],
                    "citation_ids": ["only ids already attached to those facts"],
                }
            ],
            "transitions": [
                {
                    "section_id": "selected section id",
                    "text": "short transition grounded in facts",
                    "fact_ids": ["existing fact id"],
                }
            ],
            "chart_ids": ["existing chart id"],
            "chart_recommendations": [
                {
                    "chart_type": "allowed chart type",
                    "title": "title",
                    "alt_text": "accessible description",
                    "source_fact_ids": ["existing fact id"],
                    "reason": "reason",
                    "priority": 50,
                }
            ],
            "followups": [
                {"text": "follow-up action", "fact_ids": ["existing fact id"]}
            ],
        }
        return (
            "Return exactly one JSON object matching this schema. Select and "
            "prioritize report sections, write only concise claims grounded in "
            "listed fact ids, retain failed gates/contradictions/synthetic-IV "
            "warnings, and recommend charts only from listed facts. Never invent "
            "metrics, citations, edges, estimates, source ids, or fact ids.\n\n"
            f"OUTPUT_SCHEMA:\n{json.dumps(schema, indent=2, sort_keys=True)}\n\n"
            f"NORMALIZED_INVENTORY:\n"
            f"{json.dumps(inventory, indent=2, sort_keys=True, default=str)}"
        )

    def _generate(self, backend: Any, inventory: Mapping[str, Any]) -> Any:
        if hasattr(backend, "generate_report_plan"):
            return backend.generate_report_plan(dict(inventory))
        prompt = self._prompt(inventory)
        system = (
            "You are a constrained AutoCausal report planner. Use only normalized "
            "fact/source/citation ids in the input. Output JSON only. You may not "
            "remove required caveats or present associations/predictions/synthetic "
            "instruments as identified causal effects."
        )
        generate = getattr(backend, "_generate", None)
        if callable(generate):
            return generate(prompt, system=system)
        generate = getattr(backend, "generate", None)
        if callable(generate):
            try:
                return generate(prompt, system=system)
            except TypeError:
                return generate(prompt)
        raise ReportValidationError(
            f"Director backend {type(backend).__name__} has no generation method"
        )

    def plan(
        self,
        sources: Sequence[ReportSource],
        *,
        title: str = "AutoCausal Analysis Report",
        audience: str = "technical and decision stakeholders",
        purpose: str = (
            "Summarize normalized AutoCausal evidence, uncertainty, provenance, "
            "failed gates, and recommended follow-up."
        ),
    ) -> ReportPlan:
        baseline = deterministic_report_plan(
            sources,
            self.policy,
            title=title,
            audience=audience,
            purpose=purpose,
        )
        if not self.use_slm:
            self.last_backend = "rule"
            validate_report_plan(baseline, sources, self.policy)
            return baseline
        if not self.policy.allow_slm:
            baseline.warnings.append(
                "SLM direction was requested but policy disallows it; deterministic "
                "rule planning was used."
            )
            baseline.director_actions.append(
                {
                    "action": "slm_direction",
                    "accepted": False,
                    "reason": "policy_disallowed",
                }
            )
            self.last_backend = "rule:policy_disallowed"
            validate_report_plan(baseline, sources, self.policy)
            return baseline

        try:
            backend = self._backend()
            backend_label = self._backend_label(backend)
            self.last_backend = backend_label
            inventory = self._inventory(sources, baseline)
            raw = self._generate(backend, inventory)
            proposal = _parse_json_object(raw)
            return validate_slm_proposal(
                proposal,
                baseline=baseline,
                sources=sources,
                policy=self.policy,
                backend_label=backend_label,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            baseline.director_backend = f"rule-fallback:{self.last_backend}"
            baseline.warnings.append(
                "SLM report direction was discarded and deterministic fallback "
                f"used: {self.last_error}"
            )
            baseline.director_actions.append(
                {
                    "action": "slm_direction",
                    "accepted": False,
                    "reason": self.last_error,
                    "fallback": "deterministic_rule_plan",
                }
            )
            validate_report_plan(baseline, sources, self.policy)
            return baseline


__all__ = [
    "SLMReportDirector",
    "deterministic_report_plan",
    "validate_report_plan",
    "validate_slm_proposal",
]
