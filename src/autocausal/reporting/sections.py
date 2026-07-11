"""Build render-ready sections from a validated report plan and evidence."""

from __future__ import annotations

from typing import Any, Sequence

from .adapters import (
    ASSOCIATION_CAVEAT,
    CAUSAL_CAVEAT,
    NLP_PRIVACY_CAVEAT,
    SLM_CAVEAT,
    SYNTHETIC_IV_CAVEAT,
)
from .models import (
    REPORT_SECTION_CATALOG,
    ChartSpec,
    ReportCitation,
    ReportClaim,
    ReportFact,
    ReportPlan,
    ReportPolicy,
    ReportSection,
    ReportSource,
    ReportTable,
)


SECTION_CAVEATS: dict[str, list[str]] = {
    "associations": [ASSOCIATION_CAVEAT],
    "discovery": [CAUSAL_CAVEAT],
    "causal_estimates": [
        CAUSAL_CAVEAT,
        "Confidence intervals and diagnostics quantify model uncertainty; they "
        "do not validate untestable identification assumptions.",
    ],
    "iv_evidence": [
        "Observed instruments and synthetic instruments are reported separately.",
        SYNTHETIC_IV_CAVEAT,
        "Observed Z still requires documented relevance, exclusion, and design review.",
    ],
    "refutations_sensitivity": [
        "Refutation and sensitivity checks probe robustness but do not prove "
        "causal identification."
    ],
    "automl": [
        "Predictive AutoML metrics are not causal estimates and are reported separately."
    ],
    "nlp_behavioral": [NLP_PRIVACY_CAVEAT, ASSOCIATION_CAVEAT],
    "insight_actions": [
        SLM_CAVEAT,
        "Experiment recommendations require human design review and policy approval.",
    ],
    "deep_research": [
        "Only supplied/fetched source-record ids are eligible citations.",
        "Contradictory literature evidence is retained for review.",
    ],
    "limitations": [
        CAUSAL_CAVEAT,
        "Missing report families are omitted and recorded in the report audit; "
        "absence of a section is not evidence that no issue exists.",
    ],
}


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if item))


def _included_sources(
    sources: Sequence[ReportSource], plan: ReportPlan
) -> list[ReportSource]:
    included = set(plan.included_artifacts)
    excluded = set(plan.excluded_artifacts)
    return [
        source
        for source in sources
        if source.id in included and source.id not in excluded
    ]


def _bounded_table(table: ReportTable, policy: ReportPolicy) -> ReportTable:
    if len(table.rows) <= policy.max_rows_per_table:
        return table
    footnote = table.footnote
    if footnote:
        footnote += " "
    footnote += (
        f"Showing {policy.max_rows_per_table} of {len(table.rows)} rows by report policy."
    )
    return ReportTable(
        id=table.id,
        title=table.title,
        columns=list(table.columns),
        rows=list(table.rows[: policy.max_rows_per_table]),
        provenance_ids=list(table.provenance_ids),
        footnote=footnote,
        sensitive_columns=list(table.sensitive_columns),
        category=table.category,
    )


def _source_inventory(sources: Sequence[ReportSource], *, table_id: str) -> ReportTable:
    return ReportTable(
        id=table_id,
        title="Normalized source artifact inventory",
        columns=[
            "source_id",
            "family",
            "title",
            "facts",
            "tables",
            "charts",
            "citations",
            "run_id",
        ],
        rows=[
            {
                "source_id": source.id,
                "family": source.family,
                "title": source.title,
                "facts": len(source.facts),
                "tables": len(source.tables),
                "charts": len(source.charts),
                "citations": len(source.citations),
                "run_id": source.metadata.get("run_id") or "",
            }
            for source in sources
        ],
        provenance_ids=[source.id for source in sources],
        footnote="All artifacts were normalized without retaining raw rows.",
        category="scope_provenance",
    )


def _policy_table(policy: ReportPolicy) -> ReportTable:
    values = policy.to_dict()
    keys = (
        "profile",
        "production_mode",
        "max_pages",
        "max_rows_per_table",
        "max_charts",
        "raw_data_prohibited",
        "redact_pii",
        "redact_secrets",
        "citation_integrity",
        "require_verified_citations",
        "allow_slm",
        "template",
        "theme",
        "page_size",
    )
    return ReportTable(
        id="report:policy-thresholds",
        title="Report policy and thresholds",
        columns=["setting", "value"],
        rows=[{"setting": key, "value": values.get(key)} for key in keys],
        provenance_ids=["report-policy"],
        category="technical_appendix",
    )


def _director_actions_table(plan: ReportPlan) -> ReportTable:
    rows = []
    for action in plan.director_actions:
        rows.append(
            {
                "action": action.get("action"),
                "accepted": action.get("accepted"),
                "section": action.get("section_id"),
                "detail": action.get("detail") or action.get("reason"),
                "fact_ids": action.get("fact_ids"),
            }
        )
    return ReportTable(
        id="report:director-actions",
        title="Report director action audit",
        columns=["action", "accepted", "section", "detail", "fact_ids"],
        rows=rows,
        provenance_ids=["report-director"],
        footnote=f"Director backend: {plan.director_backend}",
        category="technical_appendix",
    )


def _claims_for(
    plan: ReportPlan,
    section_id: str,
    fact_map: dict[str, ReportFact],
) -> list[ReportClaim]:
    claims = list(plan.section_summaries.get(section_id) or [])
    transition = plan.transitions.get(section_id)
    if transition is not None:
        claims.append(transition)
    # The plan validator already rejects unknown ids; this defensive filter keeps
    # section construction safe if called independently.
    return [
        claim
        for claim in claims
        if claim.fact_ids and all(fact_id in fact_map for fact_id in claim.fact_ids)
    ]


def _section_summary(section_id: str, claims: Sequence[ReportClaim]) -> str:
    if claims:
        generated = any(
            claim.generated_by not in ("rule", "template") for claim in claims
        )
        if generated:
            return (
                "SLM-directed statements below are constrained to the displayed "
                "normalized fact and provenance ids."
            )
        return "The key evidence-linked statements for this section follow."
    if section_id == "cover":
        return ""
    if section_id == "limitations":
        return "This section preserves required epistemic, safety, and completeness caveats."
    if section_id == "technical_appendix":
        return "Machine-readable run, policy, engine, trace, and director metadata."
    return "Structured evidence from normalized source artifacts."


def build_report_sections(
    sources: Sequence[ReportSource],
    plan: ReportPlan,
    policy: ReportPolicy,
) -> tuple[list[ReportSection], list[ReportCitation], list[str]]:
    """Create deterministic sections and audit notes from a validated plan."""
    selected_sources = _included_sources(sources, plan)
    fact_map = {
        fact.id: fact for source in selected_sources for fact in source.facts
    }
    facts_by_category: dict[str, list[ReportFact]] = {}
    tables_by_category: dict[str, list[ReportTable]] = {}
    for source in selected_sources:
        for fact in source.facts:
            facts_by_category.setdefault(fact.category, []).append(fact)
        for table in source.tables:
            tables_by_category.setdefault(table.category, []).append(
                _bounded_table(table, policy)
            )
    for category, facts in facts_by_category.items():
        facts.sort(key=lambda fact: (-fact.priority, fact.id))
    for category, tables in tables_by_category.items():
        tables.sort(key=lambda table: table.id)

    charts_by_category: dict[str, list[ChartSpec]] = {}
    selected_chart_ids: set[str] = set()
    for chart in sorted(
        plan.chart_specs, key=lambda item: (-item.priority, item.id)
    )[: policy.max_charts]:
        if chart.id in selected_chart_ids:
            continue
        selected_chart_ids.add(chart.id)
        charts_by_category.setdefault(chart.category, []).append(chart)

    all_citations: dict[str, ReportCitation] = {}
    for source in selected_sources:
        for citation in source.citations:
            all_citations[citation.id] = citation

    source_caveats = _dedupe(
        [caveat for source in selected_sources for caveat in source.caveats]
    )
    source_warnings = _dedupe(
        [warning for source in selected_sources for warning in source.warnings]
    )
    sections: list[ReportSection] = []
    audit_notes: list[str] = []

    for section_id in plan.section_order:
        heading = REPORT_SECTION_CATALOG[section_id]
        claims = _claims_for(plan, section_id, fact_map)
        facts = list(facts_by_category.get(section_id) or [])
        tables = list(tables_by_category.get(section_id) or [])
        charts = list(charts_by_category.get(section_id) or [])
        caveats = list(SECTION_CAVEATS.get(section_id) or [])
        section_audit: list[str] = []

        if section_id == "cover":
            summary = plan.purpose
            section_audit.append(
                f"Audience: {plan.audience}; template/theme: "
                f"{policy.template}/{policy.theme}."
            )
        elif section_id == "executive_summary":
            referenced_ids = {
                fact_id for claim in claims for fact_id in claim.fact_ids
            }
            facts = [
                fact_map[fact_id]
                for fact_id in sorted(
                    referenced_ids,
                    key=lambda fact_id: (-fact_map[fact_id].priority, fact_id),
                )
            ]
            summary = _section_summary(section_id, claims)
        elif section_id == "scope_provenance":
            tables.insert(
                0,
                _source_inventory(
                    selected_sources, table_id="report:scope-source-inventory"
                ),
            )
            summary = _section_summary(section_id, claims)
        elif section_id == "limitations":
            referenced_ids = {
                fact_id for claim in claims for fact_id in claim.fact_ids
            }
            facts = [
                fact_map[fact_id]
                for fact_id in sorted(
                    referenced_ids,
                    key=lambda fact_id: (-fact_map[fact_id].priority, fact_id),
                )
            ]
            caveats.extend(source_caveats)
            caveats.extend(source_warnings)
            summary = _section_summary(section_id, claims)
        elif section_id == "technical_appendix":
            tables.insert(
                0,
                _source_inventory(
                    selected_sources, table_id="report:appendix-source-inventory"
                ),
            )
            tables.append(_policy_table(policy))
            tables.append(_director_actions_table(plan))
            summary = _section_summary(section_id, claims)
        else:
            summary = _section_summary(section_id, claims)

        if section_id == "iv_evidence":
            has_synthetic = any(
                fact.attributes.get("synthetic_iv") for fact in facts
            )
            has_observed = any(
                not fact.attributes.get("synthetic_iv")
                for fact in facts
                if "instrument" in fact.label.lower()
                or "iv " in fact.label.lower()
            )
            section_audit.append(
                f"Observed-IV evidence present: {has_observed}; synthetic-IV "
                f"audit entries present: {has_synthetic}."
            )
        if section_id == "deep_research":
            section_audit.append(
                f"Eligible SourceRecord citations: {len(all_citations)}."
            )

        provenance = _dedupe(
            [fact.provenance_id for fact in facts]
            + [
                provenance
                for table in tables
                for provenance in table.provenance_ids
            ]
            + [
                provenance
                for chart in charts
                for provenance in chart.provenance_ids
            ]
            + [
                provenance
                for claim in claims
                for provenance in claim.provenance_ids
            ]
        )
        sections.append(
            ReportSection(
                id=section_id,
                heading=heading,
                summary=summary,
                facts=facts,
                tables=tables,
                charts=charts,
                caveats=_dedupe(caveats),
                provenance_references=provenance,
                claims=claims,
                narrative_is_slm=any(
                    claim.generated_by not in ("rule", "template")
                    for claim in claims
                ),
                audit_notes=section_audit,
            )
        )

    rendered_ids = {section.id for section in sections}
    for section_id, heading in REPORT_SECTION_CATALOG.items():
        if section_id in rendered_ids:
            continue
        has_evidence = bool(
            facts_by_category.get(section_id)
            or tables_by_category.get(section_id)
            or charts_by_category.get(section_id)
        )
        reason = (
            "excluded by the validated plan/policy"
            if has_evidence
            else "no normalized source evidence was supplied"
        )
        audit_notes.append(f"Omitted `{heading}`: {reason}.")

    for source in sources:
        if source.id not in {item.id for item in selected_sources}:
            audit_notes.append(
                f"Excluded source artifact `{source.id}` ({source.family}) by plan."
            )
    audit_notes.extend(plan.warnings)
    audit_notes.extend(source_warnings)
    audit_notes.append(
        "No raw rows, raw frames, sample values, secrets, or unsupported citation "
        "records are retained in the report bundle."
    )
    return (
        sections,
        [all_citations[citation_id] for citation_id in sorted(all_citations)],
        _dedupe(audit_notes),
    )


__all__ = ["SECTION_CAVEATS", "build_report_sections"]
