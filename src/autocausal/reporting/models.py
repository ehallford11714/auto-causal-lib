"""Typed, serialization-safe contracts for the AutoCausal report engine.

The reporting package deliberately keeps its contracts independent from the
rest of AutoCausal.  Adapters may therefore normalize optional or concurrently
developed modules without importing them at package import time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_SECTION_CATALOG: dict[str, str] = {
    "cover": "Cover",
    "executive_summary": "Executive summary",
    "scope_provenance": "Scope, data provenance, and run policy",
    "data_quality": "Data quality, cleansing ledger, and QC gates",
    "eda_readiness": "Exploratory analysis and causal readiness",
    "associations": "Correlations and associations (non-causal)",
    "discovery": "Discovery graph, agreement, stability, and FDR",
    "causal_estimates": "Causal specification, estimates, and diagnostics",
    "iv_evidence": "Instrumental-variable evidence",
    "refutations_sensitivity": "Refutations, sensitivity, and escalated gates",
    "automl": "AutoML results (predictive, not causal)",
    "nlp_behavioral": "NLP and behavioral findings",
    "insight_actions": "Insight, agentic actions, and experiments",
    "deep_research": "Literature evidence and contradictions",
    "visualizations": "Visualizations and chart specifications",
    "limitations": "Limitations, unresolved questions, and recommendations",
    "technical_appendix": "Technical appendix",
}

DEFAULT_SECTION_ORDER: tuple[str, ...] = tuple(REPORT_SECTION_CATALOG)
REPORT_THEMES: tuple[str, ...] = (
    "high_contrast",
    "professional",
    "monochrome",
)


class ReportError(RuntimeError):
    """Base error for report generation."""


class ReportValidationError(ReportError):
    """A report plan or bundle violated an integrity constraint."""


class ReportSafetyError(ReportValidationError):
    """A report would expose raw or sensitive data."""


class ReportRenderError(ReportError):
    """A requested output could not be rendered safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Convert a value to deterministic JSON-friendly structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return str(value)


def _markdown_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False)
        return text if len(text) <= 240 else text[:237] + "..."
    return str(value)


class ReportModel:
    """Common report ergonomics used by all public contracts."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

    def to_markdown(self) -> str:
        return "```json\n" + self.to_json() + "\n```\n"

    def report(self, *, as_markdown: bool = True) -> str:
        return self.to_markdown() if as_markdown else self.to_json()

    def write(self, path: str | Path, *, fmt: str | None = None) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        selected = (fmt or output.suffix.lstrip(".") or "json").lower()
        if selected in ("md", "markdown"):
            output.write_text(self.to_markdown(), encoding="utf-8")
        elif selected == "json":
            output.write_text(self.to_json(), encoding="utf-8")
        else:
            raise ValueError(f"{type(self).__name__}.write supports json or markdown")
        return output


@dataclass
class ReportCitation(ReportModel):
    """A normalized citation backed by a supplied/fetched source-record id."""

    id: str
    title: str = ""
    url: str = ""
    authors: list[str] = field(default_factory=list)
    year: str = ""
    verified: bool = False
    verification_status: str = ""
    supplied_by: str = "source_record"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        if not self.id:
            raise ValueError("ReportCitation.id cannot be empty")
        self.authors = [str(author) for author in self.authors]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "authors": list(self.authors),
            "year": self.year,
            "verified": self.verified,
            "verification_status": self.verification_status,
            "supplied_by": self.supplied_by,
            "metadata": _jsonable(self.metadata),
        }

    def to_markdown(self) -> str:
        authors = ", ".join(self.authors)
        detail = ". ".join(part for part in (authors, self.title, self.year) if part)
        target = f" [{self.url}]({self.url})" if self.url else ""
        verified = "verified" if self.verified else "unverified"
        return f"- [{self.id}] {detail or self.id}{target} ({verified})\n"


@dataclass
class ReportFact(ReportModel):
    """One normalized fact; every rendered key claim points to one or more facts."""

    id: str
    source_id: str
    provenance_id: str
    label: str
    value: Any
    category: str
    unit: str = ""
    priority: int = 50
    citation_ids: list[str] = field(default_factory=list)
    caveat: str = ""
    sensitive: bool = False
    evidence_eligible: bool = True
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("id", "source_id", "provenance_id", "label", "category"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"ReportFact.{name} cannot be empty")
        self.priority = max(0, min(100, int(self.priority)))
        self.citation_ids = list(dict.fromkeys(str(cid) for cid in self.citation_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "provenance_id": self.provenance_id,
            "label": self.label,
            "value": _jsonable(self.value),
            "category": self.category,
            "unit": self.unit,
            "priority": self.priority,
            "citation_ids": list(self.citation_ids),
            "caveat": self.caveat,
            "sensitive": self.sensitive,
            "evidence_eligible": self.evidence_eligible,
            "attributes": _jsonable(self.attributes),
        }

    def to_markdown(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        refs = f" [{self.provenance_id}]"
        citations = (
            " " + " ".join(f"[{citation}]" for citation in self.citation_ids)
            if self.citation_ids
            else ""
        )
        eligible = "" if self.evidence_eligible else " **(audit only / excluded)**"
        return (
            f"- **{self.label}:** {_markdown_scalar(self.value)}{unit}{eligible}"
            f"{refs}{citations}\n"
        )


@dataclass
class ReportTable(ReportModel):
    """A bounded structured table derived from normalized evidence."""

    id: str
    title: str
    columns: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    provenance_ids: list[str] = field(default_factory=list)
    footnote: str = ""
    sensitive_columns: list[str] = field(default_factory=list)
    category: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.title:
            raise ValueError("ReportTable id and title are required")
        self.columns = [str(column) for column in self.columns]
        self.provenance_ids = list(
            dict.fromkeys(str(item) for item in self.provenance_ids if item)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "columns": list(self.columns),
            "rows": [_jsonable(row) for row in self.rows],
            "provenance_ids": list(self.provenance_ids),
            "footnote": self.footnote,
            "sensitive_columns": list(self.sensitive_columns),
            "category": self.category,
        }

    def to_markdown(self) -> str:
        lines = [f"### {self.title}", ""]
        if not self.rows:
            lines.extend(["_No rows._", ""])
            return "\n".join(lines)
        lines.append("| " + " | ".join(self.columns) + " |")
        lines.append("|" + "|".join("---" for _ in self.columns) + "|")
        for row in self.rows:
            values = [
                _markdown_scalar(row.get(column)).replace("|", "\\|")
                for column in self.columns
            ]
            lines.append("| " + " | ".join(values) + " |")
        if self.footnote:
            lines.extend(["", f"_{self.footnote}_"])
        if self.provenance_ids:
            lines.extend(
                ["", "Provenance: " + ", ".join(f"`{p}`" for p in self.provenance_ids)]
            )
        lines.append("")
        return "\n".join(lines)


@dataclass
class ChartSpec(ReportModel):
    """A chart image or a safe specification over existing structured evidence."""

    id: str
    chart_type: str
    title: str
    alt_text: str
    source_fact_ids: list[str] = field(default_factory=list)
    source_table_id: str = ""
    image_path: str = ""
    spec: dict[str, Any] = field(default_factory=dict)
    provenance_ids: list[str] = field(default_factory=list)
    priority: int = 50
    caption: str = ""
    category: str = "visualizations"
    runtime_artifact: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not all((self.id, self.chart_type, self.title, self.alt_text)):
            raise ValueError("ChartSpec id, chart_type, title, and alt_text are required")
        self.source_fact_ids = list(
            dict.fromkeys(str(item) for item in self.source_fact_ids if item)
        )
        self.provenance_ids = list(
            dict.fromkeys(str(item) for item in self.provenance_ids if item)
        )
        self.priority = max(0, min(100, int(self.priority)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "chart_type": self.chart_type,
            "title": self.title,
            "alt_text": self.alt_text,
            "source_fact_ids": list(self.source_fact_ids),
            "source_table_id": self.source_table_id,
            "image_path": self.image_path,
            "spec": _jsonable(self.spec),
            "provenance_ids": list(self.provenance_ids),
            "priority": self.priority,
            "caption": self.caption,
            "category": self.category,
        }

    def to_markdown(self) -> str:
        lines = [f"### {self.title}", ""]
        if self.image_path:
            lines.extend([f"![{self.alt_text}]({self.image_path})", ""])
        else:
            lines.extend(
                [
                    f"_Chart unavailable; specification retained ({self.chart_type})._",
                    "",
                    "```json",
                    json.dumps(_jsonable(self.spec), indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        if self.caption:
            lines.append(self.caption)
        lines.append(f"Alt text: {self.alt_text}")
        return "\n".join(lines) + "\n"


@dataclass
class ReportClaim(ReportModel):
    """Narrative text whose evidence mapping is explicit and machine-checkable."""

    text: str
    fact_ids: list[str]
    provenance_ids: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    generated_by: str = "rule"
    label: str = ""

    def __post_init__(self) -> None:
        self.text = str(self.text).strip()
        self.fact_ids = list(dict.fromkeys(str(item) for item in self.fact_ids if item))
        self.provenance_ids = list(
            dict.fromkeys(str(item) for item in self.provenance_ids if item)
        )
        self.citation_ids = list(
            dict.fromkeys(str(item) for item in self.citation_ids if item)
        )
        if not self.text:
            raise ValueError("ReportClaim.text cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "fact_ids": list(self.fact_ids),
            "provenance_ids": list(self.provenance_ids),
            "citation_ids": list(self.citation_ids),
            "generated_by": self.generated_by,
            "label": self.label,
        }

    def to_markdown(self) -> str:
        refs = ", ".join(self.fact_ids)
        label = f"**{self.label}:** " if self.label else ""
        return f"- {label}{self.text} `[{refs}]`\n"


@dataclass
class ReportPolicy(ReportModel):
    """Safety, content, layout, citation, and SLM policy for a report run."""

    schema: str = "AutoCausalReportPolicy.v1"
    profile: str = "production"
    production_mode: bool = True
    allowed_sections: tuple[str, ...] = DEFAULT_SECTION_ORDER
    required_sections: tuple[str, ...] = (
        "cover",
        "executive_summary",
        "scope_provenance",
        "limitations",
        "technical_appendix",
    )
    max_pages: int = 80
    max_rows_per_table: int = 40
    max_charts: int = 16
    raw_data_prohibited: bool = True
    redact_pii: bool = True
    redact_secrets: bool = True
    citation_integrity: bool = True
    require_verified_citations: bool = True
    allow_slm: bool = True
    slm_permissions: tuple[str, ...] = (
        "select_sections",
        "prioritize_sections",
        "summarize_facts",
        "propose_transitions",
        "recommend_charts",
        "surface_conflicts",
        "recommend_followups",
    )
    send_fact_values_to_slm: bool = True
    template: str = "comprehensive"
    theme: str = "high_contrast"
    page_size: str = "letter"
    include_appendix: bool = True
    include_audit_notes: bool = True
    fail_closed: bool = True
    max_narrative_chars: int = 1200
    sibling_formats: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.allowed_sections = tuple(str(item) for item in self.allowed_sections)
        self.required_sections = tuple(str(item) for item in self.required_sections)
        self.slm_permissions = tuple(str(item) for item in self.slm_permissions)
        self.sibling_formats = tuple(str(item).lower() for item in self.sibling_formats)
        unknown = set(self.allowed_sections) - set(REPORT_SECTION_CATALOG)
        if unknown:
            raise ValueError(f"unknown allowed report sections: {sorted(unknown)}")
        unavailable_required = set(self.required_sections) - set(self.allowed_sections)
        if unavailable_required:
            raise ValueError(
                "required sections must also be allowed: "
                f"{sorted(unavailable_required)}"
            )
        self.max_pages = max(1, int(self.max_pages))
        self.max_rows_per_table = max(1, int(self.max_rows_per_table))
        self.max_charts = max(0, int(self.max_charts))
        self.max_narrative_chars = max(100, int(self.max_narrative_chars))
        self.page_size = self.page_size.lower()
        if self.page_size not in ("letter", "a4"):
            raise ValueError("ReportPolicy.page_size must be 'letter' or 'a4'")
        self.theme = str(self.theme).lower().replace("-", "_")
        if self.theme not in REPORT_THEMES:
            raise ValueError(
                "ReportPolicy.theme must be one of "
                + ", ".join(repr(item) for item in REPORT_THEMES)
            )
        if self.production_mode:
            self.raw_data_prohibited = True
            self.redact_pii = True
            self.redact_secrets = True
            self.citation_integrity = True
            self.fail_closed = True

    @classmethod
    def production(cls, **overrides: Any) -> "ReportPolicy":
        values: dict[str, Any] = {
            "profile": "production",
            "production_mode": True,
            "raw_data_prohibited": True,
            "redact_pii": True,
            "redact_secrets": True,
            "citation_integrity": True,
            "require_verified_citations": True,
            "fail_closed": True,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def exploratory(cls, **overrides: Any) -> "ReportPolicy":
        values: dict[str, Any] = {
            "profile": "exploratory",
            "production_mode": False,
            "max_pages": 120,
            "max_rows_per_table": 100,
            "max_charts": 30,
            "raw_data_prohibited": True,
            "redact_pii": True,
            "citation_integrity": True,
            "require_verified_citations": False,
            "fail_closed": False,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReportPolicy":
        payload = dict(value)
        payload.pop("schema", None)
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile": self.profile,
            "production_mode": self.production_mode,
            "allowed_sections": list(self.allowed_sections),
            "required_sections": list(self.required_sections),
            "max_pages": self.max_pages,
            "max_rows_per_table": self.max_rows_per_table,
            "max_charts": self.max_charts,
            "raw_data_prohibited": self.raw_data_prohibited,
            "redact_pii": self.redact_pii,
            "redact_secrets": self.redact_secrets,
            "citation_integrity": self.citation_integrity,
            "require_verified_citations": self.require_verified_citations,
            "allow_slm": self.allow_slm,
            "slm_permissions": list(self.slm_permissions),
            "send_fact_values_to_slm": self.send_fact_values_to_slm,
            "template": self.template,
            "theme": self.theme,
            "page_size": self.page_size,
            "include_appendix": self.include_appendix,
            "include_audit_notes": self.include_audit_notes,
            "fail_closed": self.fail_closed,
            "max_narrative_chars": self.max_narrative_chars,
            "sibling_formats": list(self.sibling_formats),
            "metadata": _jsonable(self.metadata),
        }

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "# Report policy",
                "",
                f"- Profile: `{self.profile}`",
                f"- Production mode: {self.production_mode}",
                f"- Template/theme: `{self.template}` / `{self.theme}`",
                f"- Page size/max pages: `{self.page_size}` / {self.max_pages}",
                f"- Table rows/charts: {self.max_rows_per_table} / {self.max_charts}",
                f"- Raw data prohibited: {self.raw_data_prohibited}",
                f"- PII redaction: {self.redact_pii}",
                f"- Citation integrity: {self.citation_integrity}",
                f"- SLM allowed: {self.allow_slm}",
                "",
            ]
        )


@dataclass
class ReportPlan(ReportModel):
    """Validated content plan produced by rules or the constrained SLM director."""

    title: str
    audience: str
    purpose: str
    section_order: list[str]
    included_artifacts: list[str] = field(default_factory=list)
    excluded_artifacts: list[str] = field(default_factory=list)
    chart_specs: list[ChartSpec] = field(default_factory=list)
    appendix_policy: dict[str, Any] = field(default_factory=dict)
    citation_policy: dict[str, Any] = field(default_factory=dict)
    redaction_policy: dict[str, Any] = field(default_factory=dict)
    section_summaries: dict[str, list[ReportClaim]] = field(default_factory=dict)
    transitions: dict[str, ReportClaim] = field(default_factory=dict)
    director_backend: str = "rule"
    director_actions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema: str = "AutoCausalReportPlan.v1"

    def __post_init__(self) -> None:
        self.title = str(self.title).strip()
        self.audience = str(self.audience).strip()
        self.purpose = str(self.purpose).strip()
        if not all((self.title, self.audience, self.purpose)):
            raise ValueError("ReportPlan title, audience, and purpose are required")
        self.section_order = list(dict.fromkeys(str(item) for item in self.section_order))
        self.included_artifacts = list(
            dict.fromkeys(str(item) for item in self.included_artifacts)
        )
        self.excluded_artifacts = list(
            dict.fromkeys(str(item) for item in self.excluded_artifacts)
        )
        self.chart_specs = [
            item if isinstance(item, ChartSpec) else ChartSpec(**dict(item))
            for item in self.chart_specs
        ]
        converted: dict[str, list[ReportClaim]] = {}
        for section_id, claims in self.section_summaries.items():
            converted[str(section_id)] = [
                claim
                if isinstance(claim, ReportClaim)
                else ReportClaim(**dict(claim))
                for claim in claims
            ]
        self.section_summaries = converted
        self.transitions = {
            str(section_id): (
                claim
                if isinstance(claim, ReportClaim)
                else ReportClaim(**dict(claim))
            )
            for section_id, claim in self.transitions.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "title": self.title,
            "audience": self.audience,
            "purpose": self.purpose,
            "section_order": list(self.section_order),
            "included_artifacts": list(self.included_artifacts),
            "excluded_artifacts": list(self.excluded_artifacts),
            "chart_specs": [chart.to_dict() for chart in self.chart_specs],
            "appendix_policy": _jsonable(self.appendix_policy),
            "citation_policy": _jsonable(self.citation_policy),
            "redaction_policy": _jsonable(self.redaction_policy),
            "section_summaries": {
                section_id: [claim.to_dict() for claim in claims]
                for section_id, claims in self.section_summaries.items()
            },
            "transitions": {
                section_id: claim.to_dict()
                for section_id, claim in self.transitions.items()
            },
            "director_backend": self.director_backend,
            "director_actions": _jsonable(self.director_actions),
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"- Audience: {self.audience}",
            f"- Purpose: {self.purpose}",
            f"- Director: `{self.director_backend}`",
            "",
            "## Section plan",
            "",
        ]
        for index, section_id in enumerate(self.section_order, 1):
            lines.append(
                f"{index}. {REPORT_SECTION_CATALOG.get(section_id, section_id)} "
                f"(`{section_id}`)"
            )
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        lines.append("")
        return "\n".join(lines)


@dataclass
class ReportSection(ReportModel):
    """One fully normalized, render-ready section."""

    id: str
    heading: str
    summary: str = ""
    facts: list[ReportFact] = field(default_factory=list)
    tables: list[ReportTable] = field(default_factory=list)
    charts: list[ChartSpec] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    provenance_references: list[str] = field(default_factory=list)
    claims: list[ReportClaim] = field(default_factory=list)
    narrative_is_slm: bool = False
    audit_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id or not self.heading:
            raise ValueError("ReportSection id and heading are required")
        self.facts = [
            item if isinstance(item, ReportFact) else ReportFact(**dict(item))
            for item in self.facts
        ]
        self.tables = [
            item if isinstance(item, ReportTable) else ReportTable(**dict(item))
            for item in self.tables
        ]
        self.charts = [
            item if isinstance(item, ChartSpec) else ChartSpec(**dict(item))
            for item in self.charts
        ]
        self.claims = [
            item if isinstance(item, ReportClaim) else ReportClaim(**dict(item))
            for item in self.claims
        ]
        self.provenance_references = list(
            dict.fromkeys(str(item) for item in self.provenance_references if item)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "heading": self.heading,
            "summary": self.summary,
            "facts": [fact.to_dict() for fact in self.facts],
            "tables": [table.to_dict() for table in self.tables],
            "charts": [chart.to_dict() for chart in self.charts],
            "caveats": list(self.caveats),
            "provenance_references": list(self.provenance_references),
            "claims": [claim.to_dict() for claim in self.claims],
            "narrative_is_slm": self.narrative_is_slm,
            "audit_notes": list(self.audit_notes),
        }

    def to_markdown(self) -> str:
        lines = [f"## {self.heading}", ""]
        if self.summary:
            label = (
                "_SLM-directed narrative over cited normalized facts:_ "
                if self.narrative_is_slm
                else ""
            )
            lines.extend([label + self.summary, ""])
        lines.extend(claim.to_markdown().rstrip() for claim in self.claims)
        if self.claims:
            lines.append("")
        lines.extend(fact.to_markdown().rstrip() for fact in self.facts)
        if self.facts:
            lines.append("")
        for table in self.tables:
            lines.append(table.to_markdown().rstrip())
            lines.append("")
        for chart in self.charts:
            lines.append(chart.to_markdown().rstrip())
            lines.append("")
        if self.caveats:
            lines.extend(["### Caveats", ""])
            lines.extend(f"- {caveat}" for caveat in self.caveats)
            lines.append("")
        if self.audit_notes:
            lines.extend(["### Audit notes", ""])
            lines.extend(f"- {note}" for note in self.audit_notes)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class ReportSource(ReportModel):
    """Normalized evidence from one source artifact."""

    id: str
    family: str
    title: str
    facts: list[ReportFact] = field(default_factory=list)
    tables: list[ReportTable] = field(default_factory=list)
    charts: list[ChartSpec] = field(default_factory=list)
    citations: list[ReportCitation] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    contains_raw_data: bool = False

    def __post_init__(self) -> None:
        if not all((self.id, self.family, self.title)):
            raise ValueError("ReportSource id, family, and title are required")
        self.facts = [
            item if isinstance(item, ReportFact) else ReportFact(**dict(item))
            for item in self.facts
        ]
        self.tables = [
            item if isinstance(item, ReportTable) else ReportTable(**dict(item))
            for item in self.tables
        ]
        self.charts = [
            item if isinstance(item, ChartSpec) else ChartSpec(**dict(item))
            for item in self.charts
        ]
        self.citations = [
            item
            if isinstance(item, ReportCitation)
            else ReportCitation(**dict(item))
            for item in self.citations
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "title": self.title,
            "facts": [fact.to_dict() for fact in self.facts],
            "tables": [table.to_dict() for table in self.tables],
            "charts": [chart.to_dict() for chart in self.charts],
            "citations": [citation.to_dict() for citation in self.citations],
            "caveats": list(self.caveats),
            "metadata": _jsonable(self.metadata),
            "warnings": list(self.warnings),
            "contains_raw_data": self.contains_raw_data,
        }

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", f"Family: `{self.family}`", ""]
        lines.extend(fact.to_markdown().rstrip() for fact in self.facts)
        for table in self.tables:
            lines.extend(["", table.to_markdown().rstrip()])
        if self.caveats:
            lines.extend(["", "## Caveats", ""])
            lines.extend(f"- {item}" for item in self.caveats)
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class ReportBundle(ReportModel):
    """All normalized evidence, plan, sections, and integrity audit data."""

    plan: ReportPlan
    policy: ReportPolicy
    sources: list[ReportSource]
    sections: list[ReportSection]
    citations: list[ReportCitation] = field(default_factory=list)
    audit_notes: list[str] = field(default_factory=list)
    director_record: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=utc_now)
    schema: str = "AutoCausalReportBundle.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ReportPlan):
            self.plan = ReportPlan(**dict(self.plan))  # type: ignore[arg-type]
        if not isinstance(self.policy, ReportPolicy):
            self.policy = ReportPolicy.from_dict(self.policy)  # type: ignore[arg-type]
        self.sources = [
            item if isinstance(item, ReportSource) else ReportSource(**dict(item))
            for item in self.sources
        ]
        self.sections = [
            item if isinstance(item, ReportSection) else ReportSection(**dict(item))
            for item in self.sections
        ]
        self.citations = [
            item
            if isinstance(item, ReportCitation)
            else ReportCitation(**dict(item))
            for item in self.citations
        ]

    @property
    def facts(self) -> list[ReportFact]:
        return [fact for source in self.sources for fact in source.facts]

    @property
    def run_ids(self) -> list[str]:
        values: list[str] = []
        for source in self.sources:
            run_id = source.metadata.get("run_id")
            if run_id:
                values.append(str(run_id))
        return list(dict.fromkeys(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "generated_at": self.generated_at,
            "plan": self.plan.to_dict(),
            "policy": self.policy.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
            "sections": [section.to_dict() for section in self.sections],
            "citations": [citation.to_dict() for citation in self.citations],
            "audit_notes": list(self.audit_notes),
            "director_record": _jsonable(self.director_record),
            "validation": _jsonable(self.validation),
            "contains_raw_data": False,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.plan.title}",
            "",
            f"**Audience:** {self.plan.audience}  ",
            f"**Purpose:** {self.plan.purpose}  ",
            f"**Generated:** {self.generated_at}  ",
            f"**Director:** `{self.plan.director_backend}`",
            "",
        ]
        for section in self.sections:
            lines.append(section.to_markdown().rstrip())
            lines.append("")
        if self.citations:
            lines.extend(["## References", ""])
            lines.extend(citation.to_markdown().rstrip() for citation in self.citations)
            lines.append("")
        if self.policy.include_audit_notes and self.audit_notes:
            lines.extend(["## Report audit", ""])
            lines.extend(f"- {note}" for note in self.audit_notes)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class ReportArtifact(ReportModel):
    """A rendered report plus integrity and provenance metadata."""

    path: Path | str
    format: str
    size_bytes: int
    sha256: str
    generated_at: str
    run_ids: list[str] = field(default_factory=list)
    package_version: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    siblings: dict[str, str] = field(default_factory=dict)
    page_count: int | None = None
    director_backend: str = "rule"
    schema: str = "AutoCausalReportArtifact.v1"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.format = self.format.lower()
        self.size_bytes = int(self.size_bytes)
        self.run_ids = list(dict.fromkeys(str(item) for item in self.run_ids if item))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "path": str(self.path),
            "format": self.format,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "generated_at": self.generated_at,
            "run_ids": list(self.run_ids),
            "package_version": self.package_version,
            "provenance": _jsonable(self.provenance),
            "warnings": list(self.warnings),
            "siblings": dict(self.siblings),
            "page_count": self.page_count,
            "director_backend": self.director_backend,
        }

    def to_markdown(self) -> str:
        siblings = (
            ", ".join(f"`{fmt}`: `{path}`" for fmt, path in self.siblings.items())
            or "none"
        )
        return "\n".join(
            [
                "# Report artifact",
                "",
                f"- Path: `{self.path}`",
                f"- Format: `{self.format}`",
                f"- Size: {self.size_bytes} bytes",
                f"- SHA-256: `{self.sha256}`",
                f"- Generated: {self.generated_at}",
                f"- Pages: {self.page_count if self.page_count is not None else 'n/a'}",
                f"- Director: `{self.director_backend}`",
                f"- Siblings: {siblings}",
                "",
            ]
        )

    def write(self, path: str | Path, *, fmt: str | None = None) -> Path:
        """Write artifact metadata, never overwrite the rendered artifact implicitly."""
        return super().write(path, fmt=fmt)


def ensure_unique_ids(items: Sequence[Any], *, kind: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        item_id = str(getattr(item, "id", ""))
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    if duplicates:
        raise ReportValidationError(
            f"duplicate {kind} ids: {sorted(duplicates)}"
        )


__all__ = [
    "ChartSpec",
    "DEFAULT_SECTION_ORDER",
    "REPORT_SECTION_CATALOG",
    "REPORT_THEMES",
    "ReportArtifact",
    "ReportBundle",
    "ReportCitation",
    "ReportClaim",
    "ReportError",
    "ReportFact",
    "ReportModel",
    "ReportPlan",
    "ReportPolicy",
    "ReportRenderError",
    "ReportSafetyError",
    "ReportSection",
    "ReportSource",
    "ReportTable",
    "ReportValidationError",
    "ensure_unique_ids",
    "utc_now",
]
