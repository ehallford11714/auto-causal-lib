"""Public orchestration API for AutoCausal report planning and rendering."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters import AdapterRegistry, normalize_report_sources
from .director import SLMReportDirector, validate_report_plan
from .models import (
    ReportArtifact,
    ReportBundle,
    ReportPlan,
    ReportPolicy,
    ReportSafetyError,
    ReportSection,
    ReportSource,
    ReportValidationError,
    utc_now,
)
from .renderers import RenderResult, render_bundle
from .sections import build_report_sections


_FORMAT_SUFFIX = {
    "pdf": ".pdf",
    "markdown": ".md",
    "md": ".md",
    "html": ".html",
    "htm": ".html",
    "json": ".json",
}
_SENSITIVE_STRING_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    re.compile(r"\b(?:sk|pk|ghp|hf|xox[baprs])_[A-Za-z0-9_-]{12,}\b"),
)


def _resolve_policy(value: ReportPolicy | Mapping[str, Any] | None) -> ReportPolicy:
    if value is None:
        return ReportPolicy.production()
    if isinstance(value, ReportPolicy):
        return ReportPolicy.from_dict(value.to_dict())
    if isinstance(value, Mapping):
        return ReportPolicy.from_dict(value)
    raise TypeError("policy must be ReportPolicy, mapping, or None")


def _normalized_format(output: Path, requested: str | None) -> tuple[Path, str]:
    selected = (requested or output.suffix.lstrip(".") or "pdf").lower()
    if selected == "htm":
        selected = "html"
    if selected == "md":
        selected = "markdown"
    if selected not in {"pdf", "markdown", "html", "json"}:
        raise ValueError(
            f"Unsupported report format `{selected}`; use pdf, markdown, html, or json."
        )
    expected = _FORMAT_SUFFIX[selected]
    if not output.suffix:
        output = output.with_suffix(expected)
    return output, selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        from autocausal.__version__ import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _scan_sensitive_strings(value: Any, *, location: str = "bundle") -> list[str]:
    findings: list[str] = []
    if isinstance(value, str):
        if "[REDACTED_" in value or "[OMITTED_RAW_" in value:
            return findings
        for pattern in _SENSITIVE_STRING_PATTERNS:
            if pattern.search(value):
                findings.append(location)
                break
    elif isinstance(value, Mapping):
        for key, item in value.items():
            findings.extend(
                _scan_sensitive_strings(item, location=f"{location}.{key}")
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            findings.extend(
                _scan_sensitive_strings(item, location=f"{location}[{index}]")
            )
    return findings


def validate_report_bundle(bundle: ReportBundle) -> dict[str, Any]:
    """Validate section/source/citation integrity and production privacy."""
    plan_validation = validate_report_plan(
        bundle.plan, bundle.sources, bundle.policy
    )
    source_facts = {
        fact.id: fact for source in bundle.sources for fact in source.facts
    }
    source_tables = {
        table.id for source in bundle.sources for table in source.tables
    }
    citation_ids = {citation.id for citation in bundle.citations}
    section_ids = [section.id for section in bundle.sections]
    if section_ids != bundle.plan.section_order:
        raise ReportValidationError(
            "Rendered section order does not match the validated plan: "
            f"{section_ids} != {bundle.plan.section_order}"
        )
    for section in bundle.sections:
        unknown_facts = {fact.id for fact in section.facts} - set(source_facts)
        if unknown_facts:
            raise ReportValidationError(
                f"Section `{section.id}` contains unknown facts: "
                f"{sorted(unknown_facts)}"
            )
        for claim in section.claims:
            if not claim.fact_ids:
                raise ReportValidationError(
                    f"Section `{section.id}` contains an ungrounded narrative claim"
                )
            if set(claim.fact_ids) - set(source_facts):
                raise ReportValidationError(
                    f"Section `{section.id}` claim references unknown facts"
                )
            if set(claim.citation_ids) - citation_ids:
                raise ReportValidationError(
                    f"Section `{section.id}` claim references unsupported citations"
                )
        for table in section.tables:
            # Report-generated inventory/policy/director tables intentionally do
            # not exist in source adapters and use a reserved id prefix.
            if (
                table.id not in source_tables
                and not table.id.startswith("report:")
            ):
                raise ReportValidationError(
                    f"Section `{section.id}` contains unknown table `{table.id}`"
                )
        if section.id in {
            "associations",
            "discovery",
            "causal_estimates",
            "iv_evidence",
            "automl",
            "nlp_behavioral",
            "insight_actions",
            "limitations",
        } and not section.caveats:
            raise ReportValidationError(
                f"Required caveats are missing from section `{section.id}`"
            )
    if bundle.policy.production_mode:
        findings = _scan_sensitive_strings(bundle.to_dict())
        if findings:
            raise ReportSafetyError(
                "Production report contains unredacted sensitive string patterns "
                f"at: {sorted(set(findings))[:12]}"
            )
    return {
        **plan_validation,
        "bundle_schema": bundle.schema,
        "render_section_count": len(bundle.sections),
        "render_citation_count": len(bundle.citations),
        "privacy_scan": "pass",
        "contains_raw_data": False,
    }


class ReportEngine:
    """Production-oriented SLM-directed report planning and rendering.

    The SLM receives only normalized facts. Invalid SLM actions are discarded
    and the deterministic rule/template plan is used.
    """

    def __init__(
        self,
        *,
        use_slm: bool = True,
        policy: ReportPolicy | Mapping[str, Any] | None = None,
        model_name: str | None = None,
        director: SLMReportDirector | None = None,
        adapter_registry: AdapterRegistry | None = None,
    ) -> None:
        self.use_slm = bool(use_slm)
        self.policy = _resolve_policy(policy)
        self.model_name = model_name
        self.adapter_registry = adapter_registry
        self.director = director or SLMReportDirector(
            use_slm=self.use_slm,
            policy=self.policy,
            model_name=model_name,
        )
        self.last_sources: list[ReportSource] = []
        self.last_plan: ReportPlan | None = None
        self.last_bundle: ReportBundle | None = None
        self.last_artifact: ReportArtifact | None = None
        self.last_warnings: list[str] = []

    def normalize(self, source: Any) -> list[ReportSource]:
        sources, warnings = normalize_report_sources(
            source,
            policy=self.policy,
            registry=self.adapter_registry,
        )
        self.last_sources = sources
        self.last_warnings = warnings
        return sources

    def plan(
        self,
        source: Any,
        *,
        title: str = "AutoCausal Analysis Report",
        audience: str = "technical and decision stakeholders",
        purpose: str = (
            "Summarize normalized AutoCausal evidence, uncertainty, provenance, "
            "failed gates, and recommended follow-up."
        ),
    ) -> ReportPlan:
        sources = self.normalize(source)
        plan = self.director.plan(
            sources,
            title=title,
            audience=audience,
            purpose=purpose,
        )
        validate_report_plan(plan, sources, self.policy)
        self.last_plan = plan
        return plan

    def build_bundle(
        self,
        source: Any,
        *,
        plan: ReportPlan | None = None,
        title: str = "AutoCausal Analysis Report",
        audience: str = "technical and decision stakeholders",
        purpose: str = (
            "Summarize normalized AutoCausal evidence, uncertainty, provenance, "
            "failed gates, and recommended follow-up."
        ),
    ) -> ReportBundle:
        sources = self.normalize(source)
        resolved_plan = plan or self.director.plan(
            sources,
            title=title,
            audience=audience,
            purpose=purpose,
        )
        validate_report_plan(resolved_plan, sources, self.policy)
        sections, citations, audit_notes = build_report_sections(
            sources, resolved_plan, self.policy
        )
        bundle = ReportBundle(
            plan=resolved_plan,
            policy=self.policy,
            sources=sources,
            sections=sections,
            citations=citations,
            audit_notes=list(dict.fromkeys(self.last_warnings + audit_notes)),
            director_record={
                "backend": resolved_plan.director_backend,
                "model_name": self.model_name,
                "slm_requested": self.use_slm,
                "slm_allowed": self.policy.allow_slm,
                "actions": list(resolved_plan.director_actions),
                "fallback_error": self.director.last_error,
            },
        )
        bundle.validation = validate_report_bundle(bundle)
        self.last_plan = resolved_plan
        self.last_bundle = bundle
        return bundle

    def validate(
        self,
        source: Any,
        *,
        plan: ReportPlan | None = None,
    ) -> dict[str, Any]:
        if isinstance(source, ReportBundle):
            validation = validate_report_bundle(source)
            self.last_bundle = source
            return validation
        bundle = self.build_bundle(source, plan=plan)
        return dict(bundle.validation)

    def generate(
        self,
        *,
        source: Any,
        output: str | Path,
        format: str | None = None,
        siblings: Sequence[str] | None = None,
        plan: ReportPlan | None = None,
        title: str = "AutoCausal Analysis Report",
        audience: str = "technical and decision stakeholders",
        purpose: str = (
            "Summarize normalized AutoCausal evidence, uncertainty, provenance, "
            "failed gates, and recommended follow-up."
        ),
    ) -> ReportArtifact:
        bundle = self.build_bundle(
            source,
            plan=plan,
            title=title,
            audience=audience,
            purpose=purpose,
        )
        output_path, selected_format = _normalized_format(Path(output), format)
        result = render_bundle(
            bundle,
            output_path,
            format=selected_format,
        )
        sibling_paths: dict[str, str] = {}
        sibling_warnings: list[str] = []
        requested_siblings = list(
            dict.fromkeys(
                str(item).lower()
                for item in (
                    list(siblings or []) + list(self.policy.sibling_formats)
                )
            )
        )
        for sibling_format in requested_siblings:
            normalized = "markdown" if sibling_format == "md" else sibling_format
            if normalized == "htm":
                normalized = "html"
            if normalized == selected_format:
                continue
            if normalized not in {"markdown", "html", "json", "pdf"}:
                raise ValueError(
                    f"Unsupported sibling format `{sibling_format}`"
                )
            sibling_path = output_path.with_suffix(_FORMAT_SUFFIX[normalized])
            sibling_result = render_bundle(
                bundle,
                sibling_path,
                format=normalized,
            )
            sibling_paths[normalized] = str(sibling_result.path)
            sibling_warnings.extend(sibling_result.warnings)

        package_version = _package_version()
        stat = result.path.stat()
        artifact = ReportArtifact(
            path=result.path,
            format=selected_format,
            size_bytes=stat.st_size,
            sha256=_sha256(result.path),
            generated_at=utc_now(),
            run_ids=bundle.run_ids,
            package_version=package_version,
            provenance={
                "bundle_schema": bundle.schema,
                "plan_schema": bundle.plan.schema,
                "policy_schema": bundle.policy.schema,
                "source_ids": [source.id for source in bundle.sources],
                "source_families": [
                    source.family for source in bundle.sources
                ],
                "fact_count": len(bundle.facts),
                "section_ids": [section.id for section in bundle.sections],
                "citation_ids": [
                    citation.id for citation in bundle.citations
                ],
                "validation": dict(bundle.validation),
                "director_actions": list(bundle.plan.director_actions),
            },
            warnings=list(
                dict.fromkeys(
                    self.last_warnings
                    + bundle.plan.warnings
                    + result.warnings
                    + sibling_warnings
                )
            ),
            siblings=sibling_paths,
            page_count=result.page_count,
            director_backend=bundle.plan.director_backend,
        )
        self.last_artifact = artifact
        return artifact

    def status(self) -> dict[str, Any]:
        return {
            "schema": "AutoCausalReportEngineStatus.v1",
            "use_slm": self.use_slm,
            "policy": self.policy.to_dict(),
            "director_backend": (
                self.last_plan.director_backend
                if self.last_plan is not None
                else self.director.last_backend
            ),
            "has_plan": self.last_plan is not None,
            "has_bundle": self.last_bundle is not None,
            "artifact": (
                self.last_artifact.to_dict()
                if self.last_artifact is not None
                else None
            ),
            "warnings": list(self.last_warnings),
        }


def generate_report(
    source: Any,
    output: str | Path,
    *,
    use_slm: bool = True,
    policy: ReportPolicy | Mapping[str, Any] | None = None,
    format: str | None = None,
    siblings: Sequence[str] | None = None,
    **kwargs: Any,
) -> ReportArtifact:
    """Functional convenience wrapper around :class:`ReportEngine`."""
    return ReportEngine(use_slm=use_slm, policy=policy).generate(
        source=source,
        output=output,
        format=format,
        siblings=siblings,
        **kwargs,
    )


__all__ = [
    "ReportEngine",
    "generate_report",
    "validate_report_bundle",
]
