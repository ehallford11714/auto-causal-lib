"""SLM-directed, provenance-safe AutoCausal report generation.

Primary API::

    from autocausal.reporting import ReportEngine, ReportPolicy

    artifact = ReportEngine(
        use_slm=True,
        policy=ReportPolicy.production(),
    ).generate(source=result, output="autocausal-report.pdf")
"""

from __future__ import annotations

from .adapters import (
    AdapterRegistry,
    ReportSourceAdapter,
    default_adapter_registry,
    normalize_report_sources,
)
from .director import (
    SLMReportDirector,
    deterministic_report_plan,
    validate_report_plan,
    validate_slm_proposal,
)
from .engine import ReportEngine, generate_report, validate_report_bundle
from .models import (
    ChartSpec,
    DEFAULT_SECTION_ORDER,
    REPORT_SECTION_CATALOG,
    REPORT_THEMES,
    ReportArtifact,
    ReportBundle,
    ReportCitation,
    ReportClaim,
    ReportError,
    ReportFact,
    ReportPlan,
    ReportPolicy,
    ReportRenderError,
    ReportSafetyError,
    ReportSection,
    ReportSource,
    ReportTable,
    ReportValidationError,
)
from .renderers import (
    RenderResult,
    render_bundle,
    render_html,
    render_json,
    render_markdown,
    render_pdf,
)
from .tools import (
    generate_approved_agentic_report,
    register_reporting_mcp_tools,
    register_reporting_skilling_tools,
    report_tool_surface,
)

__all__ = [
    "AdapterRegistry",
    "ChartSpec",
    "DEFAULT_SECTION_ORDER",
    "REPORT_SECTION_CATALOG",
    "REPORT_THEMES",
    "RenderResult",
    "ReportArtifact",
    "ReportBundle",
    "ReportCitation",
    "ReportClaim",
    "ReportEngine",
    "ReportError",
    "ReportFact",
    "ReportPlan",
    "ReportPolicy",
    "ReportRenderError",
    "ReportSafetyError",
    "ReportSection",
    "ReportSource",
    "ReportSourceAdapter",
    "ReportTable",
    "ReportValidationError",
    "SLMReportDirector",
    "default_adapter_registry",
    "deterministic_report_plan",
    "generate_report",
    "generate_approved_agentic_report",
    "normalize_report_sources",
    "register_reporting_mcp_tools",
    "register_reporting_skilling_tools",
    "render_bundle",
    "render_html",
    "render_json",
    "render_markdown",
    "render_pdf",
    "report_tool_surface",
    "validate_report_bundle",
    "validate_report_plan",
    "validate_slm_proposal",
]
