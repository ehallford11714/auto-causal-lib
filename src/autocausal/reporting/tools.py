"""Optional skilling, MCP/AgentHook, and agentic integration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .engine import ReportEngine
from .models import ReportPolicy, ReportSafetyError


_MCP_ENGINES: dict[str, ReportEngine] = {}


def _policy_from_args(args: Mapping[str, Any]) -> ReportPolicy:
    explicit = args.get("policy")
    if isinstance(explicit, Mapping):
        return ReportPolicy.from_dict(explicit)
    profile = str(args.get("profile") or "production").lower()
    if profile in {"exploratory", "explore"}:
        return ReportPolicy.exploratory()
    return ReportPolicy.production()


def register_reporting_skilling_tools(surface: Any) -> Any:
    """Register ``report.plan/generate/validate`` on an existing ToolSurface."""
    from autocausal.skilling.surface import ToolDef
    from autocausal.suites.action_protocol import ActionResult

    def _plan(source: Any, **kwargs: Any) -> ActionResult:
        engine = ReportEngine(
            use_slm=bool(kwargs.get("use_slm", True)),
            policy=_policy_from_args(kwargs),
        )
        plan = engine.plan(
            source,
            title=str(kwargs.get("title") or "AutoCausal Analysis Report"),
            audience=str(
                kwargs.get("audience") or "technical and decision stakeholders"
            ),
        )
        return ActionResult(
            name="report.plan",
            payload={"plan": plan.to_dict()},
            warnings=list(plan.warnings),
            notes=["Only normalized facts were exposed to the report director."],
        )

    def _generate(source: Any, **kwargs: Any) -> ActionResult:
        output = kwargs.get("output")
        if not output:
            return ActionResult(
                name="report.generate",
                warnings=["output is required"],
            )
        engine = ReportEngine(
            use_slm=bool(kwargs.get("use_slm", True)),
            policy=_policy_from_args(kwargs),
        )
        artifact = engine.generate(
            source=source,
            output=Path(str(output)),
            format=kwargs.get("format"),
            siblings=kwargs.get("siblings"),
        )
        return ActionResult(
            name="report.generate",
            payload={"artifact": artifact.to_dict()},
            warnings=list(artifact.warnings),
            notes=["Report artifact passed provenance/privacy validation."],
        )

    def _validate(source: Any, **kwargs: Any) -> ActionResult:
        engine = ReportEngine(
            use_slm=bool(kwargs.get("use_slm", True)),
            policy=_policy_from_args(kwargs),
        )
        validation = engine.validate(source)
        return ActionResult(
            name="report.validate",
            payload={"validation": validation},
            notes=["Validation is fail-closed under the production report policy."],
        )

    common = {
        "use_slm": {"type": "boolean", "default": True},
        "profile": {
            "type": "string",
            "enum": ["production", "exploratory"],
            "default": "production",
        },
        "policy": {"type": "object"},
    }
    surface.register(
        ToolDef(
            name="report.plan",
            description=(
                "Build a validated report plan from the current normalized report "
                "source. Raw frames are not accepted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    **common,
                    "title": {"type": "string"},
                    "audience": {"type": "string"},
                },
                "additionalProperties": True,
            },
            suite="report",
            action="plan",
            handler=_plan,
        )
    )
    surface.register(
        ToolDef(
            name="report.generate",
            description="Generate a validated PDF/Markdown/HTML/JSON report artifact.",
            parameters={
                "type": "object",
                "properties": {
                    **common,
                    "output": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["pdf", "markdown", "html", "json"],
                    },
                    "siblings": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["output"],
                "additionalProperties": True,
            },
            suite="report",
            action="generate",
            handler=_generate,
        )
    )
    surface.register(
        ToolDef(
            name="report.validate",
            description="Validate report provenance, privacy, citations, and plan.",
            parameters={
                "type": "object",
                "properties": dict(common),
                "additionalProperties": True,
            },
            suite="report",
            action="validate",
            handler=_validate,
        )
    )
    return surface


def report_tool_surface() -> Any:
    """Return a standalone ToolSurface containing reporting skills."""
    from autocausal.skilling.surface import ToolSurface

    return register_reporting_skilling_tools(ToolSurface())


def register_reporting_mcp_tools(registry: Any) -> Any:
    """Register report planning/generation/status on MCP and AgentHook registry."""
    from autocausal.mcp.registry import ToolSpec

    def _props(
        properties: dict[str, Any], required: list[str] | None = None
    ) -> dict[str, Any]:
        return {
            "properties": properties,
            "required": list(required or []),
        }

    def _session_id(args: Mapping[str, Any]) -> str:
        from autocausal.mcp.session import DEFAULT_SESSION

        return str(args.get("session_id") or DEFAULT_SESSION)

    def _engine(args: Mapping[str, Any]) -> ReportEngine:
        return ReportEngine(
            use_slm=bool(args.get("use_slm", True)),
            policy=_policy_from_args(args),
            model_name=(
                str(args.get("model_name")) if args.get("model_name") else None
            ),
        )

    def _plan(args: dict[str, Any], store: Any) -> dict[str, Any]:
        sid = _session_id(args)
        ac = store.get(sid)
        engine = _engine(args)
        plan = engine.plan(
            ac,
            title=str(args.get("title") or "AutoCausal Analysis Report"),
            audience=str(
                args.get("audience") or "technical and decision stakeholders"
            ),
        )
        _MCP_ENGINES[sid] = engine
        return {
            "ok": True,
            "tool": "autocausal_report_plan",
            "session_id": sid,
            "plan": plan.to_dict(),
            "status": engine.status(),
        }

    def _generate(args: dict[str, Any], store: Any) -> dict[str, Any]:
        sid = _session_id(args)
        output = args.get("output") or args.get("path")
        if not output:
            raise ValueError("output is required")
        ac = store.get(sid)
        engine = _engine(args)
        artifact = engine.generate(
            source=ac,
            output=Path(str(output)),
            format=(str(args["format"]) if args.get("format") else None),
            siblings=args.get("siblings"),
            title=str(args.get("title") or "AutoCausal Analysis Report"),
            audience=str(
                args.get("audience") or "technical and decision stakeholders"
            ),
        )
        _MCP_ENGINES[sid] = engine
        return {
            "ok": True,
            "tool": "autocausal_generate_report",
            "session_id": sid,
            "artifact": artifact.to_dict(),
            "status": engine.status(),
        }

    def _status(args: dict[str, Any], store: Any) -> dict[str, Any]:
        sid = _session_id(args)
        engine = _MCP_ENGINES.get(sid)
        return {
            "ok": True,
            "tool": "autocausal_report_status",
            "session_id": sid,
            "has_session": bool(store.has(sid)),
            "status": (
                engine.status()
                if engine is not None
                else {
                    "schema": "AutoCausalReportEngineStatus.v1",
                    "has_plan": False,
                    "has_bundle": False,
                    "artifact": None,
                }
            ),
        }

    common = {
        "session_id": {"type": "string"},
        "use_slm": {"type": "boolean", "default": True},
        "profile": {
            "type": "string",
            "enum": ["production", "exploratory"],
            "default": "production",
        },
        "policy": {"type": "object"},
        "model_name": {"type": "string"},
        "title": {"type": "string"},
        "audience": {"type": "string"},
    }
    registry.register(
        ToolSpec(
            name="autocausal_report_plan",
            description=(
                "Build a constrained, provenance-validated report plan for a session."
            ),
            parameters=_props(dict(common)),
            handler=_plan,
            optional_module="autocausal.reporting",
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_generate_report",
            description=(
                "Generate a headless PDF or Markdown/HTML/JSON report artifact "
                "under report policy."
            ),
            parameters=_props(
                {
                    **common,
                    "output": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["pdf", "markdown", "html", "json"],
                        "default": "pdf",
                    },
                    "siblings": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["markdown", "html", "json"],
                        },
                    },
                },
                required=["output"],
            ),
            handler=_generate,
            optional_module="autocausal.reporting",
        )
    )
    registry.register(
        ToolSpec(
            name="autocausal_report_status",
            description="Return the last report plan/artifact status for a session.",
            parameters=_props({"session_id": {"type": "string"}}),
            handler=_status,
            optional_module="autocausal.reporting",
        )
    )
    return registry


def generate_approved_agentic_report(
    source: Any,
    output: str | Path,
    *,
    approved: bool,
    use_slm: bool = True,
    policy: ReportPolicy | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Policy-approval guard suitable for an agentic/LangGraph final node."""
    if not approved:
        raise ReportSafetyError(
            "Agentic report generation requires explicit policy approval."
        )
    return ReportEngine(use_slm=use_slm, policy=policy).generate(
        source=source,
        output=output,
        **kwargs,
    )


__all__ = [
    "generate_approved_agentic_report",
    "register_reporting_mcp_tools",
    "register_reporting_skilling_tools",
    "report_tool_surface",
]
