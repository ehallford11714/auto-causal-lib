"""Connective hooks — insight loop, skilling ToolSurface, optional MCP package.

Soft-optional: missing ``autocausal.mcp`` / external MCP servers never break imports.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "insight_grail_step",
    "register_grail_skilling_tools",
    "try_register_mcp_tools",
    "grail_tool_schemas",
]


def insight_grail_step(
    *,
    text: str = "",
    context: Optional[dict[str, Any]] = None,
    max_cycles: int = 1,
    chain_length: int = 3,
    prefer_live: bool = True,
) -> dict[str, Any]:
    """Run a GRAIL memory/graph step for the insight / SLM research loop.

    Returns a dict suitable for ``InsightReport.notes`` / round_history.
    Always uses stub when live Kineteq is absent.
    """
    from autocausal.grail.adapter import GrailEngine

    eng = GrailEngine(prefer_live=prefer_live)
    goal = (text or "").strip() or "Explore causal relationships"
    ctx = dict(context or {})
    if text and "text" not in ctx:
        ctx["text"] = text

    # Lightweight path: impute + memory + graph (one cycle unless asked)
    audit = eng.impute(goal, context=ctx)
    mem = eng.memory_step(goal, context=ctx, top_k=8)
    focus = [
        str(a.value)
        for a in audit.assumptions
        if a.parameter in ("treatment", "outcome", "instrument") and a.value
    ]
    boost = eng.graph_retrieve(context=ctx, focus=focus, top_k=8)

    if max_cycles > 1:
        report = eng.run(
            goal, context=ctx, max_cycles=max_cycles, chain_length=chain_length
        )
        return {
            "stage": "grail",
            "backend": report.backend,
            "live_kineteq": report.live_kineteq,
            "report": report.to_dict(),
            "focus_columns": report.focus_columns,
            "boost_edges": report.boost_edges,
            "memory_keys": [m.key for m in report.memory[:8]],
            "notes": list(report.notes),
        }

    return {
        "stage": "grail",
        "backend": audit.backend,
        "live_kineteq": False,
        "imputation": audit.to_dict(),
        "focus_columns": focus,
        "boost_edges": boost,
        "memory_keys": [m.key for m in mem],
        "notes": list(audit.notes)
        + ["insight_grail_step: memory/graph only (max_cycles=1)."],
    }


def grail_tool_schemas() -> list[dict[str, Any]]:
    """JSON-schema-like MCP / skilling tool descriptors (``autocausal_grail_*``)."""
    from autocausal.grail.mcp_tools import MCP_TOOL_SCHEMAS

    return list(MCP_TOOL_SCHEMAS)


def register_grail_skilling_tools(surface: Any = None) -> Any:
    """Register GRAIL tools onto a skilling ``ToolSurface`` (create if None)."""
    from autocausal.grail.mcp_tools import register_on_surface

    return register_on_surface(surface)


def try_register_mcp_tools(server: Any = None) -> dict[str, Any]:
    """Attempt to register ``autocausal_grail_*`` on an MCP server object.

    Looks for ``autocausal.mcp`` connective package if ``server`` is None.
    Returns status dict; never raises for missing MCP.
    """
    from autocausal.grail.mcp_tools import try_bind_mcp

    return try_bind_mcp(server)
