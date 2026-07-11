"""MCP / skilling tool definitions for GRAIL (``autocausal_grail_*``).

When ``autocausal.mcp`` (or a caller-supplied MCP server) is present, tools
are registered. Otherwise schemas remain available for connective hooks and
the skilling ``ToolSurface``.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "MCP_TOOL_SCHEMAS",
    "TOOL_NAMES",
    "dispatch_grail_tool",
    "register_on_surface",
    "try_bind_mcp",
]

TOOL_NAMES = (
    "autocausal_grail_status",
    "autocausal_grail_impute",
    "autocausal_grail_compose",
    "autocausal_grail_run",
    "autocausal_grail_memory",
    "autocausal_grail_graph",
)

MCP_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "autocausal_grail_status",
        "description": (
            "Report GRAIL backend status (stub vs live Kineteq module/MCP). "
            "Offline stub is always available; live Kineteq is soft-optional."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "autocausal_grail_impute",
        "description": (
            "GRAIL self-imputation audit: detect underspecified goal parameters, "
            "declare ASSUMPTIONS (Kineteq grail_impute analog; stub offline)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "domain": {"type": "string", "default": "causal"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "text": {"type": "string"},
            },
            "required": ["goal"],
            "additionalProperties": True,
        },
    },
    {
        "name": "autocausal_grail_compose",
        "description": (
            "Compose a dense expert reasoning chain (Kineteq grail_compose analog). "
            "Stub returns prompts only — not a live LM conductor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "chain_length": {"type": "integer", "default": 3},
                "domain": {"type": "string", "default": "causal"},
                "columns": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["goal"],
            "additionalProperties": True,
        },
    },
    {
        "name": "autocausal_grail_run",
        "description": (
            "Run GRAIL reflective loop (impute→compose→fold→cycles). "
            "Live Kineteq when configured; else rich offline stub."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "max_cycles": {"type": "integer", "default": 2},
                "chain_length": {"type": "integer", "default": 3},
                "domain": {"type": "string", "default": "causal"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "text": {"type": "string"},
                "edges": {"type": "array"},
            },
            "required": ["goal"],
            "additionalProperties": True,
        },
    },
    {
        "name": "autocausal_grail_memory",
        "description": "GRAIL graph/episodic memory retrieve step for AutoCausal loops.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "edges": {"type": "array"},
                "top_k": {"type": "integer", "default": 8},
            },
            "required": ["query"],
            "additionalProperties": True,
        },
    },
    {
        "name": "autocausal_grail_graph",
        "description": "GRAIL graph retrieve → boost-edge candidates for discovery focus.",
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {"type": "array", "items": {"type": "string"}},
                "columns": {"type": "array", "items": {"type": "string"}},
                "edges": {"type": "array"},
                "text": {"type": "string"},
                "top_k": {"type": "integer", "default": 10},
            },
            "additionalProperties": True,
        },
    },
]


def _context_from_args(args: dict[str, Any]) -> dict[str, Any]:
    cols = args.get("columns") or []
    ctx: dict[str, Any] = {
        "columns": [{"name": c} for c in cols] if cols and isinstance(cols[0], str) else cols,
        "edges": list(args.get("edges") or []),
        "text": str(args.get("text") or args.get("goal") or args.get("query") or ""),
    }
    return ctx


def dispatch_grail_tool(name: str, arguments: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Execute one ``autocausal_grail_*`` tool; always offline-safe."""
    from autocausal.grail.adapter import GrailEngine, grail_backend_status

    args = dict(arguments or {})
    eng = GrailEngine(prefer_live=True)

    if name == "autocausal_grail_status":
        return {"ok": True, "status": grail_backend_status()}

    if name == "autocausal_grail_impute":
        goal = str(args.get("goal") or "")
        audit = eng.impute(goal, context=_context_from_args(args), domain=args.get("domain"))
        return {"ok": True, "result": audit.to_dict()}

    if name == "autocausal_grail_compose":
        goal = str(args.get("goal") or "")
        chain = eng.compose(
            goal,
            context=_context_from_args(args),
            chain_length=int(args.get("chain_length") or 3),
            domain=args.get("domain"),
        )
        return {"ok": True, "result": chain.to_dict()}

    if name == "autocausal_grail_run":
        goal = str(args.get("goal") or "")
        report = eng.run(
            goal,
            context=_context_from_args(args),
            max_cycles=int(args.get("max_cycles") or 2),
            chain_length=int(args.get("chain_length") or 3),
            domain=args.get("domain"),
        )
        return {"ok": True, "result": report.to_dict()}

    if name == "autocausal_grail_memory":
        query = str(args.get("query") or args.get("goal") or "")
        mem = eng.memory_step(
            query, context=_context_from_args(args), top_k=int(args.get("top_k") or 8)
        )
        return {"ok": True, "result": [m.to_dict() for m in mem]}

    if name == "autocausal_grail_graph":
        focus = list(args.get("focus") or [])
        boost = eng.graph_retrieve(
            context=_context_from_args(args),
            focus=focus or None,
            top_k=int(args.get("top_k") or 10),
        )
        return {"ok": True, "result": boost}

    return {"ok": False, "error": f"Unknown GRAIL tool: {name}"}


def register_on_surface(surface: Any = None) -> Any:
    """Attach GRAIL tools to skilling ToolSurface."""
    from autocausal.skilling.surface import ToolDef, ToolSurface
    from autocausal.suites.action_protocol import ActionResult
    import pandas as pd

    if surface is None:
        surface = ToolSurface()

    def _make_handler(tool_name: str):
        def handler(df: pd.DataFrame, **kwargs: Any) -> ActionResult:
            args = dict(kwargs)
            if "columns" not in args and df is not None:
                args["columns"] = [str(c) for c in df.columns]
            out = dispatch_grail_tool(tool_name, args)
            return ActionResult(
                name=tool_name,
                payload=out,
                notes=[
                    "GRAIL tool via skilling surface — exploratory scaffold, not identification."
                ],
            )

        handler.__name__ = tool_name
        return handler

    for schema in MCP_TOOL_SCHEMAS:
        name = schema["name"]
        surface.register(
            ToolDef(
                name=name,
                description=schema["description"],
                parameters=schema["parameters"],
                suite="grail",
                action=name.replace("autocausal_grail_", ""),
                handler=_make_handler(name),
            )
        )
    return surface


def try_bind_mcp(server: Any = None) -> dict[str, Any]:
    """Bind tools to MCP server / ToolRegistry if available.

    Prefers ``autocausal.mcp`` ``ToolRegistry.register`` / ``build_default_registry``
    (GRAIL tools are already registered in ``build_default_registry``).
    Also accepts a caller-supplied object with ``add_tool`` / ``register_tool``.
    """
    notes: list[str] = []
    bound: list[str] = []

    if server is None:
        try:
            from autocausal.mcp.registry import build_default_registry

            reg = build_default_registry()
            names = [n for n in reg.list_names() if n.startswith("autocausal_grail_")]
            return {
                "ok": bool(names),
                "backend": "autocausal.mcp",
                "tools": names or list(TOOL_NAMES),
                "notes": [
                    "GRAIL tools registered via build_default_registry "
                    "(also exposed on AgentHook / python -m autocausal.mcp)."
                ],
            }
        except Exception as e:
            return {
                "ok": False,
                "backend": "absent",
                "tools": list(TOOL_NAMES),
                "notes": [
                    f"autocausal.mcp bind soft-fail: {type(e).__name__}: {e}. "
                    "Schemas still available via grail_tool_schemas()."
                ],
            }

    # Caller-supplied registry or MCP server
    if hasattr(server, "register") and hasattr(server, "list_names"):
        # ToolRegistry-like
        try:
            from autocausal.mcp.registry import ToolSpec
            from autocausal.mcp.serialize import ok_payload, err_payload
            from autocausal.mcp.session import SessionStore

            for schema in MCP_TOOL_SCHEMAS:
                name = schema["name"]

                def _handler(
                    args: dict[str, Any],
                    store: SessionStore,
                    _n: str = name,
                ) -> dict[str, Any]:
                    out = dispatch_grail_tool(_n, args)
                    if not out.get("ok", True):
                        return err_payload(str(out.get("error")), tool=_n, soft=True)
                    return ok_payload(tool=_n, **{k: v for k, v in out.items() if k != "ok"})

                params = schema.get("parameters") or {}
                server.register(
                    ToolSpec(
                        name=name,
                        description=schema["description"],
                        parameters={
                            "properties": dict(params.get("properties") or {}),
                            "required": list(params.get("required") or []),
                        },
                        handler=_handler,
                        optional_module="autocausal.grail",
                    )
                )
                bound.append(name)
            return {
                "ok": True,
                "backend": "ToolRegistry",
                "tools": bound,
                "notes": ["Registered GRAIL tools onto supplied ToolRegistry."],
            }
        except Exception as e:
            notes.append(f"ToolRegistry path soft-fail: {type(e).__name__}: {e}")

    for schema in MCP_TOOL_SCHEMAS:
        name = schema["name"]

        def _handler(arguments: dict[str, Any], _n: str = name) -> dict[str, Any]:
            return dispatch_grail_tool(_n, arguments)

        try:
            if hasattr(server, "add_tool"):
                server.add_tool(
                    name,
                    schema["description"],
                    schema["parameters"],
                    _handler,
                )
                bound.append(name)
            elif hasattr(server, "register_tool"):
                server.register_tool(schema, _handler)
                bound.append(name)
            else:
                notes.append("Server lacks add_tool/register_tool/register")
                break
        except Exception as e:
            notes.append(f"{name} bind soft-fail: {type(e).__name__}: {e}")

    return {
        "ok": bool(bound),
        "backend": "mcp_server",
        "tools": bound or list(TOOL_NAMES),
        "notes": notes
        or (["Bound GRAIL MCP tools."] if bound else ["No tools bound."]),
    }
