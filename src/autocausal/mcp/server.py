"""MCP stdio server for AutoCausalLib (soft-optional ``mcp`` SDK).

Run::

    pip install autocausal[mcp]
    python -m autocausal.mcp
    # or: autocausal-mcp

Windows-friendly: uses the official MCP Python SDK stdio transport.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

from autocausal.mcp.hooks import AgentHook
from autocausal.mcp.registry import EPISTEMIC

__all__ = ["mcp_sdk_available", "create_server", "run_stdio", "main"]


def mcp_sdk_available() -> bool:
    try:
        import mcp  # noqa: F401

        return True
    except Exception:
        return False


def create_server(hook: Optional[AgentHook] = None) -> Any:
    """Build an MCP ``Server`` bound to an :class:`AgentHook`.

    Raises ``ImportError`` if the ``mcp`` package is not installed.
    """
    if not mcp_sdk_available():
        raise ImportError(
            "The 'mcp' package is required for the AutoCausal MCP server. "
            "Install with: pip install autocausal[mcp]"
        )

    from mcp.server import Server
    from mcp.types import TextContent, Tool

    hook = hook or AgentHook()
    server = Server("autocausal")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        tools: list[Tool] = []
        for spec in hook.list_tools():
            schema = spec.get("inputSchema") or {"type": "object", "properties": {}}
            desc = spec.get("description") or ""
            if spec.get("epistemic"):
                desc = f"{desc} [{EPISTEMIC}]"
            tools.append(
                Tool(
                    name=str(spec["name"]),
                    description=desc,
                    inputSchema=schema,
                )
            )
        return tools

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
        result = hook.call_tool(name, arguments or {})
        text = json.dumps(result, indent=2, default=str)
        return [TextContent(type="text", text=text)]

    return server


async def _run_stdio_async(hook: Optional[AgentHook] = None) -> None:
    from mcp.server.stdio import stdio_server

    server = create_server(hook)
    init = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init)


def run_stdio(hook: Optional[AgentHook] = None) -> None:
    """Block on the MCP stdio server (Windows-safe via SDK transport)."""
    import asyncio

    # Avoid Proactor quirks on some Windows Python builds for stdio pipes
    if sys.platform == "win32":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:
            pass
    asyncio.run(_run_stdio_async(hook))


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry for ``python -m autocausal.mcp`` / ``autocausal-mcp``."""
    argv = list(argv if argv is not None else sys.argv[1:])
    if any(a in ("-h", "--help") for a in argv):
        print(
            "AutoCausal MCP server\n\n"
            "Usage:\n"
            "  python -m autocausal.mcp\n"
            "  autocausal-mcp\n\n"
            "Install SDK:\n"
            "  pip install autocausal[mcp]\n\n"
            "Docs: docs/MCP.md\n"
            f"Epistemic: {EPISTEMIC}\n"
        )
        return 0
    if any(a in ("--list-tools", "list-tools") for a in argv):
        hook = AgentHook()
        for t in hook.list_tools():
            print(f"{t['name']}: {t['description']}")
        return 0
    if not mcp_sdk_available():
        print(
            "error: 'mcp' package not installed.\n"
            "  pip install autocausal[mcp]\n"
            "Library-first (no MCP SDK):\n"
            "  from autocausal.connective import AgentHook\n",
            file=sys.stderr,
        )
        return 1
    run_stdio()
    return 0
