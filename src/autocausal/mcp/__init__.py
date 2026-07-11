"""AutoCausal MCP connective package.

Soft-optional: importing ``autocausal.mcp`` never requires the ``mcp`` SDK.
Use :class:`~autocausal.mcp.hooks.AgentHook` for in-process agents, or
``python -m autocausal.mcp`` after ``pip install autocausal[mcp]``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentHook",
    "call_tool",
    "list_tools",
    "ToolRegistry",
    "build_default_registry",
    "SessionStore",
    "mcp_sdk_available",
    "run_stdio",
    "main",
]


def __getattr__(name: str) -> Any:
    if name in ("AgentHook", "call_tool", "list_tools", "get_default_hook"):
        from autocausal.mcp import hooks as _hooks

        return getattr(_hooks, name)
    if name in ("ToolRegistry", "build_default_registry", "ToolSpec", "EPISTEMIC"):
        from autocausal.mcp import registry as _reg

        return getattr(_reg, name)
    if name in ("SessionStore", "DEFAULT_SESSION"):
        from autocausal.mcp import session as _sess

        return getattr(_sess, name)
    if name in ("mcp_sdk_available", "create_server", "run_stdio", "main"):
        from autocausal.mcp import server as _srv

        return getattr(_srv, name)
    raise AttributeError(f"module 'autocausal.mcp' has no attribute {name!r}")
