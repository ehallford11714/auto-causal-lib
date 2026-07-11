"""Library-first connective broker for external agents (MCP-compatible surface).

Prefer this import when you do **not** need a live MCP stdio process::

    from autocausal.connective import AgentHook, call_tool

    hook = AgentHook()
    hook.call_tool("autocausal_load_dataset", {"dataset_id": "iris"})
    print(hook.call_tool("autocausal_discover", {"use_iv": False}))

The MCP server (``python -m autocausal.mcp``) uses the same :class:`AgentHook`.
"""

from __future__ import annotations

from autocausal.mcp.hooks import AgentHook, call_tool, get_default_hook, list_tools
from autocausal.mcp.registry import EPISTEMIC, ToolRegistry, build_default_registry
from autocausal.mcp.session import SessionStore

__all__ = [
    "AgentHook",
    "call_tool",
    "list_tools",
    "get_default_hook",
    "ToolRegistry",
    "build_default_registry",
    "SessionStore",
    "EPISTEMIC",
]
