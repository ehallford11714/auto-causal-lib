"""Library-first agent broker — same tool surface as the MCP server.

Use without installing the ``mcp`` SDK::

    from autocausal.connective import AgentHook
    # or: from autocausal.mcp.hooks import AgentHook

    hook = AgentHook()
    print(hook.list_tools())
    print(hook.call_tool("autocausal_load_dataset", {"dataset_id": "iris"}))
    print(hook.call_tool("autocausal_discover", {"use_iv": False}))
    print(hook.call_tool("autocausal_report", {"format": "markdown"}))
"""

from __future__ import annotations

from typing import Any, Optional

from autocausal.mcp.registry import ToolRegistry, build_default_registry
from autocausal.mcp.session import SessionStore

__all__ = ["AgentHook", "call_tool", "list_tools", "get_default_hook"]

_DEFAULT: Optional["AgentHook"] = None


class AgentHook:
    """In-process connective hook for non-MCP agents.

    Shares the same tool registry as the MCP stdio server. Sessions keep
    ``AutoCausal`` instances across multi-step tool calls.
    """

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        store: Optional[SessionStore] = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.store = store or SessionStore()

    def list_tools(self) -> list[dict[str, Any]]:
        """Return JSON-schema-like tool definitions."""
        return self.registry.schemas()

    def list_names(self) -> list[str]:
        return self.registry.list_names()

    def call_tool(self, name: str, args: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Invoke a tool by name; always returns a dict with ``ok`` bool."""
        return self.registry.invoke(name, args, store=self.store)

    def schemas(self) -> list[dict[str, Any]]:
        return self.list_tools()

    def sessions(self) -> list[dict[str, Any]]:
        return self.store.list_sessions()

    def reset(self) -> None:
        """Drop all sessions (new empty store)."""
        self.store = SessionStore()


def get_default_hook() -> AgentHook:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = AgentHook()
    return _DEFAULT


def call_tool(name: str, args: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Module-level invoke against a process-wide default AgentHook."""
    return get_default_hook().call_tool(name, args)


def list_tools() -> list[dict[str, Any]]:
    """Module-level tool schemas against the default AgentHook."""
    return get_default_hook().list_tools()
