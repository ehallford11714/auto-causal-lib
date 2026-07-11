"""JSON-safe serialization helpers for MCP / AgentHook payloads."""

from __future__ import annotations

import json
from typing import Any


def to_jsonable(obj: Any, *, max_edges: int = 50, max_list: int = 100) -> Any:
    """Best-effort conversion of library objects to JSON-serializable structures."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_edges=max_edges, max_list=max_list) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = list(obj)[:max_list]
        return [to_jsonable(x, max_edges=max_edges, max_list=max_list) for x in items]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return to_jsonable(obj.to_dict(), max_edges=max_edges, max_list=max_list)
        except Exception:
            pass
    if hasattr(obj, "to_markdown") and callable(obj.to_markdown):
        try:
            return {"markdown": obj.to_markdown()}
        except Exception:
            pass
    # DiscoveryResult-like
    if hasattr(obj, "edges") and hasattr(obj, "candidates"):
        try:
            edges = list(getattr(obj, "edges") or [])[:max_edges]
            return {
                "n_edges": len(getattr(obj, "edges") or []),
                "edges": to_jsonable(edges, max_edges=max_edges, max_list=max_list),
                "candidates": to_jsonable(
                    getattr(obj, "candidates") or {}, max_edges=max_edges, max_list=max_list
                ),
                "notes": list(getattr(obj, "notes") or [])[:40],
            }
        except Exception:
            pass
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def ok_payload(**kwargs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True}
    out.update({k: to_jsonable(v) for k, v in kwargs.items()})
    return out


def err_payload(message: str, *, tool: str = "", **kwargs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "error": str(message)}
    if tool:
        out["tool"] = tool
    out.update({k: to_jsonable(v) for k, v in kwargs.items()})
    return out
