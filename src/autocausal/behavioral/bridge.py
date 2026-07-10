"""Soft hooks to IntentIsolates / ReasonTrace when present on path.

Never hard-requires external packages.
"""

from __future__ import annotations

from typing import Any, Optional


def isolates_available() -> bool:
    try:
        import intentisolates  # noqa: F401

        return True
    except ImportError:
        return False


def reason_trace_available() -> bool:
    """Detect optional ReasonTrace-like modules if importable."""
    for mod in ("reasontrace", "ReasonTrace", "intentisolates.reason"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def soft_isolates_annotate(text: str, **kwargs: Any) -> dict[str, Any]:
    """If IntentIsolates is installed, run a light annotation; else stub."""
    if not isolates_available():
        return {
            "ok": False,
            "backend": "missing",
            "notes": ["intentisolates not installed — soft skip"],
            "data": None,
        }
    try:
        from autocausal.isolates_bridge import run_isolates_causal

        result = run_isolates_causal(text, **kwargs)
        payload = result.to_dict() if hasattr(result, "to_dict") else {"raw": str(result)}
        return {"ok": True, "backend": "intentisolates", "data": payload, "notes": []}
    except Exception as e:
        return {
            "ok": False,
            "backend": "intentisolates",
            "error": f"{type(e).__name__}: {e}",
            "notes": ["isolates annotate soft-fail"],
            "data": None,
        }


def soft_reason_trace_hook(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Optional ReasonTrace bridge — returns stub when unavailable."""
    if not reason_trace_available():
        return {
            "ok": False,
            "backend": "missing",
            "n_events": len(events),
            "notes": ["ReasonTrace not on path — soft skip"],
        }
    return {
        "ok": True,
        "backend": "reasontrace",
        "n_events": len(events),
        "notes": ["ReasonTrace detected; passthrough hook only"],
    }


__all__ = [
    "isolates_available",
    "reason_trace_available",
    "soft_isolates_annotate",
    "soft_reason_trace_hook",
]
