"""In-memory session store for MCP / AgentHook AutoCausal instances."""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from autocausal.api import AutoCausal

__all__ = ["SessionStore", "DEFAULT_SESSION"]

DEFAULT_SESSION = "default"


class SessionStore:
    """Named AutoCausal handles so multi-step agent tool calls share state."""

    def __init__(self) -> None:
        self._sessions: dict[str, AutoCausal] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._last_insight: dict[str, Any] = {}
        self._last_public: dict[str, Any] = {}
        self._last_experiments: dict[str, Any] = {}

    def new_id(self) -> str:
        return f"s_{uuid4().hex[:10]}"

    def put(self, ac: AutoCausal, session_id: Optional[str] = None) -> str:
        sid = session_id or DEFAULT_SESSION
        self._sessions[sid] = ac
        self._meta[sid] = {
            "source": getattr(ac, "source", ""),
            "n_rows": len(ac.df),
            "n_cols": len(ac.df.columns),
            "columns": [str(c) for c in ac.df.columns],
        }
        return sid

    def get(self, session_id: Optional[str] = None) -> AutoCausal:
        sid = session_id or DEFAULT_SESSION
        if sid not in self._sessions:
            raise KeyError(
                f"No AutoCausal session {sid!r}. "
                f"Call autocausal_load_dataset / autocausal_from_csv first. "
                f"Known: {sorted(self._sessions)}"
            )
        return self._sessions[sid]

    def has(self, session_id: Optional[str] = None) -> bool:
        return (session_id or DEFAULT_SESSION) in self._sessions

    def list_sessions(self) -> list[dict[str, Any]]:
        out = []
        for sid, meta in self._meta.items():
            out.append({"session_id": sid, **meta})
        return out

    def drop(self, session_id: str) -> bool:
        existed = session_id in self._sessions
        self._sessions.pop(session_id, None)
        self._meta.pop(session_id, None)
        self._last_insight.pop(session_id, None)
        self._last_public.pop(session_id, None)
        self._last_experiments.pop(session_id, None)
        return existed

    def set_insight(self, session_id: str, payload: Any) -> None:
        self._last_insight[session_id] = payload

    def get_insight(self, session_id: Optional[str] = None) -> Any:
        return self._last_insight.get(session_id or DEFAULT_SESSION)

    def set_public(self, session_id: str, payload: Any) -> None:
        self._last_public[session_id] = payload

    def get_public(self, session_id: Optional[str] = None) -> Any:
        return self._last_public.get(session_id or DEFAULT_SESSION)

    def set_experiments(self, session_id: str, payload: Any) -> None:
        self._last_experiments[session_id] = payload

    def get_experiments(self, session_id: Optional[str] = None) -> Any:
        return self._last_experiments.get(session_id or DEFAULT_SESSION)

    def refresh_meta(self, session_id: str) -> None:
        if session_id not in self._sessions:
            return
        ac = self._sessions[session_id]
        self._meta[session_id] = {
            "source": getattr(ac, "source", ""),
            "n_rows": len(ac.df),
            "n_cols": len(ac.df.columns),
            "columns": [str(c) for c in ac.df.columns],
        }
