"""Shared action protocol for suite action registries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import pandas as pd

__all__ = ["ActionResult", "ActionFn", "ActionRegistry"]


@dataclass
class ActionResult:
    """Result of one dedicated suite action."""

    name: str
    frame: Optional[pd.DataFrame] = None
    payload: dict[str, Any] = field(default_factory=dict)
    ops: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    n_affected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "payload": dict(self.payload),
            "ops": list(self.ops),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "n_affected": self.n_affected,
            "mutated_frame": self.frame is not None,
        }


ActionFn = Callable[..., ActionResult]


class ActionRegistry:
    """Name → callable registry with list/get/run helpers."""

    def __init__(self, suite: str) -> None:
        self.suite = suite
        self._actions: dict[str, ActionFn] = {}

    def register(self, name: str, fn: ActionFn) -> ActionFn:
        self._actions[name] = fn
        return fn

    def decorator(self, name: Optional[str] = None):
        def wrap(fn: ActionFn) -> ActionFn:
            key = name or fn.__name__
            self.register(key, fn)
            return fn

        return wrap

    def list(self) -> list[str]:
        return sorted(self._actions.keys())

    def get(self, name: str) -> ActionFn:
        if name not in self._actions:
            raise KeyError(f"Unknown {self.suite} action: {name!r}. Known: {self.list()}")
        return self._actions[name]

    def run(self, name: str, *args: Any, **kwargs: Any) -> ActionResult:
        return self.get(name)(*args, **kwargs)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._actions
