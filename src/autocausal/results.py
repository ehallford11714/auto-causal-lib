"""Result dataclasses (kept separate to avoid circular imports)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from autocausal.impute import ImputationReport
from autocausal.roles import ColumnRole


@dataclass
class DiscoveryResult:
    """Structured output from causal relationship discovery."""

    edges: list[dict[str, Any]]
    graph: dict[str, Any]
    roles: dict[str, ColumnRole]
    candidates: dict[str, list[str]]
    imputation: Optional[ImputationReport] = None
    method: str = "score_pc_lite"
    notes: list[str] = field(default_factory=list)
    mining: Optional[dict[str, Any]] = None
    guide: Optional[dict[str, Any]] = None
    grounding: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        roles = {k: (v.value if hasattr(v, "value") else str(v)) for k, v in self.roles.items()}
        out: dict[str, Any] = {
            "method": self.method,
            "edges": self.edges,
            "graph": self.graph,
            "roles": roles,
            "candidates": self.candidates,
            "notes": self.notes,
        }
        if self.imputation is not None:
            out["imputation"] = asdict(self.imputation)
        if self.mining is not None:
            out["mining"] = self.mining
        if self.guide is not None:
            out["guide"] = self.guide
        if self.grounding is not None:
            out["grounding"] = self.grounding
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        from autocausal.report import render_markdown_report

        return render_markdown_report(self)


@dataclass
class AutoResult:
    """Full orchestrated auto() pipeline output."""

    discovery: DiscoveryResult
    mining: Optional[dict[str, Any]] = None
    guide: Optional[dict[str, Any]] = None
    grounding: Optional[dict[str, Any]] = None
    join_log: list[dict[str, Any]] = field(default_factory=list)
    ping: Optional[dict[str, Any]] = None
    source: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ping": self.ping,
            "join_log": self.join_log,
            "mining": self.mining,
            "discovery": self.discovery.to_dict(),
            "guide": self.guide,
            "grounding": self.grounding,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        from autocausal.report import render_auto_markdown

        return render_auto_markdown(self)
