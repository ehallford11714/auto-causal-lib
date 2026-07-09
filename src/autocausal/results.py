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
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        from autocausal.report import render_markdown_report

        return render_markdown_report(self)
