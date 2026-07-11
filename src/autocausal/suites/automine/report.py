"""AutoMine report types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from autocausal.suites.base import write_report
from autocausal.suites.director import EPISTEMIC_NOTE

__all__ = ["MineReport"]


@dataclass
class MineReport:
    n_rows: int
    n_cols: int
    columns: list[dict[str, Any]] = field(default_factory=list)
    associations: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    kpis: list[str] = field(default_factory=list)
    ranked_candidates: list[dict[str, Any]] = field(default_factory=list)
    join_log: list[dict[str, Any]] = field(default_factory=list)
    datamine: Optional[dict[str, Any]] = None
    behavioral: Optional[dict[str, Any]] = None
    fabric_envelope: Optional[dict[str, Any]] = None
    action_results: list[dict[str, Any]] = field(default_factory=list)
    actions_run: list[str] = field(default_factory=list)
    slm_directives: Optional[dict[str, Any]] = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source: str = ""
    backend: str = "rule"
    mining_backend: str = "autocausal.mining"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": list(self.columns),
            "associations": list(self.associations),
            "suggestions": list(self.suggestions),
            "kpis": list(self.kpis),
            "ranked_candidates": list(self.ranked_candidates),
            "join_log": list(self.join_log),
            "datamine": self.datamine,
            "behavioral": self.behavioral,
            "fabric_envelope": self.fabric_envelope,
            "action_results": list(self.action_results),
            "actions_run": list(self.actions_run),
            "slm_directives": self.slm_directives,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "source": self.source,
            "backend": self.backend,
            "mining_backend": self.mining_backend,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_mine_report(self, *, backend: Optional[str] = None) -> dict[str, Any]:
        if self.fabric_envelope is not None:
            return self.fabric_envelope
        from autocausal.contracts import mining_to_mine_report

        return mining_to_mine_report(
            self,
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            backend=backend or self.mining_backend,
            extra_meta={"suite": "AutoMineSuite", "director_backend": self.backend},
        )

    def to_markdown(self) -> str:
        lines = [
            "# AutoMine report",
            "",
            f"- rows={self.n_rows}, cols={self.n_cols}",
            f"- director backend: `{self.backend}`",
            f"- mining backend: `{self.mining_backend}`",
            "",
            f"> {EPISTEMIC_NOTE}",
            "",
        ]
        if self.actions_run:
            lines += ["## Actions run", ""]
            for a in self.actions_run:
                lines.append(f"- `{a}`")
            lines.append("")
        if self.kpis:
            lines += ["## Suggested KPIs", ""]
            for k in self.kpis:
                lines.append(f"- `{k}`")
            lines.append("")
        lines += ["## Top associations", ""]
        if not self.associations:
            lines.append("_None above threshold._")
        else:
            lines += ["| a | b | metric | score |", "|---|---|---|---:|"]
            for a in self.associations[:25]:
                lines.append(
                    f"| `{a.get('a')}` | `{a.get('b')}` | {a.get('metric', '')} | {a.get('score', '')} |"
                )
        lines.append("")
        if self.suggestions:
            lines += ["## Suggested relationships", ""]
            for s in self.suggestions[:20]:
                lines.append(
                    f"- `{s.get('source')}` → `{s.get('target')}` "
                    f"({s.get('reason', '')}; score={s.get('score', '')})"
                )
            lines.append("")
        if self.warnings:
            lines += ["## Warnings", ""]
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""]
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        return "\n".join(lines)

    def write(self, path: Union[str, Path], *, fmt: str = "auto") -> Path:
        return write_report(self, path, fmt=fmt)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()
