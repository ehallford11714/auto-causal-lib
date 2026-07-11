"""AutoCleanse report types."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from autocausal.suites.base import write_report
from autocausal.suites.director import EPISTEMIC_NOTE

__all__ = ["CleanseOp", "CleanseReport"]


@dataclass
class CleanseOp:
    op: str
    detail: str
    columns: list[str] = field(default_factory=list)
    n_affected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleanseReport:
    n_rows_in: int
    n_rows_out: int
    n_cols_in: int
    n_cols_out: int
    operations: list[CleanseOp] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    action_results: list[dict[str, Any]] = field(default_factory=list)
    actions_run: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    slm_directives: Optional[dict[str, Any]] = None
    imputation: Optional[dict[str, Any]] = None
    qc: Optional[dict[str, Any]] = None
    missingness: Optional[dict[str, Any]] = None
    source: str = ""
    backend: str = "rule"
    schema: str = "AutoCausalCleanseReport.v2"
    dry_run: bool = False
    reversible: bool = True
    policy_profile: str = "exploratory"
    before_fingerprint: dict[str, Any] = field(default_factory=dict)
    after_fingerprint: dict[str, Any] = field(default_factory=dict)
    schema_violations: list[dict[str, Any]] = field(default_factory=list)
    range_violations: list[dict[str, Any]] = field(default_factory=list)
    pii_summary: dict[str, Any] = field(default_factory=dict)
    gate_report: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "n_cols_in": self.n_cols_in,
            "n_cols_out": self.n_cols_out,
            "operations": [o.to_dict() for o in self.operations],
            "dropped_columns": list(self.dropped_columns),
            "action_results": list(self.action_results),
            "actions_run": list(self.actions_run),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "slm_directives": self.slm_directives,
            "imputation": self.imputation,
            "qc": self.qc,
            "missingness": self.missingness,
            "source": self.source,
            "backend": self.backend,
            "dry_run": self.dry_run,
            "reversible": self.reversible,
            "policy_profile": self.policy_profile,
            "before_fingerprint": dict(self.before_fingerprint),
            "after_fingerprint": dict(self.after_fingerprint),
            "schema_violations": list(self.schema_violations),
            "range_violations": list(self.range_violations),
            "pii_summary": dict(self.pii_summary),
            "gate_report": self.gate_report,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# AutoCleanse report",
            "",
            f"- rows {self.n_rows_in} → {self.n_rows_out}",
            f"- cols {self.n_cols_in} → {self.n_cols_out}",
            f"- backend: `{self.backend}`",
            f"- policy profile: `{self.policy_profile}`",
            f"- dry run: `{self.dry_run}` · reversible: `{self.reversible}`",
            "",
            f"> {EPISTEMIC_NOTE}",
            "",
        ]
        if self.actions_run:
            lines += ["## Actions run", ""]
            for a in self.actions_run:
                lines.append(f"- `{a}`")
            lines.append("")
        lines += ["## Operations", ""]
        if not self.operations:
            lines.append("_No mutations._")
        else:
            for op in self.operations:
                lines.append(f"- `{op.op}`: {op.detail} (n={op.n_affected})")
        if self.dropped_columns:
            lines += ["", "## Dropped columns", ""]
            for c in self.dropped_columns:
                lines.append(f"- `{c}`")
        if self.schema_violations or self.range_violations:
            lines += ["", "## Validation findings", ""]
            for finding in self.schema_violations + self.range_violations:
                lines.append(
                    f"- `{finding.get('column')}`: {finding.get('issue')} "
                    f"(n={finding.get('count', 0)})"
                )
        if self.warnings:
            lines += ["", "## Warnings", ""]
            for w in self.warnings:
                lines.append(f"- {w}")
        if self.notes:
            lines += ["", "## Notes", ""]
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
