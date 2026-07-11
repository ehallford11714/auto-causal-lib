"""AutoEDA report types."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from autocausal.suites.base import write_report
from autocausal.suites.director import EPISTEMIC_NOTE

__all__ = ["RoleProposal", "EDAReport"]


@dataclass
class RoleProposal:
    outcome: Optional[str] = None
    treatment: Optional[str] = None
    instruments: list[str] = field(default_factory=list)
    confounders: list[str] = field(default_factory=list)
    time_col: Optional[str] = None
    group_col: Optional[str] = None
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EDAReport:
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    missingness: dict[str, float]
    cardinality: dict[str, int]
    numeric_summary: dict[str, dict[str, float]]
    correlations: dict[str, dict[str, float]]
    roles: RoleProposal
    readiness_score: float
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    leakage_hints: list[str] = field(default_factory=list)
    qc: Optional[dict[str, Any]] = None
    mining_profile: Optional[dict[str, Any]] = None
    plots: Optional[dict[str, Any]] = None
    action_results: list[dict[str, Any]] = field(default_factory=list)
    actions_run: list[str] = field(default_factory=list)
    slm_directives: Optional[dict[str, Any]] = None
    notes: list[str] = field(default_factory=list)
    source: str = ""
    backend: str = "rule"
    schema: str = "AutoCausalEDAReport.v2"
    association_scan: list[dict[str, Any]] = field(default_factory=list)
    missingness_patterns: dict[str, Any] = field(default_factory=dict)
    categorical_imbalance: dict[str, Any] = field(default_factory=dict)
    subgroup_imbalance: dict[str, Any] = field(default_factory=dict)
    causal_readiness: dict[str, Any] = field(default_factory=dict)
    assumption_readiness: dict[str, Any] = field(default_factory=dict)
    descriptive_findings: list[str] = field(default_factory=list)
    predictive_findings: list[str] = field(default_factory=list)
    causal_readiness_findings: list[str] = field(default_factory=list)
    gate_report: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": list(self.columns),
            "dtypes": dict(self.dtypes),
            "missingness": dict(self.missingness),
            "cardinality": dict(self.cardinality),
            "numeric_summary": dict(self.numeric_summary),
            "correlations": dict(self.correlations),
            "roles": self.roles.to_dict(),
            "readiness_score": self.readiness_score,
            "warnings": list(self.warnings),
            "suggestions": list(self.suggestions),
            "leakage_hints": list(self.leakage_hints),
            "qc": self.qc,
            "mining_profile": self.mining_profile,
            "plots": self.plots,
            "action_results": list(self.action_results),
            "actions_run": list(self.actions_run),
            "slm_directives": self.slm_directives,
            "notes": list(self.notes),
            "source": self.source,
            "backend": self.backend,
            "association_scan": list(self.association_scan),
            "missingness_patterns": dict(self.missingness_patterns),
            "categorical_imbalance": dict(self.categorical_imbalance),
            "subgroup_imbalance": dict(self.subgroup_imbalance),
            "causal_readiness": dict(self.causal_readiness),
            "assumption_readiness": dict(self.assumption_readiness),
            "descriptive_findings": list(self.descriptive_findings),
            "predictive_findings": list(self.predictive_findings),
            "causal_readiness_findings": list(
                self.causal_readiness_findings
            ),
            "gate_report": self.gate_report,
        }

    def to_gate_inputs(self) -> dict[str, Any]:
        """Machine-readable, raw-value-free inputs for production gates."""
        return {
            "schema": "AutoCausalEDAGateInputs.v1",
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "max_missing_fraction": max(
                self.missingness.values(), default=0.0
            ),
            "missingness": dict(self.missingness),
            "cardinality": dict(self.cardinality),
            "leakage_hints": list(self.leakage_hints),
            "roles": self.roles.to_dict(),
            "readiness_score": self.readiness_score,
            "categorical_imbalance": dict(self.categorical_imbalance),
            "subgroup_imbalance": dict(self.subgroup_imbalance),
            "causal_readiness": dict(self.causal_readiness),
            "assumption_readiness": dict(self.assumption_readiness),
            "association_tests": [
                {
                    "x": item.get("x"),
                    "y": item.get("y"),
                    "measure": item.get("measure"),
                    "coefficient": item.get("coefficient"),
                    "p_value": item.get("p_value"),
                    "q_value": item.get("q_value"),
                    "n": item.get("n"),
                    "identification_evidence": False,
                }
                for item in self.association_scan
            ],
            "raw_values_included": False,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# AutoEDA report (causal readiness)",
            "",
            f"- rows={self.n_rows}, cols={self.n_cols}",
            f"- readiness_score=**{self.readiness_score:.2f}**",
            f"- backend: `{self.backend}`",
            "",
            f"> {EPISTEMIC_NOTE}",
            "",
        ]
        if self.actions_run:
            lines += ["## Actions run", ""]
            for a in self.actions_run:
                lines.append(f"- `{a}`")
            lines.append("")
        lines += [
            "## Proposed roles (hypotheses — not ground truth)",
            f"- Y (outcome): `{self.roles.outcome}`",
            f"- X (treatment): `{self.roles.treatment}`",
            f"- Z (instruments): {self.roles.instruments}",
            f"- W (confounders): {self.roles.confounders}",
            "",
            "## Missingness (top)",
            "",
            "| column | missing% | cardinality |",
            "|---|---:|---:|",
        ]
        ranked = sorted(self.missingness.items(), key=lambda kv: -kv[1])[:15]
        for c, m in ranked:
            lines.append(f"| `{c}` | {m:.1%} | {self.cardinality.get(c, '')} |")
        lines.append("")
        if self.leakage_hints:
            lines += ["## Leakage hints", ""]
            for h in self.leakage_hints:
                lines.append(f"- {h}")
            lines.append("")
        if self.descriptive_findings:
            lines += ["## Descriptive findings", ""]
            lines.extend(f"- {value}" for value in self.descriptive_findings)
            lines.append("")
        if self.predictive_findings:
            lines += [
                "## Predictive findings (not causal evidence)",
                "",
            ]
            lines.extend(f"- {value}" for value in self.predictive_findings)
            lines.append("")
        if self.causal_readiness_findings:
            lines += ["## Causal-design readiness", ""]
            lines.extend(
                f"- {value}" for value in self.causal_readiness_findings
            )
            lines.append("")
        if self.warnings:
            lines += ["## Warnings", ""]
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")
        if self.suggestions:
            lines += ["## Suggestions", ""]
            for s in self.suggestions:
                lines.append(f"- {s}")
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
