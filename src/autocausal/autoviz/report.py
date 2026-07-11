"""Serializable contracts for analysis-aware visualization planning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


EPISTEMIC_CAVEAT = (
    "Visual and predictive associations are descriptive; a chart does not "
    "establish a causal effect or validate an identification strategy."
)

CHART_PLAN_TYPES = frozenset(
    {
        "distribution",
        "missingness",
        "correlation",
        "association",
        "treatment_outcome",
        "covariate_balance",
        "overlap",
        "iv_first_stage",
        "edge_stability",
        "dag",
        "network",
        "panel_trend",
        "subgroup_effects",
        "residual_diagnostics",
        "calibration",
        "roc",
        "pr",
        "feature_importance",
        "gate_dashboard",
        "evidence_matrix",
    }
)


@dataclass
class VizRecommendation:
    """One ranked visualization recommendation and its prerequisites."""

    id: str
    chart_type: str
    title: str
    priority: int
    rationale: str
    required_columns: list[str] = field(default_factory=list)
    data_requirements: dict[str, Any] = field(default_factory=dict)
    spec_hints: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    source: str = "rule"

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        self.chart_type = str(self.chart_type).strip().lower()
        self.title = str(self.title).strip()
        self.priority = int(self.priority)
        self.required_columns = list(dict.fromkeys(str(c) for c in self.required_columns))
        if not self.id:
            raise ValueError("visualization recommendation id cannot be empty")
        if self.chart_type not in CHART_PLAN_TYPES:
            raise ValueError(f"unsupported visualization plan type {self.chart_type!r}")
        if not 0 <= self.priority <= 100:
            raise ValueError("visualization priority must be between 0 and 100")
        if not self.rationale.strip():
            raise ValueError("visualization rationale cannot be empty")
        if EPISTEMIC_CAVEAT not in self.caveats:
            self.caveats.append(EPISTEMIC_CAVEAT)

    def validate_columns(self, columns: Sequence[str]) -> list[str]:
        available = {str(column) for column in columns}
        return [column for column in self.required_columns if column not in available]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        columns = ", ".join(f"`{column}`" for column in self.required_columns)
        return "\n".join(
            [
                f"# {self.title}",
                "",
                f"- id/type: `{self.id}` / `{self.chart_type}`",
                f"- priority: {self.priority}",
                f"- rationale: {self.rationale}",
                f"- columns: {columns or '(summary metadata only)'}",
                "",
                f"> {EPISTEMIC_CAVEAT}",
                "",
            ]
        )

    def report(self, *, as_markdown: bool = True) -> str:
        return self.to_markdown() if as_markdown else self.to_json()

    def write(self, path: str | Path, *, fmt: str = "auto") -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        selected = fmt.lower()
        if selected == "auto":
            selected = "json" if output.suffix.lower() == ".json" else "markdown"
        if selected not in ("json", "markdown", "md"):
            raise ValueError("VizRecommendation.write fmt must be json or markdown")
        output.write_text(
            self.to_json() if selected == "json" else self.to_markdown(),
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VizRecommendation":
        return cls(**dict(value))


@dataclass
class VizPlan:
    """Validated, deterministic visualization plan."""

    recommendations: list[VizRecommendation] = field(default_factory=list)
    frame_summary: dict[str, Any] = field(default_factory=dict)
    mode: str = "exploratory"
    planner: str = "rule"
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    schema: str = "AutoCausalVizPlan.v1"

    def __post_init__(self) -> None:
        converted: list[VizRecommendation] = []
        for item in self.recommendations:
            converted.append(
                item
                if isinstance(item, VizRecommendation)
                else VizRecommendation.from_dict(item)  # type: ignore[arg-type]
            )
        self.recommendations = sorted(
            converted, key=lambda item: (-item.priority, item.id)
        )
        self.validate()

    def validate(self) -> None:
        columns = [str(column) for column in self.frame_summary.get("columns") or []]
        seen: set[str] = set()
        for recommendation in self.recommendations:
            if recommendation.id in seen:
                raise ValueError(
                    f"duplicate visualization recommendation id {recommendation.id!r}"
                )
            seen.add(recommendation.id)
            missing = recommendation.validate_columns(columns)
            if columns and missing:
                raise ValueError(
                    f"recommendation {recommendation.id!r} references missing "
                    f"columns: {missing}"
                )
        if self.mode not in ("exploratory", "production"):
            raise ValueError("VizPlan.mode must be 'exploratory' or 'production'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mode": self.mode,
            "planner": self.planner,
            "frame_summary": dict(self.frame_summary),
            "recommendations": [item.to_dict() for item in self.recommendations],
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
            "contains_raw_values": False,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        return AutoVizReport(plan=self).to_markdown()

    def report(self, *, as_markdown: bool = True) -> str:
        return self.to_markdown() if as_markdown else self.to_json()

    def write(self, path: str | Path, *, fmt: str = "auto") -> Path:
        return AutoVizReport(plan=self).write(path, fmt=fmt)


@dataclass
class AutoVizReport:
    """Human- and machine-readable result of :class:`AutoVizSuite`."""

    plan: VizPlan
    source: str = "dataframe"
    notes: list[str] = field(default_factory=list)
    schema: str = "AutoCausalAutoVizReport.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "plan": self.plan.to_dict(),
            "notes": list(self.notes),
            "epistemic_caveat": EPISTEMIC_CAVEAT,
            "contains_raw_values": False,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        summary = self.plan.frame_summary
        lines = [
            "# AutoViz analysis plan",
            "",
            f"- source: `{self.source}`",
            f"- mode: `{self.plan.mode}`",
            f"- rows: {summary.get('n_rows', 0)}",
            f"- columns: {summary.get('n_columns', 0)}",
            f"- planner: `{self.plan.planner}`",
            "",
            f"> {EPISTEMIC_CAVEAT}",
            "",
            "## Ranked recommendations",
            "",
        ]
        if not self.plan.recommendations:
            lines.append("- No valid recommendations were produced.")
        for item in self.plan.recommendations:
            columns = ", ".join(f"`{column}`" for column in item.required_columns)
            lines.extend(
                [
                    f"### {item.priority}. {item.title}",
                    "",
                    f"- id/type: `{item.id}` / `{item.chart_type}`",
                    f"- rationale: {item.rationale}",
                    f"- columns: {columns or '(summary metadata only)'}",
                ]
            )
            if item.data_requirements:
                requirements = ", ".join(
                    f"{key}={value!r}"
                    for key, value in sorted(item.data_requirements.items())
                )
                lines.append(f"- requirements: {requirements}")
            lines.append("")
        combined_notes = list(self.plan.warnings) + list(self.notes)
        if combined_notes:
            lines.extend(["## Notes", ""])
            lines.extend(f"- {note}" for note in combined_notes)
            lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        return self.to_markdown() if as_markdown else self.to_json()

    def write(self, path: str | Path, *, fmt: str = "auto") -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        selected = fmt.lower()
        if selected == "auto":
            selected = "json" if output.suffix.lower() == ".json" else "markdown"
        if selected not in ("json", "markdown", "md"):
            raise ValueError("AutoVizReport.write fmt must be json or markdown")
        output.write_text(
            self.to_json() if selected == "json" else self.to_markdown(),
            encoding="utf-8",
        )
        return output


__all__ = [
    "AutoVizReport",
    "CHART_PLAN_TYPES",
    "EPISTEMIC_CAVEAT",
    "VizPlan",
    "VizRecommendation",
]
