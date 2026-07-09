"""FitReport contract for ML KPI loop imputers/predictors."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class FitReport:
    """Structured fit metrics for imputer / predictor stages."""

    schema: str = "FitReport.v1"
    produced_by: str = "autocausal.ml"
    produced_at: str = ""
    imputer: str = "median"
    predictor: str = "none"
    kpi_focus: list[str] = field(default_factory=list)
    outcome: Optional[str] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    torch_used: bool = False
    sklearn_used: bool = False

    def __post_init__(self) -> None:
        if not self.produced_at:
            self.produced_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "produced_by": self.produced_by,
            "produced_at": self.produced_at,
            "payload": {
                "imputer": self.imputer,
                "predictor": self.predictor,
                "kpi_focus": self.kpi_focus,
                "outcome": self.outcome,
                "metrics": self.metrics,
                "notes": self.notes,
                "torch_used": self.torch_used,
                "sklearn_used": self.sklearn_used,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Fit report",
            "",
            f"**Imputer:** `{self.imputer}`  |  **Predictor:** `{self.predictor}`",
            f"**Torch used:** {self.torch_used}  |  **Sklearn used:** {self.sklearn_used}",
            "",
        ]
        if self.kpi_focus:
            lines.append("**KPI focus:** " + ", ".join(f"`{k}`" for k in self.kpi_focus))
            lines.append("")
        if self.outcome:
            lines.append(f"**Outcome:** `{self.outcome}`")
            lines.append("")
        if self.metrics:
            lines += ["## Metrics", ""]
            for k, v in self.metrics.items():
                lines.append(f"- `{k}`: {v}")
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)
