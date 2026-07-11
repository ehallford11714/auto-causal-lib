"""Rendered chart result and report contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autocausal.autochart.specs import ChartSpec


CHART_CAVEAT = (
    "This visualization is descriptive. It must not be interpreted as evidence "
    "that a displayed relationship is causal."
)


@dataclass
class RenderedChart:
    """One rendered chart, including a dependency-free data/spec fallback."""

    spec: ChartSpec
    backend: str
    artifact: Any = None
    data_payload: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    production: bool = False
    schema: str = "AutoCausalRenderedChart.v1"

    def to_dict(self) -> dict[str, Any]:
        contains_raw_values = bool(
            self.provenance.get("contains_raw_values", False)
        )
        return {
            "schema": self.schema,
            "backend": self.backend,
            "spec": self.spec.to_dict(
                redact_filter_values=self.production and not contains_raw_values,
                redact_annotations=self.production and not contains_raw_values,
            ),
            "data": dict(self.data_payload),
            "provenance": dict(self.provenance),
            "warnings": list(self.warnings),
            "epistemic_caveat": CHART_CAVEAT,
            "contains_raw_values": contains_raw_values,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.spec.title}",
            "",
            f"- chart id/type: `{self.spec.id}` / `{self.spec.type}`",
            f"- backend: `{self.backend}`",
            f"- alt text: {self.spec.accessibility.alt_text}",
            f"- raw values included: {bool(self.provenance.get('contains_raw_values', False))}",
            "",
            f"> {CHART_CAVEAT}",
            "",
        ]
        if self.warnings:
            lines.extend(["## Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
            lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        return self.to_markdown() if as_markdown else self.to_json()

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        suffix = output.suffix.lower()
        if suffix == ".json":
            output.write_text(self.to_json(), encoding="utf-8")
            return output
        if suffix in (".md", ".markdown"):
            output.write_text(self.to_markdown(), encoding="utf-8")
            return output
        if self.backend == "plotly" and self.artifact is not None:
            if suffix == ".html":
                self.artifact.write_html(
                    str(output),
                    include_plotlyjs="cdn",
                    full_html=True,
                )
                return output
            if suffix in (".png", ".svg"):
                try:
                    self.artifact.write_image(str(output))
                except Exception as exc:
                    raise RuntimeError(
                        "Plotly static export requires a compatible image "
                        "engine such as kaleido."
                    ) from exc
                return output
        if self.backend == "matplotlib" and self.artifact is not None:
            if suffix in (".png", ".svg"):
                self.artifact.savefig(
                    output,
                    format=suffix.lstrip("."),
                    bbox_inches="tight",
                    metadata={"Description": self.spec.accessibility.alt_text},
                )
                return output
        raise ValueError(
            f"backend {self.backend!r} does not support {suffix or 'extensionless'} export"
        )

    def write(self, path: str | Path) -> Path:
        """Alias for :meth:`save`."""
        return self.save(path)

    def close(self) -> None:
        if self.backend == "matplotlib" and self.artifact is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self.artifact)
            except Exception:
                pass


@dataclass
class AutoChartReport:
    charts: list[RenderedChart] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    schema: str = "AutoCausalAutoChartReport.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "charts": [chart.to_dict() for chart in self.charts],
            "notes": list(self.notes),
            "epistemic_caveat": CHART_CAVEAT,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# AutoChart report",
            "",
            f"- charts: {len(self.charts)}",
            "",
            f"> {CHART_CAVEAT}",
            "",
        ]
        for chart in self.charts:
            lines.append(
                f"- `{chart.spec.id}`: {chart.spec.title} (backend={chart.backend})"
            )
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
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
            raise ValueError("AutoChartReport.write fmt must be json or markdown")
        output.write_text(
            self.to_json() if selected == "json" else self.to_markdown(),
            encoding="utf-8",
        )
        return output


__all__ = ["AutoChartReport", "CHART_CAVEAT", "RenderedChart"]
