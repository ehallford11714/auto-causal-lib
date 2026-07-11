"""Dataclasses for physics predictive engine + autocausal loop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class PhysicsState:
    """Normalized dynamical state for KPI / physical proxies."""

    names: list[str]
    position: list[float]
    velocity: list[float] = field(default_factory=list)
    t: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_vector(self) -> list[float]:
        return list(self.position)


@dataclass
class TrajectoryPoint:
    t: int
    state: PhysicsState
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    uncertainty: list[float] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "state": self.state.to_dict(),
            "kinetic_energy": self.kinetic_energy,
            "potential_energy": self.potential_energy,
            "uncertainty": self.uncertainty,
            "note": self.note,
        }


@dataclass
class Trajectory:
    """Multi-step rollout of physical / KPI proxies."""

    points: list[TrajectoryPoint]
    backend: str = "numpy_analytic_v1"
    system: str = "damped_oscillator"
    horizon: int = 0
    predictions: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "system": self.system,
            "horizon": self.horizon or len(self.points),
            "points": [p.to_dict() for p in self.points],
            "predictions": self.predictions,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Physics trajectory",
            "",
            f"**Backend:** `{self.backend}` · **System:** `{self.system}` · "
            f"**Horizon:** {self.horizon or len(self.points)}",
            "",
        ]
        if not self.points:
            lines.append("_Empty trajectory._")
            lines.append("")
            return "\n".join(lines)
        names = self.points[0].state.names
        header = "| t | ke | pe | " + " | ".join(names) + " | ±band |"
        sep = "|---:|---:|---:|" + "|---:|" * len(names) + "|---|"
        lines += [header, sep]
        for p in self.points:
            vals = " | ".join(f"{x:.4f}" for x in p.state.position)
            band = (
                ", ".join(f"±{u:.3f}" for u in p.uncertainty[:3])
                if p.uncertainty
                else "—"
            )
            lines.append(
                f"| {p.t} | {p.kinetic_energy:.4f} | {p.potential_energy:.4f} | "
                f"{vals} | {band} |"
            )
        lines.append("")
        if self.predictions:
            lines.append("## Next-step predictions")
            lines.append("")
            for k, v in self.predictions.items():
                lines.append(f"- `{k}`: {v}")
            lines.append("")
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


@dataclass
class PhysicalInsight:
    source: str
    target: str
    mechanism: str
    domain: str
    analogy_label: str  # literal | analogy — careful labeling for non-physics domains
    confidence: float
    evidence: str = ""
    trajectory_signal: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicalGroundingReport:
    insights: list[PhysicalInsight]
    domain: str = "mechanics-lite"
    method: str = "physics_glossary+trajectory"
    glossary_hits: list[dict[str, Any]] = field(default_factory=list)
    merged_grounding: Optional[dict[str, Any]] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "method": self.method,
            "insights": [i.to_dict() for i in self.insights],
            "glossary_hits": self.glossary_hits,
            "merged_grounding": self.merged_grounding,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Physical insight grounding",
            "",
            f"**Domain:** `{self.domain}` · **Method:** `{self.method}`",
            "",
        ]
        if not self.insights:
            lines.append("_No physical insights._")
            lines.append("")
            return "\n".join(lines)
        lines.append("| edge | mechanism | domain | label | conf |")
        lines.append("|---|---|---|---|---:|")
        for i in self.insights:
            lines.append(
                f"| `{i.source}` → `{i.target}` | {i.mechanism} | {i.domain} | "
                f"{i.analogy_label} | {i.confidence:.2f} |"
            )
        lines.append("")
        for i in self.insights:
            if i.evidence or i.trajectory_signal:
                lines.append(
                    f"- **{i.source}→{i.target}:** {i.evidence} "
                    f"{i.trajectory_signal}".strip()
                )
        lines.append("")
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


@dataclass
class PhysicsLoopResult:
    """Full autocausal physics loop output."""

    trajectory: Trajectory
    physical_grounding: PhysicalGroundingReport
    discovery: Optional[dict[str, Any]] = None
    mining: Optional[dict[str, Any]] = None
    guide: Optional[dict[str, Any]] = None
    grounding: Optional[dict[str, Any]] = None
    source: str = ""
    horizon: int = 0
    backend: str = ""
    notes: list[str] = field(default_factory=list)
    second_pass: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "horizon": self.horizon,
            "backend": self.backend,
            "second_pass": self.second_pass,
            "mining": self.mining,
            "discovery": self.discovery,
            "trajectory": self.trajectory.to_dict(),
            "physical_grounding": self.physical_grounding.to_dict(),
            "guide": self.guide,
            "grounding": self.grounding,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Autocausal physics loop",
            "",
            f"**Source:** `{self.source}` · **Horizon:** {self.horizon} · "
            f"**Backend:** `{self.backend}`",
            "",
        ]
        if self.notes:
            lines.append("## Pipeline notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        if self.mining:
            lines.append(
                f"## Mining — associations: "
                f"{len(self.mining.get('associations') or [])}"
            )
            lines.append("")
        if self.discovery:
            edges = self.discovery.get("edges") or []
            lines.append(f"## Discovery — {len(edges)} edge(s)")
            lines.append("")
            for e in edges[:12]:
                lines.append(
                    f"- `{e.get('source')}` → `{e.get('target')}` "
                    f"({e.get('type', '')}, conf={e.get('confidence', '')})"
                )
            lines.append("")
        lines.append(self.trajectory.to_markdown())
        lines.append(self.physical_grounding.to_markdown())
        if self.guide:
            lines.append("## Guide")
            lines.append("")
            lines.append(f"- Backend: `{self.guide.get('backend')}`")
            focus = self.guide.get("focus_columns") or []
            if focus:
                lines.append("- Focus: " + ", ".join(f"`{c}`" for c in focus[:12]))
            lines.append("")
        if self.grounding:
            claims = self.grounding.get("claims") or []
            lines.append(f"## Domain grounding — {len(claims)} claim(s)")
            lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()
