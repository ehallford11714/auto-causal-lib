"""Hypothesized behavioral causal edges (exploratory, with caveats)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from autocausal.behavioral.schema import CAVEAT, TraceCollection


@dataclass
class BehavioralEdge:
    """A hypothesized stimulus→response or habit→outcome edge."""

    source: str
    target: str
    kind: str  # stimulus_response | habit_outcome | exposure_compliance | reward_response
    score: float
    evidence: str = ""
    exploratory: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "score": self.score,
            "evidence": self.evidence,
            "exploratory": self.exploratory,
        }


@dataclass
class BehavioralReport:
    """Report of hypothesized behavioral causal structure."""

    trace_name: str
    edges: list[BehavioralEdge] = field(default_factory=list)
    discovery_edges: list[dict[str, Any]] = field(default_factory=list)
    mining: Optional[dict[str, Any]] = None
    panel_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    caveat: str = CAVEAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_name": self.trace_name,
            "edges": [e.to_dict() for e in self.edges],
            "discovery_edges": list(self.discovery_edges),
            "mining": self.mining,
            "panel_summary": dict(self.panel_summary),
            "notes": list(self.notes),
            "caveat": self.caveat,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Behavioral traces: `{self.trace_name}`",
            "",
            f"> {self.caveat}",
            "",
            "## Hypothesized edges",
            "",
        ]
        if not self.edges:
            lines.append("_No hypothesized edges._")
        else:
            lines.append("| source | target | kind | score | evidence |")
            lines.append("|---|---|---|---|---|")
            for e in self.edges:
                lines.append(
                    f"| {e.source} | {e.target} | {e.kind} | {e.score:.3f} | {e.evidence} |"
                )
        if self.discovery_edges:
            lines.extend(["", "## Discover edges (exploratory)", ""])
            for d in self.discovery_edges[:20]:
                lines.append(
                    f"- `{d.get('source')}` → `{d.get('target')}` "
                    f"(score={d.get('score')}, type={d.get('type')})"
                )
        if self.notes:
            lines.extend(["", "## Notes", ""])
            for n in self.notes:
                lines.append(f"- {n}")
        return "\n".join(lines) + "\n"

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, default=str)


def hypothesize_edges(
    panel: pd.DataFrame,
    *,
    collection: Optional[TraceCollection] = None,
) -> list[BehavioralEdge]:
    """Derive simple hypothesized edges from panel correlations / domain priors."""
    edges: list[BehavioralEdge] = []
    name = collection.name if collection is not None else "traces"

    def _corr(a: str, b: str) -> Optional[float]:
        if a not in panel.columns or b not in panel.columns:
            return None
        s = panel[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(s) < 3:
            return None
        if float(s[a].std()) == 0.0 or float(s[b].std()) == 0.0:
            return None
        c = float(s[a].corr(s[b]))
        if pd.isna(c):
            return None
        return c

    pairs = [
        ("habit_strength", "outcome", "habit_outcome", "habit strength proxy vs distal outcome"),
        ("compliance_rate", "outcome", "exposure_compliance", "compliance vs outcome"),
        ("mean_reward", "mean_response", "reward_response", "mean reward vs mean response"),
        ("exposure_count", "compliance_rate", "exposure_compliance", "exposure vs compliance"),
        ("mean_response", "outcome", "stimulus_response", "response rate vs outcome"),
    ]
    for src, tgt, kind, evidence in pairs:
        c = _corr(src, tgt)
        if c is None:
            continue
        edges.append(
            BehavioralEdge(
                source=src,
                target=tgt,
                kind=kind,
                score=round(abs(c), 4),
                evidence=f"{evidence} (corr={c:.3f})",
            )
        )

    # Domain priors from demo name
    if collection is not None:
        if "habit" in name:
            edges.append(
                BehavioralEdge(
                    source="action(cue)",
                    target="response(routine)",
                    kind="stimulus_response",
                    score=0.5,
                    evidence="habit-loop prior: cue → routine (exploratory)",
                )
            )
            edges.append(
                BehavioralEdge(
                    source="habit_strength",
                    target="outcome",
                    kind="habit_outcome",
                    score=0.5,
                    evidence="habit-loop prior: habit → outcome (exploratory)",
                )
            )
        if "nudge" in name:
            edges.append(
                BehavioralEdge(
                    source="action(nudge)",
                    target="response(comply)",
                    kind="stimulus_response",
                    score=0.5,
                    evidence="nudge A/B prior: arm → compliance (exploratory)",
                )
            )
        if "reinforcement" in name:
            edges.append(
                BehavioralEdge(
                    source="reward",
                    target="response",
                    kind="reward_response",
                    score=0.5,
                    evidence="reinforcement prior: reward → response rate (exploratory)",
                )
            )

    # Deduplicate by (source, target, kind)
    seen: set[tuple[str, str, str]] = set()
    uniq: list[BehavioralEdge] = []
    for e in edges:
        key = (e.source, e.target, e.kind)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq


def build_behavioral_report(
    panel: pd.DataFrame,
    *,
    collection: Optional[TraceCollection] = None,
    discovery_edges: Optional[list[dict[str, Any]]] = None,
    mining: Optional[dict[str, Any]] = None,
    notes: Optional[list[str]] = None,
) -> BehavioralReport:
    """Assemble a BehavioralReport from panel + optional discover/mine outputs."""
    from autocausal.behavioral.features import feature_summary

    edges = hypothesize_edges(panel, collection=collection)
    return BehavioralReport(
        trace_name=collection.name if collection is not None else "traces",
        edges=edges,
        discovery_edges=list(discovery_edges or []),
        mining=mining,
        panel_summary=feature_summary(panel),
        notes=list(notes or []) + [CAVEAT],
    )


__all__ = [
    "BehavioralEdge",
    "BehavioralReport",
    "hypothesize_edges",
    "build_behavioral_report",
]
