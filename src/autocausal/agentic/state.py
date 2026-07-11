"""LoopState — shared mutable state for the agentic causal FSM.

Inspired by StateFlow-style cyclic orchestration (arXiv:2403.11322) and
LangGraph-style node handoff — this is a library-owned state bag, not a
paper reimplementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

RouteDecision = Literal["continue", "stop"]

__all__ = ["LoopState", "Hypothesis", "RouteDecision", "EPISTEMIC"]

EPISTEMIC = (
    "Agentic AutoCausal loops are exploratory assistance — not causal identification. "
    "Hypotheses, edges, and SLM text are candidates for human review."
)


@dataclass
class Hypothesis:
    """One candidate causal / experimental hypothesis for the current round."""

    id: str
    statement: str
    source: str = "rule"  # rule | slm | insight | grail | vector
    priority: float = 0.5
    related_edges: list[str] = field(default_factory=list)
    related_tools: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoopState:
    """Mutable state carried across cyclic agentic nodes.

    Fields map roughly to the architecture cycle::

        hypothesize → skill/tool → validate → compact → persist → route
    """

    round: int = 0
    max_rounds: int = 3
    text: str = ""
    source: str = ""
    dataset_ids: list[str] = field(default_factory=list)

    hypotheses: list[Hypothesis] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    tool_traces: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    handles: dict[str, Any] = field(default_factory=dict)

    route: RouteDecision = "continue"
    stop_reason: str = ""
    notes: list[str] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    node_history: list[str] = field(default_factory=list)
    round_history: list[dict[str, Any]] = field(default_factory=list)

    use_slm: bool = False
    slm_backend: str = "rule"
    insight_summary: str = ""
    experiments: list[dict[str, Any]] = field(default_factory=list)
    retrieved_memories: list[dict[str, Any]] = field(default_factory=list)
    epistemic: str = EPISTEMIC

    # Opaque handle to AutoCausal instance (not serialized)
    _ac: Any = field(default=None, repr=False, compare=False)

    def edge_id(self, edge: dict[str, Any]) -> str:
        src = str(edge.get("source") or edge.get("from") or "?")
        tgt = str(edge.get("target") or edge.get("to") or "?")
        return f"{src}->{tgt}"

    def sync_edge_ids(self) -> None:
        self.edge_ids = [self.edge_id(e) for e in self.edges]

    def record_node(self, name: str) -> None:
        self.node_history.append(name)
        self.stages.append(f"r{self.round}:{name}")

    def snapshot_round(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "n_hypotheses": len(self.hypotheses),
            "n_edges": len(self.edges),
            "edge_ids": list(self.edge_ids),
            "n_tool_traces": len(self.tool_traces),
            "validation_ok": bool(self.validation.get("ok", True)),
            "route": self.route,
            "stop_reason": self.stop_reason,
            "narrative": self.narrative[:500],
            "metrics": dict(self.metrics),
            "slm_backend": self.slm_backend,
            "n_experiments": len(self.experiments),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "max_rounds": self.max_rounds,
            "text": self.text,
            "source": self.source,
            "dataset_ids": list(self.dataset_ids),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "edges": list(self.edges),
            "edge_ids": list(self.edge_ids),
            "metrics": dict(self.metrics),
            "tool_traces": list(self.tool_traces),
            "validation": dict(self.validation),
            "narrative": self.narrative,
            "handles": dict(self.handles),
            "route": self.route,
            "stop_reason": self.stop_reason,
            "notes": list(self.notes),
            "stages": list(self.stages),
            "node_history": list(self.node_history),
            "round_history": list(self.round_history),
            "use_slm": self.use_slm,
            "slm_backend": self.slm_backend,
            "insight_summary": self.insight_summary,
            "experiments": list(self.experiments),
            "retrieved_memories": list(self.retrieved_memories),
            "epistemic": self.epistemic,
        }
