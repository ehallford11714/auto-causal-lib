"""GRAIL report types — library-first, epistemically labeled.

Origin: Kineteq GRAIL (Generative Reflective Agentic Imputation Loop).
The offline stub produces the same *shape* of artifacts; it is **not** the
live Kineteq LM/MCP implementation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


EPISTEMIC = (
    "GRAIL outputs are exploratory reasoning scaffolds for AutoCausal "
    "(goal enrichment, expert-chain composition, reflective cycles). "
    "They are not causal identification. Offline stub ≠ live Kineteq GRAIL."
)


@dataclass
class Assumption:
    """Declared imputation / assumption from grail_impute-style audit."""

    parameter: str
    value: Any
    confidence: float = 0.5
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImputationAudit:
    """Self-imputation audit (Kineteq ``grail_impute`` analog)."""

    original_goal: str
    enriched_goal: str
    assumptions: list[Assumption] = field(default_factory=list)
    underspecified: list[str] = field(default_factory=list)
    domain: str = "causal"
    backend: str = "grail_stub"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_goal": self.original_goal,
            "enriched_goal": self.enriched_goal,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "underspecified": list(self.underspecified),
            "domain": self.domain,
            "backend": self.backend,
            "notes": list(self.notes),
        }


@dataclass
class ExpertStep:
    """One step in a composed expert reasoning chain (``grail_compose``)."""

    step: int
    role: str
    prompt: str
    charges: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExpertChain:
    """Dense expert prompt chain + optional mutation prompt."""

    goal: str
    steps: list[ExpertStep] = field(default_factory=list)
    mutation_prompt: str = ""
    chain_length: int = 0
    backend: str = "grail_stub"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "mutation_prompt": self.mutation_prompt,
            "chain_length": self.chain_length or len(self.steps),
            "backend": self.backend,
            "notes": list(self.notes),
        }


@dataclass
class FoldDiagnosis:
    """Lagrangian-of-intelligence style fold summary (``grail_fold`` analog).

    Offline stub computes a lightweight T/V proxy from chain charges —
    not the full Fisher-weighted Kineteq fold.
    """

    action_s: float = 0.0
    kinetic_t: float = 0.0
    potential_v: float = 0.0
    per_step: list[dict[str, Any]] = field(default_factory=list)
    directive: str = ""
    backend: str = "grail_stub"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CycleTrace:
    """One reflective cycle inside ``grail_run``."""

    cycle: int
    reflection: str
    verdict: str
    answer_delta: str = ""
    memory_keys: list[str] = field(default_factory=list)
    graph_hits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphMemoryNode:
    """Episodic / graph memory node for retrieval steps."""

    key: str
    kind: str  # edge | column | assumption | query | cycle
    content: str
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GrailReport:
    """Unified GRAIL run report for AutoCausal hooks."""

    goal: str
    domain: str = "causal"
    backend: str = "grail_stub"
    live_kineteq: bool = False
    imputation: Optional[ImputationAudit] = None
    chain: Optional[ExpertChain] = None
    fold: Optional[FoldDiagnosis] = None
    cycles: list[CycleTrace] = field(default_factory=list)
    final_answer: str = ""
    genome_id: str = ""
    memory: list[GraphMemoryNode] = field(default_factory=list)
    focus_columns: list[str] = field(default_factory=list)
    next_questions: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    boost_edges: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    epistemic: str = EPISTEMIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "domain": self.domain,
            "backend": self.backend,
            "live_kineteq": self.live_kineteq,
            "imputation": self.imputation.to_dict() if self.imputation else None,
            "chain": self.chain.to_dict() if self.chain else None,
            "fold": self.fold.to_dict() if self.fold else None,
            "cycles": [c.to_dict() for c in self.cycles],
            "final_answer": self.final_answer,
            "genome_id": self.genome_id,
            "memory": [m.to_dict() for m in self.memory],
            "focus_columns": list(self.focus_columns),
            "next_questions": list(self.next_questions),
            "search_queries": list(self.search_queries),
            "boost_edges": list(self.boost_edges),
            "notes": list(self.notes),
            "epistemic": self.epistemic,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# GRAIL report",
            "",
            f"**Backend:** `{self.backend}`"
            + (" (live Kineteq)" if self.live_kineteq else " (offline / soft)"),
            f"**Domain:** `{self.domain}`",
            f"**Goal:** {self.goal}",
            "",
            f"> {self.epistemic}",
            "",
        ]
        if self.imputation:
            lines += ["## Imputation audit", ""]
            lines.append(f"**Enriched goal:** {self.imputation.enriched_goal}")
            lines.append("")
            if self.imputation.underspecified:
                lines.append(
                    "**Underspecified:** "
                    + ", ".join(f"`{u}`" for u in self.imputation.underspecified)
                )
                lines.append("")
            for a in self.imputation.assumptions:
                lines.append(
                    f"- `{a.parameter}` = `{a.value}` "
                    f"(conf={a.confidence:.2f}) — {a.rationale}"
                )
            lines.append("")
        if self.chain and self.chain.steps:
            lines += ["## Expert chain", ""]
            for s in self.chain.steps:
                lines.append(f"{s.step}. **{s.role}** — {s.prompt[:200]}")
            if self.chain.mutation_prompt:
                lines.append("")
                lines.append(f"*Mutation:* {self.chain.mutation_prompt[:240]}")
            lines.append("")
        if self.fold:
            lines += [
                "## Fold (T/V proxy)",
                "",
                f"- S={self.fold.action_s:.3f}  T={self.fold.kinetic_t:.3f}  "
                f"V={self.fold.potential_v:.3f}",
                f"- Directive: {self.fold.directive}",
                "",
            ]
        if self.cycles:
            lines += ["## Reflective cycles", ""]
            for c in self.cycles:
                lines.append(
                    f"- Cycle {c.cycle}: **{c.verdict}** — {c.reflection[:180]}"
                )
            lines.append("")
        if self.final_answer:
            lines += ["## Final answer", "", self.final_answer, ""]
        if self.focus_columns:
            lines += ["## Focus columns", ""] + [
                f"- `{c}`" for c in self.focus_columns
            ] + [""]
        if self.next_questions:
            lines += ["## Next questions", ""] + [
                f"- {q}" for q in self.next_questions
            ] + [""]
        if self.memory:
            lines += ["## Graph / memory", ""]
            for m in self.memory[:12]:
                lines.append(f"- [{m.kind}] `{m.key}` ({m.score:.2f}): {m.content[:120]}")
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)
