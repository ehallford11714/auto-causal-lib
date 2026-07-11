"""AgenticLoopReport — structured output of the SLM-guided agentic causal loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from autocausal.agentic.state import EPISTEMIC

__all__ = ["AgenticLoopReport", "CAVEATS"]

CAVEATS = [
    "Agentic AutoCausal loops are exploratory — not causal identification.",
    "Compaction narratives are lossy; trust lossless handles (edge ids, metrics) for audit.",
    "Vector memory retrieval is heuristic similarity, not causal evidence.",
    "Optional SLM guidance is generative assistance; rule policy always works offline.",
    "Soft LangGraph / chromadb / faiss backends never hard-crash when missing.",
]


@dataclass
class AgenticLoopReport:
    """First-class report from ``AgenticCausalLoop`` / ``run_agentic_loop``."""

    summary: str = ""
    narrative: str = ""
    handles: dict[str, Any] = field(default_factory=dict)
    key_edges: list[dict[str, Any]] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    experiments: list[dict[str, Any]] = field(default_factory=list)
    round_history: list[dict[str, Any]] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    node_history: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    vector_hits: list[dict[str, Any]] = field(default_factory=list)
    persist_path: Optional[str] = None
    runtime_backend: str = "fsm"
    slm_used: bool = False
    slm_backend: str = "rule"
    source: str = ""
    n_rows: int = 0
    n_cols: int = 0
    n_rounds: int = 0
    stop_reason: str = ""
    notes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=lambda: list(CAVEATS))
    epistemic: str = EPISTEMIC
    design_refs: list[str] = field(
        default_factory=lambda: [
            "ACON arXiv:2510.00615 (compaction inspiration)",
            "MEM1 arXiv:2506.15841 / A-MEM arXiv:2502.12110 (memory budget)",
            "StateFlow arXiv:2403.11322 (cyclic FSM inspiration)",
            "HippoRAG arXiv:2405.14831 / Mem0 arXiv:2504.19413 (vector+graph LTM)",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "narrative": self.narrative,
            "handles": dict(self.handles),
            "key_edges": list(self.key_edges),
            "edge_ids": list(self.edge_ids),
            "hypotheses": list(self.hypotheses),
            "experiments": list(self.experiments),
            "round_history": list(self.round_history),
            "stages": list(self.stages),
            "node_history": list(self.node_history),
            "metrics": dict(self.metrics),
            "validation": dict(self.validation),
            "memory": dict(self.memory),
            "vector_hits": list(self.vector_hits),
            "persist_path": self.persist_path,
            "runtime_backend": self.runtime_backend,
            "slm_used": self.slm_used,
            "slm_backend": self.slm_backend,
            "source": self.source,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_rounds": self.n_rounds,
            "stop_reason": self.stop_reason,
            "notes": list(self.notes),
            "caveats": list(self.caveats),
            "epistemic": self.epistemic,
            "design_refs": list(self.design_refs),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Agentic Causal Loop Report",
            "",
            f"> {self.epistemic}",
            "",
            f"**Source:** `{self.source}`  ",
            f"**Rounds:** {self.n_rounds}  ",
            f"**Runtime:** `{self.runtime_backend}`  ",
            f"**SLM:** used={self.slm_used} backend=`{self.slm_backend}`  ",
            f"**Stop:** {self.stop_reason or '(n/a)'}",
            "",
            "## Summary",
            "",
            self.summary or self.narrative or "(empty)",
            "",
            "## Narrative (lossy compaction)",
            "",
            self.narrative or "(none)",
            "",
            "## Lossless handles",
            "",
            "```json",
            json.dumps(self.handles, indent=2, default=str)[:2000],
            "```",
            "",
            f"## Edges ({len(self.edge_ids)})",
            "",
        ]
        for eid in self.edge_ids[:20]:
            lines.append(f"- `{eid}`")
        if not self.edge_ids:
            lines.append("- (none)")
        lines += ["", "## Hypotheses", ""]
        for h in self.hypotheses[:10]:
            lines.append(f"- [{h.get('source', '?')}] {h.get('statement', '')}")
        if not self.hypotheses:
            lines.append("- (none)")
        lines += ["", "## Stages", ""]
        lines.append(", ".join(self.stages[-40:]) or "(none)")
        lines += ["", "## Caveats", ""]
        for c in self.caveats:
            lines.append(f"- {c}")
        lines += ["", "## Design inspiration (not paper clones)", ""]
        for r in self.design_refs:
            lines.append(f"- {r}")
        if self.notes:
            lines += ["", "## Notes", ""]
            for n in self.notes[:20]:
                lines.append(f"- {n}")
        lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()

    def save(self, path: Union[str, Path], *, fmt: str = "json") -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "markdown" or str(p).endswith(".md"):
            p.write_text(self.to_markdown(), encoding="utf-8")
        else:
            p.write_text(self.to_json(), encoding="utf-8")
        return p

    def write(self, path: Union[str, Path], *, fmt: str = "json") -> Path:
        """Alias for ``save`` — matches InsightReport / suite ``write`` naming."""
        return self.save(path, fmt=fmt)
