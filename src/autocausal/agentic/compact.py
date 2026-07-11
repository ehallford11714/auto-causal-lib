"""Compactor — ACON/ReSum-inspired lossy narrative + lossless handles.

Design inspiration (not a reimplementation):
- ACON (arXiv:2510.00615) — agent context compaction + distill toward smaller models
- ReSum-style summarization of long agent traces

Keeps a short prose narrative for the next round while preserving lossless
handles: edge ids, metrics, dataset ids, tool-trace refs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from autocausal.agentic.state import LoopState

__all__ = ["CompactBundle", "Compactor"]


@dataclass
class CompactBundle:
    """Result of one compaction step."""

    narrative: str
    handles: dict[str, Any] = field(default_factory=dict)
    dropped_chars: int = 0
    backend: str = "rule"  # rule | slm
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Compactor:
    """Compress round state into lossy prose + lossless handles.

    When ``use_slm=True``, attempts a soft SLM one-liner via the suite director
    path; always falls back to deterministic rule compaction.
    """

    def __init__(
        self,
        *,
        max_narrative_chars: int = 600,
        use_slm: bool = False,
        model_name: Optional[str] = None,
    ) -> None:
        self.max_narrative_chars = max(80, int(max_narrative_chars))
        self.use_slm = bool(use_slm)
        self.model_name = model_name

    def compact(self, state: LoopState) -> CompactBundle:
        handles = self._lossless_handles(state)
        rule_narrative = self._rule_narrative(state)
        backend = "rule"
        narrative = rule_narrative
        notes: list[str] = []

        if self.use_slm:
            slm_text = self._try_slm_summary(state, rule_narrative)
            if slm_text:
                narrative = slm_text
                backend = "slm"
                notes.append("SLM compaction used (soft).")
            else:
                notes.append("SLM compaction unavailable; rule narrative kept.")

        if len(narrative) > self.max_narrative_chars:
            dropped = len(narrative) - self.max_narrative_chars
            narrative = narrative[: self.max_narrative_chars - 1] + "…"
        else:
            dropped = 0

        bundle = CompactBundle(
            narrative=narrative,
            handles=handles,
            dropped_chars=dropped,
            backend=backend,
            notes=notes,
        )
        state.narrative = narrative
        state.handles = handles
        state.slm_backend = backend if self.use_slm else state.slm_backend
        return bundle

    def _lossless_handles(self, state: LoopState) -> dict[str, Any]:
        state.sync_edge_ids()
        trace_refs = []
        for i, t in enumerate(state.tool_traces[-12:]):
            trace_refs.append(
                {
                    "i": i,
                    "name": t.get("name") or t.get("tool") or "tool",
                    "ok": bool(t.get("ok", True)),
                }
            )
        return {
            "round": state.round,
            "edge_ids": list(state.edge_ids),
            "n_edges": len(state.edges),
            "metrics": dict(state.metrics),
            "dataset_ids": list(state.dataset_ids),
            "source": state.source,
            "tool_trace_refs": trace_refs,
            "hypothesis_ids": [h.id for h in state.hypotheses],
            "validation_ok": bool(state.validation.get("ok", True)),
            "route": state.route,
        }

    def _rule_narrative(self, state: LoopState) -> str:
        hyps = "; ".join(h.statement for h in state.hypotheses[:3]) or "none"
        edges = ", ".join(state.edge_ids[:6]) or "none"
        n_ok = sum(1 for t in state.tool_traces if t.get("ok", True))
        n_fail = len(state.tool_traces) - n_ok
        val = "ok" if state.validation.get("ok", True) else "issues"
        mem_n = len(state.retrieved_memories)
        parts = [
            f"Round {state.round}: hypotheses=[{hyps}].",
            f"Edges=[{edges}] (n={len(state.edges)}).",
            f"Tools ok={n_ok} fail={n_fail}; validation={val}.",
        ]
        if mem_n:
            parts.append(f"Retrieved {mem_n} prior memories.")
        if state.insight_summary:
            parts.append(f"Insight: {state.insight_summary[:160]}")
        if state.experiments:
            titles = [str(e.get("title") or e.get("name") or "?") for e in state.experiments[:3]]
            parts.append("Experiments: " + ", ".join(titles) + ".")
        return " ".join(parts)

    def _try_slm_summary(self, state: LoopState, fallback: str) -> Optional[str]:
        """Soft SLM one-liner; never raises."""
        try:
            from autocausal.suites.director import SLMAutoDirector

            director = SLMAutoDirector(use_slm=True, model_name=self.model_name)
            # Prefer a lightweight prompt via director internals if available
            prompt = (
                "Summarize this causal agent round in one short paragraph. "
                "Exploratory only, not identification.\n" + fallback
            )
            if hasattr(director, "summarize"):
                out = director.summarize(prompt)  # type: ignore[attr-defined]
                if out:
                    return str(out)[: self.max_narrative_chars]
            # Fallback: try guide-style soft call via resolve
            if hasattr(director, "direct"):
                # No frame required for narrative — skip if API needs df
                pass
            # Attempt HF generate via soft slm module
            try:
                from autocausal import slm as _slm

                if hasattr(_slm, "generate") or hasattr(_slm, "complete"):
                    gen = getattr(_slm, "generate", None) or getattr(_slm, "complete", None)
                    if callable(gen):
                        text = gen(prompt, max_new_tokens=80)
                        if text:
                            return str(text).strip()[: self.max_narrative_chars]
            except Exception:
                pass
            # Last soft path: ExperimentRecommender / insight narrator unused —
            # return None so rule narrative is kept
            _ = state
            return None
        except Exception:
            return None
