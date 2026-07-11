"""LangGraph / FSM SLM chain for AutoCausal exploratory analysis.

Library-first cyclic chain::

    hypothesize/guide(SLM|Qwen) → skill/tools → validate/discover
    → compact → insight report → route

Uses ``langgraph`` when installed (base dependency); soft-falls back to
``graph_runtime.GraphRuntime`` FSM otherwise.

Epistemic: guides analysis — does **not** identify causation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from autocausal.agentic.graph_runtime import NODE_ORDER, GraphRuntime, langgraph_available
from autocausal.agentic.loop import AgenticCausalLoop, run_agentic_loop
from autocausal.agentic.report import AgenticLoopReport

__all__ = [
    "SLM_CHAIN_NODES",
    "SLMLangGraphChain",
    "SLMChainReport",
    "run_slm_langgraph_loop",
    "langgraph_available",
]

# Extended node labels for docs / CLI (maps onto AgenticCausalLoop nodes).
SLM_CHAIN_NODES: tuple[str, ...] = (
    "hypothesize_guide",  # → hypothesize (+ SLM guide)
    "skill_tools",  # → skill
    "validate_discover",  # → validate
    "compact",
    "insight_report",  # folded into validate/compact narrative
    "route",
)


@dataclass
class SLMChainReport:
    """Wrapper around ``AgenticLoopReport`` with chain metadata."""

    agentic: AgenticLoopReport
    chain_backend: str = "fsm"
    model_name: Optional[str] = None
    insight_markdown: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.agentic.to_dict()
        d["chain_backend"] = self.chain_backend
        d["model_name"] = self.model_name
        d["insight_markdown"] = self.insight_markdown
        d["chain_notes"] = list(self.notes)
        d["epistemic"] = (
            "SLM/LangGraph chain guides exploratory analysis; "
            "not causal identification."
        )
        return d

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# AutoCausal SLM LangGraph chain report",
            "",
            f"**Chain backend:** `{self.chain_backend}`  ",
            f"**Model:** `{self.model_name or 'rule/default'}`  ",
            "",
            "> Epistemic: SLM guides analysis — **not** causal identification.",
            "",
        ]
        if self.notes:
            lines += ["## Chain notes", ""] + [f"- {n}" for n in self.notes] + [""]
        lines.append(self.agentic.to_markdown())
        if self.insight_markdown:
            lines += ["", "## Insight excerpt", "", self.insight_markdown[:4000]]
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


class SLMLangGraphChain:
    """Build and run the SLM-guided LangGraph/FSM causal chain.

    Prefer::

        from autocausal.agentic import run_slm_langgraph_loop
        report = run_slm_langgraph_loop(df, use_slm=True, max_rounds=2)
    """

    def __init__(
        self,
        *,
        use_slm: bool = True,
        model_name: Optional[str] = None,
        max_rounds: int = 2,
        persist_dir: Optional[Union[str, Path]] = None,
        prefer_langgraph: bool = True,
        ensure_qwen: bool = False,
    ) -> None:
        self.use_slm = use_slm
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.persist_dir = persist_dir
        self.prefer_langgraph = prefer_langgraph
        self.ensure_qwen = ensure_qwen
        self.last_report: Optional[SLMChainReport] = None
        self._setup_notes: list[str] = []

    def _maybe_setup_qwen(self) -> None:
        if not self.ensure_qwen and not self.model_name:
            return
        try:
            from autocausal.slm import ensure_local_qwen

            res = ensure_local_qwen(
                model_id=self.model_name,
                download=bool(self.ensure_qwen),
                set_env=True,
            )
            self.model_name = self.model_name or res.get("model_id")
            self._setup_notes.extend(res.get("notes") or [])
            if not res.get("ok") and self.ensure_qwen:
                self._setup_notes.append(
                    "Qwen download soft-failed — continuing with rule/available HF cache."
                )
        except Exception as e:
            self._setup_notes.append(f"ensure_local_qwen soft-fail: {type(e).__name__}")

    def run(
        self,
        source: Any = None,
        *,
        ac: Any = None,
        text: str = "",
        max_rounds: Optional[int] = None,
        use_slm: Optional[bool] = None,
        **kwargs: Any,
    ) -> SLMChainReport:
        if use_slm is not None:
            self.use_slm = bool(use_slm)
        self._maybe_setup_qwen()

        rounds = int(max_rounds if max_rounds is not None else self.max_rounds)
        loop = AgenticCausalLoop(
            use_slm=self.use_slm,
            model_name=self.model_name,
            max_rounds=rounds,
            persist_dir=self.persist_dir,
            prefer_langgraph=self.prefer_langgraph,
        )
        agentic = loop.run(
            source,
            ac=ac,
            text=text,
            max_rounds=rounds,
            use_slm=self.use_slm,
            **kwargs,
        )

        insight_md = ""
        try:
            from autocausal.insight import InsightSuite

            target_ac = ac or loop.ac
            if target_ac is not None:
                suite = InsightSuite.from_autocausal(
                    target_ac,
                    use_slm=self.use_slm,
                    model_name=self.model_name,
                )
                irep = suite.run(text=text or "", use_slm=self.use_slm)
                insight_md = irep.to_markdown() if hasattr(irep, "to_markdown") else ""
                if getattr(irep, "summary", None) and not agentic.summary:
                    agentic.summary = irep.summary
        except Exception as e:
            self._setup_notes.append(f"insight stage soft-fail: {type(e).__name__}: {e}")

        backend = agentic.runtime_backend or (
            "langgraph" if (self.prefer_langgraph and langgraph_available()) else "fsm"
        )
        notes = list(self._setup_notes) + [
            "Chain nodes: " + " → ".join(SLM_CHAIN_NODES),
            "Underlying runtime nodes: " + " → ".join(NODE_ORDER),
            "Epistemic: not causal identification.",
        ]
        report = SLMChainReport(
            agentic=agentic,
            chain_backend=backend,
            model_name=self.model_name or agentic.slm_backend,
            insight_markdown=insight_md,
            notes=notes,
        )
        self.last_report = report
        return report


def run_slm_langgraph_loop(
    source: Any = None,
    *,
    ac: Any = None,
    text: str = "",
    max_rounds: int = 2,
    use_slm: bool = True,
    model_name: Optional[str] = None,
    persist_dir: Optional[Union[str, Path]] = None,
    prefer_langgraph: bool = True,
    ensure_qwen: bool = False,
    **kwargs: Any,
) -> SLMChainReport:
    """One-shot SLM LangGraph/FSM chain → ``SLMChainReport``.

    Soft-falls to rule SLM + FSM when HF/langgraph unavailable.
    """
    # Accept DataFrame paths via AgenticCausalLoop helpers
    if isinstance(source, pd.DataFrame) or source is not None or ac is not None:
        chain = SLMLangGraphChain(
            use_slm=use_slm,
            model_name=model_name,
            max_rounds=max_rounds,
            persist_dir=persist_dir,
            prefer_langgraph=prefer_langgraph,
            ensure_qwen=ensure_qwen,
        )
        return chain.run(
            source,
            ac=ac,
            text=text,
            max_rounds=max_rounds,
            use_slm=use_slm,
            **kwargs,
        )
    # Empty call still returns via agentic helper for symmetry
    agentic = run_agentic_loop(
        source,
        ac=ac,
        text=text,
        max_rounds=max_rounds,
        use_slm=use_slm,
        model_name=model_name,
        persist_dir=persist_dir,
        prefer_langgraph=prefer_langgraph,
        **kwargs,
    )
    return SLMChainReport(
        agentic=agentic,
        chain_backend=agentic.runtime_backend,
        model_name=model_name,
        notes=["Empty source path — agentic fallback."],
    )
