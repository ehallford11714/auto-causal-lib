"""AgenticCausalLoop — SLM-guided cyclic causal research loop.

Architecture cycle (library-owned, SOTA-inspired)::

    hypothesize → skill/tool → validate → compact → persist → route

Reuses existing AutoCausal surfaces:
- ``insight`` research helpers / ExperimentRecommender
- ``skilling.SLMToolBroker`` for tool invocation
- ``suites`` director soft SLM
- optional GRAIL step when present

Soft deps only: langgraph, chromadb, faiss, HF SLM — never hard-crash.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union
from uuid import uuid4

import pandas as pd

from autocausal.agentic.compact import Compactor
from autocausal.agentic.graph_runtime import NODE_ORDER, GraphRuntime
from autocausal.agentic.memory import AgentMemory, MemoryItem
from autocausal.agentic.persist import EpisodeStore
from autocausal.agentic.report import AgenticLoopReport
from autocausal.agentic.state import EPISTEMIC, Hypothesis, LoopState
from autocausal.agentic.vector_memory import VectorStoreMemory, make_vector_memory

__all__ = [
    "AgenticCausalLoop",
    "run_agentic_loop",
    "EPISTEMIC",
]


def _env_slm() -> bool:
    import os

    for n in ("AUTOCAUSAL_SLM", "EMOTIVEVISION_SLM", "CAUSALIV_SLM"):
        if os.environ.get(n, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _as_autocausal(source: Any = None, *, existing: Any = None) -> Any:
    from autocausal.api import AutoCausal

    if existing is not None:
        return existing
    if isinstance(source, AutoCausal):
        return source
    if isinstance(source, pd.DataFrame):
        return AutoCausal.from_dataframe(source)
    if source is None:
        raise ValueError("Provide source (path/DataFrame/AutoCausal) or ac=")
    path = Path(str(source))
    if path.suffix.lower() == ".parquet":
        return AutoCausal.from_parquet(path)
    return AutoCausal.from_csv(path)


def _edges_from_ac(ac: Any) -> list[dict[str, Any]]:
    result = getattr(ac, "result", None)
    if result is None:
        return []
    edges = getattr(result, "edges", None) or []
    out: list[dict[str, Any]] = []
    for e in edges:
        if hasattr(e, "to_dict"):
            out.append(e.to_dict())
        elif isinstance(e, dict):
            out.append(dict(e))
        else:
            out.append(
                {
                    "source": getattr(e, "source", None),
                    "target": getattr(e, "target", None),
                    "score": getattr(e, "score", None),
                }
            )
    return out


def _candidates_from_ac(ac: Any) -> dict[str, Any]:
    result = getattr(ac, "result", None)
    if result is None:
        return {}
    c = getattr(result, "candidates", None)
    if c is None:
        return {}
    if hasattr(c, "to_dict"):
        return c.to_dict()
    if isinstance(c, dict):
        return dict(c)
    return {}


class AgenticCausalLoop:
    """SLM-guided agentic causal loop with compaction + constant-budget memory.

    Example::

        from autocausal.agentic import AgenticCausalLoop
        from autocausal import load_dataset

        loop = AgenticCausalLoop(use_slm=False, persist_dir=".autocausal_agentic")
        report = loop.run(load_dataset("iris"), max_rounds=2, text="petal drivers")
        print(report.to_markdown())
    """

    def __init__(
        self,
        *,
        use_slm: bool = False,
        model_name: Optional[str] = None,
        max_rounds: int = 3,
        persist_dir: Optional[Union[str, Path]] = None,
        vector_backend: str = "auto",
        prefer_langgraph: bool = True,
        memory_max_episodes: int = 16,
        invoke_tools: Optional[Sequence[str]] = None,
    ) -> None:
        self.use_slm = bool(use_slm) or _env_slm()
        self.model_name = model_name
        self.max_rounds = max(1, int(max_rounds))
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.prefer_langgraph = prefer_langgraph
        self.invoke_tools = list(
            invoke_tools
            or (
                "autocleanse.profile_missingness",
                "autoeda.summarize_distributions",
                "automine.mine_associations",
            )
        )
        self.memory = AgentMemory(max_episodes=memory_max_episodes)
        self.vectors = make_vector_memory(backend=vector_backend)
        self.compactor = Compactor(use_slm=self.use_slm, model_name=model_name)
        self.last_report: Optional[AgenticLoopReport] = None
        self.last_state: Optional[LoopState] = None
        self.ac: Any = None
        self._episode_store: Optional[EpisodeStore] = None
        if self.persist_dir is not None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._episode_store = EpisodeStore.open(self.persist_dir / "episodes.jsonl")

    # ------------------------------------------------------------------ nodes

    def _node_hypothesize(self, state: LoopState) -> LoopState:
        ac = state._ac
        hyps: list[Hypothesis] = []

        # Retrieve prior vector memories (HippoRAG/Mem0-inspired soft LTM)
        query = state.text or state.narrative or state.insight_summary or "causal edges"
        try:
            hits = self.vectors.query(query, k=5)
            state.retrieved_memories = hits
            for h in hits[:3]:
                hyps.append(
                    Hypothesis(
                        id=f"vec-{uuid4().hex[:6]}",
                        statement=f"Prior memory: {h.get('text', '')[:200]}",
                        source="vector",
                        priority=float(h.get("score") or 0.3),
                        meta={"vector_id": h.get("id")},
                    )
                )
        except Exception as e:
            state.notes.append(f"vector query soft-fail: {type(e).__name__}")

        # Episodic context
        ctx = self.memory.episodic.as_context(max_chars=600)
        if ctx:
            self.memory.working.add(
                MemoryItem.make(ctx, kind="working", round=state.round, score=0.4)
            )

        # Ensure mine/discover for edge-based hypotheses
        try:
            if getattr(ac, "result", None) is None:
                ac.mine()
                if ac.df.isna().any().any():
                    ac.impute(method="auto")
                ac.discover(use_iv=False, min_abs_corr=0.15, qc="off")
            edges = _edges_from_ac(ac)
            state.edges = edges
            state.sync_edge_ids()
            for e in edges[:5]:
                eid = state.edge_id(e)
                score = float(e.get("score") or e.get("weight") or 0.5)
                hyps.append(
                    Hypothesis(
                        id=f"edge-{uuid4().hex[:6]}",
                        statement=f"Candidate edge {eid} (score={score:.3f})",
                        source="rule",
                        priority=score,
                        related_edges=[eid],
                    )
                )
        except Exception as e:
            state.notes.append(f"hypothesize discover soft-fail: {type(e).__name__}: {e}")

        # Soft insight / experiment recommender
        try:
            from autocausal.insight.experiments import ExperimentRecommender

            rec = ExperimentRecommender(use_slm=state.use_slm, model_name=self.model_name)
            plan = rec.recommend(
                edges=state.edges,
                mining={},
                candidates=_candidates_from_ac(ac),
                text=state.text,
                round_index=state.round,
                max_rounds=state.max_rounds,
            )
            state.experiments = [r.to_dict() for r in plan.recommendations[:8]]
            state.slm_backend = getattr(plan, "backend", None) or state.slm_backend
            for r in plan.recommendations[:3]:
                hyps.append(
                    Hypothesis(
                        id=f"exp-{uuid4().hex[:6]}",
                        statement=str(getattr(r, "title", None) or r.to_dict().get("title") or "experiment"),
                        source="slm" if state.use_slm else "insight",
                        priority=float(getattr(r, "priority", None) or 0.5),
                        related_tools=list(getattr(r, "tools", None) or [])[:4],
                    )
                )
            if getattr(plan, "stop", False):
                state.notes.append(plan.stop_reason or "Recommender suggested stop.")
        except Exception as e:
            state.notes.append(f"experiment recommend soft-fail: {type(e).__name__}")

        # Soft GRAIL embellishment
        try:
            from autocausal.grail import insight_grail_step

            g = insight_grail_step(
                text=state.text or "agentic loop",
                context={"edges": state.edge_ids[:10], "round": state.round},
                max_cycles=1,
            )
            if isinstance(g, dict) and g:
                state.notes.append("GRAIL soft step attached.")
                hyps.append(
                    Hypothesis(
                        id=f"grail-{uuid4().hex[:6]}",
                        statement=str(g.get("summary") or g.get("note") or "GRAIL reflective step"),
                        source="grail",
                        priority=0.45,
                        meta={"grail": {k: g[k] for k in list(g)[:6]}},
                    )
                )
        except Exception:
            pass

        # Soft SLM narrative hypothesis
        if state.use_slm and state.text:
            hyps.append(
                Hypothesis(
                    id=f"slm-{uuid4().hex[:6]}",
                    statement=f"Investigate: {state.text[:180]}",
                    source="slm",
                    priority=0.55,
                )
            )

        if not hyps:
            hyps.append(
                Hypothesis(
                    id=f"default-{uuid4().hex[:6]}",
                    statement="Explore strongest associations and role candidates.",
                    source="rule",
                    priority=0.4,
                )
            )

        hyps.sort(key=lambda h: h.priority, reverse=True)
        state.hypotheses = hyps[:8]
        for h in state.hypotheses[:3]:
            self.memory.working.add(
                MemoryItem.make(
                    h.statement,
                    kind="working",
                    round=state.round,
                    score=h.priority,
                    handles={"hypothesis_id": h.id},
                )
            )
        state.metrics["n_hypotheses"] = len(state.hypotheses)
        return state

    def _node_skill(self, state: LoopState) -> LoopState:
        ac = state._ac
        df = ac.df
        traces: list[dict[str, Any]] = []

        try:
            from autocausal.skilling.broker import SLMToolBroker

            broker = SLMToolBroker(use_slm=state.use_slm, model_name=self.model_name)
            # Prefer tools named by hypotheses; else default invoke list
            wanted: list[str] = []
            for h in state.hypotheses:
                wanted.extend(h.related_tools)
            tools = [t for t in wanted if t] or list(self.invoke_tools)

            # Prefer skill-directed tool sequence when available
            if hasattr(broker, "select_tools"):
                try:
                    selected = broker.select_tools(
                        "skill:automine",
                        text=state.text,
                        context={"hypotheses": [h.statement for h in state.hypotheses[:3]]},
                    )
                    names = [
                        str(c.get("name") or c.get("tool") or "")
                        for c in (selected or [])
                        if isinstance(c, dict)
                    ]
                    names = [n for n in names if n]
                    if names:
                        tools = names[:5]
                except Exception:
                    pass

            for name in tools[:5]:
                try:
                    result = broker.invoke(name, df=df)
                    traces.append(
                        result.to_dict() if hasattr(result, "to_dict") else dict(result)
                    )
                except Exception as e:
                    traces.append({"name": name, "ok": False, "warnings": [str(e)]})
        except Exception as e:
            state.notes.append(f"skilling soft-fail: {type(e).__name__}: {e}")
            # Minimal offline fallback: mine + discover
            try:
                ac.mine()
                ac.discover(use_iv=False, min_abs_corr=0.15, qc="off")
                traces.append({"name": "fallback.mine_discover", "ok": True})
            except Exception as e2:
                traces.append(
                    {"name": "fallback.mine_discover", "ok": False, "warnings": [str(e2)]}
                )

        # Soft suite director pass
        try:
            from autocausal.suites.director import SLMAutoDirector, frame_profile

            director = SLMAutoDirector(use_slm=state.use_slm, model_name=self.model_name)
            profile = frame_profile(df)
            state.metrics["frame"] = {
                "n_rows": profile.get("n_rows"),
                "n_cols": profile.get("n_cols"),
            }
            if hasattr(director, "direct"):
                try:
                    directives = director.direct("mine", df, text=state.text)
                    if directives is not None and hasattr(directives, "to_dict"):
                        traces.append(
                            {
                                "name": "suites.director",
                                "ok": True,
                                "payload": directives.to_dict(),
                            }
                        )
                        state.slm_backend = getattr(
                            directives, "backend", state.slm_backend
                        )
                except TypeError:
                    pass
        except Exception as e:
            state.notes.append(f"director soft-fail: {type(e).__name__}")

        state.tool_traces = traces
        state.metrics["n_tool_ok"] = sum(1 for t in traces if t.get("ok", True))
        state.metrics["n_tool_fail"] = sum(1 for t in traces if not t.get("ok", True))
        return state

    def _node_validate(self, state: LoopState) -> LoopState:
        ac = state._ac
        validation: dict[str, Any] = {"ok": True, "checks": []}

        # QC soft
        try:
            from autocausal.qc import validate_frame

            qc = validate_frame(ac.df)
            qc_dict = qc.to_dict() if hasattr(qc, "to_dict") else {"raw": str(qc)}
            ok = True
            if isinstance(qc_dict, dict):
                issues = qc_dict.get("issues") or qc_dict.get("errors") or []
                ok = len(issues) == 0 or qc_dict.get("ok", True)
            validation["checks"].append({"name": "qc", "ok": ok, "detail": qc_dict})
            if not ok:
                validation["ok"] = False
        except Exception as e:
            validation["checks"].append(
                {"name": "qc", "ok": True, "detail": f"soft-skip: {type(e).__name__}"}
            )

        # Re-discover soft
        try:
            ac.discover(use_iv=False, min_abs_corr=0.15, qc="off")
            edges = _edges_from_ac(ac)
            state.edges = edges
            state.sync_edge_ids()
            validation["checks"].append(
                {"name": "discover", "ok": True, "n_edges": len(edges)}
            )
            state.metrics["n_edges"] = len(edges)
        except Exception as e:
            validation["checks"].append(
                {"name": "discover", "ok": False, "detail": str(e)}
            )
            validation["ok"] = False

        # Soft refute if present
        try:
            if state.edges:
                from autocausal.suite_tools import refute as _refute

                ref = _refute(state.edges[0], method="placebo", df=ac.df)
                validation["checks"].append(
                    {
                        "name": "refute",
                        "ok": True,
                        "detail": ref.to_dict()
                        if hasattr(ref, "to_dict")
                        else str(ref)[:300],
                    }
                )
        except Exception as e:
            validation["checks"].append(
                {
                    "name": "refute",
                    "ok": True,
                    "detail": f"soft-skip: {type(e).__name__}",
                }
            )

        # Insight summary soft
        try:
            from autocausal.insight import InsightSuite

            report = InsightSuite.from_autocausal(ac, use_slm=False).run(
                text=state.text, use_slm=False
            )
            state.insight_summary = getattr(report, "summary", "") or ""
            if not state.experiments and getattr(report, "experiments_recommended", None):
                state.experiments = list(report.experiments_recommended)[:8]
            validation["checks"].append({"name": "insight", "ok": True})
        except Exception as e:
            validation["checks"].append(
                {"name": "insight", "ok": True, "detail": f"soft-skip: {type(e).__name__}"}
            )

        state.validation = validation
        return state

    def _node_compact(self, state: LoopState) -> LoopState:
        bundle = self.compactor.compact(state)
        state.notes.extend(bundle.notes)
        # Promote to episodic (MEM1 constant budget)
        score = 0.5 + 0.1 * float(state.metrics.get("n_edges") or 0)
        self.memory.promote_working_to_episodic(
            narrative=bundle.narrative,
            handles=bundle.handles,
            round=state.round,
            score=min(1.0, score),
        )
        # Store in vector LTM
        try:
            self.vectors.add(
                bundle.narrative,
                kind="episode",
                meta={"round": state.round, "n_edges": len(state.edge_ids)},
            )
            if state.insight_summary:
                self.vectors.add(
                    state.insight_summary,
                    kind="insight",
                    meta={"round": state.round},
                )
            for ex in state.experiments[:3]:
                title = str(ex.get("title") or ex.get("name") or "")
                if title:
                    self.vectors.add(title, kind="experiment", meta={"round": state.round})
        except Exception as e:
            state.notes.append(f"vector add soft-fail: {type(e).__name__}")
        return state

    def _node_persist(self, state: LoopState) -> LoopState:
        if self._episode_store is None:
            state.notes.append("Persist skipped (no persist_dir).")
            return state
        try:
            self._episode_store.append(
                {
                    "kind": "episode",
                    "round": state.round,
                    "narrative": state.narrative,
                    "handles": state.handles,
                    "edge_ids": list(state.edge_ids),
                    "metrics": dict(state.metrics),
                    "validation_ok": bool(state.validation.get("ok", True)),
                    "source": state.source,
                }
            )
        except Exception as e:
            state.notes.append(f"persist soft-fail: {type(e).__name__}: {e}")
        return state

    def _node_route(self, state: LoopState) -> LoopState:
        # Stop conditions
        if state.round >= state.max_rounds - 1:
            state.route = "stop"
            state.stop_reason = f"Completed max_rounds={state.max_rounds}."
            return state
        if not state.validation.get("ok", True) and state.round > 0:
            # Soft: continue once more unless edges empty
            if not state.edges:
                state.route = "stop"
                state.stop_reason = "Validation issues and no edges."
                return state
        if state.round > 0 and not state.edges:
            state.route = "stop"
            state.stop_reason = "No exploratory edges found."
            return state
        # Plateau: same edge set as previous round
        if len(state.round_history) >= 1:
            prev = state.round_history[-1]
            prev_ids = set(prev.get("edge_ids") or [])
            if prev_ids and prev_ids == set(state.edge_ids) and state.round >= 1:
                state.route = "stop"
                state.stop_reason = "Edge set unchanged vs prior round (plateau)."
                return state
        state.route = "continue"
        state.stop_reason = ""
        return state

    def _build_runtime(self) -> GraphRuntime:
        rt = GraphRuntime(prefer_langgraph=self.prefer_langgraph)
        rt.register("hypothesize", self._node_hypothesize)
        rt.register("skill", self._node_skill)
        rt.register("validate", self._node_validate)
        rt.register("compact", self._node_compact)
        rt.register("persist", self._node_persist)
        rt.register("route", self._node_route)
        return rt

    def run(
        self,
        source: Any = None,
        *,
        ac: Any = None,
        text: str = "",
        max_rounds: Optional[int] = None,
        use_slm: Optional[bool] = None,
        dataset_ids: Optional[list[str]] = None,
        **_kwargs: Any,
    ) -> AgenticLoopReport:
        """Run the cyclic agentic loop → ``AgenticLoopReport``."""
        if use_slm is not None:
            self.use_slm = bool(use_slm) or _env_slm()
            self.compactor.use_slm = self.use_slm

        ac = _as_autocausal(source, existing=ac)
        self.ac = ac
        rounds = max(1, int(max_rounds if max_rounds is not None else self.max_rounds))

        state = LoopState(
            round=0,
            max_rounds=rounds,
            text=text or "",
            source=str(getattr(ac, "source", "memory")),
            dataset_ids=list(dataset_ids or []),
            use_slm=self.use_slm,
            _ac=ac,
        )

        runtime = self._build_runtime()
        result = runtime.run_until(state, max_rounds=rounds, order=NODE_ORDER)
        state = result.state
        self.last_state = state

        n_rounds = len(state.round_history) or (state.round + 1)
        summary = (
            state.narrative
            or f"Agentic loop finished after {n_rounds} round(s); "
            f"{len(state.edge_ids)} edge handles; route={state.route}."
        )
        report = AgenticLoopReport(
            summary=summary,
            narrative=state.narrative,
            handles=dict(state.handles),
            key_edges=list(state.edges)[:30],
            edge_ids=list(state.edge_ids),
            hypotheses=[h.to_dict() for h in state.hypotheses],
            experiments=list(state.experiments),
            round_history=list(state.round_history),
            stages=list(state.stages),
            node_history=list(state.node_history),
            metrics=dict(state.metrics),
            validation=dict(state.validation),
            memory=self.memory.to_dict(),
            vector_hits=list(state.retrieved_memories),
            persist_path=str(self._episode_store.path) if self._episode_store else None,
            runtime_backend=result.backend,
            slm_used=bool(self.use_slm and state.slm_backend == "slm"),
            slm_backend=state.slm_backend,
            source=state.source,
            n_rows=int(len(ac.df)),
            n_cols=int(ac.df.shape[1]),
            n_rounds=n_rounds,
            stop_reason=state.stop_reason,
            notes=list(state.notes) + list(result.notes),
        )
        self.last_report = report
        return report


def run_agentic_loop(
    source: Any = None,
    *,
    ac: Any = None,
    text: str = "",
    max_rounds: int = 3,
    use_slm: bool = False,
    model_name: Optional[str] = None,
    persist_dir: Optional[Union[str, Path]] = None,
    vector_backend: str = "auto",
    prefer_langgraph: bool = True,
    **kwargs: Any,
) -> AgenticLoopReport:
    """One-shot helper: build ``AgenticCausalLoop`` and run."""
    loop = AgenticCausalLoop(
        use_slm=use_slm,
        model_name=model_name,
        max_rounds=max_rounds,
        persist_dir=persist_dir,
        vector_backend=vector_backend,
        prefer_langgraph=prefer_langgraph,
    )
    return loop.run(source, ac=ac, text=text, max_rounds=max_rounds, use_slm=use_slm, **kwargs)
