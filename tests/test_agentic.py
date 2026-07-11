"""Offline tests for autocausal.agentic (no HF / langgraph / chroma required)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from autocausal.agentic import (
    NODE_ORDER,
    AgenticCausalLoop,
    AgenticLoopReport,
    AgentMemory,
    Compactor,
    EpisodicMemory,
    GraphRuntime,
    LoopState,
    VectorStoreMemory,
    WorkingMemory,
    langgraph_available,
    run_agentic_loop,
)
from autocausal.agentic.compact import CompactBundle
from autocausal.agentic.memory import MemoryItem
from autocausal.agentic.persist import EpisodeStore, load_episodes, persist_episode
from autocausal.agentic.state import Hypothesis
from autocausal.connective import AgentHook
from autocausal.datasets import load_dataset
from autocausal.mcp.registry import build_default_registry


def test_module_exports_and_node_order():
    assert NODE_ORDER[0] == "hypothesize"
    assert NODE_ORDER[-1] == "route"
    assert "compact" in NODE_ORDER
    assert isinstance(langgraph_available(), bool)


def test_working_and_episodic_budget():
    mem = AgentMemory(max_working=4, max_episodes=3, max_chars_total=500)
    for i in range(6):
        mem.working.add(MemoryItem.make(f"w{i}" * 20, kind="working", score=float(i)))
    assert len(mem.working.items) <= 4

    for i in range(5):
        mem.episodic.add(
            MemoryItem.make(f"episode-{i}-" + ("x" * 40), kind="episodic", score=0.1 * i, round=i)
        )
    assert len(mem.episodic) <= 3
    # Links form between consecutive retained notes
    eps = list(mem.episodic.episodes)
    assert any(e.links for e in eps)


def test_compactor_lossy_and_lossless():
    state = LoopState(
        round=1,
        edges=[{"source": "a", "target": "b", "score": 0.9}],
        tool_traces=[{"name": "automine.mine_associations", "ok": True}],
        metrics={"n_edges": 1},
        dataset_ids=["iris"],
        hypotheses=[Hypothesis(id="h1", statement="a→b candidate", source="rule")],
    )
    state.sync_edge_ids()
    bundle = Compactor(use_slm=False).compact(state)
    assert isinstance(bundle, CompactBundle)
    assert "a->b" in bundle.handles["edge_ids"]
    assert bundle.handles["dataset_ids"] == ["iris"]
    assert len(bundle.narrative) > 10
    assert bundle.backend == "rule"


def test_vector_memory_numpy_fallback():
    store = VectorStoreMemory(backend="numpy")
    store.add("petal length associated with petal width", kind="insight")
    store.add("sepal length candidate confounder", kind="insight")
    store.add("random unrelated finance revenue spend", kind="experiment")
    hits = store.query("petal width drivers", k=2)
    assert store.backend == "numpy"
    assert len(hits) >= 1
    assert "petal" in hits[0]["text"].lower() or hits[0]["score"] >= 0


def test_persist_jsonl(tmp_path: Path):
    path = tmp_path / "episodes.jsonl"
    row = persist_episode(path, narrative="round0 summary", handles={"edge_ids": ["x->y"]}, round=0)
    assert row["narrative"] == "round0 summary"
    loaded = load_episodes(path)
    assert len(loaded) == 1
    store = EpisodeStore.open(path)
    store.append({"kind": "episode", "round": 1, "narrative": "r1"})
    assert len(store.read_all()) == 2


def test_graph_runtime_fsm_cycle():
    seen: list[str] = []

    def _mk(name: str):
        def _fn(state: LoopState) -> LoopState:
            seen.append(name)
            if name == "route":
                state.route = "stop"
                state.stop_reason = "test stop"
            return state

        return _fn

    rt = GraphRuntime(prefer_langgraph=False)
    for n in NODE_ORDER:
        rt.register(n, _mk(n))
    state = LoopState(max_rounds=1)
    result = rt.run_cycle(state)
    assert result.backend == "fsm"
    assert seen == list(NODE_ORDER)
    assert result.state.route == "stop"


def test_run_agentic_loop_iris_offline(tmp_path: Path):
    df = load_dataset("iris")
    report = run_agentic_loop(
        df,
        text="what relates to petal width?",
        max_rounds=2,
        use_slm=False,
        persist_dir=tmp_path / "agentic",
        prefer_langgraph=False,
        vector_backend="numpy",
    )
    assert isinstance(report, AgenticLoopReport)
    assert report.n_rounds >= 1
    assert report.runtime_backend == "fsm"
    assert report.epistemic
    assert "exploratory" in report.summary.lower() or len(report.narrative) > 0
    md = report.to_markdown()
    assert "Agentic Causal Loop" in md
    payload = report.to_dict()
    assert "handles" in payload
    assert (tmp_path / "agentic" / "episodes.jsonl").exists()


def test_autocausal_agentic_loop_method():
    from autocausal import AutoCausal

    ac = AutoCausal.from_dataframe(load_dataset("iris"))
    report = ac.agentic_loop(
        text="iris edges",
        max_rounds=1,
        use_slm=False,
        prefer_langgraph=False,
        vector_backend="numpy",
    )
    assert isinstance(report, AgenticLoopReport)
    assert report.n_rows == 150


def test_mcp_agentic_tool_registered_and_runs():
    reg = build_default_registry()
    assert "autocausal_agentic_loop" in reg.list_names()

    hook = AgentHook()
    loaded = hook.call_tool("autocausal_load_dataset", {"dataset_id": "iris"})
    assert loaded["ok"] is True
    out = hook.call_tool(
        "autocausal_agentic_loop",
        {
            "max_rounds": 1,
            "use_slm": False,
            "prefer_langgraph": False,
            "text": "offline agentic",
        },
    )
    assert out["ok"] is True
    assert out.get("agentic")
    assert out.get("n_rounds", 0) >= 1


def test_package_lazy_import():
    import autocausal as ac

    assert ac.AgenticCausalLoop is AgenticCausalLoop
    assert callable(ac.run_agentic_loop)
