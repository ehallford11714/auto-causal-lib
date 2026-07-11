"""Offline tests for autocausal.grail (stub path — no Kineteq required)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from autocausal.grail import (
    GrailEngine,
    GrailStub,
    TOOL_NAMES,
    dispatch_grail_tool,
    grail_backend_status,
    insight_grail_step,
    register_grail_skilling_tools,
    run_grail,
    try_register_mcp_tools,
)
from autocausal.guides import direct, list_guides, resolve_backends


def _ctx() -> dict[str, Any]:
    return {
        "text": "Does campaign_spend cause revenue?",
        "columns": [
            {"name": "campaign_spend"},
            {"name": "revenue"},
            {"name": "instrument_z"},
            {"name": "age"},
        ],
        "edges": [
            {
                "source": "campaign_spend",
                "target": "revenue",
                "confidence": 0.8,
                "score": 0.6,
            }
        ],
        "candidates": {
            "treatment": ["campaign_spend"],
            "outcome": ["revenue"],
            "instrument": ["instrument_z"],
        },
    }


def test_grail_backend_status_offline():
    st = grail_backend_status()
    assert st["stub"] is True
    assert st["preferred"] in ("grail_stub", "kineteq_module", "kineteq_mcp")
    assert "epistemic" in st


def test_stub_impute_compose_fold_run():
    stub = GrailStub()
    ctx = _ctx()
    audit = stub.impute("Does spend cause revenue?", context=ctx)
    assert audit.enriched_goal
    assert audit.assumptions
    assert audit.backend == "grail_stub"

    chain = stub.compose("Does spend cause revenue?", context=ctx, chain_length=3)
    assert len(chain.steps) == 3
    fold = stub.fold(chain)
    assert "directive" in fold.to_dict()

    report = stub.run("Does spend cause revenue?", context=ctx, max_cycles=2)
    assert report.live_kineteq is False
    assert report.backend == "grail_stub"
    assert len(report.cycles) == 2
    assert report.focus_columns
    md = report.to_markdown()
    assert "GRAIL report" in md
    assert "NOT" in report.epistemic or "not" in report.epistemic.lower()


def test_engine_run_and_convenience():
    eng = GrailEngine(prefer_live=False)
    report = eng.run("causal goal", context=_ctx(), max_cycles=1)
    assert report.backend == "grail_stub"
    r2 = run_grail("causal goal", context=_ctx(), prefer_live=False, max_cycles=1)
    assert r2.genome_id


def test_memory_and_graph():
    eng = GrailEngine(prefer_live=False)
    mem = eng.memory_step("revenue", context=_ctx(), top_k=5)
    assert mem
    boost = eng.graph_retrieve(context=_ctx(), focus=["campaign_spend"], top_k=5)
    assert isinstance(boost, list)


def test_guide_backend_grail():
    ids = {r["id"] for r in list_guides()}
    assert "kineteq_grail" in ids
    assert resolve_backends(["grail"]) == ["kineteq_grail"]
    plan = direct(_ctx(), backends=["grail", "rule"])
    assert "grail_stub" in plan.backends or "kineteq_grail" in plan.backends or any(
        "grail" in b for b in plan.backends
    )
    assert plan.focus_columns or plan.next_questions


def test_dispatch_mcp_tool_names():
    assert "autocausal_grail_run" in TOOL_NAMES
    st = dispatch_grail_tool("autocausal_grail_status", {})
    assert st["ok"] is True
    out = dispatch_grail_tool(
        "autocausal_grail_run",
        {"goal": "Does spend cause revenue?", "columns": ["campaign_spend", "revenue"]},
    )
    assert out["ok"] is True
    assert out["result"]["backend"] == "grail_stub"


def test_insight_grail_step():
    step = insight_grail_step(text="Does spend cause revenue?", context=_ctx())
    assert step["stage"] == "grail"
    assert step["focus_columns"] is not None


def test_skilling_and_mcp_register():
    surface = register_grail_skilling_tools()
    names = surface.list_names()
    assert any(n.startswith("autocausal_grail_") for n in names)
    # Execute one tool via surface handler
    df = pd.DataFrame({"campaign_spend": [1, 2], "revenue": [3, 4]})
    tool = surface.get("autocausal_grail_impute")
    result = tool.handler(df, goal="Does spend cause revenue?")
    assert result.payload.get("ok") is True

    bind = try_register_mcp_tools()
    assert "tools" in bind
    # When autocausal.mcp is present, GRAIL tools should be listed
    if bind.get("ok"):
        assert any("grail" in t for t in bind["tools"])


def test_mcp_registry_includes_grail():
    from autocausal.mcp.registry import build_default_registry

    reg = build_default_registry()
    names = reg.list_names()
    for t in TOOL_NAMES:
        assert t in names, f"missing {t}"
    out = reg.invoke("autocausal_grail_status", {})
    assert out.get("ok") is True
