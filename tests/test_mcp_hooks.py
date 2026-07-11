"""Offline unit tests for MCP / AgentHook connective (no live MCP client)."""

from __future__ import annotations

import json

import pytest

from autocausal.connective import AgentHook, call_tool, list_tools
from autocausal.mcp.hooks import AgentHook as HookFromMcp
from autocausal.mcp.registry import build_default_registry
from autocausal.mcp.serialize import err_payload, ok_payload, to_jsonable
from autocausal.mcp.session import SessionStore


EXPECTED_TOOLS = {
    "autocausal_list_datasets",
    "autocausal_load_dataset",
    "autocausal_from_csv",
    "autocausal_cleanse",
    "autocausal_eda",
    "autocausal_mine",
    "autocausal_discover",
    "autocausal_insight_loop",
    "autocausal_recommend_experiments",
    "autocausal_public_mine",
    "autocausal_report",
    "autocausal_skilling_list",
    "autocausal_session_status",
    "autocausal_list_tools",
}


def test_registry_has_expected_tools():
    reg = build_default_registry()
    names = set(reg.list_names())
    assert EXPECTED_TOOLS <= names
    schemas = reg.schemas()
    assert all("name" in s and "inputSchema" in s for s in schemas)


def test_agent_hook_list_and_unknown_tool():
    hook = AgentHook()
    names = hook.list_names()
    assert "autocausal_load_dataset" in names
    bad = hook.call_tool("no_such_tool", {})
    assert bad["ok"] is False
    assert "Unknown tool" in bad["error"]


def test_load_discover_report_iris():
    hook = AgentHook()
    loaded = hook.call_tool("autocausal_load_dataset", {"dataset_id": "iris"})
    assert loaded["ok"] is True
    assert loaded["n_rows"] == 150
    assert "session_id" in loaded

    mined = hook.call_tool("autocausal_mine", {"use_suite": False, "min_score": 0.1})
    assert mined["ok"] is True

    disc = hook.call_tool(
        "autocausal_discover",
        {"use_iv": False, "min_abs_corr": 0.25, "qc": "off"},
    )
    assert disc["ok"] is True
    assert disc.get("n_edges", 0) >= 0

    report = hook.call_tool("autocausal_report", {"format": "markdown"})
    assert report["ok"] is True
    assert "markdown" in report
    assert len(report["markdown"]) > 20

    status = hook.call_tool("autocausal_session_status", {})
    assert status["ok"] is True
    assert status["sessions"]


def test_list_datasets_and_list_tools():
    hook = AgentHook()
    ds = hook.call_tool("autocausal_list_datasets", {})
    assert ds["ok"] is True
    assert ds["n"] >= 1
    assert any(d.get("id") == "iris" for d in ds["datasets"])

    tools = hook.call_tool("autocausal_list_tools", {})
    assert tools["ok"] is True
    assert tools["n"] >= len(EXPECTED_TOOLS)


def test_module_level_call_tool():
    # fresh default may share process state — use dedicated hook for isolation
    hook = HookFromMcp()
    r = hook.call_tool("autocausal_list_datasets", {})
    assert r["ok"]
    schemas = list_tools()
    assert isinstance(schemas, list)
    assert any(s["name"] == "autocausal_discover" for s in schemas)


def test_recommend_experiments_soft():
    hook = AgentHook()
    hook.call_tool("autocausal_load_dataset", {"dataset_id": "iris"})
    hook.call_tool("autocausal_discover", {"use_iv": False, "qc": "off", "min_abs_corr": 0.3})
    rec = hook.call_tool("autocausal_recommend_experiments", {"use_slm": False})
    assert rec["ok"] is True
    assert "experiments" in rec


def test_skilling_list_soft():
    hook = AgentHook()
    out = hook.call_tool("autocausal_skilling_list", {})
    # soft: ok True with catalog, or ok False with soft error — never crash
    assert "ok" in out
    if out["ok"]:
        assert "catalog" in out


def test_cleanse_eda_soft_paths():
    hook = AgentHook()
    hook.call_tool("autocausal_load_dataset", {"dataset_id": "iris"})
    c = hook.call_tool("autocausal_cleanse", {"use_slm": False})
    assert c["ok"] is True
    e = hook.call_tool("autocausal_eda", {"use_slm": False})
    assert e["ok"] is True


def test_serialize_helpers():
    assert ok_payload(x=1)["ok"] is True
    assert err_payload("boom", tool="t")["ok"] is False
    assert to_jsonable({"a": [1, 2]}) == {"a": [1, 2]}
    assert json.dumps(to_jsonable({"n": 1}))


def test_session_store():
    from autocausal.api import AutoCausal
    import pandas as pd

    store = SessionStore()
    ac = AutoCausal.from_dataframe(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
    sid = store.put(ac, "s1")
    assert sid == "s1"
    assert store.get("s1") is ac
    assert store.list_sessions()[0]["n_rows"] == 2


def test_mcp_package_import_without_sdk():
    """Importing autocausal.mcp must not require the mcp package."""
    import autocausal.mcp as m

    assert m.AgentHook is not None
    # mcp_sdk_available is soft
    avail = m.mcp_sdk_available()
    assert isinstance(avail, bool)


def test_mcp_main_list_tools():
    from autocausal.mcp.server import main

    assert main(["--list-tools"]) == 0
    assert main(["--help"]) == 0


def test_connective_reexports():
    from autocausal import connective

    assert connective.AgentHook is AgentHook
    assert callable(connective.call_tool)
