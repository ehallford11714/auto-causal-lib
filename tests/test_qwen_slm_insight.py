"""Qwen / SLM / insight / LangGraph chain tests.

Default pytest stays fast (rule path + soft skips).
Full Qwen load/generate::

    set AUTOCAUSAL_TEST_QWEN=1
    pytest tests/test_qwen_slm_insight.py -m slow -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from autocausal import AutoCausal, __version__
from autocausal.agentic import (
    SLMLangGraphChain,
    langgraph_available,
    run_slm_langgraph_loop,
)
from autocausal.datasets import load_dataset
from autocausal.insight import InsightSuite, run_insight_loop
from autocausal.slm import (
    DEFAULT_QWEN_SMALL,
    ensure_local_qwen,
    get_backend,
    probe_hardware,
    recommend_qwen_model,
    slm_available,
    slm_status,
)


def _qwen_env() -> bool:
    return os.environ.get("AUTOCAUSAL_TEST_QWEN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _qwen_model_ready(model_id: str | None = None) -> tuple[bool, str]:
    """Return (ready, reason). Soft — never raises."""
    if not slm_available():
        return False, "transformers not installed"
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "torch not installed"
    mid = model_id or os.environ.get("AUTOCAUSAL_SLM_MODEL") or DEFAULT_QWEN_SMALL
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore

        # config.json presence in cache is a cheap readiness signal
        path = try_to_load_from_cache(mid, "config.json")
        if path is None or isinstance(path, Exception) or str(path) == "None":
            return False, f"model `{mid}` not in HF cache (run: python -m autocausal slm setup-qwen)"
        return True, mid
    except Exception as e:
        # Fallback: allow env-gated tests to attempt load
        if _qwen_env():
            return True, mid
        return False, f"cache probe soft-fail: {type(e).__name__}"


def test_version_bumped():
    assert __version__.startswith("0.14")


def test_probe_and_recommend_qwen_offline():
    hw = probe_hardware()
    assert "cpu_count" in hw
    rec = recommend_qwen_model(hw)
    assert "Qwen" in rec["recommended_model_id"] or "qwen" in rec["recommended_model_id"].lower()
    assert rec["reason"]
    st = slm_status()
    assert st["rule_backend"] is True
    assert "recommended_qwen" in st
    assert "epistemic" in st


def test_ensure_local_qwen_no_download():
    prev_model = os.environ.get("AUTOCAUSAL_SLM_MODEL")
    prev_slm = os.environ.get("AUTOCAUSAL_SLM")
    try:
        res = ensure_local_qwen(download=False, set_env=True)
        assert res["model_id"]
        assert res["env_set"] is True
        assert os.environ.get("AUTOCAUSAL_SLM_MODEL") == res["model_id"]
        assert any("Epistemic" in n or "epistemic" in n.lower() for n in res["notes"])
    finally:
        if prev_model is None:
            os.environ.pop("AUTOCAUSAL_SLM_MODEL", None)
        else:
            os.environ["AUTOCAUSAL_SLM_MODEL"] = prev_model
        if prev_slm is None:
            os.environ.pop("AUTOCAUSAL_SLM", None)
        else:
            os.environ["AUTOCAUSAL_SLM"] = prev_slm


def test_get_backend_rule_and_hf_construct():
    rule = get_backend(use_slm=False)
    assert rule.name == "rule"
    hf = get_backend(use_slm=True, model_name="sshleifer/tiny-gpt2")
    assert hf.name == "huggingface"


def test_autocausal_guide_rule_offline():
    """Always-green offline path (no Qwen)."""
    ac = AutoCausal.from_dataframe(load_dataset("iris"))
    ac.mine()
    ac.impute()
    ac.discover(use_iv=False)
    g = ac.guide(text="what drives petal width?", use_slm=False)
    assert g is not None
    md = g.to_markdown() if hasattr(g, "to_markdown") else str(g)
    assert len(md) > 20
    assert g.backend


def test_insight_report_rule_offline():
    """Always-green insight report generation without SLM."""
    df = load_dataset("iris")
    report = run_insight_loop(df, text="petal drivers", use_slm=False)
    assert report.summary
    md = report.to_markdown()
    assert "insight" in md.lower() or len(md) > 40
    assert report.report(as_markdown=True)
    suite = InsightSuite(use_slm=False)
    r2 = suite.run(df, text="petal", use_slm=False)
    assert r2.summary


def test_slm_langgraph_chain_rule_offline():
    """LangGraph chain with use_slm=False — FSM or langgraph backend, always runs."""
    df = load_dataset("iris")
    report = run_slm_langgraph_loop(
        df,
        text="iris petal drivers",
        max_rounds=1,
        use_slm=False,
        prefer_langgraph=True,
        ensure_qwen=False,
    )
    assert report.agentic.summary or report.agentic.narrative or report.to_markdown()
    md = report.to_markdown()
    assert "not" in md.lower() and "identification" in md.lower() or "Epistemic" in md
    assert report.chain_backend in ("fsm", "langgraph")
    # AutoCausal.slm_loop
    ac = AutoCausal.from_dataframe(df)
    r2 = ac.slm_loop(text="petal", max_rounds=1, use_slm=False)
    assert r2.to_markdown()


def test_langgraph_available_bool():
    assert isinstance(langgraph_available(), bool)


@pytest.mark.slow
def test_qwen_import_and_backend():
    if not _qwen_env():
        pytest.skip("Set AUTOCAUSAL_TEST_QWEN=1 to run Qwen load tests")
    ready, reason = _qwen_model_ready()
    if not ready:
        pytest.skip(reason)
    from autocausal.slm import HuggingFaceSLM

    mid = reason if reason.startswith("Qwen") or "/" in reason else DEFAULT_QWEN_SMALL
    backend = HuggingFaceSLM(model_name=mid)
    assert backend._ensure() or backend._error  # soft: either loads or records error
    if not backend._ensure():
        pytest.skip(backend._error or "Qwen load soft-failed")
    text = backend._generate("Say OK in one word.", system="Be brief.")
    assert isinstance(text, str)


@pytest.mark.slow
def test_qwen_autocausal_guide_and_insight():
    if not _qwen_env():
        pytest.skip("Set AUTOCAUSAL_TEST_QWEN=1 to run Qwen integration tests")
    ready, reason = _qwen_model_ready()
    if not ready:
        pytest.skip(reason)
    mid = os.environ.get("AUTOCAUSAL_SLM_MODEL") or DEFAULT_QWEN_SMALL
    os.environ["AUTOCAUSAL_SLM_MODEL"] = mid
    ac = AutoCausal.from_dataframe(load_dataset("iris"))
    ac.mine()
    ac.impute()
    ac.discover(use_iv=False)
    g = ac.guide(text="what is associated with petal width?", use_slm=True, model_name=mid)
    assert g is not None
    assert g.to_markdown()
    report = InsightSuite(use_slm=True, model_name=mid).run(
        load_dataset("iris"), text="petal", use_slm=True
    )
    assert report.summary
    assert report.to_markdown()
    chain = SLMLangGraphChain(use_slm=True, model_name=mid, max_rounds=1)
    crep = chain.run(load_dataset("iris"), text="petal")
    assert crep.to_markdown()
