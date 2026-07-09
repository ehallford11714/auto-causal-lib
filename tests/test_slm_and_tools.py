"""Offline RuleBackend SLM + suite_tools tests (no torch/gensim/nltk required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autocausal import AutoCausal
from autocausal.cli import main
from autocausal.slm import (
    RuleBackend,
    create_from_context,
    get_backend,
    guide_pipeline,
    infer_from_results,
    slm_status,
)
from autocausal.suite_tools import (
    invoke_tool,
    list_tools,
    tool_catalog,
    validate_pipeline,
)


def _iv_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    d = 0.8 * z + rng.normal(0, 0.3, size=n)
    y = 1.2 * d + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({"z_iv": z, "treat_x": d, "outcome_y": y, "age": rng.normal(40, 10, n)})


def test_rule_backend_create_and_infer():
    df = _iv_frame()
    ac = AutoCausal.from_dataframe(df)
    ac.mine()
    ac.run()
    cre = ac.create(text="Does treat_x cause outcome_y using lottery instrument?")
    assert cre.questions
    assert cre.backend == "rule"
    inf = ac.interpret()
    assert inf.narrative
    assert inf.caveats


def test_create_from_context_standalone():
    res = create_from_context(
        {
            "text": "random assignment causes sales",
            "columns": [{"name": "z_assign"}, {"name": "treat"}, {"name": "revenue"}],
            "emotion": "happy",
            "intent": "inform",
        }
    )
    assert any("instrument" in (i.get("name") or "").lower() or "random" in str(i).lower() for i in res.instruments) or res.questions
    assert any(m.get("role") == "affect_context" for m in res.morphemes)


def test_infer_from_results_weak_iv_caveat():
    res = infer_from_results(
        {
            "edges": [{"source": "d", "target": "y", "confidence": 0.4, "type": "iv_2sls"}],
            "iv": {"coef": 0.5, "first_stage_f": 3.0},
        }
    )
    assert any("weak" in c.lower() for c in res.caveats)


def test_get_backend_default_is_rule():
    b = get_backend(use_slm=False)
    assert isinstance(b, RuleBackend)
    g = guide_pipeline({"columns": [{"name": "z_iv"}], "edges": []})
    assert g.backend == "rule"


def test_slm_status_never_raises():
    st = slm_status()
    assert st["rule_backend"] is True
    assert "huggingface_ready" in st


def test_list_tools_without_optional_deps():
    tools = list_tools()
    ids = {t.id for t in tools}
    assert "builtin_2sls" in ids
    assert "nltk" in ids
    assert "gensim" in ids
    assert "dowhy" in ids
    # missing optional packages should be stub/missing, not crash
    for t in tools:
        assert t.status in ("available", "stub", "missing", "unknown")


def test_invoke_nltk_and_gensim_soft():
    nlp = invoke_tool("nltk", text="Does treatment cause outcome revenue?")
    assert nlp.ok
    assert nlp.backend in ("nltk", "builtin_regex")
    gen = invoke_tool("gensim", texts=["treatment causes sales", "campaign spend lifts revenue"])
    assert gen.ok
    assert gen.backend in ("gensim", "builtin_bow")


def test_text_z_and_builtin_2sls():
    tags = invoke_tool("text_z", text="lottery randomized rainfall instrument")
    assert tags.ok
    assert tags.data["n"] >= 2
    df = _iv_frame()
    iv = invoke_tool("builtin_2sls", df=df, y="outcome_y", d="treat_x", z="z_iv")
    assert iv.ok
    assert "coef" in iv.data
    assert "first_stage_f" in iv.data


def test_validate_pipeline():
    df = _iv_frame(80)
    report = validate_pipeline(
        {"edges": [{"source": "treat_x", "target": "outcome_y", "confidence": 0.5}]},
        df=df,
        claims_text="treat_x causes outcome_y",
        y="outcome_y",
        d="treat_x",
        z="z_iv",
    )
    assert report.checks
    assert "weak_iv_f" in {c["id"] for c in report.checks}
    assert report.score >= 0


def test_cli_tools_list_and_create():
    assert main(["tools", "list"]) == 0
    assert main(["slm-status"]) == 0
    assert main(["create", "--text", "lottery assignment for spend"]) == 0


def test_cli_tools_validate_csv(tmp_path):
    df = _iv_frame(50)
    path = tmp_path / "iv.csv"
    df.to_csv(path, index=False)
    assert main(["tools", "validate", "--csv", str(path), "--y", "outcome_y", "--d", "treat_x", "--z", "z_iv"]) == 0


def test_tool_catalog_structure():
    cat = tool_catalog()
    assert cat["n"] >= 10
    assert "causal" in cat["by_category"]
    assert "nlp" in cat["by_category"]
