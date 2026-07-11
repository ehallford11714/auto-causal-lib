"""Offline tests for direction guide backends + DirectionPlan merge."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from autocausal import AutoCausal
from autocausal.cli import main as cli_main
from autocausal.guides import (
    DirectionPlan,
    HuggingFaceSLMGuide,
    KineteqPivotEmbeddingGuide,
    LLMIntentGuide,
    RetracementGuide,
    RuleGuide,
    direct,
    list_guides,
    merge_guide_results,
    resolve_backends,
)
from autocausal.guides.types import GuideResult, GuideSuggestion


def _ctx() -> dict[str, Any]:
    return {
        "text": "what causes revenue from campaign spend?",
        "columns": [
            {"name": "campaign_spend"},
            {"name": "revenue"},
            {"name": "instrument_z"},
            {"name": "age"},
            {"name": "noise"},
            {"name": "spend_lag1"},
        ],
        "associations": [
            {"a": "campaign_spend", "b": "revenue", "metric": "pearson", "score": 0.72},
            {"a": "age", "b": "revenue", "metric": "pearson", "score": 0.2},
        ],
        "edges": [
            {
                "source": "campaign_spend",
                "target": "revenue",
                "confidence": 0.8,
                "score": 0.6,
            },
            {
                "source": "revenue",
                "target": "campaign_spend",
                "confidence": 0.3,
                "score": 0.25,
            },
            {
                "source": "noise",
                "target": "revenue",
                "confidence": 0.05,
                "score": 0.05,
            },
        ],
        "candidates": {
            "instrument": ["instrument_z"],
            "confounder": ["age"],
            "treatment": ["campaign_spend"],
            "outcome": ["revenue"],
        },
    }


def test_list_guides_offline():
    rows = list_guides()
    ids = {r["id"] for r in rows}
    assert {"rule", "huggingface", "llmintent", "retracement", "kineteq_pivot", "kineteq_grail"} <= ids
    rule = next(r for r in rows if r["id"] == "rule")
    assert rule["available"] is True


def test_cli_guides_list(capsys):
    assert cli_main(["guides", "list"]) == 0
    out = capsys.readouterr().out
    assert "rule" in out
    assert "llmintent" in out
    assert "kineteq_pivot" in out


def test_rule_guide_via_registry():
    g = RuleGuide().guide(_ctx())
    assert g.backend == "rule"
    assert g.available
    assert g.focus_columns


def test_llmintent_stub_or_real():
    g = LLMIntentGuide().guide(_ctx())
    assert g.focus_columns
    assert g.to_dict()
    # Either real llmintent or soft stub
    assert g.backend in ("llmintent", "llmintent_stub")
    if not g.available:
        assert "not installed" in " ".join(g.notes).lower() or "stub" in g.backend


def test_retracement_stub_or_real():
    g = RetracementGuide().guide(_ctx())
    assert g.lag_hints  # spend_lag1
    assert any(h.get("column") == "spend_lag1" for h in g.lag_hints)
    assert g.boost_edges or g.suppress_edges
    assert g.backend in ("retracement", "retracement_stub")


def test_kineteq_pivot_fallback():
    g = KineteqPivotEmbeddingGuide().guide(_ctx())
    # Without MCP/module → local fallback labeled not Kineteq
    assert g.backend == "pivot_fallback" or g.backend.startswith("kineteq")
    assert g.focus_columns or g.related_variables
    assert any("fallback" in n.lower() or "kineteq" in n.lower() for n in g.notes)


def test_mock_adapters_merge():
    """Merge rule + mock llmintent/retracement/kineteq contributions."""
    mocks = [
        GuideResult(
            backend="llmintent",
            available=True,
            focus_columns=["campaign_spend", "revenue"],
            treatment=["campaign_spend"],
            outcome=["revenue"],
            instruments=["instrument_z"],
            next_questions=["Does spend cause revenue?"],
            boost_edges=[
                {
                    "source": "campaign_spend",
                    "target": "revenue",
                    "reason": "mock",
                    "backend": "llmintent",
                }
            ],
            notes=["mock llmintent"],
        ),
        GuideResult(
            backend="retracement",
            available=True,
            focus_columns=["spend_lag1", "campaign_spend"],
            lag_hints=[{"column": "spend_lag1", "kind": "lag"}],
            suppress_edges=[
                {
                    "source": "revenue",
                    "target": "campaign_spend",
                    "reason": "reverse",
                    "backend": "retracement",
                }
            ],
            notes=["mock retracement"],
        ),
        GuideResult(
            backend="pivot_fallback",
            available=False,
            focus_columns=["campaign_spend", "instrument_z"],
            related_variables=["campaign_spend", "instrument_z"],
            instruments=["instrument_z"],
            notes=["mock kineteq fallback"],
        ),
        RuleGuide().guide(_ctx()),
    ]
    plan = merge_guide_results(
        mocks, requested=["llmintent", "retracement", "kineteq_pivot", "rule"]
    )
    assert isinstance(plan, DirectionPlan)
    assert "campaign_spend" in plan.focus_columns
    assert "instrument_z" in plan.candidate_z
    assert plan.boost_edges
    assert plan.suppress_edges
    assert plan.to_markdown()
    gr = plan.as_guide_result()
    assert gr.focus_columns


def test_direct_multi_backend_offline():
    plan = direct(
        _ctx(),
        backends=["llmintent", "retracement", "kineteq_pivot", "rule"],
    )
    assert plan.focus_columns
    assert plan.contributions
    assert plan.to_json()
    data = json.loads(plan.to_json())
    assert "focus_columns" in data
    assert "candidate_z" in data


def test_resolve_backends_aliases():
    assert resolve_backends(["intent", "retrace", "pivot"]) == [
        "llmintent",
        "retracement",
        "kineteq_pivot",
    ]
    assert "huggingface" in resolve_backends(["rule"], use_slm=True)


def test_api_direct_second_pass():
    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame(
        {
            "campaign_spend": rng.normal(size=n),
            "revenue": rng.normal(size=n),
            "instrument_z": rng.normal(size=n),
            "age": rng.normal(40, 5, n),
        }
    )
    df["revenue"] = df["revenue"] + 1.2 * df["campaign_spend"]
    ac = AutoCausal.from_dataframe(df)
    plan = ac.direct(
        text="does campaign spend cause revenue?",
        backends=["rule", "kineteq_pivot"],
        second_pass=True,
    )
    assert ac.direction_plan is plan
    assert ac.guide_result is not None
    assert ac.result is not None


def test_auto_with_guide_backends(tmp_path):
    rng = np.random.default_rng(2)
    n = 60
    df = pd.DataFrame(
        {
            "treatment": rng.integers(0, 2, n),
            "revenue": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    result = AutoCausal.auto(
        str(path),
        text="what causes revenue?",
        guide_backends=["rule", "kineteq_pivot"],
        second_pass=True,
    )
    assert result.direction_plan is not None
    assert result.guide is not None
    assert result.to_markdown()


@pytest.mark.skipif(
    not LLMIntentGuide().available(),
    reason="llmintent not installed on path",
)
def test_llmintent_optional_integration():
    g = LLMIntentGuide().guide(_ctx())
    assert g.available
    assert g.backend == "llmintent"
    assert g.focus_columns or g.next_questions


def test_huggingface_soft_unavailable_ok():
    # Should not raise even if transformers missing
    g = HuggingFaceSLMGuide().guide(_ctx())
    assert g.suggestions or g.focus_columns or g.notes
