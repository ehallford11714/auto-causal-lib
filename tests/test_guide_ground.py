"""Grounding + rule guide offline; HF skipped unless env set."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from autocausal import AutoCausal
from autocausal.grounding import ground_edges
from autocausal.slm import RuleGuide, guide_pipeline, get_guide


def test_rule_guide_offline():
    ctx = {
        "text": "what causes revenue?",
        "columns": [{"name": "treatment"}, {"name": "revenue"}, {"name": "age"}, {"name": "noise"}],
        "associations": [{"a": "treatment", "b": "revenue", "metric": "pearson", "score": 0.7}],
        "edges": [
            {"source": "treatment", "target": "revenue", "confidence": 0.8, "score": 0.6},
            {"source": "noise", "target": "revenue", "confidence": 0.05, "score": 0.05},
        ],
        "candidates": {"instrument": ["z"], "confounder": ["age"], "treatment": ["treatment"], "outcome": ["revenue"]},
    }
    g = RuleGuide().guide(ctx)
    assert g.backend == "rule"
    assert g.focus_columns
    assert g.validate_edges
    assert g.to_markdown()
    g2 = guide_pipeline(ctx, use_slm=False)
    assert g2.backend == "rule"


def test_grounding_offline():
    edges = [
        {"source": "treatment", "target": "revenue", "confidence": 0.7},
        {"source": "campaign", "target": "conversion", "confidence": 0.6},
        {"source": "foo_bar", "target": "baz_qux", "confidence": 0.4},
    ]
    report = ground_edges(edges, use_web=False)
    labels = {c.label for c in report.claims}
    assert "documented" in labels or "plausible" in labels
    assert any(c.label == "unsupported" for c in report.claims)
    assert report.to_markdown()


def test_guide_api_pipeline():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "treatment": rng.integers(0, 2, n),
            "revenue": rng.normal(size=n),
            "instrument_z": rng.normal(size=n),
            "age": rng.normal(40, 5, n),
        }
    )
    df["revenue"] = df["revenue"] + 1.5 * df["treatment"]
    ac = AutoCausal.from_dataframe(df)
    ac.mine().impute().discover(use_iv=True, min_abs_corr=0.05)
    assert ac.mining is not None
    g = ac.guide(text="what causes revenue?")
    assert g.focus_columns
    gr = ac.ground()
    assert gr.claims


@pytest.mark.skipif(
    os.environ.get("AUTOCAUSAL_SLM_TEST", "").strip() not in ("1", "true", "yes"),
    reason="HF SLM download skipped unless AUTOCAUSAL_SLM_TEST=1",
)
def test_hf_slm_optional():
    guide = get_guide(use_slm=True, model_name="sshleifer/tiny-gpt2")
    result = guide.guide(
        {
            "text": "causes Y",
            "columns": [{"name": "x"}, {"name": "y"}],
            "edges": [{"source": "x", "target": "y", "confidence": 0.5}],
            "associations": [],
            "candidates": {},
        }
    )
    assert result.suggestions
