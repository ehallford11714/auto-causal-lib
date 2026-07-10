"""Offline insight suite tests (rule path); SLM tests skip without HF."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autocausal import AutoCausal, InsightReport, InsightSuite, run_insight_loop
from autocausal.cli import main
from autocausal.insight import demo_insight, synthesize_insight
from autocausal.insight.report import RoleHypotheses
from autocausal.slm import slm_available


def _frame(n: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    treat_x = 0.75 * z + rng.normal(0, 0.3, size=n)
    outcome_y = 1.15 * treat_x + rng.normal(0, 0.4, size=n)
    return pd.DataFrame(
        {
            "z_iv": z,
            "treat_x": treat_x,
            "outcome_y": outcome_y,
            "age": rng.normal(40, 10, n),
        }
    )


def test_insight_report_writers(tmp_path: Path):
    report = InsightReport(
        summary="Test summary",
        key_edges=[{"source": "a", "target": "b", "type": "score", "score": 0.5, "confidence": 0.4}],
        role_hypotheses=RoleHypotheses(treatment=["a"], outcome=["b"]),
        source="unit",
        n_rows=10,
        n_cols=2,
    )
    md = report.to_markdown()
    assert "insight report" in md.lower()
    assert "identification" in md.lower() or "exploratory" in md.lower()
    d = report.to_dict()
    assert d["summary"] == "Test summary"
    assert "treatment_X" in d["role_hypotheses"]

    md_path = report.write(tmp_path / "r.md")
    assert md_path.exists()
    assert "Test summary" in md_path.read_text(encoding="utf-8")
    js_path = report.write(tmp_path / "r.json")
    payload = json.loads(js_path.read_text(encoding="utf-8"))
    assert payload["key_edges"][0]["source"] == "a"


def test_run_insight_loop_rule_offline():
    df = _frame()
    report = run_insight_loop(df, text="Does treat_x cause outcome_y?", use_slm=False)
    assert isinstance(report, InsightReport)
    assert report.summary
    assert report.stages
    assert report.slm_used is False
    assert any("identification" in c.lower() or "exploratory" in c.lower() for c in report.caveats)
    md = report.to_markdown()
    assert "Role hypotheses" in md or "role" in md.lower()


def test_insight_suite_from_prebuilt_autocausal():
    ac = AutoCausal.from_dataframe(_frame(), source="prebuilt")
    ac.mine()
    ac.impute()
    ac.discover()
    suite = InsightSuite.from_autocausal(ac, use_slm=False)
    report = suite.run(text="focus", use_slm=False)
    assert report.summary
    assert "prebuilt" in (report.source or "")


def test_synthesize_rule_only():
    report = synthesize_insight(
        edges=[{"source": "treat_x", "target": "outcome_y", "type": "score", "score": 0.8, "confidence": 0.7}],
        candidates={"treatment": ["treat_x"], "outcome": ["outcome_y"], "instrument": ["z_iv"]},
        source="synth",
        n_rows=10,
        n_cols=4,
        stages=["synthesize"],
        data_sources=["synth"],
        use_slm=False,
    )
    assert "treat_x" in report.summary or report.key_edges
    assert report.role_hypotheses.treatment == ["treat_x"]
    assert report.slm_used is False


def test_demo_insight_offline():
    report = demo_insight(use_slm=False, max_rounds=1, research_loop=False)
    assert report.n_rows > 0
    assert report.summary


def test_cli_insight_demo():
    rc = main(["insight", "demo", "--no-slm", "--rounds", "1", "--format", "json"])
    assert rc == 0


def test_cli_insight_run(tmp_path: Path):
    csv = tmp_path / "tiny.csv"
    _frame(40).to_csv(csv, index=False)
    out = tmp_path / "out.md"
    rc = main(["insight", "run", "--csv", str(csv), "--no-slm", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "insight" in body.lower()


def test_public_api_imports():
    from autocausal.insight import InsightSuite, InsightReport, run_insight_loop

    assert InsightSuite and InsightReport and run_insight_loop


@pytest.mark.skipif(not slm_available(), reason="HuggingFace transformers not installed")
def test_insight_slm_soft_when_available():
    """Runs only when transformers importable; still soft if torch load fails."""
    report = demo_insight(use_slm=True, max_rounds=1, research_loop=False)
    assert report.summary
    assert isinstance(report.caveats, list)
