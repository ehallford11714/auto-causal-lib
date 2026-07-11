"""Offline tests for AutoCleanse / AutoEDA / AutoMine suites (rule path).

SLM path soft-skips when HuggingFace/torch unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autocausal import (
    AutoCausal,
    AutoCleanseSuite,
    AutoEDASuite,
    AutoMineSuite,
    SLMAutoDirector,
)
from autocausal.suites import auto_cleanse, auto_eda, auto_mine
from autocausal.suites.director import resolve_suite_slm
from autocausal.slm import slm_available


def _messy_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "unit_id": np.arange(n),
            "treat_x": rng.integers(0, 2, size=n),
            "instrument_z": rng.normal(0, 1, size=n),
            "confound_age": rng.normal(40, 10, size=n),
            "outcome_y": rng.normal(0, 1, size=n),
            "spend": rng.normal(100, 20, size=n),
            "revenue": rng.normal(200, 50, size=n),
            "const_col": 1,
            "almost_missing": [np.nan] * (n - 2) + [1.0, 2.0],
            "obj_num": [str(x) for x in rng.normal(0, 1, size=n)],
        }
    )
    # inject missing + duplicate
    df.loc[0:5, "spend"] = np.nan
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


def test_resolve_suite_slm_defaults_try_on():
    assert resolve_suite_slm(None) is True
    assert resolve_suite_slm(True) is True
    assert resolve_suite_slm(False) is False


def test_director_rule_path():
    df = _messy_df()
    d = SLMAutoDirector(use_slm=False).direct("cleanse", df, text="prep spend→revenue")
    assert d.backend == "rule"
    assert d.stage == "cleanse"
    assert "const_col" in d.drop_columns or "almost_missing" in d.drop_columns
    assert d.to_dict()["continue"] is True
    md = d.to_markdown()
    assert "generative assistance" in md.lower() or "SLM" in md


def test_autocleanse_suite_rule():
    df = _messy_df()
    suite = AutoCleanseSuite(df, use_slm=False).run()
    assert suite.frame is not None and suite.report is not None
    assert suite.report.n_rows_out <= suite.report.n_rows_in
    assert "const_col" in suite.report.dropped_columns or suite.frame["spend"].isna().sum() == 0
    assert suite.report.slm_directives is not None
    assert suite.report.to_dict()["backend"]
    md = suite.to_markdown()
    assert "AutoCleanse" in md
    ac = suite.to_autocausal()
    assert isinstance(ac, AutoCausal)
    assert ac.cleanse_report is not None


def test_autoeda_suite_rule():
    df = _messy_df()
    suite = AutoEDASuite(df, use_slm=False, include_plots=True).run()
    assert suite.report is not None
    r = suite.report
    assert r.n_rows == len(df)
    assert r.roles.outcome is not None or r.roles.treatment is not None
    assert 0.0 <= r.readiness_score <= 1.0
    assert r.slm_directives is not None
    assert "Missingness" in r.to_markdown()
    d = r.to_dict()
    assert "cardinality" in d


def test_automine_suite_rule():
    df = _messy_df()
    suite = AutoMineSuite(df, use_slm=False, join_public=None, try_datamine=True).run()
    assert suite.report is not None
    r = suite.report
    assert r.n_cols >= 2
    assert isinstance(r.associations, list)
    fabric = r.to_mine_report()
    assert "MineReport" in str(fabric.get("schema", fabric))
    assert r.slm_directives is not None


def test_functional_helpers():
    df = _messy_df(40)
    frame, crep = auto_cleanse(df, use_slm=False)
    assert len(frame) > 0 and crep.n_cols_out > 0
    erep = auto_eda(frame, use_slm=False)
    assert erep.n_rows == len(frame)
    mrep = auto_mine(frame, use_slm=False, join_public=None)
    assert mrep.n_rows == len(frame)


def test_fluent_autocausal_chain(tmp_path: Path):
    df = _messy_df(60)
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    ac = (
        AutoCausal.from_csv(path)
        .cleanse(use_slm=False)
        .eda(use_slm=False)
        .automine(use_slm=False, join_public=None)
    )
    assert ac.cleanse_report is not None
    assert ac.eda_report is not None
    assert ac.mine_report is not None
    assert ac.mining is not None
    result = ac.discover(qc="off", use_iv=False, min_abs_corr=0.05)
    assert result is not None
    assert len(ac.df) > 0


def test_report_write(tmp_path: Path):
    df = _messy_df(30)
    suite = AutoCleanseSuite(df, use_slm=False).run()
    md_path = tmp_path / "c.md"
    json_path = tmp_path / "c.json"
    suite.write(md_path)
    suite.write(json_path)
    assert "AutoCleanse" in md_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "operations" in payload


def test_imports_from_top_level():
    import autocausal as ac

    assert ac.AutoCleanseSuite is AutoCleanseSuite
    assert ac.AutoEDASuite is AutoEDASuite
    assert ac.AutoMineSuite is AutoMineSuite
    assert ac.SLMAutoDirector is SLMAutoDirector


@pytest.mark.skipif(not slm_available(), reason="transformers not installed")
def test_slm_director_soft_path():
    """When HF is installed, director may return huggingface or rule+hf_*."""
    df = _messy_df(20)
    d = SLMAutoDirector(use_slm=True).direct("eda", df, text="what drives revenue?")
    assert d.stage == "eda"
    assert d.backend  # any backend string
    # Must not raise; generative flag optional
    _ = d.to_dict()


def test_suite_cli_cleanse(tmp_path: Path):
    from autocausal.cli import main

    df = _messy_df(25)
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "out.md"
    rc = main(["suite", "cleanse", "--csv", str(csv), "--no-slm", "-o", str(out)])
    assert rc == 0
    assert out.exists()
    assert "AutoCleanse" in out.read_text(encoding="utf-8")
