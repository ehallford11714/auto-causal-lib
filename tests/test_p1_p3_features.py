"""Offline tests for 0.8 P1–P3 features."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from autocausal import AutoCausal, __version__
from autocausal.contracts import (
    SCHEMA_CAUSAL_EDGE,
    SCHEMA_FABRIC_BUNDLE,
    SCHEMA_MINE_REPORT,
    SCHEMA_SEARCH_DAG,
)
from autocausal.discovery import discover_ensemble, discover_relationships
from autocausal.impute import impute_dataframe
from autocausal.ingest import load_sqlalchemy
from autocausal.join import align, suggest_keys
from autocausal.mining import mine
from autocausal.panel import PanelSpec, panel_lag
from autocausal.qc import validate_frame
from autocausal.roles import infer_column_roles
from autocausal.suite_tools import refute


def _toy_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = 0.7 * z + rng.normal(scale=0.4, size=n)
    y = 0.6 * x + 0.2 * z + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"z": z, "x": x, "y": y, "noise": rng.normal(size=n)})


def test_version_0_8_or_newer():
    # 0.8 P1–P3 APIs remain; suites + skilling ship in 0.9.x; backends in 0.11
    assert (
        __version__.startswith("0.8")
        or __version__.startswith("0.9")
        or __version__.startswith("0.10")
        or __version__.startswith("0.11")
    )


def test_fabric_contracts_mine_edges_bundle():
    df = _toy_df()
    ac = AutoCausal.from_dataframe(df)
    ac.mine()
    result = ac.discover(qc="off", use_iv=False)
    mr = ac.mining.to_mine_report()
    assert mr["schema"] == SCHEMA_MINE_REPORT
    assert "payload" in mr and mr["payload"]["backend"]
    edges = result.to_causal_edges()
    assert isinstance(edges, list)
    if edges:
        assert edges[0]["schema"] == SCHEMA_CAUSAL_EDGE
        assert "source" in edges[0]["payload"]
    bundle = ac.to_fabric_bundle()
    assert bundle["schema"] == SCHEMA_FABRIC_BUNDLE
    assert bundle["payload"]["mine_report"]["schema"] == SCHEMA_MINE_REPORT
    dag = result.to_search_dag()
    assert dag["schema"] == SCHEMA_SEARCH_DAG
    assert "nodes" in dag["payload"]


def test_discovery_stability_honest_confidence():
    df = _toy_df(120)
    roles = infer_column_roles(df)
    res = discover_relationships(
        df, roles=roles, use_iv=False, stability=True, bootstrap_n=8, seed=1
    )
    assert res.stability_enabled
    assert res.bootstrap_n == 8
    for e in res.edges:
        if e.get("type") == "association":
            assert "stability" in e
            assert "confidence_raw" in e or e.get("confidence") is not None
            # honest: confidence <= raw when both present
            if "confidence_raw" in e:
                assert e["confidence"] <= e["confidence_raw"] + 1e-9


def test_discover_ensemble_consensus():
    df = _toy_df(100)
    roles = infer_column_roles(df)
    res = discover_ensemble(df, roles=roles, use_iv=False, min_methods=1, bootstrap_n=4)
    assert res.method == "consensus"
    assert "score_pc_lite" in (res.ensemble_methods or [])
    ac = AutoCausal.from_dataframe(df)
    r2 = ac.discover_ensemble(qc="off", use_iv=False, min_abs_corr=0.1)
    assert r2.method == "consensus"


def test_qc_gate_id_leakage_and_hook():
    df = pd.DataFrame(
        {
            "user_id": list(range(40)),
            "x": np.linspace(0, 1, 40),
            "y": np.linspace(1, 2, 40),
        }
    )
    report = validate_frame(df)
    codes = {i.code for i in report.issues}
    assert "id_leakage_high_cardinality" in codes
    assert report.blocked

    ac = AutoCausal.from_dataframe(df)
    with pytest.raises(ValueError, match="QC blocked"):
        ac.discover(qc="block", use_iv=False)

    # warn mode should not raise
    ac2 = AutoCausal.from_dataframe(_toy_df())
    ac2.discover(qc="warn", use_iv=False)
    assert ac2.qc_report is not None


def test_enrich_from_text_merges_into_guide_context():
    df = _toy_df()
    ac = AutoCausal.from_dataframe(df).mine()
    ac.enrich_from_text("Randomized treatment increases revenue, instrument lottery")
    assert ac.nlp_hints is not None
    ctx = ac._guide_context("Randomized treatment increases revenue, instrument lottery")
    assert "nlp_hints" in ctx
    assert ctx.get("candidates") is not None
    guide = ac.guide(text="Does treatment cause revenue?", use_slm=False)
    assert guide is not None
    assert ac.nlp_hints is not None


def test_panel_spec_and_lag_features():
    rng = np.random.default_rng(2)
    rows = []
    for unit in range(5):
        for t in range(6):
            rows.append({"unit_id": unit, "year": 2000 + t, "y": rng.normal(), "x": rng.normal()})
    df = pd.DataFrame(rows)
    spec = PanelSpec(entity="unit_id", time="year", treatment=None, outcome="y")
    assert spec.validate(df) == []
    feat = panel_lag(df, spec, ["y", "x"], periods=1)
    assert any(c.startswith("lag1_") for c in feat.created)
    ac = AutoCausal.from_dataframe(df)
    ac.set_panel("unit_id", "year", outcome="y")
    ac.panel_features(["y"], kind="lag")
    assert "lag1_y" in ac.df.columns


def test_causaliv_request_soft():
    ac = AutoCausal.from_dataframe(_toy_df())
    ac.mine()
    ac.discover(qc="off", use_iv=True)
    spec = ac.to_causaliv_request()
    assert spec["schema"] == "CausalIVRequest.v1"
    assert "notes" in spec
    assert spec["soft"] is True
    assert "causaliv_available" in spec


def test_sensitivity_on_autocausal():
    ac = AutoCausal.from_dataframe(_toy_df())
    ac.discover(qc="off", use_iv=False)
    sens = ac.sensitivity(text="marketing conversion")
    assert sens.domain_hint
    assert ac.result is not None
    assert ac.result.sensitivity_report is not None


def test_join_align_generic():
    a = pd.DataFrame({"id": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"id": [2, 3, 4], "y": [9.0, 8.0, 7.0]})
    keys = suggest_keys([a, b])
    assert "id" in keys
    out, report = align([a, b], keys="id", how="outer")
    assert len(out) == 4
    assert report.n_frames == 2
    ac = AutoCausal.from_dataframe(a)
    ac.join_frames(b, keys="id", how="left")
    assert "y" in ac.df.columns


def test_refute_placebo_and_soft_dowhy():
    df = _toy_df(60)
    edge = {"source": "x", "target": "y", "score": 0.5}
    r = refute(edge, method="placebo", df=df, seed=0)
    assert r.ok
    assert r.backend == "builtin"
    assert "placebo_corr" in r.data
    soft = refute(edge, method="dowhy", df=df)
    assert soft.ok
    # Missing → soft_skip; installed → real DoWhy backend
    assert soft.soft_skip is True or str(soft.backend).startswith("dowhy")


def test_sql_chunksize_sample_n(tmp_path):
    db = tmp_path / "t.db"
    url = f"sqlite:///{db}"
    from sqlalchemy import create_engine, text

    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE t (a INTEGER, b REAL)"))
        for i in range(50):
            conn.execute(text("INSERT INTO t (a, b) VALUES (:a, :b)"), {"a": i, "b": float(i)})
    eng.dispose()

    df = load_sqlalchemy(url, table="t", chunksize=10, sample_n=15, sample_seed=0)
    assert len(df) == 15
    ac = AutoCausal.from_sqlalchemy(url, table="t", chunksize=20, sample_n=12, sample_seed=1)
    assert len(ac.df) == 12


def test_imputation_mechanism_diagnostics():
    df = _toy_df(40)
    df.loc[0:10, "y"] = np.nan
    # make missingness depend on x → MAR hint
    df.loc[df["x"] > df["x"].median(), "noise"] = np.nan
    _, report = impute_dataframe(df, method="median_mode")
    assert report.mechanism_hint in {
        "MCAR_plausible",
        "MAR_suspected",
        "MNAR_possible",
        "unknown",
        "none",
    }
    assert report.mechanism_notes
    assert "mean_missing" in report.diagnostics
    d = report.to_dict()
    assert "mechanism_hint" in d


def test_auto_result_fabric_and_py_typed():
    from pathlib import Path

    typed = Path(__file__).resolve().parents[1] / "src" / "autocausal" / "py.typed"
    assert typed.is_file()
    df = _toy_df(50)
    # write temp csv for auto()
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    ar = AutoCausal.auto(path, text="x causes y", use_slm=False, second_pass=False)
    assert ar.sensitivity_report is not None
    bundle = ar.to_fabric_bundle(n_rows=len(df), n_cols=len(df.columns))
    assert bundle["schema"] == SCHEMA_FABRIC_BUNDLE
    # round-trip json
    json.loads(ar.to_json())


def test_mine_report_n_rows():
    df = _toy_df(30)
    mr = mine(df)
    assert mr.n_rows == 30
    env = mr.to_mine_report()
    assert env["payload"]["n_rows"] == 30
