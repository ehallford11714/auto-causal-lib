"""IV discovery: real instruments, auto_instrument, candidate injection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autocausal import AutoCausal, load_dataset
from autocausal.discovery import NAME_IV_HINTS, propose_candidates
from autocausal.iv import AUTO_INSTRUMENT_COL, synthesize_auto_instrument
from autocausal.roles import infer_column_roles
from autocausal.public_suite import ensure_bundled_public_data, load_public


def _iv_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    age = rng.normal(40, 10, size=n)
    treatment = ((0.85 * z + 0.25 * rng.normal(size=n)) > 0).astype(int)
    outcome = 1.5 * treatment + 0.4 * age + 0.25 * rng.normal(size=n)
    return pd.DataFrame({"z": z, "age": age, "treatment": treatment, "outcome": outcome})


def test_name_heuristics_include_extras():
    for hint in ("instrument_z", "assignment", "rand", "randomized", "encourage"):
        assert any(hint == h or hint in h or h in hint for h in NAME_IV_HINTS) or hint in NAME_IV_HINTS


def test_propose_candidates_finds_instrument_z():
    df = pd.DataFrame(
        {
            "treatment": [0, 1] * 30,
            "revenue": np.linspace(1, 2, 60),
            "instrument_z": np.linspace(-1, 1, 60),
            "age": np.linspace(20, 60, 60),
        }
    )
    roles = infer_column_roles(df)
    cands, notes = propose_candidates(roles, [], list(df.columns), mat=df)
    assert "instrument_z" in cands["instrument"]
    assert "treatment" in cands["treatment"]
    assert "revenue" in cands["outcome"]


def test_propose_candidates_no_fake_iv_by_default():
    df = pd.DataFrame(
        {
            "treatment": [0, 1] * 40,
            "outcome": np.linspace(0, 1, 80),
            "age": np.linspace(20, 60, 80),
            "bmi": np.linspace(18, 35, 80),
        }
    )
    roles = infer_column_roles(df)
    cands, notes = propose_candidates(roles, [], list(df.columns), mat=df)
    assert cands["instrument"] == []
    assert any("No instrument" in n or "instrument" in n.lower() for n in notes)


def test_synthetic_iv_edges_no_skip():
    ac = AutoCausal.from_dataframe(_iv_frame())
    result = ac.discover(use_iv=True, auto_instrument=False, qc="off", min_abs_corr=0.05)
    assert "z" in (result.candidates.get("instrument") or [])
    iv_edges = [e for e in result.edges if e.get("type") == "iv_2sls"]
    assert iv_edges, "expected IV edges when z/treatment/outcome present"
    assert not any("IV pass skipped" in n for n in result.notes)


def test_iv_demo_dataset_produces_iv_edges():
    df = load_dataset("iv_demo", allow_network=False)
    assert {"z", "treatment", "outcome"} <= set(df.columns)
    ac = AutoCausal.from_dataframe(df)
    result = ac.discover(use_iv=True, auto_instrument=False, qc="off", min_abs_corr=0.05)
    assert "z" in (result.candidates.get("instrument") or [])
    iv_edges = [e for e in result.edges if e.get("type") == "iv_2sls"]
    assert iv_edges
    assert not any("IV pass skipped" in n for n in result.notes)


def test_candidates_injection_overrides():
    df = _iv_frame().rename(columns={"z": "lottery_assign"})
    ac = AutoCausal.from_dataframe(df)
    result = ac.discover(
        use_iv=True,
        auto_instrument=False,
        qc="off",
        min_abs_corr=0.05,
        candidates={
            "treatment": ["treatment"],
            "outcome": ["outcome"],
            "instrument": ["lottery_assign"],
        },
    )
    assert result.candidates["instrument"][0] == "lottery_assign"
    assert any(e.get("type") == "iv_2sls" for e in result.edges)


def test_set_iv_roles_and_helper():
    df = _iv_frame().drop(columns=["z"])
    ac = AutoCausal.from_dataframe(df)
    ac.set_iv_roles(treatment="treatment", outcome="outcome")
    ac.auto_add_instrument(treatment="treatment", seed=1)
    assert AUTO_INSTRUMENT_COL in ac.df.columns
    result = ac.discover(use_iv=True, auto_instrument=False, qc="off", min_abs_corr=0.05)
    assert AUTO_INSTRUMENT_COL in (result.candidates.get("instrument") or [])
    assert any(e.get("type") == "iv_2sls" for e in result.edges)
    assert any("SYNTHETIC" in n or "exploratory" in n.lower() for n in result.notes)


def test_auto_instrument_default_avoids_skip():
    """Missing Z → auto_instrument_z added; IV pass attempts (not skipped for missing Z)."""
    df = pd.DataFrame(
        {
            "treatment": [0, 1] * 50,
            "outcome": np.linspace(0, 3, 100) + np.array([0, 1.2] * 50),
            "age": np.linspace(20, 60, 100),
        }
    )
    ac = AutoCausal.from_dataframe(df)
    result = ac.discover(use_iv=True, auto_instrument=True, qc="off", min_abs_corr=0.05)
    assert AUTO_INSTRUMENT_COL in (result.candidates.get("instrument") or [])
    assert not any("IV pass skipped" in n for n in result.notes)
    assert any("SYNTHETIC" in n or "auto-generated" in n.lower() or "exploratory" in n.lower() for n in result.notes)
    # May or may not produce edges if n/cols weak, but should not skip for missing instrument
    assert any(e.get("type") == "iv_2sls" for e in result.edges) or any(
        "first stage" in n.lower() or "2SLS" in n or "IV edges" in n for n in result.notes
    )


def test_auto_instrument_false_skips_without_z():
    df = pd.DataFrame(
        {
            "treatment": [0, 1] * 40,
            "outcome": np.linspace(0, 1, 80),
            "age": np.linspace(20, 60, 80),
        }
    )
    ac = AutoCausal.from_dataframe(df)
    result = ac.discover(use_iv=True, auto_instrument=False, qc="off", min_abs_corr=0.05)
    assert result.candidates.get("instrument") == [] or AUTO_INSTRUMENT_COL not in (
        result.candidates.get("instrument") or []
    )
    assert any("IV pass skipped" in n or "No instrument" in n for n in result.notes)


def test_iris_auto_instrument_attempts_iv():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df)
    result = ac.discover(use_iv=True, auto_instrument=True, qc="off", min_abs_corr=0.1)
    # Iris has no real IV — auto path should still attempt, with epistemic notes
    blob = " ".join(result.notes).lower()
    assert "identification" in blob or "exploratory" in blob or "synthetic" in blob
    if AUTO_INSTRUMENT_COL in (result.candidates.get("instrument") or []):
        assert any("synthetic" in n.lower() or "exploratory" in n.lower() for n in result.notes)


def test_iris_without_auto_instrument_documents_skip():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df)
    result = ac.discover(use_iv=True, auto_instrument=False, qc="off", min_abs_corr=0.15)
    # No invented name-heuristic IVs on iris morphology columns
    instruments = result.candidates.get("instrument") or []
    assert AUTO_INSTRUMENT_COL not in instruments
    assert not any(i.startswith("auto_instrument") for i in instruments)


def test_marketing_demo_has_instruments():
    ensure_bundled_public_data(force=True)
    df = load_public("marketing_demo", allow_network=False)
    assert "instrument_z" in df.columns or "assignment" in df.columns
    ac = AutoCausal.from_dataframe(df)
    result = ac.discover(use_iv=True, auto_instrument=False, qc="off", min_abs_corr=0.05)
    assert result.candidates.get("instrument")
    assert any(e.get("type") == "iv_2sls" for e in result.edges)


def test_synthesize_helper_notes():
    df = _iv_frame().drop(columns=["z"])
    out, notes = synthesize_auto_instrument(df, "treatment", seed=2)
    assert AUTO_INSTRUMENT_COL in out.columns
    assert any("SYNTHETIC" in n for n in notes)
