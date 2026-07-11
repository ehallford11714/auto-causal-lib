"""Production-readiness: safe defaults, mode gates, synthetic IV honesty."""

from __future__ import annotations

import pytest

from autocausal import AutoCausal, load_dataset
from autocausal.iv import AUTO_INSTRUMENT_COL
from autocausal.production import (
    SYNTHETIC_CONFIDENCE_CAP,
    apply_mode_defaults,
    production_checklist,
)


def test_discover_default_no_auto_iv_on_iris():
    """Default discover on iris must not invent iv_2sls via auto_instrument."""
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df)
    result = ac.discover(use_iv=True, qc="off", min_abs_corr=0.1)
    assert result.mode == "exploratory"
    instruments = result.candidates.get("instrument") or []
    assert AUTO_INSTRUMENT_COL not in instruments
    assert not any(str(i).startswith("auto_instrument") for i in instruments)
    syn_iv = [
        e
        for e in result.edges
        if e.get("type") == "iv_2sls"
        and (e.get("auto_instrument") or e.get("synthetic") or e.get("identification") == "none")
    ]
    assert syn_iv == []
    # Default path should not emit synthetic IV edges at all on iris
    assert not any(
        e.get("type") == "iv_2sls" and str(e.get("instrument") or "").startswith("auto_instrument")
        for e in result.edges
    )


def test_auto_instrument_true_tags_synthetic():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df)
    result = ac.discover(
        use_iv=True, auto_instrument=True, qc="off", min_abs_corr=0.1
    )
    assert AUTO_INSTRUMENT_COL in (result.candidates.get("instrument") or []) or any(
        "SYNTHETIC" in n or "synthetic" in n.lower() for n in result.notes
    )
    iv_edges = [e for e in result.edges if e.get("type") == "iv_2sls"]
    for e in iv_edges:
        assert e.get("auto_instrument") is True or e.get("synthetic") is True
        assert e.get("identification") == "none"
        assert float(e.get("confidence") or 0) <= SYNTHETIC_CONFIDENCE_CAP + 1e-9
    md = result.report()
    assert "EPISTEMIC" in md
    assert "Synthetic IV" in md or "synthetic" in md.lower()


def test_production_mode_refuses_synthetic_iv():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df, mode="production")
    with pytest.raises(ValueError, match="refuses auto_instrument"):
        ac.discover(use_iv=True, auto_instrument=True, qc="block")


def test_production_mode_no_synthetic_edges():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df)
    result = ac.discover(mode="production", use_iv=True, qc="block", min_abs_corr=0.1)
    assert result.mode == "production"
    assert not any(
        e.get("type") == "iv_2sls"
        and (
            e.get("auto_instrument")
            or e.get("synthetic")
            or str(e.get("instrument") or "").startswith("auto_instrument")
        )
        for e in result.edges
    )
    assert AUTO_INSTRUMENT_COL not in (result.candidates.get("instrument") or [])


def test_production_estimate_requires_explicit_roles():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df, mode="production")
    ac.discover(use_iv=False, qc="block", min_abs_corr=0.1)
    with pytest.raises(ValueError, match="explicit y="):
        ac.estimate()
    # With explicit roles, estimate proceeds
    cols = list(ac.df.columns)
    est = ac.estimate(y=cols[0], d=cols[1], backend="builtin_ols")
    assert est is not None


def test_production_refuses_auto_add_instrument():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df, mode="production")
    with pytest.raises(ValueError, match="refuses auto_add_instrument"):
        ac.auto_add_instrument(treatment=list(df.columns)[0])


def test_apply_mode_defaults_production():
    s = apply_mode_defaults(mode="production")
    assert s.mode == "production"
    assert s.auto_instrument is False
    assert s.allow_iv_fallback is False
    assert s.qc == "block"
    assert s.stability is True
    assert s.ensemble is True


def test_production_checklist_default_auto_instrument():
    cl = production_checklist(production=True)
    assert cl["schema"] == "AutoCausalProductionChecklist.v1"
    by_id = {c["id"]: c for c in cl["checks"]}
    assert by_id["default_auto_instrument_false"]["ok"] is True
    assert by_id["discover_relationships_auto_instrument_false"]["ok"] is True


def test_doctor_production_flag():
    from autocausal.doctor import doctor_report

    r = doctor_report(production=True)
    assert "production_checklist" in r
    assert r.get("production_ok") is True
    assert r["version"].startswith("0.14.")
