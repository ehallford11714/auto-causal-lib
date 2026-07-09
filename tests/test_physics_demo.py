"""Offline tests for physics Streamlit demo helpers (no Streamlit required)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from autocausal.apps import physics_demo_path
from autocausal.apps.samples import (
    load_demo_frame,
    synthetic_affect,
    synthetic_kpi_panel,
    synthetic_markets,
    synthetic_oscillator,
)
from autocausal.physics import PhysicsCausalSuite


def test_synthetic_samples_shapes():
    osc = synthetic_oscillator(n=40)
    assert "position" in osc.columns and len(osc) == 40
    kpi = synthetic_kpi_panel(n=30)
    assert "revenue" in kpi.columns
    mkt = synthetic_markets(n=30)
    assert "price" in mkt.columns and "volatility" in mkt.columns
    aff = synthetic_affect(n=25)
    assert "valence" in aff.columns and "arousal" in aff.columns


def test_load_demo_frame_and_physics_loop():
    df = load_demo_frame("oscillator", n=50, seed=7)
    assert isinstance(df, pd.DataFrame)
    suite = PhysicsCausalSuite.from_dataframe(df, prefer_nfs=False)
    result = suite.loop(
        horizon=3,
        text="what drives outcome?",
        domain="mechanics-lite",
        second_pass=False,
        min_abs_corr=0.05,
    )
    assert len(result.trajectory.points) == 3
    assert result.physical_grounding is not None
    assert "kinetic_energy" in result.trajectory.points[0].to_dict()


def test_load_demo_markets_loop():
    df = load_demo_frame("markets", n=40, seed=2)
    result = PhysicsCausalSuite.from_dataframe(
        df, system="drift_diffusion", prefer_nfs=False
    ).loop(horizon=2, domain="markets-as-dynamics", second_pass=False, min_abs_corr=0.05)
    assert result.horizon == 2
    assert len(result.trajectory.points) == 2
    # glossary may or may not hit depending on edge set; report must serialize
    assert result.physical_grounding.to_markdown()


def test_physics_demo_path_exists():
    path = Path(physics_demo_path())
    assert path.name == "physics_streamlit.py"
    assert path.is_file()


def test_cli_physics_ui_missing_streamlit_message(monkeypatch):
    """Without streamlit, `physics ui` exits 1 with install hint (no import of app)."""
    import builtins
    import sys

    from autocausal.cli import main

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "streamlit" or name.startswith("streamlit."):
            raise ImportError("no streamlit in test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # also clear any cached streamlit
    for key in list(sys.modules):
        if key == "streamlit" or key.startswith("streamlit."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    rc = main(["physics", "ui", "--port", "8518"])
    assert rc == 1


def test_write_bundled_sample(tmp_path, monkeypatch):
    from autocausal.apps import samples as samples_mod

    monkeypatch.setattr(
        samples_mod,
        "bundled_sample_path",
        lambda kind="oscillator": tmp_path / f"{kind}_demo.csv",
    )
    df = load_demo_frame("kpi_panel", n=20, seed=1, write_bundled=True)
    assert (tmp_path / "kpi_panel_demo.csv").is_file()
    assert len(df) == 20
