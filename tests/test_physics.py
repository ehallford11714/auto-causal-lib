"""Offline tests for physics engine + autocausal physics loop."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autocausal import AutoCausal
from autocausal.physics import (
    PhysicsCausalSuite,
    PhysicsEngine,
    ground_physical,
    state_from_dataframe,
    try_nextframeseq_npe,
)


def _synth_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    position = np.cos(t) + 0.05 * rng.normal(size=n)
    velocity = -np.sin(t) + 0.05 * rng.normal(size=n)
    force = -0.3 * position + 0.1 * rng.normal(size=n)
    energy = 0.5 * velocity**2 + 0.15 * position**2
    treatment = (force > 0).astype(float)
    outcome = 0.8 * treatment + 0.2 * energy + 0.1 * rng.normal(size=n)
    return pd.DataFrame(
        {
            "position": position,
            "velocity": velocity,
            "force": force,
            "energy": energy,
            "treatment": treatment,
            "outcome": outcome,
            "noise": rng.normal(size=n),
        }
    )


def test_rollout_damped_oscillator():
    df = _synth_df()
    state = state_from_dataframe(df, columns=["position", "velocity", "force"])
    eng = PhysicsEngine(system="damped_oscillator", prefer_nfs=False, seed=1)
    traj = eng.rollout(state, horizon=5)
    assert len(traj.points) == 5
    assert traj.system == "damped_oscillator"
    assert traj.predictions
    assert all(len(p.uncertainty) == 3 for p in traj.points)
    assert traj.to_markdown()
    assert "position" in traj.predictions


def test_rollout_systems_and_edge_coupling():
    state = state_from_dataframe(_synth_df(), columns=["position", "velocity"])
    edges = [{"source": "position", "target": "velocity", "score": 0.5, "confidence": 0.6}]
    for system in ("damped_oscillator", "drift_diffusion", "linear_ode"):
        eng = PhysicsEngine(system=system, prefer_nfs=False).fit_from_edges(edges)
        traj = eng.rollout(state, horizon=3)
        assert traj.horizon == 3
        assert len(traj.points) == 3


def test_ground_physical_mechanics():
    edges = [
        {"source": "force", "target": "velocity", "confidence": 0.7, "score": 0.5},
        {"source": "energy", "target": "stability", "confidence": 0.4, "score": 0.2},
        {"source": "foo", "target": "bar", "confidence": 0.3, "score": 0.1},
    ]
    eng = PhysicsEngine(prefer_nfs=False)
    state = state_from_dataframe(_synth_df(), columns=["force", "velocity", "energy"])
    traj = eng.rollout(state, horizon=2)
    report = ground_physical(edges, traj, domain="mechanics-lite")
    assert report.insights
    assert any(i.analogy_label == "literal" for i in report.insights)
    assert report.to_markdown()


def test_ground_physical_markets_analogy():
    edges = [{"source": "price", "target": "return", "confidence": 0.6, "score": 0.4}]
    report = ground_physical(edges, None, domain="markets-as-dynamics")
    assert report.insights
    assert report.insights[0].analogy_label == "analogy"


def test_physics_suite_loop():
    df = _synth_df()
    suite = PhysicsCausalSuite.from_dataframe(df, prefer_nfs=False)
    result = suite.loop(
        horizon=4,
        text="what drives outcome?",
        domain="auto",
        second_pass=True,
        min_abs_corr=0.05,
    )
    assert result.horizon == 4
    assert len(result.trajectory.points) == 4
    assert result.discovery is not None
    assert result.physical_grounding.insights is not None
    assert result.guide is not None
    assert result.to_markdown()
    assert "mine" in " ".join(result.notes)


def test_physics_suite_rollout_only():
    suite = PhysicsCausalSuite.from_dataframe(_synth_df(), prefer_nfs=False)
    traj = suite.rollout(horizon=3, use_edges=False)
    assert len(traj.points) == 3


def test_autocausal_physics_loop_api():
    ac = AutoCausal.from_dataframe(_synth_df())
    result = ac.physics_loop(horizon=3, text="force → outcome?", second_pass=False)
    assert ac.physics_result is result
    assert len(result.trajectory.points) == 3


def test_auto_physics_flag(tmp_path):
    path = tmp_path / "phys.csv"
    _synth_df().to_csv(path, index=False)
    out = AutoCausal.auto(str(path), physics=True, physics_horizon=2, second_pass=False)
    assert out.physics is not None
    assert out.physics.get("horizon") == 2
    assert out.to_markdown()


def test_try_nextframeseq_soft():
    # Must not raise whether or not NFS is installed
    cls = try_nextframeseq_npe()
    assert cls is None or callable(cls) or isinstance(cls, type)


def test_cli_physics_rollout(tmp_path):
    from autocausal.cli import main

    path = tmp_path / "cli_phys.csv"
    _synth_df(n=40).to_csv(path, index=False)
    rc = main(["physics", "rollout", "--csv", str(path), "--horizon", "2", "--format", "json"])
    assert rc == 0
