"""Analytic physics predictive engine (demo-ready v1) + soft NextFrameSeq NPE hook."""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from autocausal.physics.types import PhysicsState, Trajectory, TrajectoryPoint

SystemKind = Literal["damped_oscillator", "drift_diffusion", "linear_ode"]

__all__ = [
    "PhysicsEngine",
    "state_from_dataframe",
    "state_from_kpis",
    "try_nextframeseq_npe",
]


def try_nextframeseq_npe() -> Optional[Any]:
    """Soft-import NextFrameSeq NeuralPhysicsEngine if installed."""
    try:
        from nextframeseq.physics.npe import NeuralPhysicsEngine  # type: ignore

        return NeuralPhysicsEngine
    except Exception:
        return None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(str(c))
    return cols


def state_from_dataframe(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    row: Union[int, str] = -1,
) -> PhysicsState:
    """Build a PhysicsState from the last (or chosen) row of numeric columns."""
    nums = list(columns) if columns else _numeric_columns(df)
    if not nums:
        raise ValueError("No numeric columns available for physics state")
    work = df[nums].apply(pd.to_numeric, errors="coerce")
    # z-score per column for stable dynamics
    means = work.mean()
    stds = work.std().replace(0, 1.0).fillna(1.0)
    normed = (work - means) / stds
    if isinstance(row, str):
        idx = int(row)
    else:
        idx = int(row) if row >= 0 else len(normed) + int(row)
    idx = max(0, min(len(normed) - 1, idx))
    pos = [float(x) if np.isfinite(x) else 0.0 for x in normed.iloc[idx].tolist()]
    # velocity from finite difference vs previous row when available
    vel = [0.0] * len(pos)
    if idx > 0:
        prev = [float(x) if np.isfinite(x) else 0.0 for x in normed.iloc[idx - 1].tolist()]
        vel = [p - q for p, q in zip(pos, prev)]
    return PhysicsState(
        names=list(nums),
        position=pos,
        velocity=vel,
        t=float(idx),
        meta={
            "means": {c: float(means[c]) for c in nums},
            "stds": {c: float(stds[c]) for c in nums},
            "row": idx,
        },
    )


def state_from_kpis(
    kpis: dict[str, float],
    *,
    velocity: Optional[dict[str, float]] = None,
) -> PhysicsState:
    names = list(kpis.keys())
    pos = [float(kpis[n]) for n in names]
    vel = [float((velocity or {}).get(n, 0.0)) for n in names]
    return PhysicsState(names=names, position=pos, velocity=vel)


class PhysicsEngine:
    """
    Demo-ready analytic dynamics over tabular KPI / feature state.

    Systems:
      - damped_oscillator: x'' + c x' + k x = 0 (per dim, coupled lightly)
      - drift_diffusion:   x' = μ + σ ε  (Euler–Maruyama mean path; σ for bands)
      - linear_ode:        x' = A x      (fit A from edges or identity damping)

    If NextFrameSeq ``nextframeseq.physics`` is importable, ``prefer_nfs=True``
    records the NPE class as an available backend note (tabular path still uses
    the local numpy engine; graph NPE needs morpheme graphs).
    """

    def __init__(
        self,
        *,
        system: SystemKind = "damped_oscillator",
        damping: float = 0.85,
        stiffness: float = 0.15,
        drift: float = 0.0,
        diffusion: float = 0.05,
        dt: float = 1.0,
        coupling: float = 0.05,
        prefer_nfs: bool = True,
        edge_weights: Optional[dict[tuple[str, str], float]] = None,
        seed: Optional[int] = 0,
    ) -> None:
        self.system: SystemKind = system
        self.damping = float(damping)
        self.stiffness = float(stiffness)
        self.drift = float(drift)
        self.diffusion = float(diffusion)
        self.dt = float(dt)
        self.coupling = float(coupling)
        self.edge_weights = dict(edge_weights or {})
        self._rng = np.random.default_rng(seed)
        self._nfs_cls = try_nextframeseq_npe() if prefer_nfs else None
        self.backend = "numpy_analytic_v1"
        if self._nfs_cls is not None:
            self.backend = "numpy_analytic_v1+nfs_available"

    @property
    def nfs_available(self) -> bool:
        return self._nfs_cls is not None

    def fit_from_edges(self, edges: list[dict[str, Any]]) -> "PhysicsEngine":
        """Use discovery edge scores as soft coupling weights between variables."""
        for e in edges:
            src, tgt = e.get("source"), e.get("target")
            if not src or not tgt:
                continue
            score = float(e.get("score") or e.get("confidence") or 0.0)
            self.edge_weights[(str(src), str(tgt))] = score
        return self

    def _force_from_edges(self, names: list[str], x: np.ndarray) -> np.ndarray:
        f = np.zeros_like(x)
        name_to_i = {n: i for i, n in enumerate(names)}
        for (src, tgt), w in self.edge_weights.items():
            i = name_to_i.get(src)
            j = name_to_i.get(tgt)
            if i is None or j is None:
                continue
            # force on target from source state (NPE-like message pass)
            f[j] += self.coupling * w * x[i]
        return f

    def step(self, state: PhysicsState) -> PhysicsState:
        x = np.asarray(state.position, dtype=float)
        v = np.asarray(
            state.velocity if state.velocity else [0.0] * len(x),
            dtype=float,
        )
        if v.shape != x.shape:
            v = np.zeros_like(x)
        names = list(state.names)
        force = self._force_from_edges(names, x)
        dt = self.dt

        if self.system == "drift_diffusion":
            # mean path: x' = drift + coupling force; band from diffusion
            dx = (self.drift + force) * dt
            x_new = x + dx
            v_new = dx / max(dt, 1e-9)
        elif self.system == "linear_ode":
            # x' = -stiffness * x - (1-damping)*v + force  (stable linear)
            dx = (-self.stiffness * x - (1.0 - self.damping) * v + force) * dt
            x_new = x + dx
            v_new = dx / max(dt, 1e-9)
        else:
            # damped oscillator (matches NFS NPE Newtonian proxy)
            # v' = -c*v - k*x + F; x' = v
            v_new = v * self.damping - self.stiffness * x + force * dt
            x_new = x + v_new * dt

        return PhysicsState(
            names=names,
            position=[float(z) for z in x_new],
            velocity=[float(z) for z in v_new],
            t=float(state.t) + dt,
            meta=dict(state.meta),
        )

    def rollout(
        self,
        state: PhysicsState,
        horizon: int = 5,
        *,
        uncertainty_scale: Optional[float] = None,
    ) -> Trajectory:
        """Roll state forward ``horizon`` steps; attach uncertainty band stub."""
        h = max(0, int(horizon))
        scale = (
            float(uncertainty_scale)
            if uncertainty_scale is not None
            else max(self.diffusion, 0.02)
        )
        points: list[TrajectoryPoint] = []
        cur = state
        notes = [
            f"system={self.system} damping={self.damping} stiffness={self.stiffness}",
        ]
        if self._nfs_cls is not None:
            notes.append(
                "NextFrameSeq NeuralPhysicsEngine importable "
                "(graph NPE path available; tabular uses local analytic engine)."
            )
        elif self.backend.startswith("numpy"):
            notes.append("NextFrameSeq not installed — using local numpy analytic engine.")

        for t in range(1, h + 1):
            cur = self.step(cur)
            x = np.asarray(cur.position, dtype=float)
            v = np.asarray(cur.velocity or [0.0] * len(x), dtype=float)
            ke = float(0.5 * np.sum(v**2))
            pe = float(0.5 * self.stiffness * np.sum(x**2))
            # uncertainty grows ~ sqrt(t) * scale (stub band, not calibrated)
            band = [float(scale * np.sqrt(t) * (1.0 + 0.1 * abs(xi))) for xi in x]
            points.append(
                TrajectoryPoint(
                    t=t,
                    state=cur,
                    kinetic_energy=ke,
                    potential_energy=pe,
                    uncertainty=band,
                    note=f"step t+{t}",
                )
            )

        predictions: dict[str, Any] = {}
        if points:
            last = points[-1]
            for name, val, unc in zip(
                last.state.names, last.state.position, last.uncertainty
            ):
                predictions[name] = {
                    "mean": round(val, 6),
                    "lo": round(val - unc, 6),
                    "hi": round(val + unc, 6),
                    "horizon": h,
                }

        return Trajectory(
            points=points,
            backend=self.backend,
            system=self.system,
            horizon=h,
            predictions=predictions,
            notes=notes,
        )

    def predict_next(
        self,
        state: PhysicsState,
        *,
        steps: int = 1,
    ) -> dict[str, Any]:
        """Convenience: one-shot next KPI/proxy prediction with uncertainty band."""
        traj = self.rollout(state, horizon=max(1, steps))
        return traj.predictions
