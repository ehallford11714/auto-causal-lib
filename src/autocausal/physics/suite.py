"""Autocausal physics loop suite — observe → mine → discover → rollout → ground → guide."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import pandas as pd

from autocausal.physics.engine import PhysicsEngine, state_from_dataframe
from autocausal.physics.grounding import ground_physical, merge_with_domain_grounding
from autocausal.physics.types import PhysicsLoopResult, Trajectory

SystemKind = Literal["damped_oscillator", "drift_diffusion", "linear_ode"]

__all__ = ["PhysicsCausalSuite"]


class PhysicsCausalSuite:
    """
    End-to-end autocausal + physics predictive loop.

    Flow::

        observe/load → mine → impute → discover → physics rollout
        → physical ground → guide → optional second pass
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        source: str = "memory",
        system: SystemKind = "damped_oscillator",
        prefer_nfs: bool = True,
    ) -> None:
        from autocausal.api import AutoCausal

        self._ac = AutoCausal(df, source=source)
        self.system: SystemKind = system
        self.prefer_nfs = prefer_nfs
        self.engine = PhysicsEngine(system=system, prefer_nfs=prefer_nfs)
        self.last_result: Optional[PhysicsLoopResult] = None
        self.last_trajectory: Optional[Trajectory] = None

    @property
    def ac(self) -> Any:
        return self._ac

    @property
    def df(self) -> pd.DataFrame:
        return self._ac.df

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        system: SystemKind = "damped_oscillator",
        prefer_nfs: bool = True,
        **read_csv_kwargs: Any,
    ) -> "PhysicsCausalSuite":
        from autocausal.ingest import load_csv

        df = load_csv(path, **read_csv_kwargs)
        return cls(df, source=f"csv:{path}", system=system, prefer_nfs=prefer_nfs)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        source: str = "dataframe",
        system: SystemKind = "damped_oscillator",
        prefer_nfs: bool = True,
    ) -> "PhysicsCausalSuite":
        return cls(df, source=source, system=system, prefer_nfs=prefer_nfs)

    @classmethod
    def from_autocausal(
        cls,
        ac: Any,
        *,
        system: SystemKind = "damped_oscillator",
        prefer_nfs: bool = True,
    ) -> "PhysicsCausalSuite":
        suite = cls(
            ac.df,
            source=getattr(ac, "source", "autocausal"),
            system=system,
            prefer_nfs=prefer_nfs,
        )
        suite._ac = ac
        return suite

    def rollout(
        self,
        *,
        horizon: int = 5,
        columns: Optional[Sequence[str]] = None,
        system: Optional[SystemKind] = None,
        use_edges: bool = True,
    ) -> Trajectory:
        """Physics-only rollout from current frame (optionally coupled by discovery edges)."""
        if system is not None:
            self.system = system
            self.engine = PhysicsEngine(system=system, prefer_nfs=self.prefer_nfs)
        edges: list[dict[str, Any]] = []
        if use_edges and self._ac.result is not None:
            edges = list(self._ac.result.edges or [])
            self.engine.fit_from_edges(edges)
        state = state_from_dataframe(self._ac.df, columns=columns)
        traj = self.engine.rollout(state, horizon=horizon)
        self.last_trajectory = traj
        return traj

    def loop(
        self,
        *,
        horizon: int = 5,
        text: Optional[str] = None,
        domain: Union[str, Sequence[str]] = "auto",
        system: Optional[SystemKind] = None,
        use_slm: bool = False,
        second_pass: bool = True,
        use_web_ground: bool = False,
        impute_method: str = "auto",
        columns: Optional[Sequence[str]] = None,
        **discover_kwargs: Any,
    ) -> PhysicsLoopResult:
        """
        Full loop: mine → impute → discover → rollout → physical ground → guide
        → optional second-pass discover/rollout.
        """
        notes: list[str] = []
        if system is not None:
            self.system = system
            self.engine = PhysicsEngine(system=system, prefer_nfs=self.prefer_nfs)

        self._ac.mine()
        mining = self._ac.mining
        notes.append("observe: loaded tabular frame")
        notes.append("mine: column profiles + associations")
        self._ac.impute(method=impute_method)  # type: ignore[arg-type]
        notes.append(f"impute: method={impute_method}")
        result = self._ac.discover(**discover_kwargs)
        notes.append(f"discover: {len(result.edges)} edge(s)")

        self.engine.fit_from_edges(list(result.edges or []))
        state = state_from_dataframe(self._ac.df, columns=columns)
        traj = self.engine.rollout(state, horizon=horizon)
        self.last_trajectory = traj
        notes.append(
            f"physics rollout: system={traj.system} backend={traj.backend} horizon={horizon}"
        )

        phys = ground_physical(list(result.edges or []), traj, domain=domain)
        domain_g = self._ac.ground(use_web=use_web_ground)
        phys = merge_with_domain_grounding(phys, domain_g)
        notes.append(f"physical ground: {len(phys.insights)} insight(s)")

        guide = self._ac.guide(text=text, use_slm=use_slm)
        notes.append(f"guide: backend={getattr(guide, 'backend', '')}")

        did_second = False
        if second_pass and guide is not None and getattr(guide, "focus_columns", None):
            focus = [c for c in guide.focus_columns if c in self._ac.df.columns]
            if len(focus) >= 2:
                notes.append(f"second-pass focus: {focus[:12]}")
                result = self._ac.discover(focus_columns=focus, **discover_kwargs)
                self.engine.fit_from_edges(list(result.edges or []))
                focus_num = [
                    c
                    for c in focus
                    if c in self._ac.df.columns
                    and pd.api.types.is_numeric_dtype(self._ac.df[c])
                ]
                state2 = state_from_dataframe(
                    self._ac.df,
                    columns=focus_num or columns,
                )
                traj = self.engine.rollout(state2, horizon=horizon)
                self.last_trajectory = traj
                phys = ground_physical(list(result.edges or []), traj, domain=domain)
                domain_g = self._ac.ground(use_web=use_web_ground)
                phys = merge_with_domain_grounding(phys, domain_g)
                guide = self._ac.guide(text=text, use_slm=use_slm)
                did_second = True
                notes.append("second-pass: rediscover + rerollout + re-ground + guide")

        out = PhysicsLoopResult(
            trajectory=traj,
            physical_grounding=phys,
            discovery=result.to_dict() if result is not None else None,
            mining=mining.to_dict() if mining is not None and hasattr(mining, "to_dict") else None,
            guide=guide.to_dict() if guide is not None else None,
            grounding=domain_g.to_dict() if domain_g is not None else None,
            source=self._ac.source,
            horizon=horizon,
            backend=traj.backend,
            notes=notes,
            second_pass=did_second,
        )
        self.last_result = out
        return out
