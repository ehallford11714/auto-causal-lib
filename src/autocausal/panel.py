"""Panel / longitudinal helpers and soft DiD / IV handoff notes.

Epistemic honesty: panel helpers prepare structure for DiD/IV; they do not
estimate treatment effects. Soft handoff notes point to CausalIV / DoWhy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence, Union

import pandas as pd

__all__ = [
    "PanelSpec",
    "PanelFeatures",
    "panel_lag",
    "panel_diff",
    "panel_within",
    "did_handoff_notes",
    "iv_handoff_notes",
]


@dataclass
class PanelSpec:
    """Describe a panel / longitudinal layout."""

    entity: str
    time: str
    treatment: Optional[str] = None
    outcome: Optional[str] = None
    covariates: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Return list of validation problems (empty if ok)."""
        problems: list[str] = []
        for col in (self.entity, self.time):
            if col not in df.columns:
                problems.append(f"missing panel column: {col}")
        for col in [self.treatment, self.outcome, *self.covariates]:
            if col and col not in df.columns:
                problems.append(f"missing declared column: {col}")
        if self.entity in df.columns and self.time in df.columns:
            dup = df.duplicated([self.entity, self.time]).sum()
            if dup:
                problems.append(f"{int(dup)} duplicate entity-time rows")
        return problems

    def sort_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in (self.entity, self.time) if c in df.columns]
        if not cols:
            return df.copy()
        return df.sort_values(cols).reset_index(drop=True)


@dataclass
class PanelFeatures:
    """Result of panel-aware feature helpers."""

    df: pd.DataFrame
    spec: PanelSpec
    created: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "created": list(self.created),
            "notes": list(self.notes),
            "n_rows": len(self.df),
            "n_cols": len(self.df.columns),
        }


def panel_lag(
    df: pd.DataFrame,
    spec: PanelSpec,
    columns: Sequence[str],
    *,
    periods: int = 1,
    prefix: str = "lag",
) -> PanelFeatures:
    """Add within-entity lags for ``columns``."""
    work = spec.sort_frame(df)
    created: list[str] = []
    for col in columns:
        if col not in work.columns:
            continue
        name = f"{prefix}{periods}_{col}"
        work[name] = work.groupby(spec.entity, sort=False)[col].shift(periods)
        created.append(name)
    notes = [
        f"Created {len(created)} lag feature(s) (periods={periods}).",
        "Lags are structural features — not causal estimates.",
    ]
    return PanelFeatures(df=work, spec=spec, created=created, notes=notes)


def panel_diff(
    df: pd.DataFrame,
    spec: PanelSpec,
    columns: Sequence[str],
    *,
    periods: int = 1,
    prefix: str = "d",
) -> PanelFeatures:
    """First-difference within entity."""
    work = spec.sort_frame(df)
    created: list[str] = []
    for col in columns:
        if col not in work.columns:
            continue
        name = f"{prefix}{periods}_{col}"
        work[name] = work.groupby(spec.entity, sort=False)[col].diff(periods)
        created.append(name)
    notes = [
        f"Created {len(created)} difference feature(s).",
        "First-differences remove entity FE under linearity — still not identified ATE.",
    ]
    return PanelFeatures(df=work, spec=spec, created=created, notes=notes)


def panel_within(
    df: pd.DataFrame,
    spec: PanelSpec,
    columns: Sequence[str],
    *,
    prefix: str = "within",
) -> PanelFeatures:
    """Entity demeaning (within transform) for numeric columns."""
    work = spec.sort_frame(df)
    created: list[str] = []
    for col in columns:
        if col not in work.columns:
            continue
        if not pd.api.types.is_numeric_dtype(work[col]):
            continue
        name = f"{prefix}_{col}"
        work[name] = work[col] - work.groupby(spec.entity, sort=False)[col].transform("mean")
        created.append(name)
    notes = [
        f"Created {len(created)} within-transformed column(s).",
        "Within transform is a FE preparation step — not a DiD estimate.",
    ]
    return PanelFeatures(df=work, spec=spec, created=created, notes=notes)


def did_handoff_notes(spec: PanelSpec) -> list[str]:
    """Soft notes for difference-in-differences handoff (no estimation)."""
    return [
        "DiD handoff (soft): AutoCausal does not estimate DiD ATT/ATE.",
        f"Suggested entity=`{spec.entity}`, time=`{spec.time}`, "
        f"treatment=`{spec.treatment}`, outcome=`{spec.outcome}`.",
        "Prefer CausalIV / DoWhy / EconML for parallel-trends DiD with covariates: "
        f"{spec.covariates or '(none declared)'}.",
        "Check: no anticipation, parallel trends, no spillover — not verified here.",
    ]


def iv_handoff_notes(
    *,
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    instrument: Optional[str] = None,
    confounders: Optional[Sequence[str]] = None,
) -> list[str]:
    """Soft notes for IV handoff to CausalIV / suite_tools."""
    return [
        "IV handoff (soft): use AutoCausal.to_causaliv_request() for a structured spec.",
        f"Suggested Y=`{outcome}`, D=`{treatment}`, Z=`{instrument}`, "
        f"W={list(confounders or [])}.",
        "IV requires relevance + exclusion + exchangeability — not verified by AutoCausal.",
        "Install causaliv or use suite_tools / numpy 2SLS lite for exploratory IV only.",
    ]
