"""Dedicated AutoEDA actions — callable + SLM-selectable registry.

Library-first::

    from autocausal.suites.autoeda import EDAActions
    roles = EDAActions.suggest_roles(df)
    print(EDAActions.list())
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from autocausal.suites.action_protocol import ActionRegistry, ActionResult
from autocausal.suites.autoeda.report import RoleProposal

__all__ = [
    "EDA_REGISTRY",
    "EDAActions",
    "summarize_distributions",
    "correlation_matrix",
    "cardinality_report",
    "suggest_roles",
    "qc_snapshot",
    "leakage_hints",
    "mining_profile",
]

EDA_REGISTRY = ActionRegistry("autoeda")


def summarize_distributions(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
) -> ActionResult:
    """Numeric mean/std/min/max (+ dtypes / missingness snapshot)."""
    cols = list(columns) if columns else list(df.columns)
    cols = [c for c in cols if c in df.columns]
    dtypes = {str(c): str(df[c].dtype) for c in cols}
    missingness = {str(c): float(df[c].isna().mean()) for c in cols}
    numeric_summary: dict[str, dict[str, float]] = {}
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        numeric_summary[str(c)] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "nunique": float(s.nunique()),
        }
    return ActionResult(
        name="summarize_distributions",
        payload={
            "dtypes": dtypes,
            "missingness": missingness,
            "numeric_summary": numeric_summary,
        },
        n_affected=len(numeric_summary),
    )


EDA_REGISTRY.register("summarize_distributions", summarize_distributions)


def correlation_matrix(
    df: pd.DataFrame,
    *,
    max_cols: int = 12,
    columns: Optional[Sequence[str]] = None,
) -> ActionResult:
    """Pairwise numeric correlations."""
    if columns:
        numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    corr_cols = numeric_cols[:max_cols]
    correlations: dict[str, dict[str, float]] = {}
    if len(corr_cols) >= 2:
        cm = df[corr_cols].corr(numeric_only=True)
        for a in cm.columns:
            correlations[str(a)] = {
                str(b): float(cm.loc[a, b])
                for b in cm.columns
                if a != b and np.isfinite(cm.loc[a, b])
            }
    return ActionResult(
        name="correlation_matrix",
        payload={"correlations": correlations, "columns": [str(c) for c in corr_cols]},
        n_affected=len(correlations),
    )


EDA_REGISTRY.register("correlation_matrix", correlation_matrix)


def cardinality_report(df: pd.DataFrame) -> ActionResult:
    """Per-column nunique + missingness."""
    cardinality = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns}
    missingness = {str(c): float(df[c].isna().mean()) for c in df.columns}
    return ActionResult(
        name="cardinality_report",
        payload={"cardinality": cardinality, "missingness": missingness},
        n_affected=len(cardinality),
    )


EDA_REGISTRY.register("cardinality_report", cardinality_report)


def _score_role(name: str, series: pd.Series, role: str) -> float:
    cl = name.lower()
    score = 0.0
    if role == "outcome":
        if any(k in cl for k in ("y", "outcome", "sales", "return", "target", "revenue", "churn")):
            score += 2.0
        if pd.api.types.is_numeric_dtype(series):
            score += 0.5
    elif role == "treatment":
        if any(k in cl for k in ("x", "treat", "campaign", "exposure", "spend", "demand")):
            score += 2.0
        if series.nunique(dropna=True) <= 3:
            score += 0.3
    elif role == "instrument":
        if any(k in cl for k in ("z", "instrument", "iv", "lottery", "assign", "random")):
            score += 2.5
    elif role == "confounder":
        if any(k in cl for k in ("age", "size", "macro", "control", "covar", "region", "gender")):
            score += 1.5
    return score


def suggest_roles(
    df: pd.DataFrame,
    *,
    role_hypotheses: Optional[dict[str, list[str]]] = None,
) -> ActionResult:
    """Propose X/Y/Z/W role hypotheses from names + optional SLM hints."""
    scores_y: dict[str, float] = {}
    scores_x: dict[str, float] = {}
    scores_z: dict[str, float] = {}
    scores_w: dict[str, float] = {}
    for c in df.columns:
        s = df[c]
        scores_y[c] = _score_role(str(c), s, "outcome")
        scores_x[c] = _score_role(str(c), s, "treatment")
        scores_z[c] = _score_role(str(c), s, "instrument")
        scores_w[c] = _score_role(str(c), s, "confounder")

    hyp = role_hypotheses or {}
    for c in hyp.get("outcome") or []:
        if c in scores_y:
            scores_y[c] += 2.0
    for c in hyp.get("treatment") or []:
        if c in scores_x:
            scores_x[c] += 2.0
    for c in hyp.get("instrument") or []:
        if c in scores_z:
            scores_z[c] += 2.0
    for c in hyp.get("confounder") or []:
        if c in scores_w:
            scores_w[c] += 1.5

    def _best(scores: dict[str, float], exclude: set[str]) -> Optional[str]:
        cands = [(c, v) for c, v in scores.items() if c not in exclude and v > 0]
        if not cands:
            for c in df.columns:
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c]):
                    return str(c)
            return None
        cands.sort(key=lambda kv: kv[1], reverse=True)
        return str(cands[0][0])

    y = _best(scores_y, set())
    x = _best(scores_x, {y} if y else set())
    used = {c for c in (y, x) if c}
    z_sorted = sorted(
        ((c, v) for c, v in scores_z.items() if c not in used and v > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    instruments = [str(c) for c, _ in z_sorted[:3]]
    used |= set(instruments)
    confounders = [
        str(c)
        for c, v in sorted(scores_w.items(), key=lambda kv: -kv[1])
        if c not in used and v > 0
    ][:5]
    time_col = next(
        (
            str(c)
            for c in df.columns
            if any(k in str(c).lower() for k in ("date", "time", "year", "month"))
            or str(df[c].dtype).startswith("datetime")
        ),
        None,
    )
    group_col = next(
        (
            str(c)
            for c in df.columns
            if any(k in str(c).lower() for k in ("group", "unit", "region"))
        ),
        None,
    )
    roles = RoleProposal(
        outcome=y,
        treatment=x,
        instruments=instruments,
        confounders=confounders,
        time_col=time_col,
        group_col=group_col,
        scores={
            **{f"y:{k}": v for k, v in scores_y.items()},
            **{f"x:{k}": v for k, v in scores_x.items()},
            **{f"z:{k}": v for k, v in scores_z.items()},
        },
    )
    return ActionResult(
        name="suggest_roles",
        payload={"roles": roles.to_dict()},
        notes=["Role proposals are hypotheses — not ground truth."],
        n_affected=1,
    )


EDA_REGISTRY.register("suggest_roles", suggest_roles)


def qc_snapshot(df: pd.DataFrame) -> ActionResult:
    try:
        from autocausal.qc import validate_frame

        report = validate_frame(df)
        return ActionResult(
            name="qc_snapshot",
            payload={"qc": report.to_dict()},
            notes=["QC is a hygiene gate — not identification."],
            n_affected=len(report.issues),
        )
    except Exception as e:
        return ActionResult(
            name="qc_snapshot",
            warnings=[f"QC soft-fail: {type(e).__name__}: {e}"],
        )


EDA_REGISTRY.register("qc_snapshot", qc_snapshot)


def leakage_hints(
    df: pd.DataFrame,
    *,
    outcome: Optional[str] = None,
    correlations: Optional[dict[str, dict[str, float]]] = None,
) -> ActionResult:
    """Name-based + high-corr leakage hints."""
    hints: list[str] = []
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ("target_leak", "future_", "y_true", "ground_truth", "label_encoded")):
            hints.append(f"Column name `{c}` looks like label leakage.")
    if outcome and correlations and outcome in correlations:
        for other, r in correlations[outcome].items():
            if abs(r) >= 0.99 and other != outcome:
                hints.append(
                    f"`{other}` |corr|={abs(r):.3f} with outcome `{outcome}` — possible leakage."
                )
    return ActionResult(
        name="leakage_hints",
        payload={"leakage_hints": hints},
        notes=["Leakage hints are exploratory — review before modeling."],
        n_affected=len(hints),
    )


EDA_REGISTRY.register("leakage_hints", leakage_hints)


def mining_profile(df: pd.DataFrame) -> ActionResult:
    """Soft hook to ``autocausal.mining.profile_dataframe``."""
    try:
        from autocausal.mining import profile_dataframe

        return ActionResult(
            name="mining_profile",
            payload={"mining_profile": profile_dataframe(df)},
            n_affected=df.shape[1],
        )
    except Exception as e:
        return ActionResult(
            name="mining_profile",
            warnings=[f"Mining profile soft-fail: {type(e).__name__}: {e}"],
        )


EDA_REGISTRY.register("mining_profile", mining_profile)


class EDAActions:
    """Namespace for dedicated EDA actions + registry."""

    registry = EDA_REGISTRY
    summarize_distributions = staticmethod(summarize_distributions)
    correlation_matrix = staticmethod(correlation_matrix)
    cardinality_report = staticmethod(cardinality_report)
    suggest_roles = staticmethod(suggest_roles)
    qc_snapshot = staticmethod(qc_snapshot)
    leakage_hints = staticmethod(leakage_hints)
    mining_profile = staticmethod(mining_profile)

    @classmethod
    def list(cls) -> list[str]:
        return cls.registry.list()

    @classmethod
    def run(cls, name: str, df: pd.DataFrame, **kwargs: Any) -> ActionResult:
        return cls.registry.run(name, df, **kwargs)

    @classmethod
    def default_sequence(cls) -> list[str]:
        return [
            "summarize_distributions",
            "correlation_matrix",
            "cardinality_report",
            "suggest_roles",
            "qc_snapshot",
            "leakage_hints",
            "mining_profile",
        ]
