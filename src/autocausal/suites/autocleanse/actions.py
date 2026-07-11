"""Dedicated AutoCleanse actions — callable + SLM-selectable registry.

Library-first::

    from autocausal.suites.autocleanse import CleanseActions
    out = CleanseActions.impute(df, method="auto")
    print(CleanseActions.list())
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from autocausal.suites.action_protocol import ActionRegistry, ActionResult

__all__ = [
    "CLEANSE_REGISTRY",
    "CleanseActions",
    "profile_missingness",
    "coerce_types",
    "drop_duplicates",
    "drop_high_null_cols",
    "drop_constant_cols",
    "flag_outliers",
    "impute",
    "strip_id_leakage",
    "qc_snapshot",
]

CLEANSE_REGISTRY = ActionRegistry("autocleanse")

ImputeMethod = Literal["auto", "median_mode", "knn", "none"]

_ID_NAME_RE = re.compile(
    r"(^id$|_id$|^uuid$|guid|ssn|email|phone|account_?number|customer_?key|row_?id|index$)",
    re.I,
)


def profile_missingness(df: pd.DataFrame) -> ActionResult:
    """Profile per-column missingness (read-only)."""
    miss = {str(c): float(df[c].isna().mean()) for c in df.columns}
    n_miss_cols = sum(1 for v in miss.values() if v > 0)
    return ActionResult(
        name="profile_missingness",
        payload={"missingness": miss, "n_cols_with_missing": n_miss_cols},
        notes=["Missingness profile is descriptive only."],
        n_affected=n_miss_cols,
    )


CLEANSE_REGISTRY.register("profile_missingness", profile_missingness)


def coerce_types(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
) -> ActionResult:
    """Coerce object columns to numeric/datetime when majority parseable."""
    out = df.copy()
    ops: list[dict[str, Any]] = []
    targets = list(columns) if columns else list(out.columns)
    for c in targets:
        if c not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[c]) or pd.api.types.is_datetime64_any_dtype(out[c]):
            continue
        if out[c].dtype == object or str(out[c].dtype) == "string":
            converted = pd.to_numeric(out[c], errors="coerce")
            if converted.notna().mean() >= 0.5:
                n_chg = int(converted.notna().sum())
                out[c] = converted
                ops.append({"op": "coerce_numeric", "columns": [c], "n_affected": n_chg})
            else:
                dt = pd.to_datetime(out[c], errors="coerce", utc=True)
                if dt.notna().mean() >= 0.5:
                    n_chg = int(dt.notna().sum())
                    out[c] = dt
                    ops.append({"op": "coerce_datetime", "columns": [c], "n_affected": n_chg})
    return ActionResult(
        name="coerce_types",
        frame=out,
        ops=ops,
        n_affected=len(ops),
    )


CLEANSE_REGISTRY.register("coerce_types", coerce_types)


def drop_duplicates(df: pd.DataFrame) -> ActionResult:
    """Drop exact duplicate rows."""
    before = len(df)
    out = df.drop_duplicates()
    n = before - len(out)
    ops = [{"op": "drop_duplicates", "columns": [], "n_affected": n, "detail": "exact rows"}] if n else []
    return ActionResult(name="drop_duplicates", frame=out, ops=ops, n_affected=n)


CLEANSE_REGISTRY.register("drop_duplicates", drop_duplicates)


def drop_high_null_cols(
    df: pd.DataFrame,
    *,
    max_missing_frac: float = 0.95,
    columns: Optional[Sequence[str]] = None,
) -> ActionResult:
    """Drop columns with missingness ≥ threshold."""
    out = df.copy()
    dropped: list[str] = []
    ops: list[dict[str, Any]] = []
    targets = list(columns) if columns else list(out.columns)
    for c in targets:
        if c not in out.columns:
            continue
        miss = float(out[c].isna().mean()) if len(out) else 0.0
        if miss >= max_missing_frac:
            out = out.drop(columns=[c])
            dropped.append(c)
            ops.append(
                {
                    "op": "drop_high_null_cols",
                    "columns": [c],
                    "n_affected": 1,
                    "detail": f"missing={miss:.0%}",
                }
            )
    return ActionResult(
        name="drop_high_null_cols",
        frame=out,
        ops=ops,
        payload={"dropped_columns": dropped},
        n_affected=len(dropped),
    )


CLEANSE_REGISTRY.register("drop_high_null_cols", drop_high_null_cols)


def drop_constant_cols(df: pd.DataFrame) -> ActionResult:
    """Drop columns with ≤1 unique non-null value."""
    out = df.copy()
    dropped: list[str] = []
    ops: list[dict[str, Any]] = []
    for c in list(out.columns):
        if out[c].nunique(dropna=True) <= 1:
            out = out.drop(columns=[c])
            dropped.append(c)
            ops.append({"op": "drop_constant_cols", "columns": [c], "n_affected": 1})
    return ActionResult(
        name="drop_constant_cols",
        frame=out,
        ops=ops,
        payload={"dropped_columns": dropped},
        n_affected=len(dropped),
    )


CLEANSE_REGISTRY.register("drop_constant_cols", drop_constant_cols)


def flag_outliers(
    df: pd.DataFrame,
    *,
    z: float = 5.0,
    columns: Optional[Sequence[str]] = None,
    winsorize: bool = True,
) -> ActionResult:
    """Flag (and optionally winsorize) numeric outliers by z-score."""
    out = df.copy()
    ops: list[dict[str, Any]] = []
    flagged: dict[str, int] = {}
    targets = list(columns) if columns else [
        c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])
    ]
    for c in targets:
        if c not in out.columns or not pd.api.types.is_numeric_dtype(out[c]):
            continue
        s = out[c].astype(float)
        mu, sd = float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0
        if sd < 1e-12 or not np.isfinite(sd):
            continue
        mask = ((s - mu) / sd).abs() > z
        n_out = int(mask.sum())
        if n_out == 0:
            continue
        flagged[c] = n_out
        if winsorize:
            lo, hi = mu - z * sd, mu + z * sd
            out[c] = s.clip(lo, hi)
            ops.append(
                {
                    "op": "winsorize_z",
                    "columns": [c],
                    "n_affected": n_out,
                    "detail": f"z>{z}",
                }
            )
        else:
            ops.append({"op": "flag_outliers", "columns": [c], "n_affected": n_out})
    return ActionResult(
        name="flag_outliers",
        frame=out if winsorize else None,
        ops=ops,
        payload={"flagged": flagged, "winsorize": winsorize},
        n_affected=sum(flagged.values()),
    )


CLEANSE_REGISTRY.register("flag_outliers", flag_outliers)


def impute(
    df: pd.DataFrame,
    *,
    method: ImputeMethod = "auto",
    columns: Optional[Sequence[str]] = None,
) -> ActionResult:
    """Impute missing values via ``autocausal.impute`` (or no-op if method=none)."""
    if method == "none" or not df.isna().any().any():
        return ActionResult(
            name="impute",
            frame=df.copy(),
            notes=["No imputation applied."],
            n_affected=0,
        )
    try:
        from autocausal.impute import impute_dataframe

        work = df.copy()
        if columns:
            # imputer is frame-wide; still fine — only missing cols change
            pass
        out, irep = impute_dataframe(work, method=method)  # type: ignore[arg-type]
        ops: list[dict[str, Any]] = []
        for col_info in getattr(irep, "columns", []) or []:
            ops.append(
                {
                    "op": f"impute_{getattr(col_info, 'strategy', 'auto')}",
                    "columns": [str(getattr(col_info, "column", ""))],
                    "n_affected": int(getattr(col_info, "missing_before", 0) or 0),
                }
            )
        return ActionResult(
            name="impute",
            frame=out,
            ops=ops,
            payload={"imputation": irep.to_dict() if hasattr(irep, "to_dict") else None},
            n_affected=len(ops),
        )
    except Exception as e:
        return ActionResult(
            name="impute",
            frame=df.copy(),
            warnings=[f"Imputer soft-fail: {type(e).__name__}: {e}"],
        )


CLEANSE_REGISTRY.register("impute", impute)


def strip_id_leakage(
    df: pd.DataFrame,
    *,
    drop: bool = False,
    max_cardinality_ratio: float = 0.95,
) -> ActionResult:
    """Detect ID-like / leakage-named columns; optionally drop them."""
    n = max(len(df), 1)
    flagged: list[str] = []
    ops: list[dict[str, Any]] = []
    out = df.copy()
    for c in list(out.columns):
        cl = str(c)
        nunq = int(out[c].nunique(dropna=True))
        id_like = bool(_ID_NAME_RE.search(cl)) and nunq >= max_cardinality_ratio * n
        leak_name = any(
            k in cl.lower() for k in ("target_leak", "future_", "y_true", "ground_truth")
        )
        if id_like or leak_name:
            flagged.append(cl)
            if drop:
                out = out.drop(columns=[c])
                ops.append({"op": "strip_id_leakage", "columns": [cl], "n_affected": 1})
    return ActionResult(
        name="strip_id_leakage",
        frame=out if drop else None,
        ops=ops,
        payload={"flagged_columns": flagged, "dropped": drop},
        notes=[
            "ID/leakage flags are hygiene hints — not causal proof.",
        ]
        if flagged
        else [],
        n_affected=len(flagged),
    )


CLEANSE_REGISTRY.register("strip_id_leakage", strip_id_leakage)


def qc_snapshot(df: pd.DataFrame) -> ActionResult:
    """Run ``autocausal.qc.validate_frame`` (read-only)."""
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


CLEANSE_REGISTRY.register("qc_snapshot", qc_snapshot)


class CleanseActions:
    """Namespace for dedicated cleanse actions + registry."""

    registry = CLEANSE_REGISTRY
    profile_missingness = staticmethod(profile_missingness)
    coerce_types = staticmethod(coerce_types)
    drop_duplicates = staticmethod(drop_duplicates)
    drop_high_null_cols = staticmethod(drop_high_null_cols)
    drop_constant_cols = staticmethod(drop_constant_cols)
    flag_outliers = staticmethod(flag_outliers)
    impute = staticmethod(impute)
    strip_id_leakage = staticmethod(strip_id_leakage)
    qc_snapshot = staticmethod(qc_snapshot)

    @classmethod
    def list(cls) -> list[str]:
        return cls.registry.list()

    @classmethod
    def run(cls, name: str, df: pd.DataFrame, **kwargs: Any) -> ActionResult:
        return cls.registry.run(name, df, **kwargs)

    @classmethod
    def default_sequence(cls) -> list[str]:
        return [
            "profile_missingness",
            "coerce_types",
            "drop_high_null_cols",
            "drop_constant_cols",
            "drop_duplicates",
            "strip_id_leakage",
            "flag_outliers",
            "impute",
            "qc_snapshot",
        ]
