"""Imputation backends for the KPI ML loop (median / sklearn / torch)."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from autocausal.ml.construct import ImputerKind, sklearn_available, torch_available
from autocausal.ml.fit_report import FitReport


def median_impute(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    from autocausal.impute import impute_dataframe

    out, report = impute_dataframe(df, method="median_mode")
    return out, {
        "method": "median_mode",
        "imputed_columns": report.imputed_columns,
        "total_missing_before": report.total_missing_before,
        "total_missing_after": report.total_missing_after,
    }


def sklearn_impute(
    df: pd.DataFrame, columns: Optional[list[str]] = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Iterative-style numeric impute via sklearn if present; else median."""
    if not sklearn_available():
        out, meta = median_impute(df)
        meta["fallback"] = "median_no_sklearn"
        return out, meta
    try:
        from sklearn.impute import SimpleImputer
        import numpy as np

        out = df.copy()
        cols = [c for c in (columns or list(df.columns)) if c in out.columns]
        num_cols = [
            c
            for c in cols
            if pd.api.types.is_numeric_dtype(pd.to_numeric(out[c], errors="coerce"))
        ]
        if not num_cols:
            return median_impute(df)
        mat = out[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        imp = SimpleImputer(strategy="median")
        filled = imp.fit_transform(mat)
        for i, c in enumerate(num_cols):
            out[c] = filled[:, i]
        # categoricals via mode
        for c in out.columns:
            if out[c].isna().any() and c not in num_cols:
                mode = out[c].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else "__MISSING__"
                out[c] = out[c].fillna(fill)
        return out, {
            "method": "sklearn_simple_median",
            "columns": num_cols,
            "n_imputed_cells": int(np.isnan(mat).sum()),
        }
    except Exception as e:
        out, meta = median_impute(df)
        meta["fallback"] = f"median_after_sklearn_error:{type(e).__name__}"
        return out, meta


def torch_impute(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    *,
    epochs: int = 40,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not torch_available():
        out, meta = median_impute(df)
        meta["fallback"] = "median_no_torch"
        return out, meta
    from autocausal.ml.torch_models import TorchMLPImputer

    imputer = TorchMLPImputer(epochs=epochs)
    out = imputer.fit_transform(df, columns=columns)
    # fill any remaining non-numeric gaps
    if out.isna().any().any():
        out, med = median_impute(out)
        extra = {"median_cleanup": med}
    else:
        extra = {}
    meta = imputer.stats.to_dict()
    meta.update(extra)
    meta["method"] = "torch_mlp"
    return out, meta


def apply_imputer(
    df: pd.DataFrame,
    kind: ImputerKind,
    columns: Optional[list[str]] = None,
    *,
    epochs: int = 40,
) -> tuple[pd.DataFrame, dict[str, Any], FitReport]:
    notes: list[str] = []
    torch_used = False
    sklearn_used = False
    if kind == "torch_mlp":
        out, meta = torch_impute(df, columns=columns, epochs=epochs)
        torch_used = meta.get("method") == "torch_mlp"
        if not torch_used:
            notes.append(meta.get("fallback", "torch fallback"))
    elif kind in ("iterative", "sklearn"):
        out, meta = sklearn_impute(df, columns=columns)
        sklearn_used = "sklearn" in str(meta.get("method", ""))
        if not sklearn_used:
            notes.append(meta.get("fallback", "sklearn fallback"))
    else:
        out, meta = median_impute(df)

    report = FitReport(
        imputer=str(meta.get("method", kind)),
        kpi_focus=list(columns or [])[:16],
        metrics={
            k: v
            for k, v in meta.items()
            if k in ("train_mae", "heldout_mask_mae", "epochs", "n_features", "n_rows", "total_missing_before", "total_missing_after", "n_imputed_cells")
        },
        notes=notes + list(meta.get("notes") or []) if isinstance(meta.get("notes"), list) else notes,
        torch_used=torch_used,
        sklearn_used=sklearn_used,
    )
    return out, meta, report
