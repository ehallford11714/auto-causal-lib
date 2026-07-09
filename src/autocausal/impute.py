"""Missing-value imputation with a structured report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from autocausal.roles import ColumnRole, infer_column_roles


ImputeMethod = Literal["auto", "median_mode", "knn"]


@dataclass
class ColumnImputation:
    column: str
    strategy: str
    missing_before: int
    missing_after: int
    fill_value: Any = None


@dataclass
class ImputationReport:
    method: str
    columns: list[ColumnImputation] = field(default_factory=list)
    total_missing_before: int = 0
    total_missing_after: int = 0

    @property
    def imputed_columns(self) -> list[str]:
        return [c.column for c in self.columns if c.missing_before > 0]


def _mode_value(s: pd.Series) -> Any:
    m = s.mode(dropna=True)
    if len(m) == 0:
        return None
    return m.iloc[0]


def _median_mode_impute(df: pd.DataFrame, roles: dict[str, ColumnRole]) -> tuple[pd.DataFrame, list[ColumnImputation]]:
    out = df.copy()
    details: list[ColumnImputation] = []
    for col in out.columns:
        miss = int(out[col].isna().sum())
        if miss == 0:
            continue
        role = roles.get(col, ColumnRole.UNKNOWN)
        if role == ColumnRole.NUMERIC:
            fill = float(out[col].median()) if out[col].notna().any() else 0.0
            strategy = "median"
            out[col] = out[col].fillna(fill)
        elif role == ColumnRole.DATETIME:
            # leave datetime gaps; discovery skips them
            fill = None
            strategy = "skip_datetime"
        elif role in (ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN, ColumnRole.TEXT, ColumnRole.ID):
            fill = _mode_value(out[col])
            strategy = "mode"
            if fill is None:
                fill = "__MISSING__"
                strategy = "mode_placeholder"
            out[col] = out[col].fillna(fill)
        else:
            fill = _mode_value(out[col])
            strategy = "mode"
            if fill is None:
                fill = 0
                strategy = "zero"
            out[col] = out[col].fillna(fill)
        details.append(
            ColumnImputation(
                column=str(col),
                strategy=strategy,
                missing_before=miss,
                missing_after=int(out[col].isna().sum()),
                fill_value=fill,
            )
        )
    return out, details


def _knn_lite_impute(df: pd.DataFrame, roles: dict[str, ColumnRole], *, k: int = 5) -> tuple[pd.DataFrame, list[ColumnImputation]]:
    """Simple distance-weighted KNN on numeric columns; categoricals fall back to mode."""
    out = df.copy()
    details: list[ColumnImputation] = []
    num_cols = [c for c, r in roles.items() if r == ColumnRole.NUMERIC and c in out.columns]
    if not num_cols:
        return _median_mode_impute(out, roles)

    num = out[num_cols].apply(pd.to_numeric, errors="coerce")
    # standardize with nanmean/nanstd
    mu = num.mean()
    sigma = num.std(ddof=0).replace(0, 1.0)
    z = (num - mu) / sigma

    for col in num_cols:
        miss_idx = z.index[z[col].isna()]
        miss = len(miss_idx)
        if miss == 0:
            continue
        observed = z.index[z[col].notna()]
        if len(observed) == 0:
            fill = float(mu[col]) if pd.notna(mu[col]) else 0.0
            out.loc[miss_idx, col] = fill
            details.append(
                ColumnImputation(str(col), "knn_fallback_mean", miss, int(out[col].isna().sum()), fill)
            )
            continue
        feat_cols = [c for c in num_cols if c != col]
        for i in miss_idx:
            row = z.loc[i, feat_cols]
            cand = z.loc[observed, feat_cols]
            # fill feature nan with 0 in z-space for distance only
            row_f = row.fillna(0.0)
            cand_f = cand.fillna(0.0)
            dist = np.sqrt(((cand_f - row_f) ** 2).sum(axis=1).to_numpy())
            kk = min(k, len(dist))
            if kk == 0:
                fill = float(mu[col]) if pd.notna(mu[col]) else 0.0
            else:
                nn = np.argpartition(dist, kk - 1)[:kk]
                nn_idx = observed[nn]
                vals = num.loc[nn_idx, col].to_numpy(dtype=float)
                weights = 1.0 / (dist[nn] + 1e-6)
                fill = float(np.average(vals, weights=weights))
            out.at[i, col] = fill
        details.append(
            ColumnImputation(
                column=str(col),
                strategy=f"knn_k{k}",
                missing_before=miss,
                missing_after=int(out[col].isna().sum()),
                fill_value="row-wise",
            )
        )

    # categoricals / rest via mode
    roles2 = infer_column_roles(out)
    out2, cat_details = _median_mode_impute(out, roles2)
    # only keep cat details for cols not already knn'd
    knn_cols = {d.column for d in details}
    for d in cat_details:
        if d.column not in knn_cols:
            details.append(d)
    return out2, details


def impute_dataframe(
    df: pd.DataFrame,
    *,
    method: ImputeMethod = "auto",
    knn_k: int = 5,
) -> tuple[pd.DataFrame, ImputationReport]:
    roles = infer_column_roles(df)
    before = int(df.isna().sum().sum())
    if method == "auto":
        # knn if mostly numeric and small-ish; else median/mode
        n_num = sum(1 for r in roles.values() if r == ColumnRole.NUMERIC)
        use_knn = n_num >= 2 and len(df) <= 5000 and before > 0
        method_used: str = "knn" if use_knn else "median_mode"
    else:
        method_used = method

    if method_used == "knn":
        out, details = _knn_lite_impute(df, roles, k=knn_k)
    else:
        out, details = _median_mode_impute(df, roles)
        method_used = "median_mode"

    after = int(out.isna().sum().sum())
    report = ImputationReport(
        method=method_used,
        columns=details,
        total_missing_before=before,
        total_missing_after=after,
    )
    return out, report
