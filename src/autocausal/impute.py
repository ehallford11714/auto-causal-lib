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
    mechanism_hint: str = "unknown"
    mechanism_notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def imputed_columns(self) -> list[str]:
        return [c.column for c in self.columns if c.missing_before > 0]

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


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


def _mechanism_diagnostics(df: pd.DataFrame) -> tuple[str, list[str], dict[str, Any]]:
    """Heuristic MCAR / MAR / MNAR hints (not formal Little's MCAR test).

    Epistemic honesty: these are exploratory diagnostics only.
    """
    notes: list[str] = [
        "Missingness mechanism hints are exploratory heuristics — not Little's MCAR / formal tests.",
    ]
    if df.empty or not df.isna().any().any():
        return "none", notes + ["No missing values observed."], {"mean_missing": 0.0}

    miss = df.isna()
    rates = miss.mean()
    mean_miss = float(rates.mean())
    # pairwise missingness correlation among columns with some missing
    miss_cols = [c for c in df.columns if miss[c].any()]
    pair_corr = 0.0
    n_pairs = 0
    if len(miss_cols) >= 2:
        mmat = miss[miss_cols].astype(float)
        corr = mmat.corr()
        vals: list[float] = []
        for i, a in enumerate(miss_cols):
            for b in miss_cols[i + 1 :]:
                v = corr.loc[a, b]
                if v == v:
                    vals.append(abs(float(v)))
        if vals:
            pair_corr = float(np.mean(vals))
            n_pairs = len(vals)

    # association of missingness with other observed numeric columns (MAR hint)
    mar_hits = 0
    mar_checked = 0
    num = df.select_dtypes(include=[np.number])
    for col in miss_cols[:8]:
        mask = miss[col]
        if mask.sum() < 3 or (~mask).sum() < 3:
            continue
        for other in num.columns:
            if other == col:
                continue
            mar_checked += 1
            a = num.loc[mask, other].dropna()
            b = num.loc[~mask, other].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            # mean shift heuristic
            if abs(float(a.mean()) - float(b.mean())) > 0.25 * (float(b.std()) + 1e-6):
                mar_hits += 1
                break

    # high missing + name hints → soft MNAR suspicion
    mnar_name = any(
        any(h in str(c).lower() for h in ("income", "salary", "age", "sensitive", "private"))
        for c in miss_cols
    )

    if mean_miss < 0.02 and pair_corr < 0.1 and mar_hits == 0:
        hint = "MCAR_plausible"
        notes.append("Low missingness with weak structure → MCAR plausible (not proven).")
    elif mar_hits > 0 or pair_corr >= 0.2:
        hint = "MAR_suspected"
        notes.append(
            f"Missingness correlates with observed fields "
            f"(mar_hits={mar_hits}/{max(mar_checked, 1)}, miss_corr≈{pair_corr:.2f}) → MAR suspected."
        )
    elif mnar_name or mean_miss >= 0.35:
        hint = "MNAR_possible"
        notes.append(
            "High missingness and/or sensitive-looking columns → MNAR possible; "
            "imputation may bias causal estimates."
        )
    else:
        hint = "unknown"
        notes.append("Insufficient signal to prefer MCAR/MAR/MNAR.")

    diag = {
        "mean_missing": round(mean_miss, 4),
        "missing_corr_mean": round(pair_corr, 4),
        "n_miss_pairs": n_pairs,
        "mar_hits": mar_hits,
        "mar_checked": mar_checked,
        "mnar_name_hint": mnar_name,
        "columns_with_missing": [str(c) for c in miss_cols[:20]],
    }
    return hint, notes, diag


def impute_dataframe(
    df: pd.DataFrame,
    *,
    method: ImputeMethod = "auto",
    knn_k: int = 5,
) -> tuple[pd.DataFrame, ImputationReport]:
    roles = infer_column_roles(df)
    before = int(df.isna().sum().sum())
    mechanism_hint, mech_notes, diagnostics = _mechanism_diagnostics(df)
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
        mechanism_hint=mechanism_hint,
        mechanism_notes=mech_notes,
        diagnostics=diagnostics,
    )
    return out, report
