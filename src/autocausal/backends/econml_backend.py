"""Soft EconML estimation adapters (LinearDML / CausalForestDML)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from autocausal.backends._common import resolve_roles, soft_import, soft_skip_result

INSTALL = "pip install 'auto-causal-lib[causal-extra]'  # includes econml"


def available() -> bool:
    return soft_import("econml") is not None


def estimate(
    df: pd.DataFrame,
    *,
    y: Optional[str] = None,
    d: Optional[str] = None,
    x: Optional[list[str]] = None,
    candidates: Optional[dict[str, list[str]]] = None,
    method: str = "econml_linear_dml",
    random_state: int = 0,
    **_kwargs: Any,
) -> dict[str, Any]:
    method_l = (method or "econml_linear_dml").lower().strip()
    if method_l in ("econml", "linear_dml", "econml_linear_dml"):
        method_l = "econml_linear_dml"
    elif method_l in ("causal_forest", "econml_causal_forest", "causalforestdml"):
        method_l = "econml_causal_forest"

    notes = [
        "EconML CATE estimates require unconfoundedness; mined roles are exploratory.",
    ]
    if not available():
        out = soft_skip_result(method=method_l, module="econml", install=INSTALL, notes=notes)
        out["estimate"] = None
        return out

    roles = resolve_roles(df, y=y, d=d, x=x, candidates=candidates)
    yy, dd, xx = roles["y"], roles["d"], roles["x"]
    if not yy or not dd or yy not in df.columns or dd not in df.columns:
        return {
            "ok": True,
            "soft_skip": True,
            "method": method_l,
            "backend": "econml",
            "estimate": None,
            "data": {},
            "notes": notes + ["Could not resolve treatment/outcome columns."],
            "error": None,
        }
    cols = [yy, dd] + list(xx)
    work = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 40:
        return {
            "ok": True,
            "soft_skip": True,
            "method": method_l,
            "backend": "econml",
            "estimate": None,
            "data": {"n": len(work)},
            "notes": notes + ["Need ≥40 complete rows for EconML."],
            "error": None,
        }
    try:
        Y = work[yy].to_numpy(dtype=float)
        T = work[dd].to_numpy(dtype=float)
        X = work[xx].to_numpy(dtype=float) if xx else np.ones((len(work), 1), dtype=float)
        if not xx:
            notes.append("No controls — used intercept-only X.")

        if method_l == "econml_causal_forest":
            from econml.dml import CausalForestDML
            from sklearn.ensemble import GradientBoostingRegressor

            est = CausalForestDML(
                model_y=GradientBoostingRegressor(
                    n_estimators=30,
                    max_depth=2,
                    random_state=int(random_state),
                ),
                model_t=GradientBoostingRegressor(
                    n_estimators=30,
                    max_depth=2,
                    random_state=int(random_state),
                ),
                n_estimators=40,
                min_samples_leaf=5,
                random_state=int(random_state),
            )
            backend = "econml.CausalForestDML"
        else:
            from econml.dml import LinearDML
            from sklearn.linear_model import LassoCV

            est = LinearDML(
                model_y=LassoCV(cv=3, max_iter=2000),
                model_t=LassoCV(cv=3, max_iter=2000),
                random_state=int(random_state),
            )
            backend = "econml.LinearDML"

        est.fit(Y, T, X=X)
        cate = np.asarray(est.effect(X), dtype=float).reshape(-1)
        ate = float(np.nanmean(cate))
        ate_se = float(np.nanstd(cate) / max(np.sqrt(len(cate)), 1.0))
        return {
            "ok": True,
            "soft_skip": False,
            "method": method_l,
            "backend": backend,
            "estimate": {
                "ate": round(ate, 6),
                "ate_se_heterogeneity": round(ate_se, 6),
                "cate_mean": round(ate, 6),
                "cate_std": round(float(np.nanstd(cate)), 6),
                "cate_p10": round(float(np.nanpercentile(cate, 10)), 6),
                "cate_p90": round(float(np.nanpercentile(cate, 90)), 6),
                "y": yy,
                "d": dd,
                "x": xx,
                "n": len(work),
            },
            "data": {"n_cate": int(len(cate))},
            "notes": notes,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "soft_skip": False,
            "method": method_l,
            "backend": "econml",
            "estimate": None,
            "data": {},
            "notes": notes,
            "error": f"{type(e).__name__}: {e}",
        }
