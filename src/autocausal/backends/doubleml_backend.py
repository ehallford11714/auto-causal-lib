"""Soft DoubleML estimation adapter (PLR / ATE)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from autocausal.backends._common import resolve_roles, soft_import, soft_skip_result

INSTALL = "pip install 'auto-causal-lib[causal-extra]'  # includes DoubleML"


def available() -> bool:
    return soft_import("doubleml") is not None


def estimate(
    df: pd.DataFrame,
    *,
    y: Optional[str] = None,
    d: Optional[str] = None,
    x: Optional[list[str]] = None,
    candidates: Optional[dict[str, list[str]]] = None,
    n_folds: int = 3,
    random_state: int = 0,
    **_kwargs: Any,
) -> dict[str, Any]:
    notes = [
        "DoubleML PLR ATE is valid under unconfoundedness + Neyman orthogonality assumptions.",
        "Exploratory when roles are mined rather than designed.",
    ]
    if not available():
        out = soft_skip_result(method="doubleml", module="doubleml", install=INSTALL, notes=notes)
        out["estimate"] = None
        return out

    roles = resolve_roles(df, y=y, d=d, x=x, candidates=candidates)
    yy, dd, xx = roles["y"], roles["d"], roles["x"]
    if not yy or not dd or yy not in df.columns or dd not in df.columns:
        return {
            "ok": True,
            "soft_skip": True,
            "method": "doubleml",
            "backend": "doubleml",
            "estimate": None,
            "data": {},
            "notes": notes + ["Could not resolve treatment/outcome columns."],
            "error": None,
        }
    cols = [yy, dd] + list(xx)
    work = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 30:
        return {
            "ok": True,
            "soft_skip": True,
            "method": "doubleml",
            "backend": "doubleml",
            "estimate": None,
            "data": {"n": len(work)},
            "notes": notes + ["Need ≥30 complete rows for DoubleML."],
            "error": None,
        }
    try:
        import doubleml as dml
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LassoCV

        Y = work[[yy]].to_numpy(dtype=float)
        D = work[[dd]].to_numpy(dtype=float)
        if xx:
            X = work[xx].to_numpy(dtype=float)
        else:
            X = np.ones((len(work), 1), dtype=float)
            notes.append("No controls — used intercept-only X (weak identification risk).")

        data = dml.DoubleMLData.from_arrays(X, Y, D)
        ml_l = LassoCV(cv=3, max_iter=2000)
        ml_m = LassoCV(cv=3, max_iter=2000)
        try:
            # Prefer RF if sklearn available and n not tiny
            if len(work) >= 80:
                ml_l = RandomForestRegressor(
                    n_estimators=40,
                    max_depth=4,
                    random_state=int(random_state),
                )
                ml_m = RandomForestRegressor(
                    n_estimators=40,
                    max_depth=4,
                    random_state=int(random_state),
                )
        except Exception:
            pass
        model = dml.DoubleMLPLR(data, ml_l, ml_m, n_folds=max(2, int(n_folds)))
        model.fit()
        coef = float(np.asarray(model.coef).reshape(-1)[0])
        se = float(np.asarray(model.se).reshape(-1)[0]) if model.se is not None else None
        pval = None
        try:
            pval = float(np.asarray(model.pval).reshape(-1)[0])
        except Exception:
            pval = None
        return {
            "ok": True,
            "soft_skip": False,
            "method": "doubleml",
            "backend": "doubleml.DoubleMLPLR",
            "estimate": {
                "ate": round(coef, 6),
                "se": round(se, 6) if se is not None else None,
                "pvalue": round(pval, 6) if pval is not None else None,
                "y": yy,
                "d": dd,
                "x": xx,
                "n": len(work),
            },
            "data": {"summary": str(model.summary)[:500] if hasattr(model, "summary") else None},
            "notes": notes,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "soft_skip": False,
            "method": "doubleml",
            "backend": "doubleml",
            "estimate": None,
            "data": {},
            "notes": notes,
            "error": f"{type(e).__name__}: {e}",
        }
