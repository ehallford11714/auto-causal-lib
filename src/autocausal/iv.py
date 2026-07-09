"""Optional CausalIVSuite integration + numpy 2SLS lite."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _numpy_2sls(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    controls: np.ndarray | None = None,
) -> dict[str, float]:
    """Minimal two-stage least squares: D ~ Z (+W), then Y ~ Dhat (+W)."""
    n = len(y)
    if controls is not None and controls.size:
        W = np.column_stack([np.ones(n), controls])
    else:
        W = np.ones((n, 1))
    Z = np.column_stack([W, z.reshape(-1, 1) if z.ndim == 1 else z])
    # first stage
    pi, *_ = np.linalg.lstsq(Z, d, rcond=None)
    d_hat = Z @ pi
    # second stage
    X2 = np.column_stack([W, d_hat.reshape(-1, 1)])
    beta, *_ = np.linalg.lstsq(X2, y, rcond=None)
    # residual variance / crude se
    resid = y - X2 @ beta
    dof = max(n - X2.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    try:
        xtx_inv = np.linalg.inv(X2.T @ X2)
        se = float(np.sqrt(sigma2 * xtx_inv[-1, -1]))
    except np.linalg.LinAlgError:
        se = float("nan")
    coef = float(beta[-1])
    tstat = coef / se if se and se == se and se > 0 else 0.0
    # rough p via erfc
    from math import erfc

    pval = float(erfc(abs(tstat) / np.sqrt(2.0)))
    # first-stage F approx
    ss_model = ((d_hat - d.mean()) ** 2).sum()
    ss_res = ((d - d_hat) ** 2).sum()
    f_stat = float((ss_model / 1) / (ss_res / max(n - Z.shape[1], 1))) if ss_res > 0 else 0.0
    return {
        "coef": coef,
        "se": se,
        "pvalue": pval,
        "first_stage_f": f_stat,
    }


def _try_causaliv(y: np.ndarray, d: np.ndarray, z: np.ndarray) -> dict[str, float] | None:
    """Attempt CausalIVSuite / causaliv / twosls if importable."""
    for mod_name, attr in (
        ("causaliv", "two_sls"),
        ("causaliv.twosls", "two_sls"),
        ("twosls", "fit"),
        ("causalivsuite", "two_sls"),
    ):
        try:
            import importlib

            mod = importlib.import_module(mod_name)
            fn = getattr(mod, attr, None)
            if fn is None:
                continue
            out = fn(y=y, d=d, z=z)
            if isinstance(out, dict):
                return {
                    "coef": float(out.get("coef", out.get("beta", 0.0))),
                    "se": float(out.get("se", out.get("std_err", float("nan")))),
                    "pvalue": float(out.get("pvalue", out.get("p", 1.0))),
                    "first_stage_f": float(out.get("first_stage_f", out.get("f", 0.0))),
                }
        except Exception:
            continue
    return None


def try_iv_edges(
    mat: pd.DataFrame,
    candidates: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    notes: list[str] = []
    edges: list[dict[str, Any]] = []
    treatments = candidates.get("treatment") or []
    outcomes = candidates.get("outcome") or []
    instruments = candidates.get("instrument") or []
    confounders = candidates.get("confounder") or []

    if not (treatments and outcomes and instruments):
        notes.append("IV pass skipped: need treatment, outcome, and instrument candidates.")
        return edges, notes

    used_suite = False
    for z in instruments[:3]:
        for d in treatments[:2]:
            for y in outcomes[:2]:
                if len({z, d, y}) < 3:
                    continue
                if any(c not in mat.columns for c in (z, d, y)):
                    continue
                frame = mat[[z, d, y] + [c for c in confounders if c in mat.columns and c not in (z, d, y)]].dropna()
                if len(frame) < 20:
                    continue
                yv = frame[y].to_numpy(dtype=float)
                dv = frame[d].to_numpy(dtype=float)
                zv = frame[z].to_numpy(dtype=float)
                ctrl_cols = [c for c in confounders if c in frame.columns and c not in (z, d, y)]
                controls = frame[ctrl_cols].to_numpy(dtype=float) if ctrl_cols else None

                suite = _try_causaliv(yv, dv, zv)
                if suite is not None:
                    used_suite = True
                    res = suite
                    method = "causaliv"
                else:
                    res = _numpy_2sls(yv, dv, zv, controls)
                    method = "numpy_2sls_lite"

                conf = float(min(1.0, abs(res["coef"]) / (1.0 + abs(res.get("se", 1.0))) * 0.5 + 0.1))
                if res.get("first_stage_f", 0) < 5:
                    conf *= 0.5
                edges.append(
                    {
                        "source": d,
                        "target": y,
                        "score": round(abs(res["coef"]), 4),
                        "confidence": round(conf, 4),
                        "pvalue": round(float(res.get("pvalue", 1.0)), 4),
                        "type": "iv_2sls",
                        "instrument": z,
                        "orientation": method,
                        "first_stage_f": round(float(res.get("first_stage_f", 0.0)), 3),
                    }
                )

    if used_suite:
        notes.append("IV edges estimated via CausalIVSuite/causaliv.")
    elif edges:
        notes.append("IV edges estimated via numpy 2SLS lite (CausalIVSuite not installed).")
    return edges, notes
