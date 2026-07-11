"""Soft gCastle NOTEARS discovery adapter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from autocausal.backends._common import adjacency_to_edges, numeric_frame, soft_import, soft_skip_result

INSTALL = "pip install 'auto-causal-lib[causal-extra]'  # includes gcastle"


def available() -> bool:
    return soft_import("castle") is not None


def discover(
    df: pd.DataFrame,
    *,
    columns: Optional[list[str]] = None,
    threshold: float = 0.3,
    **_kwargs: Any,
) -> dict[str, Any]:
    notes = [
        "gCastle NOTEARS is continuous DAG learning — exploratory; sensitive to scale/hyperparams.",
    ]
    if not available():
        return soft_skip_result(
            method="gcastle_notears", module="castle", install=INSTALL, notes=notes
        )

    clean, cols = numeric_frame(df, columns)
    if len(cols) < 2 or len(clean) < 10:
        return {
            "ok": True,
            "soft_skip": True,
            "method": "gcastle_notears",
            "backend": "castle",
            "edges": [],
            "data": {},
            "notes": notes + ["Need ≥2 numeric columns and ≥10 complete rows."],
            "error": None,
        }
    # Cap for cost
    if len(cols) > 10:
        cols = cols[:10]
        clean = clean[cols]
        notes.append("NOTEARS capped to first 10 numeric columns for cost.")
    try:
        from castle.algorithms import Notears

        # Standardize for numerical stability
        X = clean[cols].to_numpy(dtype=float)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        nt = Notears()
        nt.learn(X)
        adj = np.asarray(nt.causal_matrix, dtype=float)
        edges = adjacency_to_edges(adj, cols, method="gcastle_notears", threshold=threshold)
        return {
            "ok": True,
            "soft_skip": False,
            "method": "gcastle_notears",
            "backend": "castle.Notears",
            "edges": edges,
            "data": {"n_edges": len(edges), "threshold": threshold},
            "notes": notes,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "soft_skip": False,
            "method": "gcastle_notears",
            "backend": "castle",
            "edges": [],
            "data": {},
            "notes": notes,
            "error": f"{type(e).__name__}: {e}",
        }
