"""Soft LiNGAM / DirectLiNGAM discovery adapter."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from autocausal.backends._common import adjacency_to_edges, numeric_frame, soft_import, soft_skip_result

INSTALL = "pip install 'auto-causal-lib[causal-extra]'  # includes lingam"


def available() -> bool:
    return soft_import("lingam") is not None


def discover(
    df: pd.DataFrame,
    *,
    columns: Optional[list[str]] = None,
    method: str = "direct_lingam",
    **_kwargs: Any,
) -> dict[str, Any]:
    notes = [
        "LiNGAM assumes linear non-Gaussian acyclic SEM — exploratory when assumptions fail.",
    ]
    if not available():
        return soft_skip_result(method="lingam", module="lingam", install=INSTALL, notes=notes)

    clean, cols = numeric_frame(df, columns)
    if len(cols) < 2 or len(clean) < 10:
        return {
            "ok": True,
            "soft_skip": True,
            "method": "lingam",
            "backend": "lingam",
            "edges": [],
            "data": {},
            "notes": notes + ["Need ≥2 numeric columns and ≥10 complete rows."],
            "error": None,
        }
    try:
        from lingam import DirectLiNGAM

        model = DirectLiNGAM()
        model.fit(clean[cols].to_numpy(dtype=float))
        adj = np.asarray(model.adjacency_matrix_, dtype=float)
        # LiNGAM adjacency: adj[i,j] nonzero means j -> i (column causes row) in some versions;
        # DirectLiNGAM docs: adjacency_matrix_[i,j] = effect of j on i → j→i, so source=j target=i
        # Our adjacency_to_edges uses adj[i,j] as i→j; transpose to match.
        edges = adjacency_to_edges(adj.T, cols, method="lingam", threshold=0.05)
        for e in edges:
            e["method"] = "direct_lingam" if "direct" in method else "lingam"
        return {
            "ok": True,
            "soft_skip": False,
            "method": "lingam",
            "backend": "lingam.DirectLiNGAM",
            "edges": edges,
            "data": {
                "n_edges": len(edges),
                "causal_order": [cols[i] for i in list(getattr(model, "causal_order_", []) or [])],
            },
            "notes": notes,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "soft_skip": False,
            "method": "lingam",
            "backend": "lingam",
            "edges": [],
            "data": {},
            "notes": notes,
            "error": f"{type(e).__name__}: {e}",
        }
