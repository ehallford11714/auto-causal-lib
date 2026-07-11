"""Soft causal-learn discovery adapters (PC / GES / FCI)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from autocausal.backends._common import adjacency_to_edges, numeric_frame, soft_import, soft_skip_result

INSTALL = "pip install 'auto-causal-lib[causal-extra]'  # includes causal-learn"


def available() -> bool:
    return soft_import("causallearn") is not None


def _graph_to_adj(g: Any, p: int) -> np.ndarray:
    """Best-effort extract of adjacency from causal-learn GeneralGraph / ndarray."""
    if isinstance(g, np.ndarray):
        return np.asarray(g, dtype=float)
    # CausalGraph / GeneralGraph often expose graph as ndarray via .graph
    if hasattr(g, "graph"):
        arr = np.asarray(g.graph, dtype=float)
        if arr.ndim == 2:
            return arr
    if hasattr(g, "G") and hasattr(g.G, "graph"):
        return np.asarray(g.G.graph, dtype=float)
    # Fallback: try nx-style edges
    adj = np.zeros((p, p), dtype=float)
    edges = getattr(g, "get_graph_edges", None)
    if callable(edges):
        for e in edges() or []:
            try:
                i = int(getattr(e, "node1", e[0]).get_name().replace("X", "")) - 1
                j = int(getattr(e, "node2", e[1]).get_name().replace("X", "")) - 1
                adj[i, j] = 1.0
            except Exception:
                continue
    return adj


def _orient_from_cl_matrix(mat: np.ndarray) -> np.ndarray:
    """causal-learn uses coded entries; convert to simple directed weights.

    Common encoding (approx):
      -1 / 1 patterns for directed;  -1/-1 undirected; 2 for latent.
    We emit abs weight 1 for any non-zero directed-ish link.
    """
    p = mat.shape[0]
    adj = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            a = mat[i, j]
            b = mat[j, i]
            # directed i -> j often: mat[i,j]=-1 and mat[j,i]=1 (or similar)
            if a != 0 and b != 0:
                if a == -1 and b == 1:
                    adj[i, j] = 1.0
                elif a == 1 and b == -1:
                    adj[j, i] = 1.0
                else:
                    # undirected / ambiguous — emit both lightly
                    adj[i, j] = 0.5
                    adj[j, i] = 0.5
            elif a != 0:
                adj[i, j] = abs(float(a)) if abs(float(a)) <= 2 else 1.0
    return adj


def discover_pc(
    df: pd.DataFrame,
    *,
    columns: Optional[list[str]] = None,
    alpha: float = 0.05,
    **_kwargs: Any,
) -> dict[str, Any]:
    notes = [
        "causal-learn PC is exploratory; independence tests ≠ identification.",
    ]
    if not available():
        return soft_skip_result(method="causal_learn_pc", module="causallearn", install=INSTALL, notes=notes)

    clean, cols = numeric_frame(df, columns)
    if len(cols) < 2 or len(clean) < 10:
        return {
            "ok": True,
            "soft_skip": True,
            "method": "causal_learn_pc",
            "backend": "causallearn",
            "edges": [],
            "data": {},
            "notes": notes + ["Need ≥2 numeric columns and ≥10 complete rows."],
            "error": None,
        }
    try:
        from causallearn.search.ConstraintBased.PC import pc

        data = clean[cols].to_numpy(dtype=float)
        cg = pc(data, alpha=alpha, indep_test="fisherz", stable=True, show_progress=False)
        raw = _graph_to_adj(cg, len(cols))
        adj = _orient_from_cl_matrix(raw) if raw.shape == (len(cols), len(cols)) else raw
        edges = adjacency_to_edges(adj, cols, method="causal_learn_pc", threshold=0.25)
        return {
            "ok": True,
            "soft_skip": False,
            "method": "causal_learn_pc",
            "backend": "causallearn.pc",
            "edges": edges,
            "data": {"n_edges": len(edges), "n_nodes": len(cols), "alpha": alpha},
            "notes": notes,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "soft_skip": False,
            "method": "causal_learn_pc",
            "backend": "causallearn",
            "edges": [],
            "data": {},
            "notes": notes,
            "error": f"{type(e).__name__}: {e}",
        }


def discover_ges(
    df: pd.DataFrame,
    *,
    columns: Optional[list[str]] = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    notes = ["causal-learn GES is score-based and exploratory."]
    if not available():
        return soft_skip_result(method="causal_learn_ges", module="causallearn", install=INSTALL, notes=notes)

    clean, cols = numeric_frame(df, columns)
    if len(cols) < 2 or len(clean) < 10:
        return {
            "ok": True,
            "soft_skip": True,
            "method": "causal_learn_ges",
            "backend": "causallearn",
            "edges": [],
            "data": {},
            "notes": notes + ["Need ≥2 numeric columns and ≥10 complete rows."],
            "error": None,
        }
    try:
        from causallearn.search.ScoreBased.GES import ges

        data = clean[cols].to_numpy(dtype=float)
        record = ges(data, score_func="local_score_BIC")
        g = record.get("G") if isinstance(record, dict) else record
        raw = _graph_to_adj(g, len(cols))
        adj = _orient_from_cl_matrix(raw) if raw.shape == (len(cols), len(cols)) else raw
        edges = adjacency_to_edges(adj, cols, method="causal_learn_ges", threshold=0.25)
        return {
            "ok": True,
            "soft_skip": False,
            "method": "causal_learn_ges",
            "backend": "causallearn.ges",
            "edges": edges,
            "data": {"n_edges": len(edges), "n_nodes": len(cols)},
            "notes": notes,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "soft_skip": False,
            "method": "causal_learn_ges",
            "backend": "causallearn",
            "edges": [],
            "data": {},
            "notes": notes,
            "error": f"{type(e).__name__}: {e}",
        }


def discover_fci(
    df: pd.DataFrame,
    *,
    columns: Optional[list[str]] = None,
    alpha: float = 0.05,
    **_kwargs: Any,
) -> dict[str, Any]:
    notes = [
        "causal-learn FCI allows latent confounding; edges are PAG-style (exploratory).",
    ]
    if not available():
        return soft_skip_result(method="causal_learn_fci", module="causallearn", install=INSTALL, notes=notes)

    clean, cols = numeric_frame(df, columns)
    if len(cols) < 2 or len(clean) < 10:
        return {
            "ok": True,
            "soft_skip": True,
            "method": "causal_learn_fci",
            "backend": "causallearn",
            "edges": [],
            "data": {},
            "notes": notes + ["Need ≥2 numeric columns and ≥10 complete rows."],
            "error": None,
        }
    # Cap columns for cost — FCI is expensive
    if len(cols) > 12:
        cols = cols[:12]
        clean = clean[cols]
        notes.append("FCI capped to first 12 numeric columns for cost.")
    try:
        from causallearn.search.ConstraintBased.FCI import fci

        data = clean[cols].to_numpy(dtype=float)
        g, _edges = fci(data, independence_test_method="fisherz", alpha=alpha, verbose=False)
        raw = _graph_to_adj(g, len(cols))
        adj = _orient_from_cl_matrix(raw) if raw.shape == (len(cols), len(cols)) else raw
        edges = adjacency_to_edges(adj, cols, method="causal_learn_fci", threshold=0.25)
        return {
            "ok": True,
            "soft_skip": False,
            "method": "causal_learn_fci",
            "backend": "causallearn.fci",
            "edges": edges,
            "data": {"n_edges": len(edges), "n_nodes": len(cols), "alpha": alpha},
            "notes": notes,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "soft_skip": False,
            "method": "causal_learn_fci",
            "backend": "causallearn",
            "edges": [],
            "data": {},
            "notes": notes,
            "error": f"{type(e).__name__}: {e}",
        }


def discover(
    df: pd.DataFrame,
    *,
    method: str = "causal_learn_pc",
    **kwargs: Any,
) -> dict[str, Any]:
    m = (method or "causal_learn_pc").lower().strip()
    if m in ("causal_learn_ges", "ges"):
        return discover_ges(df, **kwargs)
    if m in ("causal_learn_fci", "fci"):
        return discover_fci(df, **kwargs)
    return discover_pc(df, **kwargs)
