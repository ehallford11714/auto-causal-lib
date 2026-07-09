"""Exploratory causal relationship discovery (heuristic PC-lite + scores)."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from autocausal.iv import try_iv_edges
from autocausal.roles import ColumnRole, numeric_matrix


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def _partial_corr(data: np.ndarray, i: int, j: int, cond: list[int]) -> float:
    """Partial correlation via residual correlation (OLS)."""
    n, p = data.shape
    if not cond:
        return _pearson(data[:, i], data[:, j])
    Z = data[:, cond]
    # add intercept
    Z = np.column_stack([np.ones(n), Z])
    try:
        beta_i, *_ = np.linalg.lstsq(Z, data[:, i], rcond=None)
        beta_j, *_ = np.linalg.lstsq(Z, data[:, j], rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    ri = data[:, i] - Z @ beta_i
    rj = data[:, j] - Z @ beta_j
    return _pearson(ri, rj)


def _fisher_z_pvalue(r: float, n: int, n_cond: int) -> float:
    """Two-sided Fisher z test for partial correlation == 0."""
    df = n - n_cond - 3
    if df <= 0:
        return 1.0
    r = float(np.clip(r, -0.999999, 0.999999))
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(df)
    from math import erfc

    # normal survival for |z|/se
    stat = abs(z) / se
    p = erfc(stat / np.sqrt(2.0))
    return float(p)


def _direction_score(mat: pd.DataFrame, a: str, b: str) -> tuple[str, str, float]:
    """Orient a—b by comparing simple regression R^2 (a→b vs b→a)."""
    x = mat[a].to_numpy(dtype=float)
    y = mat[b].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 5:
        return a, b, 0.0

    def r2(pred: np.ndarray, target: np.ndarray) -> float:
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        if ss_tot < 1e-12:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    # y ~ x
    X = np.column_stack([np.ones(len(x)), x])
    b1, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2_ab = r2(X @ b1, y)
    # x ~ y
    Y = np.column_stack([np.ones(len(y)), y])
    b2, *_ = np.linalg.lstsq(Y, x, rcond=None)
    r2_ba = r2(Y @ b2, x)
    if r2_ab >= r2_ba:
        return a, b, r2_ab - r2_ba
    return b, a, r2_ba - r2_ab


def propose_candidates(
    roles: dict[str, ColumnRole],
    edges: list[dict[str, Any]],
    mat_cols: list[str],
) -> dict[str, list[str]]:
    """Heuristic treatment / outcome / instrument / confounder candidates."""
    numeric = [c for c, r in roles.items() if r == ColumnRole.NUMERIC and c in mat_cols]
    binary = [
        c
        for c, r in roles.items()
        if r in (ColumnRole.BOOLEAN, ColumnRole.CATEGORICAL) and c in mat_cols
    ]
    # degree in undirected sense
    deg: dict[str, int] = {c: 0 for c in mat_cols}
    for e in edges:
        deg[e["source"]] = deg.get(e["source"], 0) + 1
        deg[e["target"]] = deg.get(e["target"], 0) + 1

    # outcome: high-degree numeric often named like y/revenue/score
    name_out = ("y", "outcome", "target", "revenue", "sales", "conversion", "score", "label")
    name_treat = ("t", "treat", "treatment", "x", "dose", "exposure", "campaign", "price")
    name_iv = ("z", "iv", "instrument", "assign", "lottery", "exog")

    outcomes = [c for c in numeric if any(h in c.lower() for h in name_out)]
    treatments = [c for c in (binary + numeric) if any(h in c.lower() for h in name_treat)]
    instruments = [c for c in mat_cols if any(h in c.lower() for h in name_iv)]

    if not outcomes and numeric:
        outcomes = sorted(numeric, key=lambda c: deg.get(c, 0), reverse=True)[:2]
    if not treatments:
        treatments = sorted(binary or numeric, key=lambda c: deg.get(c, 0), reverse=True)[:2]

    # confounders: nodes adjacent to both a treatment and outcome
    confounders: list[str] = []
    tset, oset = set(treatments), set(outcomes)
    neighbors: dict[str, set[str]] = {c: set() for c in mat_cols}
    for e in edges:
        neighbors[e["source"]].add(e["target"])
        neighbors[e["target"]].add(e["source"])
    for c in mat_cols:
        if c in tset or c in oset:
            continue
        nbs = neighbors.get(c, set())
        if nbs & tset and nbs & oset:
            confounders.append(c)
    if not confounders:
        confounders = [c for c in numeric if c not in tset and c not in oset][:5]

    return {
        "treatment": treatments[:5],
        "outcome": outcomes[:5],
        "instrument": instruments[:5],
        "confounder": confounders[:8],
    }


def discover_relationships(
    df: pd.DataFrame,
    *,
    roles: dict[str, ColumnRole],
    alpha: float = 0.05,
    max_cond_size: int = 2,
    min_abs_corr: float = 0.15,
    use_iv: bool = True,
) -> "DiscoveryResult":
    from autocausal.results import DiscoveryResult

    mat, cols = numeric_matrix(df, roles)
    notes: list[str] = [
        "Exploratory heuristic discovery (PC-lite + score orientation). "
        "Not a guarantee of causal identification.",
    ]
    if len(cols) < 2:
        return DiscoveryResult(
            edges=[],
            graph={"nodes": list(df.columns), "edges": []},
            roles=roles,
            candidates={"treatment": [], "outcome": [], "instrument": [], "confounder": []},
            method="score_pc_lite",
            notes=notes + ["Fewer than 2 usable columns for discovery."],
        )

    # drop rows with any nan in matrix
    clean = mat.dropna()
    if len(clean) < 10:
        notes.append(f"Only {len(clean)} complete rows after encoding; results unstable.")
    data = clean.to_numpy(dtype=float)
    n, p = data.shape
    idx = {c: i for i, c in enumerate(cols)}

    # start with complete undirected graph among cols
    adj: dict[tuple[str, str], dict[str, Any]] = {}
    for a, b in combinations(cols, 2):
        i, j = idx[a], idx[b]
        r = _pearson(data[:, i], data[:, j])
        if abs(r) < min_abs_corr * 0.5:
            # still keep weak edges initially; PC will prune
            pass
        key = (a, b) if a < b else (b, a)
        adj[key] = {"corr": r, "pvalue": _fisher_z_pvalue(r, n, 0)}

    # PC-style conditional independence pruning
    for cond_size in range(0, max_cond_size + 1):
        to_remove: list[tuple[str, str]] = []
        for (a, b), meta in list(adj.items()):
            others = [c for c in cols if c != a and c != b]
            if len(others) < cond_size:
                continue
            best_p = 0.0
            best_r = meta["corr"]
            for cond in combinations(others, cond_size):
                ci = [idx[c] for c in cond]
                r = _partial_corr(data, idx[a], idx[b], ci)
                pv = _fisher_z_pvalue(r, n, cond_size)
                if pv > best_p:
                    best_p = pv
                    best_r = r
                if pv > alpha:
                    to_remove.append((a, b))
                    break
            else:
                meta["corr"] = best_r
                meta["pvalue"] = best_p if cond_size > 0 else meta["pvalue"]
        for key in to_remove:
            k = key if key[0] < key[1] else (key[1], key[0])
            adj.pop(k, None)

    edges: list[dict[str, Any]] = []
    for (a, b), meta in adj.items():
        r = float(meta["corr"])
        if abs(r) < min_abs_corr and meta.get("pvalue", 1.0) > alpha:
            continue
        src, tgt, asym = _direction_score(clean, a, b)
        conf = float(min(1.0, abs(r) * (1.0 - float(meta.get("pvalue", 1.0))) + 0.05 * asym))
        edges.append(
            {
                "source": src,
                "target": tgt,
                "score": round(abs(r), 4),
                "confidence": round(conf, 4),
                "pvalue": round(float(meta.get("pvalue", 1.0)), 4),
                "type": "association",
                "orientation": "score_r2",
            }
        )

    edges.sort(key=lambda e: e["confidence"], reverse=True)

    candidates = propose_candidates(roles, edges, cols)

    if use_iv:
        iv_edges, iv_notes = try_iv_edges(clean, candidates)
        edges.extend(iv_edges)
        notes.extend(iv_notes)

    graph = {
        "directed": True,
        "nodes": [{"id": c, "role": roles.get(c, ColumnRole.UNKNOWN).value} for c in df.columns],
        "edges": [
            {
                "source": e["source"],
                "target": e["target"],
                "score": e["score"],
                "confidence": e["confidence"],
                "type": e.get("type", "association"),
            }
            for e in edges
        ],
        "candidates": candidates,
    }

    return DiscoveryResult(
        edges=edges,
        graph=graph,
        roles=roles,
        candidates=candidates,
        method="score_pc_lite",
        notes=notes,
    )
