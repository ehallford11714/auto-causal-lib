"""Exploratory causal relationship discovery (heuristic PC-lite + scores).

Supports:
- ``score_pc_lite`` (default)
- lightweight ``corr_skeleton`` and ``mi_stub`` methods
- bootstrap stability scores (honest confidence)
- multi-method consensus via ``discover_ensemble``
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from autocausal.iv import try_iv_edges
from autocausal.roles import ColumnRole, numeric_matrix


DiscoveryMethod = Literal["score_pc_lite", "corr_skeleton", "mi_stub"]


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

    X = np.column_stack([np.ones(len(x)), x])
    b1, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2_ab = r2(X @ b1, y)
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
    deg: dict[str, int] = {c: 0 for c in mat_cols}
    for e in edges:
        deg[e["source"]] = deg.get(e["source"], 0) + 1
        deg[e["target"]] = deg.get(e["target"], 0) + 1

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


def _edge_key(src: str, tgt: str, *, undirected: bool = True) -> tuple[str, str]:
    if undirected:
        return (src, tgt) if src <= tgt else (tgt, src)
    return (src, tgt)


def _pc_lite_edges(
    clean: pd.DataFrame,
    cols: list[str],
    *,
    alpha: float,
    max_cond_size: int,
    min_abs_corr: float,
) -> list[dict[str, Any]]:
    data = clean.to_numpy(dtype=float)
    n, p = data.shape
    idx = {c: i for i, c in enumerate(cols)}

    adj: dict[tuple[str, str], dict[str, Any]] = {}
    for a, b in combinations(cols, 2):
        i, j = idx[a], idx[b]
        r = _pearson(data[:, i], data[:, j])
        key = (a, b) if a < b else (b, a)
        adj[key] = {"corr": r, "pvalue": _fisher_z_pvalue(r, n, 0)}

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
                "method": "score_pc_lite",
            }
        )
    edges.sort(key=lambda e: e["confidence"], reverse=True)
    return edges


def _corr_skeleton_edges(
    clean: pd.DataFrame,
    cols: list[str],
    *,
    min_abs_corr: float,
) -> list[dict[str, Any]]:
    """Lightweight correlation-threshold skeleton (no CI pruning)."""
    edges: list[dict[str, Any]] = []
    for a, b in combinations(cols, 2):
        r = float(clean[a].corr(clean[b]))
        if not np.isfinite(r) or abs(r) < min_abs_corr:
            continue
        src, tgt, asym = _direction_score(clean, a, b)
        conf = float(min(1.0, abs(r) + 0.05 * asym))
        edges.append(
            {
                "source": src,
                "target": tgt,
                "score": round(abs(r), 4),
                "confidence": round(conf, 4),
                "pvalue": None,
                "type": "association",
                "orientation": "score_r2",
                "method": "corr_skeleton",
            }
        )
    edges.sort(key=lambda e: e["confidence"], reverse=True)
    return edges


def _mi_stub_edges(
    clean: pd.DataFrame,
    cols: list[str],
    *,
    min_score: float = 0.12,
    n_bins: int = 4,
) -> list[dict[str, Any]]:
    """Lightweight binned mutual-information stub (no sklearn required)."""
    from math import log

    edges: list[dict[str, Any]] = []
    binned: dict[str, pd.Series] = {}
    for c in cols:
        s = pd.to_numeric(clean[c], errors="coerce")
        try:
            binned[c] = pd.qcut(s, q=n_bins, duplicates="drop").astype(str)
        except ValueError:
            binned[c] = s.round(4).astype(str)

    def _nmi(x: pd.Series, y: pd.Series) -> float:
        n = len(x)
        if n == 0:
            return 0.0
        joint = pd.crosstab(x, y)
        px = joint.sum(axis=1).to_numpy(dtype=float) / n
        py = joint.sum(axis=0).to_numpy(dtype=float) / n
        pxy = joint.to_numpy(dtype=float) / n
        mi = 0.0
        for i in range(pxy.shape[0]):
            for j in range(pxy.shape[1]):
                p = pxy[i, j]
                if p <= 0:
                    continue
                mi += p * log(p / (px[i] * py[j] + 1e-15) + 1e-15)
        hx = -sum(p * log(p + 1e-15) for p in px if p > 0)
        hy = -sum(p * log(p + 1e-15) for p in py if p > 0)
        denom = max((hx + hy) / 2.0, 1e-12)
        return float(min(1.0, mi / denom))

    for a, b in combinations(cols, 2):
        score = _nmi(binned[a], binned[b])
        if score < min_score:
            continue
        src, tgt, asym = _direction_score(clean, a, b)
        conf = float(min(1.0, score + 0.03 * asym))
        edges.append(
            {
                "source": src,
                "target": tgt,
                "score": round(score, 4),
                "confidence": round(conf, 4),
                "pvalue": None,
                "type": "association",
                "orientation": "score_r2",
                "method": "mi_stub",
            }
        )
    edges.sort(key=lambda e: e["confidence"], reverse=True)
    return edges


def _run_method(
    method: DiscoveryMethod,
    clean: pd.DataFrame,
    cols: list[str],
    *,
    alpha: float,
    max_cond_size: int,
    min_abs_corr: float,
) -> list[dict[str, Any]]:
    if method == "corr_skeleton":
        return _corr_skeleton_edges(clean, cols, min_abs_corr=min_abs_corr)
    if method == "mi_stub":
        return _mi_stub_edges(clean, cols, min_score=max(0.08, min_abs_corr * 0.8))
    return _pc_lite_edges(
        clean, cols, alpha=alpha, max_cond_size=max_cond_size, min_abs_corr=min_abs_corr
    )


def _bootstrap_stability(
    clean: pd.DataFrame,
    cols: list[str],
    base_edges: list[dict[str, Any]],
    *,
    method: DiscoveryMethod,
    bootstrap_n: int,
    alpha: float,
    max_cond_size: int,
    min_abs_corr: float,
    seed: int = 0,
) -> dict[tuple[str, str], float]:
    """Fraction of bootstrap resamples where undirected edge reappears."""
    if bootstrap_n <= 0 or not base_edges or len(clean) < 8:
        return {}
    rng = np.random.default_rng(seed)
    targets = {_edge_key(e["source"], e["target"]) for e in base_edges}
    counts = {k: 0 for k in targets}
    n = len(clean)
    for _ in range(bootstrap_n):
        idx = rng.integers(0, n, size=n)
        sample = clean.iloc[idx]
        try:
            boot_edges = _run_method(
                method,
                sample,
                cols,
                alpha=alpha,
                max_cond_size=max_cond_size,
                min_abs_corr=min_abs_corr,
            )
        except Exception:
            continue
        present = {_edge_key(e["source"], e["target"]) for e in boot_edges}
        for k in targets:
            if k in present:
                counts[k] += 1
    return {k: counts[k] / bootstrap_n for k in counts}


def _apply_stability(
    edges: list[dict[str, Any]],
    stability: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """Attach stability and honest confidence = min(confidence, stability)."""
    out: list[dict[str, Any]] = []
    for e in edges:
        key = _edge_key(e["source"], e["target"])
        stab = float(stability.get(key, e.get("stability", 0.0) or 0.0))
        raw_conf = float(e.get("confidence") or 0.0)
        honest = min(raw_conf, stab) if stab > 0 else raw_conf * 0.5
        ne = dict(e)
        ne["stability"] = round(stab, 4)
        ne["confidence_raw"] = round(raw_conf, 4)
        ne["confidence"] = round(float(honest), 4)
        ne["confidence_note"] = (
            "confidence capped by bootstrap stability (exploratory, not identification)"
        )
        out.append(ne)
    out.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return out


def consensus_edges(
    method_edges: dict[str, list[dict[str, Any]]],
    *,
    min_methods: int = 2,
) -> list[dict[str, Any]]:
    """Merge multi-method edges into a consensus graph (undirected agreement)."""
    votes: dict[tuple[str, str], dict[str, Any]] = {}
    for method, edges in method_edges.items():
        for e in edges:
            key = _edge_key(e["source"], e["target"])
            slot = votes.setdefault(
                key,
                {
                    "source": e["source"],
                    "target": e["target"],
                    "methods": [],
                    "scores": [],
                    "confidences": [],
                    "stabilities": [],
                },
            )
            # prefer orientation from higher-confidence vote
            if float(e.get("confidence") or 0) >= max(slot["confidences"] or [0]):
                slot["source"] = e["source"]
                slot["target"] = e["target"]
            slot["methods"].append(method)
            slot["scores"].append(float(e.get("score") or 0))
            slot["confidences"].append(float(e.get("confidence") or 0))
            if e.get("stability") is not None:
                slot["stabilities"].append(float(e["stability"]))

    out: list[dict[str, Any]] = []
    for key, slot in votes.items():
        n_agree = len(set(slot["methods"]))
        if n_agree < min_methods and len(method_edges) > 1:
            continue
        mean_score = float(np.mean(slot["scores"])) if slot["scores"] else 0.0
        mean_conf = float(np.mean(slot["confidences"])) if slot["confidences"] else 0.0
        # consensus bonus for multi-method agreement
        agree_frac = n_agree / max(len(method_edges), 1)
        conf = min(1.0, mean_conf * (0.7 + 0.3 * agree_frac))
        stab = float(np.mean(slot["stabilities"])) if slot["stabilities"] else agree_frac
        out.append(
            {
                "source": slot["source"],
                "target": slot["target"],
                "score": round(mean_score, 4),
                "confidence": round(conf, 4),
                "stability": round(stab, 4),
                "type": "consensus",
                "orientation": "multi_method",
                "method": "consensus",
                "methods": sorted(set(slot["methods"])),
                "n_methods": n_agree,
                "agreement": round(agree_frac, 4),
            }
        )
    out.sort(key=lambda e: (e.get("n_methods", 0), e.get("confidence", 0)), reverse=True)
    return out


def discover_relationships(
    df: pd.DataFrame,
    *,
    roles: dict[str, ColumnRole],
    alpha: float = 0.05,
    max_cond_size: int = 2,
    min_abs_corr: float = 0.15,
    use_iv: bool = True,
    method: DiscoveryMethod = "score_pc_lite",
    stability: bool = False,
    bootstrap_n: int = 20,
    seed: int = 0,
) -> "DiscoveryResult":
    from autocausal.results import DiscoveryResult

    mat, cols = numeric_matrix(df, roles)
    notes: list[str] = [
        "Exploratory heuristic discovery (PC-lite / lightweight methods). "
        "Not a guarantee of causal identification.",
    ]
    if len(cols) < 2:
        return DiscoveryResult(
            edges=[],
            graph={"nodes": list(df.columns), "edges": []},
            roles=roles,
            candidates={"treatment": [], "outcome": [], "instrument": [], "confounder": []},
            method=method,
            notes=notes + ["Fewer than 2 usable columns for discovery."],
        )

    clean = mat.dropna()
    if len(clean) < 10:
        notes.append(f"Only {len(clean)} complete rows after encoding; results unstable.")

    edges = _run_method(
        method,
        clean,
        cols,
        alpha=alpha,
        max_cond_size=max_cond_size,
        min_abs_corr=min_abs_corr,
    )

    if stability:
        n_boot = max(1, int(bootstrap_n))
        stab_map = _bootstrap_stability(
            clean,
            cols,
            edges,
            method=method,
            bootstrap_n=n_boot,
            alpha=alpha,
            max_cond_size=max_cond_size,
            min_abs_corr=min_abs_corr,
            seed=seed,
        )
        edges = _apply_stability(edges, stab_map)
        notes.append(
            f"Bootstrap stability enabled (n={n_boot}); "
            "confidence capped by per-edge stability (honest, exploratory)."
        )

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
                "stability": e.get("stability"),
                "type": e.get("type", "association"),
                "method": e.get("method", method),
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
        method=method,
        notes=notes,
        stability_enabled=stability,
        bootstrap_n=int(bootstrap_n) if stability else 0,
    )


def discover_ensemble(
    df: pd.DataFrame,
    *,
    roles: dict[str, ColumnRole],
    methods: Optional[Sequence[DiscoveryMethod]] = None,
    alpha: float = 0.05,
    max_cond_size: int = 2,
    min_abs_corr: float = 0.15,
    use_iv: bool = True,
    stability: bool = False,
    bootstrap_n: int = 10,
    min_methods: int = 2,
    seed: int = 0,
) -> "DiscoveryResult":
    """Run multiple lightweight discovery methods and return a consensus graph."""
    from autocausal.results import DiscoveryResult

    methods = list(methods or ("score_pc_lite", "corr_skeleton", "mi_stub"))
    mat, cols = numeric_matrix(df, roles)
    notes: list[str] = [
        f"Multi-method ensemble discovery: {methods}. Consensus ≠ identification.",
    ]
    if len(cols) < 2:
        return DiscoveryResult(
            edges=[],
            graph={"nodes": list(df.columns), "edges": []},
            roles=roles,
            candidates={"treatment": [], "outcome": [], "instrument": [], "confounder": []},
            method="consensus",
            notes=notes + ["Fewer than 2 usable columns."],
            ensemble_methods=list(methods),
        )

    clean = mat.dropna()
    method_edges: dict[str, list[dict[str, Any]]] = {}
    for i, m in enumerate(methods):
        edges_m = _run_method(
            m,  # type: ignore[arg-type]
            clean,
            cols,
            alpha=alpha,
            max_cond_size=max_cond_size,
            min_abs_corr=min_abs_corr,
        )
        if stability:
            stab_map = _bootstrap_stability(
                clean,
                cols,
                edges_m,
                method=m,  # type: ignore[arg-type]
                bootstrap_n=max(1, int(bootstrap_n)),
                alpha=alpha,
                max_cond_size=max_cond_size,
                min_abs_corr=min_abs_corr,
                seed=seed + i,
            )
            edges_m = _apply_stability(edges_m, stab_map)
        method_edges[m] = edges_m

    edges = consensus_edges(method_edges, min_methods=min_methods)
    # if consensus empty (strict), fall back to union of pc_lite
    if not edges and method_edges:
        primary = method_edges.get("score_pc_lite") or next(iter(method_edges.values()))
        edges = list(primary)
        notes.append("Consensus empty under min_methods; fell back to primary method edges.")

    candidates = propose_candidates(roles, edges, cols)
    if use_iv:
        iv_edges, iv_notes = try_iv_edges(clean, candidates)
        edges.extend(iv_edges)
        notes.extend(iv_notes)

    if stability:
        notes.append(
            f"Per-method bootstrap stability (n={bootstrap_n}); "
            "consensus confidence reflects agreement + stability."
        )

    graph = {
        "directed": True,
        "nodes": [{"id": c, "role": roles.get(c, ColumnRole.UNKNOWN).value} for c in df.columns],
        "edges": [
            {
                "source": e["source"],
                "target": e["target"],
                "score": e["score"],
                "confidence": e["confidence"],
                "stability": e.get("stability"),
                "type": e.get("type", "consensus"),
                "methods": e.get("methods"),
            }
            for e in edges
        ],
        "candidates": candidates,
        "method_edges": {k: len(v) for k, v in method_edges.items()},
    }

    return DiscoveryResult(
        edges=edges,
        graph=graph,
        roles=roles,
        candidates=candidates,
        method="consensus",
        notes=notes,
        stability_enabled=stability,
        bootstrap_n=int(bootstrap_n) if stability else 0,
        ensemble_methods=list(methods),
        method_edges={k: list(v) for k, v in method_edges.items()},
    )
