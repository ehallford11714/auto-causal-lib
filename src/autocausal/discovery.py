"""Exploratory causal relationship discovery (heuristic PC-lite + scores).

Supports:
- ``score_pc_lite`` (default)
- lightweight ``corr_skeleton`` and binned NMI (``mi`` / ``mi_binned``;
  ``mi_stub`` is a backward-compat alias)
- bootstrap stability scores (honest confidence)
- multi-method consensus via ``discover_ensemble``
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from autocausal.iv import (
    AUTO_INSTRUMENT_COL,
    merge_role_candidates,
    synthesize_auto_instrument,
    try_iv_edges,
)
from autocausal.roles import ColumnRole, numeric_matrix

__all__ = [
    "DiscoveryMethod",
    "BUILTIN_METHODS",
    "EXTERNAL_METHODS",
    "NAME_IV_HINTS",
    "NAME_OUT_HINTS",
    "NAME_TREAT_HINTS",
    "discover_relationships",
    "discover_ensemble",
    "propose_candidates",
    "consensus_edges",
]

DiscoveryMethod = Literal[
    "score_pc_lite",
    "corr_skeleton",
    "mi",
    "mi_binned",
    "mi_stub",
    "causal_learn_pc",
    "causal_learn_ges",
    "causal_learn_fci",
    "lingam",
    "direct_lingam",
    "gcastle_notears",
]

BUILTIN_METHODS = frozenset(
    {"score_pc_lite", "corr_skeleton", "mi", "mi_binned", "mi_stub"}
)
EXTERNAL_METHODS = frozenset(
    {
        "causal_learn_pc",
        "causal_learn_ges",
        "causal_learn_fci",
        "lingam",
        "direct_lingam",
        "gcastle_notears",
    }
)


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


# Instrument name heuristics (substring match on lowercased column names).
# Prefer real IV-named columns over inventing proxies (see allow_iv_fallback /
# auto_instrument for explicit opt-in behaviors).
NAME_IV_HINTS = (
    "z",
    "iv",
    "instrument",
    "instrument_z",
    "assign",
    "assignment",
    "lottery",
    "exog",
    "rand",
    "randomized",
    "encourage",
)
NAME_OUT_HINTS = (
    "y",
    "outcome",
    "target",
    "revenue",
    "sales",
    "conversion",
    "score",
    "label",
)
NAME_TREAT_HINTS = (
    "t",
    "treat",
    "treatment",
    "x",
    "dose",
    "exposure",
    "campaign",
    "price",
)


def _name_hits(col: str, hints: Sequence[str]) -> bool:
    low = col.lower()
    return any(h in low for h in hints)


def propose_candidates(
    roles: dict[str, ColumnRole],
    edges: list[dict[str, Any]],
    mat_cols: list[str],
    *,
    allow_iv_fallback: bool = False,
    mat: Optional[pd.DataFrame] = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Heuristic treatment / outcome / instrument / confounder candidates.

    Instruments are **name-gated** by default (``NAME_IV_HINTS``). When treatments
    and outcomes exist but no instrument names match:

    - default: leave ``instrument`` empty and emit a clear note (prefer real
      columns / ``iv_demo`` / ``set_iv_roles`` / ``auto_instrument``).
    - ``allow_iv_fallback=True``: propose weak numeric correlates of treatment
      as instrument *candidates* only (still not identification).
    """
    notes: list[str] = []
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

    outcomes = [c for c in numeric if _name_hits(c, NAME_OUT_HINTS)]
    treatments = [c for c in (binary + numeric) if _name_hits(c, NAME_TREAT_HINTS)]
    instruments = [c for c in mat_cols if _name_hits(c, NAME_IV_HINTS)]

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

    if treatments and outcomes and not instruments:
        notes.append(
            "No instrument columns matched name heuristics "
            f"{NAME_IV_HINTS[:6]}… — IV needs a Z column "
            "(load `iv_demo`, join `instruments_demo`, pass candidates=/set_iv_roles, "
            "or use auto_instrument=True)."
        )
        if allow_iv_fallback and mat is not None and treatments:
            # Weak correlate fallback — never silent; never claim identification.
            dcol = treatments[0]
            if dcol in mat.columns:
                d = pd.to_numeric(mat[dcol], errors="coerce")
                scored: list[tuple[float, str]] = []
                for c in numeric:
                    if c in tset or c in oset or c in instruments:
                        continue
                    s = pd.to_numeric(mat[c], errors="coerce")
                    pair = pd.concat([d, s], axis=1).dropna()
                    if len(pair) < 20:
                        continue
                    r = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                    if r == r and abs(r) >= 0.15:
                        scored.append((abs(r), c))
                scored.sort(reverse=True)
                instruments = [c for _, c in scored[:2]]
                if instruments:
                    notes.append(
                        "allow_iv_fallback=True: weak correlate(s) proposed as "
                        f"instrument candidates {instruments} — exploratory only, "
                        "not a real IV design."
                    )

    return (
        {
            "treatment": treatments[:5],
            "outcome": outcomes[:5],
            "instrument": instruments[:5],
            "confounder": confounders[:8],
        },
        notes,
    )


def _apply_iv_pass(
    clean: pd.DataFrame,
    roles: dict[str, ColumnRole],
    edges: list[dict[str, Any]],
    cols: list[str],
    *,
    use_iv: bool,
    auto_instrument: bool,
    allow_iv_fallback: bool,
    candidates_override: Optional[dict[str, Sequence[str]]],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, list[str]], list[str]]:
    """Propose/merge candidates, optionally synthesize Z, run IV edges."""
    notes: list[str] = []
    proposed, prop_notes = propose_candidates(
        roles, edges, cols, allow_iv_fallback=allow_iv_fallback, mat=clean
    )
    notes.extend(prop_notes)
    candidates = merge_role_candidates(proposed, candidates_override)

    work = clean
    if (
        use_iv
        and auto_instrument
        and (candidates.get("treatment") or [])
        and (candidates.get("outcome") or [])
        and not (candidates.get("instrument") or [])
    ):
        treat = candidates["treatment"][0]
        work, auto_notes = synthesize_auto_instrument(
            work, treat, seed=seed, col=AUTO_INSTRUMENT_COL
        )
        notes.extend(auto_notes)
        if AUTO_INSTRUMENT_COL in work.columns:
            candidates = merge_role_candidates(
                candidates, {"instrument": [AUTO_INSTRUMENT_COL]}
            )
            if AUTO_INSTRUMENT_COL not in cols:
                cols = list(cols) + [AUTO_INSTRUMENT_COL]

    if use_iv:
        iv_edges, iv_notes = try_iv_edges(work, candidates)
        edges = list(edges) + iv_edges
        notes.extend(iv_notes)

    return work, edges, candidates, notes


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


def _mi_binned_edges(
    clean: pd.DataFrame,
    cols: list[str],
    *,
    min_score: float = 0.12,
    n_bins: int = 4,
) -> list[dict[str, Any]]:
    """Cheap binned normalized mutual information (no sklearn required).

    Quantile-bins each column, estimates NMI from the joint contingency table,
    and orients edges via regression R² asymmetry. Labeled ``mi_binned``;
    ``mi`` / ``mi_stub`` are caller aliases for the same path.
    """
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
                "method": "mi_binned",
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
    """Backward-compat wrapper for :func:`_mi_binned_edges`."""
    return _mi_binned_edges(clean, cols, min_score=min_score, n_bins=n_bins)


def _run_external_method(
    method: str,
    clean: pd.DataFrame,
    cols: list[str],
    *,
    alpha: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Soft-call optional discovery backends; return (edges, notes)."""
    from autocausal.engines import discover_with

    raw = discover_with(clean, method=method, columns=cols, alpha=alpha)
    notes = list(raw.get("notes") or [])
    if raw.get("soft_skip"):
        notes.append(f"{method}: soft-skipped (missing or insufficient data).")
        return [], notes
    if raw.get("error"):
        notes.append(f"{method} error: {raw['error']}")
        return [], notes
    return list(raw.get("edges") or []), notes


def _run_method(
    method: DiscoveryMethod | str,
    clean: pd.DataFrame,
    cols: list[str],
    *,
    alpha: float,
    max_cond_size: int,
    min_abs_corr: float,
) -> list[dict[str, Any]]:
    m = str(method)
    if m in EXTERNAL_METHODS:
        edges, _notes = _run_external_method(m, clean, cols, alpha=alpha)
        return edges
    if m == "corr_skeleton":
        return _corr_skeleton_edges(clean, cols, min_abs_corr=min_abs_corr)
    if m in ("mi", "mi_binned", "mi_stub"):
        return _mi_binned_edges(clean, cols, min_score=max(0.08, min_abs_corr * 0.8))
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
    use_iv: Optional[bool] = None,
    auto_instrument: bool = False,
    allow_iv_fallback: bool = False,
    candidates: Optional[dict[str, Sequence[str]]] = None,
    method: DiscoveryMethod | str = "score_pc_lite",
    stability: Optional[bool] = None,
    bootstrap_n: Optional[int] = None,
    seed: int = 0,
    mode: str = "exploratory",
    policy: Optional[Any] = None,
) -> "DiscoveryResult":
    from autocausal.production import (
        EPISTEMIC,
        ProductionGateError,
        apply_mode_defaults,
        is_production,
    )
    from autocausal.results import DiscoveryResult

    settings = apply_mode_defaults(
        mode=mode,
        policy=policy,
        auto_instrument=auto_instrument,
        allow_iv_fallback=allow_iv_fallback,
        use_iv=use_iv,
        stability=stability,
        bootstrap_n=bootstrap_n,
    )
    auto_instrument = settings.auto_instrument
    allow_iv_fallback = settings.allow_iv_fallback
    use_iv = settings.use_iv
    stability = settings.stability
    bootstrap_n = settings.bootstrap_n

    mat, cols = numeric_matrix(df, roles)
    method_s = str(method)
    notes: list[str] = [
        EPISTEMIC,
        "Heuristic discovery (PC-lite / lightweight / soft backends) is alpha — "
        "not a guarantee of causal identification. Gate production with "
        "mode='production' (ensemble+stability+QC block).",
    ]
    notes.extend(settings.notes)
    if len(cols) < 2:
        return DiscoveryResult(
            edges=[],
            graph={"nodes": list(df.columns), "edges": []},
            roles=roles,
            candidates={"treatment": [], "outcome": [], "instrument": [], "confounder": []},
            method=method_s,
            notes=notes + ["Fewer than 2 usable columns for discovery."],
            mode=settings.mode,
        )

    clean = mat.dropna()
    if len(clean) < 10:
        notes.append(f"Only {len(clean)} complete rows after encoding; results unstable.")

    if method_s in EXTERNAL_METHODS:
        edges, ext_notes = _run_external_method(method_s, clean, cols, alpha=alpha)
        notes.extend(ext_notes)
        if not edges:
            if is_production(settings.mode):
                raise ProductionGateError(
                    f"Production discovery engine `{method_s}` produced no edges "
                    "or was unavailable; heuristic fallback refused.",
                    code="discovery_engine_failed",
                    recommendations=[
                        "Install/verify the requested engine or choose an available "
                        "reviewed ensemble."
                    ],
                )
            # Fall back to pc_lite so callers still get a graph
            notes.append(f"{method_s} produced no edges — fell back to score_pc_lite.")
            edges = _pc_lite_edges(
                clean, cols, alpha=alpha, max_cond_size=max_cond_size, min_abs_corr=min_abs_corr
            )
            method_s = "score_pc_lite"
    else:
        edges = _run_method(
            method_s,
            clean,
            cols,
            alpha=alpha,
            max_cond_size=max_cond_size,
            min_abs_corr=min_abs_corr,
        )

    if stability and method_s in BUILTIN_METHODS:
        n_boot = max(1, int(bootstrap_n))
        stab_map = _bootstrap_stability(
            clean,
            cols,
            edges,
            method=method_s,  # type: ignore[arg-type]
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
    elif stability and method_s not in BUILTIN_METHODS:
        notes.append("Bootstrap stability skipped for external soft backends (cost).")

    clean, edges, cand, iv_notes = _apply_iv_pass(
        clean,
        roles,
        edges,
        cols,
        use_iv=use_iv,
        auto_instrument=auto_instrument,
        allow_iv_fallback=allow_iv_fallback,
        candidates_override=candidates,
        seed=seed,
    )
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
                "method": e.get("method", method_s),
            }
            for e in edges
        ],
        "candidates": cand,
    }

    return DiscoveryResult(
        edges=edges,
        graph=graph,
        roles=roles,
        candidates=cand,
        method=method_s,
        notes=notes,
        stability_enabled=stability,
        bootstrap_n=int(bootstrap_n) if stability else 0,
        mode=settings.mode,
    )


def discover_ensemble(
    df: pd.DataFrame,
    *,
    roles: dict[str, ColumnRole],
    methods: Optional[Sequence[DiscoveryMethod | str]] = None,
    alpha: float = 0.05,
    max_cond_size: int = 2,
    min_abs_corr: float = 0.15,
    use_iv: Optional[bool] = None,
    auto_instrument: bool = False,
    allow_iv_fallback: bool = False,
    candidates: Optional[dict[str, Sequence[str]]] = None,
    stability: Optional[bool] = None,
    bootstrap_n: Optional[int] = None,
    min_methods: Optional[int] = None,
    seed: int = 0,
    include_optional: bool = True,
    mode: str = "exploratory",
    policy: Optional[Any] = None,
) -> "DiscoveryResult":
    """Run multiple discovery methods and return a consensus graph.

    When ``methods`` is omitted and ``include_optional`` is True, installed soft
    backends (causal-learn PC/GES, lingam, gcastle) are appended automatically.
    """
    from autocausal.production import EPISTEMIC, apply_mode_defaults, is_production
    from autocausal.results import DiscoveryResult

    settings = apply_mode_defaults(
        mode=mode,
        policy=policy,
        auto_instrument=auto_instrument,
        allow_iv_fallback=allow_iv_fallback,
        use_iv=use_iv,
        stability=stability,
        bootstrap_n=bootstrap_n,
        ensemble=True,
        min_methods=min_methods,
    )
    auto_instrument = settings.auto_instrument
    allow_iv_fallback = settings.allow_iv_fallback
    use_iv = settings.use_iv
    stability = settings.stability
    bootstrap_n = settings.bootstrap_n
    min_methods = settings.min_methods

    if methods is None:
        methods_list: list[str] = ["score_pc_lite", "corr_skeleton", "mi_binned"]
        if include_optional:
            try:
                from autocausal.engines import optional_ensemble_methods

                for m in optional_ensemble_methods(installed_only=True):
                    if m not in methods_list:
                        methods_list.append(m)
            except Exception:
                pass
        methods = methods_list
    else:
        methods = list(methods)
    mat, cols = numeric_matrix(df, roles)
    notes: list[str] = [
        EPISTEMIC,
        f"Multi-method ensemble discovery: {list(methods)}. Consensus ≠ identification.",
    ]
    notes.extend(settings.notes)
    if len(cols) < 2:
        return DiscoveryResult(
            edges=[],
            graph={"nodes": list(df.columns), "edges": []},
            roles=roles,
            candidates={"treatment": [], "outcome": [], "instrument": [], "confounder": []},
            method="consensus",
            notes=notes + ["Fewer than 2 usable columns."],
            ensemble_methods=list(methods),
            mode=settings.mode,
        )

    clean = mat.dropna()
    method_edges: dict[str, list[dict[str, Any]]] = {}
    for i, m in enumerate(methods):
        m_s = str(m)
        if m_s in EXTERNAL_METHODS:
            edges_m, ext_notes = _run_external_method(m_s, clean, cols, alpha=alpha)
            notes.extend(ext_notes)
        else:
            edges_m = _run_method(
                m_s,
                clean,
                cols,
                alpha=alpha,
                max_cond_size=max_cond_size,
                min_abs_corr=min_abs_corr,
            )
        if stability and m_s in BUILTIN_METHODS:
            stab_map = _bootstrap_stability(
                clean,
                cols,
                edges_m,
                method=m_s,  # type: ignore[arg-type]
                bootstrap_n=max(1, int(bootstrap_n)),
                alpha=alpha,
                max_cond_size=max_cond_size,
                min_abs_corr=min_abs_corr,
                seed=seed + i,
            )
            edges_m = _apply_stability(edges_m, stab_map)
        method_edges[m_s] = edges_m

    edges = consensus_edges(method_edges, min_methods=min_methods)
    # Exploratory can fall back; production fails closed with an empty candidate set.
    if not edges and method_edges:
        if is_production(settings.mode):
            notes.append(
                "PRODUCTION: consensus empty under min_methods; primary fallback refused."
            )
        else:
            primary = method_edges.get("score_pc_lite") or next(iter(method_edges.values()))
            edges = list(primary)
            notes.append(
                "EXPLORATORY fallback: consensus empty under min_methods; "
                "used primary method edges."
            )

    clean, edges, cand, iv_notes = _apply_iv_pass(
        clean,
        roles,
        edges,
        cols,
        use_iv=use_iv,
        auto_instrument=auto_instrument,
        allow_iv_fallback=allow_iv_fallback,
        candidates_override=candidates,
        seed=seed,
    )
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
        "candidates": cand,
        "method_edges": {k: len(v) for k, v in method_edges.items()},
    }

    return DiscoveryResult(
        edges=edges,
        graph=graph,
        roles=roles,
        candidates=cand,
        method="consensus",
        notes=notes,
        stability_enabled=stability,
        bootstrap_n=int(bootstrap_n) if stability else 0,
        ensemble_methods=list(methods),
        method_edges={k: list(v) for k, v in method_edges.items()},
        mode=settings.mode,
    )
