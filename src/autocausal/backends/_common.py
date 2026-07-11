"""Shared helpers for soft causal backends."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def soft_import(module: str) -> Any:
    try:
        import importlib

        return importlib.import_module(module)
    except Exception:
        return None


def soft_skip_result(
    *,
    method: str,
    module: str,
    install: str,
    notes: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "soft_skip": True,
        "method": method,
        "backend": "missing",
        "edges": [],
        "data": {},
        "notes": list(notes or [])
        + [f"{module} not installed — soft-skip. {install}"],
        "error": None,
    }


def numeric_frame(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    cols = list(columns) if columns else [str(c) for c in df.columns]
    keep: list[str] = []
    work = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 5:
            work[c] = s
            keep.append(c)
    clean = work.dropna()
    return clean, keep


def adjacency_to_edges(
    adj: np.ndarray,
    cols: list[str],
    *,
    method: str,
    threshold: float = 0.01,
    directed: bool = True,
) -> list[dict[str, Any]]:
    """Map a weighted adjacency matrix to AutoCausal edge dicts."""
    edges: list[dict[str, Any]] = []
    p = len(cols)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            w = float(adj[i, j]) if np.isfinite(adj[i, j]) else 0.0
            if abs(w) < threshold:
                continue
            if not directed and i > j:
                continue
            score = min(1.0, abs(w))
            edges.append(
                {
                    "source": cols[i],
                    "target": cols[j],
                    "score": round(score, 4),
                    "confidence": round(min(0.95, 0.35 + 0.5 * score), 4),
                    "pvalue": None,
                    "type": "association",
                    "orientation": method,
                    "method": method,
                    "weight": round(w, 4),
                }
            )
    edges.sort(key=lambda e: e.get("confidence", 0), reverse=True)
    return edges


def resolve_roles(
    df: pd.DataFrame,
    *,
    y: Optional[str] = None,
    d: Optional[str] = None,
    x: Optional[list[str]] = None,
    candidates: Optional[dict[str, list[str]]] = None,
) -> dict[str, Any]:
    """Resolve treatment/outcome/controls from args or mined candidates."""
    cands = candidates or {}
    outcome = y or (cands.get("outcome") or [None])[0]
    treatment = d or (cands.get("treatment") or [None])[0]
    if outcome is None or treatment is None:
        numeric = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8
        ]
        if outcome is None and numeric:
            outcome = numeric[-1]
        if treatment is None and len(numeric) >= 2:
            treatment = numeric[0] if numeric[0] != outcome else numeric[1]
    # ``x=None`` means "unspecified" → candidates then numeric auto-fill.
    # ``x=[]`` means "explicitly no confounders" (production estimate path).
    controls = list(x) if x is not None else list(cands.get("confounder") or [])
    if x is None and not controls and outcome and treatment:
        controls = [
            c
            for c in df.columns
            if c not in (outcome, treatment)
            and (
                pd.api.types.is_numeric_dtype(df[c])
                or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8
            )
        ][:8]
    return {
        "y": outcome,
        "d": treatment,
        "x": [c for c in controls if c not in (outcome, treatment) and c in df.columns],
    }
