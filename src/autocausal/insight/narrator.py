"""Insight narrator — deterministic summary + optional SLM narrative."""

from __future__ import annotations

from typing import Any, Optional

from autocausal.insight.report import InsightReport, RoleHypotheses


def synthesize_summary(
    *,
    edges: list[dict[str, Any]],
    roles: RoleHypotheses,
    n_rows: int,
    n_cols: int,
    stages: list[str],
    guide_backend: str,
    experiments: Optional[list[dict[str, Any]]] = None,
    rounds: int = 1,
) -> str:
    """Offline rule narrator — always available."""
    n_edges = len(edges)
    top = ""
    if edges:
        ranked = sorted(
            edges,
            key=lambda e: float(e.get("confidence") or e.get("score") or 0),
            reverse=True,
        )
        e0 = ranked[0]
        top = (
            f" Strongest exploratory edge: `{e0.get('source')}` → `{e0.get('target')}` "
            f"(type={e0.get('type')}, score={e0.get('score')})."
        )
    x = ", ".join(f"`{c}`" for c in roles.treatment[:4]) or "—"
    y = ", ".join(f"`{c}`" for c in roles.outcome[:4]) or "—"
    z = ", ".join(f"`{c}`" for c in roles.instrument[:4]) or "—"
    n_exp = len(experiments or [])
    round_bit = f" across {rounds} research round(s)" if rounds > 1 else ""
    return (
        f"Insight suite ran stages [{', '.join(stages) or 'n/a'}] on "
        f"{n_rows}×{n_cols} table{round_bit} (guide=`{guide_backend}`). "
        f"Found {n_edges} exploratory edge(s). "
        f"Role hypotheses — X: {x}; Y: {y}; Z: {z}.{top} "
        f"{n_exp} experiment recommendation(s) proposed. "
        "These are discovery candidates, not identified causal effects."
    )


def optional_slm_narrative(
    context: dict[str, Any],
    *,
    use_slm: bool,
    model_name: Optional[str] = None,
) -> tuple[Optional[str], bool, str]:
    """Return (narrative, slm_used, backend_name). Soft-fails to (None, False, rule)."""
    if not use_slm:
        return None, False, "rule"
    try:
        from autocausal.slm import get_backend, slm_available

        if not slm_available():
            return None, False, "rule"
        backend = get_backend(use_slm=True, model_name=model_name)
        result = backend.infer(context)
        narrative = (getattr(result, "narrative", None) or getattr(result, "raw_text", "") or "").strip()
        if not narrative:
            return None, False, getattr(result, "backend", "rule")
        return narrative, True, getattr(result, "backend", "huggingface")
    except Exception:
        return None, False, "rule"


def build_insight_report(
    *,
    edges: list[dict[str, Any]],
    candidates: dict[str, Any],
    source: str,
    n_rows: int,
    n_cols: int,
    stages: list[str],
    data_sources: list[str],
    guide: Optional[dict[str, Any]] = None,
    mining: Optional[dict[str, Any]] = None,
    discovery: Optional[dict[str, Any]] = None,
    nlp_hints: Optional[dict[str, Any]] = None,
    notes: Optional[list[str]] = None,
    use_slm: bool = False,
    model_name: Optional[str] = None,
    experiments_recommended: Optional[list[dict[str, Any]]] = None,
    relationships_mined_further: Optional[list[dict[str, Any]]] = None,
    round_history: Optional[list[dict[str, Any]]] = None,
    text: str = "",
) -> InsightReport:
    roles = RoleHypotheses.from_candidates(candidates)
    guide_backend = (guide or {}).get("backend") or ("huggingface" if use_slm else "rule")
    rounds = max(1, len(round_history or []) or 1)
    summary = synthesize_summary(
        edges=edges,
        roles=roles,
        n_rows=n_rows,
        n_cols=n_cols,
        stages=stages,
        guide_backend=str(guide_backend),
        experiments=experiments_recommended,
        rounds=rounds,
    )
    narrative, slm_used, nb = optional_slm_narrative(
        {
            "text": text,
            "edges": edges[:20],
            "candidates": candidates,
            "iv": None,
        },
        use_slm=use_slm,
        model_name=model_name,
    )
    if slm_used:
        guide_backend = nb

    key = sorted(
        edges,
        key=lambda e: float(e.get("confidence") or e.get("score") or 0),
        reverse=True,
    )[:25]

    return InsightReport(
        summary=summary,
        key_edges=key,
        role_hypotheses=roles,
        data_sources=list(data_sources),
        guide_backend=str(guide_backend),
        guide=guide,
        slm_narrative=narrative,
        slm_used=slm_used,
        nlp_hints=nlp_hints,
        mining=mining,
        discovery=discovery,
        stages=list(stages),
        notes=list(notes or []),
        source=source,
        n_rows=n_rows,
        n_cols=n_cols,
        experiments_recommended=list(experiments_recommended or []),
        relationships_mined_further=list(relationships_mined_further or []),
        round_history=list(round_history or []),
    )


# Back-compat aliases used in docs / early API sketches
synthesize_insight = build_insight_report
rule_narrate = synthesize_summary


def resolve_use_slm(use_slm: Optional[bool] = None) -> bool:
    """Honor explicit flag; else AUTOCAUSAL_SLM / related env (soft)."""
    import os

    if use_slm is False:
        return False
    if use_slm is True:
        return True
    for n in ("AUTOCAUSAL_SLM", "EMOTIVEVISION_SLM", "CAUSALIV_SLM"):
        if os.environ.get(n, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False
