"""Direction-steering guide backends for AutoCausalLib.

Soft-optional integrations:
- rule (always)
- huggingface (transformers)
- llmintent (research/LLMIntent)
- retracement (llmintent.retracement or stub)
- kineteq_pivot (MCP / module / local hashing fallback)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from autocausal.guides.huggingface import HuggingFaceSLMGuide
from autocausal.guides.kineteq_guide import (
    KineteqPivotEmbeddingGuide,
    kineteq_mcp_configured,
    kineteq_module_importable,
)
from autocausal.guides.llmintent_guide import LLMIntentGuide, llmintent_importable
from autocausal.guides.retracement_guide import RetracementGuide, retracement_importable
from autocausal.guides.rule import RuleGuide
from autocausal.guides.types import DirectionPlan, GuideResult, uniq

__all__ = [
    "DirectionPlan",
    "GuideResult",
    "RuleGuide",
    "HuggingFaceSLMGuide",
    "LLMIntentGuide",
    "RetracementGuide",
    "KineteqPivotEmbeddingGuide",
    "BACKEND_ALIASES",
    "DEFAULT_BACKENDS",
    "list_guides",
    "get_guide",
    "resolve_backends",
    "merge_guide_results",
    "direct",
    "guides_status",
]

BACKEND_ALIASES: dict[str, str] = {
    "rule": "rule",
    "rules": "rule",
    "huggingface": "huggingface",
    "hf": "huggingface",
    "slm": "huggingface",
    "llmintent": "llmintent",
    "intent": "llmintent",
    "retracement": "retracement",
    "retrace": "retracement",
    "kineteq_pivot": "kineteq_pivot",
    "kineteq": "kineteq_pivot",
    "pivot": "kineteq_pivot",
    "pivot_embed": "kineteq_pivot",
}

DEFAULT_BACKENDS: tuple[str, ...] = ("rule",)


def _make(name: str, *, model_name: Optional[str] = None) -> Any:
    if name == "rule":
        return RuleGuide()
    if name == "huggingface":
        return HuggingFaceSLMGuide(model_name=model_name)
    if name == "llmintent":
        return LLMIntentGuide(model_name=model_name)
    if name == "retracement":
        return RetracementGuide()
    if name == "kineteq_pivot":
        return KineteqPivotEmbeddingGuide()
    raise KeyError(f"Unknown guide backend: {name}")


def resolve_backends(
    backends: Optional[Sequence[str]] = None,
    *,
    use_slm: bool = False,
) -> list[str]:
    """Normalize backend names; append huggingface when use_slm."""
    if backends is None or len(list(backends)) == 0:
        names = list(DEFAULT_BACKENDS)
    else:
        names = []
        for b in backends:
            key = BACKEND_ALIASES.get(str(b).strip().lower(), str(b).strip().lower())
            if key not in names:
                names.append(key)
    if use_slm and "huggingface" not in names:
        names.append("huggingface")
    # Always keep rule as safety net if empty after filter
    return names or ["rule"]


def get_guide(name: str, *, model_name: Optional[str] = None) -> Any:
    key = BACKEND_ALIASES.get(name.strip().lower(), name.strip().lower())
    return _make(key, model_name=model_name)


def list_guides() -> list[dict[str, Any]]:
    """Availability snapshot for CLI `guides list`."""
    ret_ok, ret_src = retracement_importable()
    kin_ok, kin_src = kineteq_module_importable()
    mcp = kineteq_mcp_configured()
    return [
        {
            "id": "rule",
            "name": "RuleGuide",
            "available": True,
            "priority": 0,
            "detail": "Deterministic offline heuristics (always on)",
        },
        {
            "id": "huggingface",
            "name": "HuggingFaceSLM",
            "available": HuggingFaceSLMGuide().available(),
            "priority": 1,
            "detail": "Optional transformers SLM (pip install autocausal[slm])",
        },
        {
            "id": "llmintent",
            "name": "LLMIntentGuide",
            "available": llmintent_importable(),
            "priority": 2,
            "detail": "Intent/morpheme/report APIs (pip install -e ../LLMIntent)",
        },
        {
            "id": "retracement",
            "name": "RetracementGuide",
            "available": ret_ok,
            "priority": 3,
            "detail": (
                f"Bound via {ret_src}"
                if ret_ok
                else "Stub until llmintent.retracement / retracement package present"
            ),
        },
        {
            "id": "kineteq_pivot",
            "name": "KineteqPivotEmbeddingGuide",
            "available": kin_ok or mcp,
            "priority": 4,
            "detail": (
                f"module={kin_src or '—'}; mcp={'on' if mcp else 'off'}; "
                "else local pivot_fallback (not Kineteq)"
            ),
        },
    ]


def guides_status() -> dict[str, Any]:
    return {
        "backends": list_guides(),
        "aliases": dict(BACKEND_ALIASES),
        "default": list(DEFAULT_BACKENDS),
        "env": {
            "AUTOCAUSAL_LLMINTENT_MODEL": bool(
                __import__("os").environ.get("AUTOCAUSAL_LLMINTENT_MODEL")
            ),
            "AUTOCAUSAL_LLMINTENT_HEAVY": __import__("os").environ.get(
                "AUTOCAUSAL_LLMINTENT_HEAVY", ""
            ),
            "KINETEQ_MCP_URL": bool(__import__("os").environ.get("KINETEQ_MCP_URL")),
            "AUTOCAUSAL_KINETEQ_MCP": __import__("os").environ.get(
                "AUTOCAUSAL_KINETEQ_MCP", ""
            ),
        },
    }


def merge_guide_results(
    results: Sequence[GuideResult],
    *,
    requested: Optional[Sequence[str]] = None,
) -> DirectionPlan:
    """Merge per-backend GuideResults into a DirectionPlan (later backends refine)."""
    plan = DirectionPlan()
    if requested:
        plan.backends = list(requested)

    focus_scores: dict[str, float] = {}
    for gr in results:
        if gr.backend not in plan.backends:
            plan.backends.append(gr.backend)
        plan.contributions.append(gr.to_dict())
        if not gr.available:
            plan.unavailable.append(gr.backend)

        for c in gr.focus_columns:
            focus_scores[c] = focus_scores.get(c, 0.0) + 1.0
        for c in gr.related_variables:
            focus_scores[c] = focus_scores.get(c, 0.0) + 0.5

        plan.candidate_z.extend(gr.instruments)
        plan.confounders.extend(gr.confounders)
        plan.treatment.extend(gr.treatment)
        plan.outcome.extend(gr.outcome)
        plan.boost_edges.extend(gr.boost_edges or [
            {"source": e.get("source"), "target": e.get("target"), "reason": "validate", "backend": gr.backend}
            for e in gr.validate_edges
        ])
        plan.suppress_edges.extend(gr.suppress_edges or [
            {"source": e.get("source"), "target": e.get("target"), "reason": "drop", "backend": gr.backend}
            for e in gr.drop_edges
        ])
        plan.search_queries.extend(gr.search_queries)
        plan.next_questions.extend(gr.next_questions)
        plan.related_variables.extend(gr.related_variables)
        plan.lag_hints.extend(gr.lag_hints)
        for n in gr.notes:
            plan.rationale.append(f"[{gr.backend}] {n}")
        plan.notes.extend(gr.notes)
        # ML Model Hub construction hints (last non-empty wins for imputer/predictor)
        if getattr(gr, "kpi_focus", None):
            plan.kpi_focus.extend(list(gr.kpi_focus))
        if getattr(gr, "imputer", None):
            plan.imputer = gr.imputer
        if getattr(gr, "predictor", None):
            plan.predictor = gr.predictor

    ranked = sorted(focus_scores.items(), key=lambda kv: kv[1], reverse=True)
    plan.focus_columns = [c for c, _ in ranked[:24]]
    plan.candidate_z = uniq(plan.candidate_z, limit=12)
    plan.treatment = uniq(plan.treatment, limit=8)
    plan.outcome = uniq(plan.outcome, limit=8)
    plan.confounders = uniq(plan.confounders, limit=12)
    plan.boost_edges = uniq(plan.boost_edges, limit=30)
    plan.suppress_edges = uniq(plan.suppress_edges, limit=30)
    plan.search_queries = uniq(plan.search_queries, limit=12)
    plan.next_questions = uniq(plan.next_questions, limit=12)
    plan.related_variables = uniq(plan.related_variables, limit=20)
    plan.lag_hints = uniq(plan.lag_hints, limit=20)
    plan.unavailable = uniq(plan.unavailable, limit=20)
    plan.rationale = uniq(plan.rationale, limit=40)
    plan.notes = uniq(plan.notes, limit=40)
    plan.kpi_focus = uniq(plan.kpi_focus, limit=16)
    return plan


def direct(
    context: dict[str, Any],
    *,
    backends: Optional[Sequence[str]] = None,
    use_slm: bool = False,
    model_name: Optional[str] = None,
    include_unavailable: bool = True,
) -> DirectionPlan:
    """
    Run selected guide backends and merge into a DirectionPlan.

    Soft-fails: missing packages still contribute stub/fallback results when
    include_unavailable=True (default), so callers always get a usable plan.
    """
    names = resolve_backends(backends, use_slm=use_slm)
    results: list[GuideResult] = []
    for name in names:
        try:
            guide = _make(name, model_name=model_name)
        except KeyError as e:
            results.append(
                GuideResult(
                    backend=str(name),
                    available=False,
                    notes=[str(e)],
                )
            )
            continue
        try:
            gr = guide.guide(context)
        except Exception as e:
            gr = GuideResult(
                backend=name,
                available=False,
                notes=[f"guide soft-fail: {type(e).__name__}: {e}"],
            )
        if gr.available or include_unavailable:
            results.append(gr)
    if not results:
        results.append(RuleGuide().guide(context))
        names = ["rule"]
    return merge_guide_results(results, requested=names)
