"""RuleGuide adapter — wraps autocausal.slm.RuleBackend."""

from __future__ import annotations

import os
from typing import Any

from autocausal.guides.types import GuideResult, GuideSuggestion


def _env_torch() -> bool:
    return os.environ.get("AUTOCAUSAL_TORCH", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _torch_ok() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


class RuleGuide:
    """Always-available deterministic guide (default)."""

    name = "rule"

    def available(self) -> bool:
        return True

    def guide(self, context: dict[str, Any]) -> GuideResult:
        from autocausal.slm import RuleBackend

        base = RuleBackend().guide(context)
        kpis = [str(k) for k in (context.get("kpis") or [])]
        # Prefer torch MLP when env asks and torch is installed; else median.
        if _env_torch() and _torch_ok():
            imputer, predictor = "torch_mlp", "torch_mlp"
        else:
            imputer, predictor = "median", "sklearn_rf"
        notes = list(base.notes) + [
            f"ModelConstructPlan hint: imputer={imputer} predictor={predictor}",
        ]
        return GuideResult(
            backend=base.backend,
            suggestions=[
                GuideSuggestion(
                    action=s.action,
                    detail=s.detail,
                    priority=s.priority,
                    meta=dict(s.meta or {}),
                )
                for s in base.suggestions
            ],
            focus_columns=list(base.focus_columns) or list(kpis),
            drop_edges=list(base.drop_edges),
            validate_edges=list(base.validate_edges),
            instruments=list(base.instruments),
            confounders=list(base.confounders),
            search_queries=list(base.search_queries),
            suppress_edges=[
                {"source": e.get("source"), "target": e.get("target"), "reason": "rule_drop"}
                for e in base.drop_edges
            ],
            boost_edges=[
                {"source": e.get("source"), "target": e.get("target"), "reason": "rule_validate"}
                for e in base.validate_edges
            ],
            raw_text=base.raw_text,
            notes=notes,
            available=True,
            imputer=imputer,
            predictor=predictor,
            kpi_focus=kpis[:16],
        )
