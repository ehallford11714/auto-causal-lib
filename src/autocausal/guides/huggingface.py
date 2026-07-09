"""HuggingFace SLM guide adapter — wraps autocausal.slm.HuggingFaceSLM."""

from __future__ import annotations

from typing import Any, Optional

from autocausal.guides.types import GuideResult, GuideSuggestion


class HuggingFaceSLMGuide:
    """Optional HuggingFace transformers guide (soft-fail if deps missing)."""

    name = "huggingface"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except Exception:
            return False

    def guide(self, context: dict[str, Any]) -> GuideResult:
        from autocausal.slm import HuggingFaceSLM

        base = HuggingFaceSLM(model_name=self.model_name).guide(context)
        ok = "unavailable" not in base.backend and "error" not in base.backend
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
            focus_columns=list(base.focus_columns),
            drop_edges=list(base.drop_edges),
            validate_edges=list(base.validate_edges),
            instruments=list(base.instruments),
            confounders=list(base.confounders),
            search_queries=list(base.search_queries),
            suppress_edges=[
                {"source": e.get("source"), "target": e.get("target"), "reason": "hf_drop"}
                for e in base.drop_edges
            ],
            boost_edges=[
                {"source": e.get("source"), "target": e.get("target"), "reason": "hf_validate"}
                for e in base.validate_edges
            ],
            raw_text=base.raw_text,
            notes=list(base.notes),
            available=ok,
        )
