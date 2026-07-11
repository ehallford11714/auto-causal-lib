"""Validated structured-output adapter for the existing local Qwen/HF backend."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional


class StructuredOutputError(ValueError):
    pass


def _validated_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError("SLM response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise StructuredOutputError("SLM response must be a JSON object")
    return payload


class HuggingFaceResearchSLM:
    """Research prompts over ``autocausal.slm.HuggingFaceSLM``.

    Every response is parsed as JSON and then validated by the consuming stage.
    A failed parse raises so the workflow can record a rule fallback.
    """

    def __init__(self, *, model_name: Optional[str] = None) -> None:
        from autocausal.slm import get_backend

        self.backend = get_backend(use_slm=True, model_name=model_name)
        self.model_name = model_name
        self.tokens_used = 0

    def _call(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        generate = getattr(self.backend, "_generate", None)
        if not callable(generate):
            raise StructuredOutputError(
                "configured SLM backend does not expose structured generation"
            )
        compact = json.dumps(payload, sort_keys=True, default=str)[:24_000]
        prompt = (
            f"Task: {task}\n"
            "Return one JSON object only. Do not add citations or identifiers that "
            "are absent from the input. Treat literature as external context, not "
            "causal identification.\n\n"
            f"INPUT={compact}"
        )
        text = generate(
            prompt,
            system=(
                "You are a cautious systematic-research assistant. "
                "Use only supplied records and emit strict JSON."
            ),
        )
        self.tokens_used += max(1, (len(prompt) + len(str(text))) // 4)
        return _validated_object(text)

    def plan_questions(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._call(
            "Enrich the rule-first research agenda using the supplied question schema.",
            payload,
        )

    def screen_sources(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._call(
            "Return keep_source_ids and exclude entries using only supplied source_ids.",
            payload,
        )

    def adjudicate_match(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._call(
            "Adjudicate semantic retrieval relevance and relation for this precomputed candidate.",
            payload,
        )

    def extract_evidence(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._call(
            "Return exact copied evidence spans from supplied source text.",
            payload,
        )

    def synthesize(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._call(
            "Summarize only the supplied claims/spans; return narrative and unresolved_claim_ids.",
            payload,
        )

    def expand_queries(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._call(
            "Suggest bounded query expansions for unresolved questions using supplied terms only.",
            payload,
        )


def resolve_research_slm(
    *,
    use_slm: bool,
    backend: Any = None,
    model_name: Optional[str] = None,
) -> Any:
    if not use_slm:
        return None
    return (
        backend
        if backend is not None
        else HuggingFaceResearchSLM(model_name=model_name)
    )


__all__ = [
    "HuggingFaceResearchSLM",
    "StructuredOutputError",
    "resolve_research_slm",
]
