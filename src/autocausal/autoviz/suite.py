"""Library-first AutoViz orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import pandas as pd

from autocausal.autoviz.planner import AutoVizPlanner, StructuredPlanEnricher
from autocausal.autoviz.report import AutoVizReport


def _serialize(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


class AutoVizSuite:
    """Inspect a frame and connected analysis artifacts, then rank charts.

    ``source`` may be a DataFrame, an ``AutoCausal`` instance, or any source
    accepted by ``autocausal.suites.base.resolve_frame``.
    """

    def __init__(
        self,
        source: Any,
        *,
        mode: Optional[str] = None,
        roles: Optional[Mapping[str, Any]] = None,
        panel: Any = None,
        candidates: Optional[Mapping[str, Sequence[str]]] = None,
        edges: Optional[Sequence[Mapping[str, Any]]] = None,
        model_metrics: Any = None,
        gate_results: Any = None,
        feature_importance: Optional[Sequence[Mapping[str, Any]]] = None,
        residual_columns: Optional[Sequence[str]] = None,
        sensitive_columns: Optional[Sequence[str]] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
    ) -> None:
        from autocausal.suites.base import resolve_frame

        frame, source_label, ac = resolve_frame(source, table=table, query=query)
        self.frame: pd.DataFrame = frame
        self.source = source_label
        self.ac = ac
        self.mode = str(mode or getattr(ac, "mode", "exploratory")).lower()

        result = getattr(ac, "result", None)
        manifest = getattr(ac, "run_manifest", None)
        privacy = getattr(manifest, "privacy", {}) if manifest is not None else {}
        inferred_sensitive = list((privacy or {}).get("pii_columns") or [])

        self.roles = roles or getattr(ac, "roles", None)
        self.panel = panel if panel is not None else getattr(ac, "panel_spec", None)
        self.candidates = candidates or getattr(result, "candidates", None) or {}
        self.edges = edges or getattr(result, "edges", None) or []
        self.model_metrics = _serialize(model_metrics)
        model_payload = (
            self.model_metrics
            if isinstance(self.model_metrics, Mapping)
            else {}
        )
        self.gate_results = (
            gate_results
            if gate_results is not None
            else getattr(manifest, "gates", None)
            or model_payload.get("gates")
        )
        self.feature_importance = (
            feature_importance
            if feature_importance is not None
            else model_payload.get("feature_importance")
        )
        self.residual_columns = residual_columns
        self.sensitive_columns = list(
            dict.fromkeys([*inferred_sensitive, *(sensitive_columns or [])])
        )

    @classmethod
    def from_autocausal(cls, ac: Any, **kwargs: Any) -> "AutoVizSuite":
        return cls(ac, **kwargs)

    def run(
        self,
        *,
        use_slm: bool = False,
        slm_enricher: Optional[StructuredPlanEnricher] = None,
    ) -> AutoVizReport:
        if use_slm and self.ac is not None:
            policy = getattr(self.ac, "policy", None)
            if policy is not None and not bool(getattr(policy, "allow_slm", True)):
                raise ValueError(
                    "SLM visualization enrichment is disabled by the active policy."
                )
        planner = AutoVizPlanner(
            self.frame,
            mode=self.mode,
            roles=self.roles,
            panel=self.panel,
            candidates=self.candidates,
            edges=self.edges,
            model_metrics=self.model_metrics,
            gate_results=self.gate_results,
            feature_importance=self.feature_importance,
            residual_columns=self.residual_columns,
            sensitive_columns=self.sensitive_columns,
        )
        plan = planner.plan(use_slm=use_slm, slm_enricher=slm_enricher)
        notes = [
            "The rule planner remains available without plotting or model dependencies.",
            "Recommendations describe diagnostics; they do not infer causality.",
        ]
        if self.mode == "production":
            notes.append(
                "Production plan contains schema and aggregate metadata only; "
                "sensitive sample values were not included."
            )
        return AutoVizReport(plan=plan, source=self.source, notes=notes)

    def report(self, **kwargs: Any) -> AutoVizReport:
        """Alias for :meth:`run`, matching other suite entry points."""
        return self.run(**kwargs)


__all__ = ["AutoVizSuite"]
