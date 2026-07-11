"""Rule-first, analysis-aware visualization planning."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from autocausal.roles import ColumnRole, infer_column_roles
from autocausal.autoviz.report import VizPlan, VizRecommendation


StructuredPlanEnricher = Callable[[dict[str, Any]], Any]


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        return dict(converted) if isinstance(converted, Mapping) else {}
    return {}


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "results"):
        value = getattr(value, "results")
    if isinstance(value, Mapping):
        value = value.get("results") or value.get("gates") or [value]
    output: list[dict[str, Any]] = []
    for item in value or []:
        payload = _as_dict(item)
        if payload:
            output.append(payload)
    return output


def _role_name(value: Any) -> str:
    if isinstance(value, ColumnRole):
        return value.value
    return str(getattr(value, "value", value)).lower()


@dataclass
class PlannerContext:
    """Inputs beyond the frame that make the plan analysis-aware."""

    panel: Any = None
    candidates: dict[str, list[str]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)
    model_metrics: dict[str, Any] = field(default_factory=dict)
    gate_results: list[dict[str, Any]] = field(default_factory=list)
    feature_importance: list[dict[str, Any]] = field(default_factory=list)
    residual_columns: list[str] = field(default_factory=list)


class AutoVizPlanner:
    """Build ranked chart recommendations from schema and analysis artifacts.

    The rule planner is always available.  An optional SLM/Qwen adapter may add
    structured recommendations, but every added recommendation is validated
    against the same type and column constraints before it enters the plan.
    Raw frame values are never included in the enrichment payload.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        mode: str = "exploratory",
        roles: Optional[Mapping[str, Any]] = None,
        panel: Any = None,
        candidates: Optional[Mapping[str, Sequence[str]]] = None,
        edges: Optional[Sequence[Mapping[str, Any]]] = None,
        model_metrics: Any = None,
        gate_results: Any = None,
        feature_importance: Optional[Sequence[Mapping[str, Any]]] = None,
        residual_columns: Optional[Sequence[str]] = None,
        sensitive_columns: Optional[Sequence[str]] = None,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("AutoVizPlanner frame must be a pandas DataFrame")
        normalized_mode = str(mode).lower()
        if normalized_mode not in ("exploratory", "production"):
            raise ValueError("mode must be 'exploratory' or 'production'")
        self.frame = frame
        self.mode = normalized_mode
        inferred = roles or infer_column_roles(frame)
        self.roles = {str(key): _role_name(value) for key, value in inferred.items()}
        self.sensitive_columns = {str(value) for value in sensitive_columns or []}
        normalized_candidates: dict[str, list[str]] = {}
        for key, values in (candidates or {}).items():
            normalized_candidates[str(key).lower()] = [
                str(value) for value in values if str(value) in frame.columns
            ]
        self.context = PlannerContext(
            panel=panel,
            candidates=normalized_candidates,
            edges=[dict(edge) for edge in edges or []],
            model_metrics=_as_dict(model_metrics),
            gate_results=_as_list_of_dicts(gate_results),
            feature_importance=[dict(item) for item in feature_importance or []],
            residual_columns=[
                str(value)
                for value in residual_columns or []
                if str(value) in frame.columns
            ],
        )
        self._recommendations: list[VizRecommendation] = []

    def _add(
        self,
        identifier: str,
        chart_type: str,
        title: str,
        priority: int,
        rationale: str,
        *,
        columns: Optional[Sequence[str]] = None,
        requirements: Optional[dict[str, Any]] = None,
        hints: Optional[dict[str, Any]] = None,
        caveats: Optional[Sequence[str]] = None,
    ) -> None:
        valid_columns = [
            str(column)
            for column in (columns or [])
            if str(column) in self.frame.columns
        ]
        self._recommendations.append(
            VizRecommendation(
                id=identifier,
                chart_type=chart_type,
                title=title,
                priority=priority,
                rationale=rationale,
                required_columns=valid_columns,
                data_requirements=dict(requirements or {}),
                spec_hints=dict(hints or {}),
                caveats=list(caveats or []),
            )
        )

    def _candidate(self, *names: str) -> list[str]:
        output: list[str] = []
        for name in names:
            for value in self.context.candidates.get(name, []):
                if value not in output:
                    output.append(value)
        return output

    def _panel_columns(self) -> tuple[Optional[str], Optional[str]]:
        panel = self.context.panel
        if panel is None:
            return None, None
        if isinstance(panel, Mapping):
            return (
                panel.get("entity") or panel.get("group") or panel.get("group_col"),
                panel.get("time") or panel.get("time_col"),
            )
        return (
            getattr(panel, "entity", None) or getattr(panel, "group", None),
            getattr(panel, "time", None) or getattr(panel, "time_col", None),
        )

    def _build_rule_plan(self) -> None:
        numeric = [
            column for column, role in self.roles.items() if role == "numeric"
        ]
        categorical = [
            column
            for column, role in self.roles.items()
            if role in ("categorical", "boolean")
        ]
        datetime_columns = [
            column for column, role in self.roles.items() if role == "datetime"
        ]
        missingness = self.frame.isna().mean()
        cardinality = self.frame.nunique(dropna=True)

        for index, column in enumerate(numeric[:6]):
            self._add(
                f"distribution:{column}",
                "distribution",
                f"Distribution of {column}",
                84 - index,
                "Inspect scale, skew, outliers, and support before modeling.",
                columns=[column],
                requirements={"minimum_non_missing": 2},
                hints={"x": column, "aggregation": "histogram"},
            )

        if categorical:
            column = min(categorical, key=lambda item: int(cardinality.get(item, 0)))
            self._add(
                f"distribution:{column}",
                "distribution",
                f"Category distribution for {column}",
                78,
                "Check representation and rare categories before subgroup analysis.",
                columns=[column],
                requirements={"maximum_cardinality": 32},
                hints={"x": column, "aggregation": "count"},
            )

        columns_with_missing = [
            str(column) for column, value in missingness.items() if float(value) > 0
        ]
        if columns_with_missing:
            self._add(
                "missingness:overview",
                "missingness",
                "Missingness overview",
                96,
                "Missingness can change the estimand and invalidate complete-case comparisons.",
                columns=columns_with_missing[:30],
                requirements={"summary_only": True},
                hints={"aggregation": "missing_fraction"},
            )

        if len(numeric) >= 2:
            self._add(
                "association:numeric",
                "correlation",
                "Numeric association matrix",
                82,
                "Screen dependence, redundancy, and possible multicollinearity.",
                columns=numeric[:20],
                requirements={"minimum_numeric_columns": 2},
                hints={"aggregation": "correlation"},
            )
        if numeric and categorical:
            self._add(
                "association:mixed",
                "association",
                "Mixed-type association summary",
                69,
                "Compare relationships that a numeric-only correlation matrix omits.",
                columns=(numeric[:8] + categorical[:8]),
                requirements={"supports_mixed_types": True},
                hints={"aggregation": "association"},
            )

        treatment = self._candidate("treatment", "treatments", "exposure")
        outcome = self._candidate("outcome", "outcomes", "target")
        instruments = self._candidate("instrument", "instruments")
        covariates = self._candidate(
            "confounder", "confounders", "covariate", "covariates"
        )
        groups = self._candidate("group", "groups", "subgroup", "subgroups")
        if not groups:
            groups = [
                column
                for column in categorical
                if column not in set(treatment + outcome)
                and int(cardinality.get(column, 0)) <= 12
            ][:1]

        if treatment and outcome:
            self._add(
                "design:treatment_outcome",
                "treatment_outcome",
                "Treatment–outcome descriptive relationship",
                93,
                "Inspect support and descriptive differences for the proposed design roles.",
                columns=[treatment[0], outcome[0]],
                requirements={"explicit_roles_recommended": self.mode == "production"},
                hints={"x": treatment[0], "y": outcome[0]},
                caveats=[
                    "Treatment-group differences may be confounded and are not effect estimates."
                ],
            )
        if treatment and covariates:
            self._add(
                "design:covariate_balance",
                "covariate_balance",
                "Covariate balance by treatment",
                95,
                "Assess measured baseline imbalance before interpreting adjusted estimates.",
                columns=[treatment[0], *covariates[:12]],
                requirements={"binary_or_low_cardinality_treatment": True},
                hints={"x": treatment[0], "aggregation": "standardized_difference"},
            )
            self._add(
                "design:overlap",
                "overlap",
                "Treatment overlap and positivity",
                94,
                "Reveal weak common support where effect estimates require extrapolation.",
                columns=[treatment[0], *covariates[:12]],
                requirements={"propensity_scores_or_covariates": True},
                hints={"color": treatment[0]},
            )
        if treatment and outcome and instruments:
            self._add(
                "design:iv_first_stage",
                "iv_first_stage",
                "Instrument first-stage diagnostic",
                98,
                "Inspect instrument relevance without treating relevance as proof of validity.",
                columns=[instruments[0], treatment[0], outcome[0]],
                requirements={"observed_instrument_required_for_causal_claims": True},
                hints={"x": instruments[0], "y": treatment[0]},
                caveats=[
                    "A strong first stage does not establish exclusion or independence."
                ],
            )

        edges = self.context.edges
        edge_columns = {
            str(value)
            for edge in edges
            for value in (edge.get("source"), edge.get("target"))
            if value is not None and str(value) in self.frame.columns
        }
        if edges:
            self._add(
                "discovery:network",
                "dag",
                "Discovery network",
                76,
                "Summarize proposed edge structure and evidence annotations.",
                columns=sorted(edge_columns),
                requirements={"edge_count": len(edges)},
                hints={"aggregation": "edge_list"},
                caveats=[
                    "Discovered arrows are hypotheses and may not be causally oriented."
                ],
            )
        if any(edge.get("stability") is not None for edge in edges):
            self._add(
                "discovery:edge_stability",
                "edge_stability",
                "Bootstrap edge stability",
                91,
                "Expose unstable discoveries and sensitivity to resampling.",
                columns=sorted(edge_columns),
                requirements={"stability_metadata": True},
                hints={"aggregation": "edge_stability"},
            )

        entity_column, time_column = self._panel_columns()
        time_column = (
            str(time_column)
            if time_column is not None and str(time_column) in self.frame.columns
            else (datetime_columns[0] if datetime_columns else None)
        )
        entity_column = (
            str(entity_column)
            if entity_column is not None and str(entity_column) in self.frame.columns
            else None
        )
        trend_target = (outcome or numeric or [None])[0]
        if time_column and trend_target:
            columns = [time_column, trend_target]
            if entity_column:
                columns.append(entity_column)
            self._add(
                "panel:trend",
                "panel_trend",
                "Panel/time trend",
                88,
                "Check secular trends, timing, and panel support before temporal analysis.",
                columns=columns,
                requirements={"ordered_time": True},
                hints={
                    "x": time_column,
                    "y": trend_target,
                    "facet": entity_column,
                    "aggregation": "mean",
                },
            )

        if treatment and outcome and groups:
            self._add(
                "effects:subgroups",
                "subgroup_effects",
                "Subgroup effect summary",
                80,
                "Expose heterogeneity and sparse subgroups for follow-up validation.",
                columns=[treatment[0], outcome[0], groups[0]],
                requirements={"minimum_group_size": 10},
                hints={"x": groups[0], "y": outcome[0], "color": treatment[0]},
                caveats=[
                    "Subgroup contrasts are exploratory and require multiplicity controls."
                ],
            )

        metric_text = " ".join(
            str(key).lower() for key in self.context.model_metrics.keys()
        )
        task_text = str(
            self.context.model_metrics.get("task")
            or self.context.model_metrics.get("task_type")
            or ""
        ).lower()
        if self.context.residual_columns or any(
            token in metric_text for token in ("rmse", "mae", "r2", "residual")
        ):
            self._add(
                "model:residuals",
                "residual_diagnostics",
                "Residual diagnostics",
                86,
                "Check predictive misspecification, heteroskedasticity, and outliers.",
                columns=self.context.residual_columns[:3],
                requirements={"predictions_or_residuals": True},
                hints={"aggregation": "residual_summary"},
            )
        classification = (
            "classification" in task_text
            or "binary" in task_text
            or "multiclass" in task_text
            or any(
                token in metric_text
                for token in ("auc", "roc", "brier", "log_loss", "accuracy")
            )
        )
        if classification:
            for identifier, chart_type, title, priority in (
                ("model:calibration", "calibration", "Calibration curve", 90),
                ("model:roc", "roc", "ROC curve", 77),
                ("model:pr", "pr", "Precision–recall curve", 79),
            ):
                self._add(
                    identifier,
                    chart_type,
                    title,
                    priority,
                    "Evaluate predictive discrimination and probability quality on held-out data.",
                    requirements={"out_of_fold_predictions": True},
                    hints={"aggregation": "out_of_fold"},
                    caveats=[
                        "Predictive performance does not imply a causal relationship."
                    ],
                )
        if self.context.feature_importance or "importance" in metric_text:
            self._add(
                "model:feature_importance",
                "feature_importance",
                "Predictive feature importance",
                73,
                "Summarize model reliance with uncertainty and correlation caveats.",
                requirements={"held_out_or_permutation_importance": True},
                hints={"aggregation": "importance"},
                caveats=[
                    "Feature importance is predictive, not a causal attribution."
                ],
            )

        if self.context.gate_results:
            failed = sum(
                1
                for gate in self.context.gate_results
                if gate.get("ok") is False
                or str(gate.get("status", "")).lower() in ("fail", "escalate")
            )
            self._add(
                "production:gates",
                "gate_dashboard",
                "Evidence and production gate dashboard",
                100,
                "Make failed, warning, skipped, and overridden gates visible at handoff.",
                requirements={
                    "gate_count": len(self.context.gate_results),
                    "failed_gate_count": failed,
                },
                hints={"aggregation": "gate_status"},
            )

    def _safe_summary(self) -> dict[str, Any]:
        missingness = self.frame.isna().mean()
        cardinality = self.frame.nunique(dropna=True)
        return {
            "n_rows": int(len(self.frame)),
            "n_columns": int(len(self.frame.columns)),
            "columns": [str(column) for column in self.frame.columns],
            "roles": dict(self.roles),
            "missingness": {
                str(column): round(float(value), 6)
                for column, value in missingness.items()
            },
            "cardinality": {
                str(column): int(value) for column, value in cardinality.items()
            },
            "sensitive_columns": sorted(self.sensitive_columns),
            "contains_raw_values": False,
            "redacted": self.mode == "production",
        }

    def _apply_structured_enrichment(
        self,
        plan: VizPlan,
        enricher: StructuredPlanEnricher,
    ) -> VizPlan:
        payload = {
            "schema": "AutoCausalVizEnrichmentRequest.v1",
            "frame_summary": plan.frame_summary,
            "analysis": {
                "candidates": self.context.candidates,
                "edges": [
                    {
                        key: edge.get(key)
                        for key in (
                            "source",
                            "target",
                            "type",
                            "stability",
                            "evidence_grade",
                        )
                    }
                    for edge in self.context.edges[:100]
                ],
                "metric_names": sorted(self.context.model_metrics.keys()),
                "gates": [
                    {
                        key: gate.get(key)
                        for key in ("id", "ok", "status", "stage", "severity")
                    }
                    for gate in self.context.gate_results[:100]
                ],
            },
            "existing_plan": [
                recommendation.to_dict()
                for recommendation in plan.recommendations
            ],
            "contains_raw_values": False,
        }
        try:
            response = enricher(payload)
        except Exception as exc:
            plan.warnings.append(
                f"Structured plan enrichment failed safely: {type(exc).__name__}: {exc}"
            )
            return plan
        if isinstance(response, Mapping):
            response = response.get("recommendations") or response.get("plan") or []
        if not isinstance(response, Sequence) or isinstance(response, (str, bytes)):
            plan.warnings.append(
                "Structured plan enrichment was ignored because it did not return a list."
            )
            return plan
        existing_ids = {item.id for item in plan.recommendations}
        accepted: list[VizRecommendation] = []
        for raw in response:
            if not isinstance(raw, Mapping):
                continue
            try:
                value = dict(raw)
                value["source"] = "slm"
                recommendation = VizRecommendation.from_dict(value)
                if recommendation.id in existing_ids:
                    continue
                missing = recommendation.validate_columns(self.frame.columns)
                if missing:
                    raise ValueError(f"unknown columns: {missing}")
                accepted.append(recommendation)
                existing_ids.add(recommendation.id)
            except Exception as exc:
                plan.warnings.append(
                    "Rejected invalid enriched visualization recommendation: "
                    f"{type(exc).__name__}: {exc}"
                )
        plan.recommendations.extend(accepted)
        plan.recommendations.sort(key=lambda item: (-item.priority, item.id))
        if accepted:
            plan.planner = "rule+structured_enricher"
        plan.validate()
        return plan

    def plan(
        self,
        *,
        use_slm: bool = False,
        slm_enricher: Optional[StructuredPlanEnricher] = None,
    ) -> VizPlan:
        self._recommendations = []
        self._build_rule_plan()
        plan = VizPlan(
            recommendations=self._recommendations,
            frame_summary=self._safe_summary(),
            mode=self.mode,
            metadata={
                "n_edges": len(self.context.edges),
                "n_gates": len(self.context.gate_results),
                "has_panel_metadata": self.context.panel is not None,
                "contains_raw_values": False,
            },
        )
        if use_slm and slm_enricher is None:
            plan.warnings.append(
                "SLM enrichment requested without an adapter; deterministic rule plan retained."
            )
        elif use_slm and slm_enricher is not None:
            plan = self._apply_structured_enrichment(plan, slm_enricher)
        return plan


__all__ = ["AutoVizPlanner", "PlannerContext", "StructuredPlanEnricher"]
