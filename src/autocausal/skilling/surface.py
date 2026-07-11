"""ToolSurface — JSON-schema-like tool definitions for SLM / broker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from autocausal.suites.action_protocol import ActionResult

__all__ = ["ToolDef", "ToolSurface", "suite_tool_surface"]

EPISTEMIC = (
    "Tool outcomes are exploratory assistance — not causal identification. "
    "SLM-selected tools are generative proposals executed only when feasible."
)

Handler = Callable[..., ActionResult]


@dataclass
class ToolDef:
    """One callable tool the SLM / broker can invoke."""

    name: str
    description: str
    parameters: dict[str, Any]
    suite: str
    handler: Optional[Handler] = field(default=None, repr=False)
    action: str = ""
    epistemic: str = EPISTEMIC

    def schema(self) -> dict[str, Any]:
        """JSON-schema-like public description (no handler)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "suite": self.suite,
            "action": self.action or self.name.split(".", 1)[-1],
            "epistemic": self.epistemic,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.schema()


class ToolSurface:
    """Collection of tools the SLM can call."""

    def __init__(self, tools: Optional[list[ToolDef]] = None) -> None:
        self._tools: dict[str, ToolDef] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}. Known: {self.list_names()}")
        return self._tools[name]

    def list_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def list_tools(self, *, suite: Optional[str] = None) -> list[ToolDef]:
        tools = list(self._tools.values())
        if suite:
            tools = [t for t in tools if t.suite == suite]
        return sorted(tools, key=lambda t: t.name)

    def schemas(self, *, suite: Optional[str] = None) -> list[dict[str, Any]]:
        return [t.schema() for t in self.list_tools(suite=suite)]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def merge(self, other: "ToolSurface") -> "ToolSurface":
        out = ToolSurface(list(self._tools.values()))
        for t in other._tools.values():
            out.register(t)
        return out


def _param_object(props: dict[str, Any], required: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(required or []),
        "additionalProperties": True,
    }


def _wrap_action(action_fn: Handler, action_name: str) -> Handler:
    def handler(df: pd.DataFrame, **kwargs: Any) -> ActionResult:
        return action_fn(df, **kwargs)

    handler.__name__ = action_name
    return handler


def suite_tool_surface() -> ToolSurface:
    """Build the default surface wrapping AutoCleanse / AutoEDA / AutoMine actions
    plus soft discover / insight / experiment tools.
    """
    from autocausal.suites.autocleanse.actions import CLEANSE_REGISTRY, CleanseActions
    from autocausal.suites.autoeda.actions import EDA_REGISTRY, EDAActions
    from autocausal.suites.automine.actions import MINE_REGISTRY, MineActions

    surface = ToolSurface()

    cleanse_meta = {
        "profile_missingness": ("Profile per-column missingness (read-only).", {}),
        "coerce_types": (
            "Coerce object columns to numeric/datetime when majority parseable.",
            {"columns": {"type": "array", "items": {"type": "string"}}},
        ),
        "drop_duplicates": ("Drop exact duplicate rows.", {}),
        "drop_high_null_cols": (
            "Drop columns with missingness above threshold.",
            {"max_missing_frac": {"type": "number", "default": 0.95}},
        ),
        "drop_constant_cols": ("Drop constant columns.", {}),
        "flag_outliers": (
            "Flag/winsorize numeric outliers by z-score.",
            {
                "z": {"type": "number", "default": 5.0},
                "columns": {"type": "array", "items": {"type": "string"}},
                "winsorize": {"type": "boolean", "default": True},
            },
        ),
        "impute": (
            "Impute missing values via autocausal.impute.",
            {
                "method": {
                    "type": "string",
                    "enum": ["auto", "median_mode", "knn", "none"],
                    "default": "auto",
                }
            },
        ),
        "strip_id_leakage": (
            "Detect ID-like / leakage-named columns; optionally drop.",
            {"drop": {"type": "boolean", "default": False}},
        ),
        "qc_snapshot": ("Run QC validate_frame (read-only).", {}),
    }
    for name in CleanseActions.list():
        desc, props = cleanse_meta.get(name, (f"AutoCleanse action `{name}`.", {}))
        surface.register(
            ToolDef(
                name=f"autocleanse.{name}",
                description=desc,
                parameters=_param_object(props),
                suite="autocleanse",
                action=name,
                handler=_wrap_action(CLEANSE_REGISTRY.get(name), name),
            )
        )

    eda_meta = {
        "summarize_distributions": ("Numeric summaries + dtypes/missingness.", {}),
        "correlation_matrix": (
            "Pairwise numeric correlations.",
            {"max_cols": {"type": "integer", "default": 12}},
        ),
        "cardinality_report": ("Per-column cardinality.", {}),
        "suggest_roles": ("Propose X/Y/Z/W role hypotheses.", {}),
        "qc_snapshot": ("QC snapshot for causal readiness.", {}),
        "leakage_hints": ("Name/corr leakage hints.", {}),
        "mining_profile": ("Soft mining column profile.", {}),
    }
    for name in EDAActions.list():
        desc, props = eda_meta.get(name, (f"AutoEDA action `{name}`.", {}))
        surface.register(
            ToolDef(
                name=f"autoeda.{name}",
                description=desc,
                parameters=_param_object(props),
                suite="autoeda",
                action=name,
                handler=_wrap_action(EDA_REGISTRY.get(name), name),
            )
        )

    mine_meta = {
        "mine_associations": (
            "Mine associations via autocausal.mining.",
            {"min_score": {"type": "number", "default": 0.15}},
        ),
        "mine_kpi_hints": ("Suggest KPI-like columns.", {}),
        "join_public_sources": (
            "Join offline public suite sources.",
            {"sources": {"type": "array", "items": {"type": "string"}}},
        ),
        "mine_behavioral": ("Soft behavioral-trace mining.", {}),
        "rank_candidates": ("Rank association/relationship candidates.", {}),
        "to_mine_report": ("Export Fabric MineReport.v1 envelope.", {}),
    }
    for name in MineActions.list():
        desc, props = mine_meta.get(name, (f"AutoMine action `{name}`.", {}))
        surface.register(
            ToolDef(
                name=f"automine.{name}",
                description=desc,
                parameters=_param_object(props),
                suite="automine",
                action=name,
                handler=_wrap_action(MINE_REGISTRY.get(name), name),
            )
        )

    # Soft discover / insight / experiment / GRAIL tools
    def _discover_tool(df: pd.DataFrame, **kwargs: Any) -> ActionResult:
        from autocausal.api import AutoCausal

        ac = AutoCausal.from_dataframe(df)
        result = ac.discover(
            qc=kwargs.get("qc", "off"),
            use_iv=bool(kwargs.get("use_iv", False)),
            min_abs_corr=float(kwargs.get("min_abs_corr", 0.15)),
        )
        return ActionResult(
            name="discover",
            payload={"edges": list(result.edges or []), "n_edges": len(result.edges or [])},
            notes=["Discovery is exploratory — not identification."],
            n_affected=len(result.edges or []),
        )

    surface.register(
        ToolDef(
            name="autocausal.discover",
            description="Run exploratory causal discovery on a frame.",
            parameters=_param_object(
                {
                    "qc": {"type": "string", "enum": ["off", "warn", "block"]},
                    "use_iv": {"type": "boolean", "default": False},
                    "min_abs_corr": {"type": "number", "default": 0.15},
                }
            ),
            suite="autocausal",
            action="discover",
            handler=_discover_tool,
        )
    )

    def _insight_tool(df: pd.DataFrame, **kwargs: Any) -> ActionResult:
        try:
            from autocausal.insight import InsightSuite

            report = InsightSuite(use_slm=False).run(df, text=str(kwargs.get("text") or ""))
            return ActionResult(
                name="insight_run",
                payload={"report": report.to_dict() if hasattr(report, "to_dict") else str(report)},
                notes=["Insight suite soft tool — exploratory."],
            )
        except Exception as e:
            return ActionResult(
                name="insight_run",
                warnings=[f"Insight soft-fail: {type(e).__name__}: {e}"],
            )

    surface.register(
        ToolDef(
            name="insight.run",
            description="Run InsightSuite once (rule path).",
            parameters=_param_object({"text": {"type": "string"}}),
            suite="insight",
            action="run",
            handler=_insight_tool,
        )
    )

    def _experiment_tool(df: pd.DataFrame, **kwargs: Any) -> ActionResult:
        try:
            from autocausal.insight.experiments import ExperimentRecommender

            plan = ExperimentRecommender(use_slm=False).recommend(
                text=str(kwargs.get("text") or ""),
                edges=list(kwargs.get("edges") or []),
            )
            return ActionResult(
                name="experiment_recommend",
                payload={"plan": plan.to_dict()},
                notes=["Experiment recommendations are proposals only."],
            )
        except Exception as e:
            return ActionResult(
                name="experiment_recommend",
                warnings=[f"Experiment soft-fail: {type(e).__name__}: {e}"],
            )

    surface.register(
        ToolDef(
            name="insight.experiment_recommend",
            description="Recommend next experiments (rule ExperimentRecommender).",
            parameters=_param_object({"text": {"type": "string"}}),
            suite="insight",
            action="experiment_recommend",
            handler=_experiment_tool,
        )
    )

    # Soft GRAIL tools (autocausal_grail_*) — never hard-require Kineteq
    try:
        from autocausal.grail import register_grail_skilling_tools

        register_grail_skilling_tools(surface)
    except Exception:
        pass

    # Reporting tools are lazy and operate only on normalized report artifacts.
    try:
        from autocausal.reporting.tools import register_reporting_skilling_tools

        register_reporting_skilling_tools(surface)
    except Exception:
        pass

    return surface
