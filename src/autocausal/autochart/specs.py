"""Typed chart specifications and schema-aware validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd


SUPPORTED_CHART_TYPES = frozenset(
    {
        "distribution",
        "histogram",
        "bar",
        "line",
        "scatter",
        "box",
        "heatmap",
        "missingness",
        "correlation",
        "association",
        "treatment_outcome",
        "covariate_balance",
        "overlap",
        "iv_first_stage",
        "edge_stability",
        "dag",
        "network",
        "panel_trend",
        "subgroup_effects",
        "residual_diagnostics",
        "calibration",
        "roc",
        "pr",
        "feature_importance",
        "gate_dashboard",
        "evidence_matrix",
    }
)

FILTER_OPERATORS = frozenset(
    {"eq", "ne", "lt", "le", "gt", "ge", "in", "not_in", "isna", "notna"}
)


class ChartSpecError(ValueError):
    """Raised when a chart specification is unsafe or incompatible."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("Invalid ChartSpec: " + "; ".join(self.errors))


@dataclass(frozen=True)
class ChartFilter:
    column: str
    operator: str = "eq"
    value: Any = None

    def __post_init__(self) -> None:
        if self.operator not in FILTER_OPERATORS:
            raise ValueError(f"unsupported chart filter operator {self.operator!r}")

    def to_dict(self, *, redact_value: bool = False) -> dict[str, Any]:
        return {
            "column": self.column,
            "operator": self.operator,
            "value": "[REDACTED]" if redact_value and self.value is not None else self.value,
        }


@dataclass(frozen=True)
class ChartAnnotation:
    text: str
    x: Any = None
    y: Any = None
    kind: str = "note"


@dataclass
class AccessibilitySpec:
    alt_text: str = ""
    palette: str = "colorblind_safe"
    show_labels: bool = True
    long_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChartSpec:
    """Backend-neutral visualization contract.

    ``provenance`` should describe where analytical inputs came from.  It must
    not be used to claim that a visual association is causally identified.
    """

    id: str
    type: str
    title: str
    x: Optional[str] = None
    y: Optional[str | list[str]] = None
    color: Optional[str] = None
    facet: Optional[str] = None
    aggregation: Optional[str] = None
    filters: list[ChartFilter] = field(default_factory=list)
    annotations: list[ChartAnnotation] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    accessibility: AccessibilitySpec = field(default_factory=AccessibilitySpec)
    max_rows: int = 5_000
    max_cardinality: int = 50
    deterministic_sample: bool = True
    random_state: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = "AutoCausalChartSpec.v1"

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        self.type = str(self.type).strip().lower()
        self.title = str(self.title).strip()
        self.max_rows = max(1, int(self.max_rows))
        self.max_cardinality = max(2, int(self.max_cardinality))
        self.random_state = int(self.random_state)
        self.filters = [
            item if isinstance(item, ChartFilter) else ChartFilter(**dict(item))
            for item in self.filters
        ]
        self.annotations = [
            item
            if isinstance(item, ChartAnnotation)
            else ChartAnnotation(**dict(item))
            for item in self.annotations
        ]
        if isinstance(self.accessibility, Mapping):
            self.accessibility = AccessibilitySpec(**dict(self.accessibility))
        if not self.id:
            raise ValueError("ChartSpec.id cannot be empty")
        if not self.title:
            raise ValueError("ChartSpec.title cannot be empty")
        if self.type not in SUPPORTED_CHART_TYPES:
            raise ValueError(f"unsupported chart type {self.type!r}")
        if not self.accessibility.alt_text:
            axes = ", ".join(self.referenced_columns())
            self.accessibility.alt_text = (
                f"{self.title}. Descriptive {self.type} chart"
                + (f" using {axes}" if axes else "")
                + "; it does not establish causality."
            )

    @property
    def y_columns(self) -> list[str]:
        if self.y is None:
            return []
        if isinstance(self.y, str):
            return [self.y]
        return [str(value) for value in self.y]

    def referenced_columns(self) -> list[str]:
        values: list[str] = []
        for value in (self.x, *self.y_columns, self.color, self.facet):
            if value is not None and str(value) not in values:
                values.append(str(value))
        for item in self.filters:
            if item.column not in values:
                values.append(item.column)
        return values

    def validate(
        self,
        frame: pd.DataFrame,
        *,
        production: bool = False,
        raise_on_error: bool = True,
    ) -> list[str]:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("ChartSpec.validate frame must be a pandas DataFrame")
        errors: list[str] = []
        columns = {str(column) for column in frame.columns}
        missing = [
            column for column in self.referenced_columns() if column not in columns
        ]
        if missing:
            errors.append(f"unknown columns: {missing}")
        if len(frame) > self.max_rows and not self.deterministic_sample:
            errors.append(
                f"frame has {len(frame)} rows, above max_rows={self.max_rows}, "
                "and deterministic sampling is disabled"
            )
        for column in (self.x, self.color, self.facet):
            if column is None or column not in columns:
                continue
            series = frame[column]
            if (
                not pd.api.types.is_numeric_dtype(series)
                and int(series.nunique(dropna=True)) > self.max_cardinality
            ):
                errors.append(
                    f"column {column!r} cardinality exceeds "
                    f"max_cardinality={self.max_cardinality}"
                )
        if self.type in ("scatter", "iv_first_stage") and (
            self.x is None or len(self.y_columns) != 1
        ):
            errors.append(f"{self.type} requires one x and one y column")
        if self.type == "treatment_outcome" and (
            self.x is None or len(self.y_columns) != 1
        ):
            errors.append("treatment_outcome requires treatment x and outcome y")
        if self.type == "panel_trend" and (
            self.x is None or len(self.y_columns) < 1
        ):
            errors.append("panel_trend requires time x and at least one y")
        if self.type in ("calibration", "roc", "pr") and len(self.y_columns) > 2:
            errors.append(f"{self.type} supports at most two y columns")
        if production and self.metadata.get("allow_raw_values") is True:
            errors.append(
                "production rendering requires an explicit renderer-level "
                "allow_raw_values override, not ChartSpec metadata"
            )
        if errors and raise_on_error:
            raise ChartSpecError(errors)
        return errors

    def to_dict(
        self,
        *,
        redact_filter_values: bool = False,
        redact_annotations: bool = False,
    ) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "x": self.x,
            "y": self.y,
            "color": self.color,
            "facet": self.facet,
            "aggregation": self.aggregation,
            "filters": [
                item.to_dict(redact_value=redact_filter_values)
                for item in self.filters
            ],
            "annotations": [
                (
                    {
                        "text": "[REDACTED_ANNOTATION]",
                        "x": None,
                        "y": None,
                        "kind": item.kind,
                    }
                    if redact_annotations
                    else asdict(item)
                )
                for item in self.annotations
            ],
            "provenance": dict(self.provenance),
            "accessibility": self.accessibility.to_dict(),
            "max_rows": self.max_rows,
            "max_cardinality": self.max_cardinality,
            "deterministic_sample": self.deterministic_sample,
            "random_state": self.random_state,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChartSpec":
        payload = dict(value)
        payload.pop("schema", None)
        return cls(**payload)

    def to_report_chart_spec(
        self,
        *,
        alt_text: Optional[str] = None,
        image_path: str = "",
        source_fact_ids: Optional[Sequence[str]] = None,
        priority: int = 50,
        caption: str = "",
        category: str = "visualizations",
    ) -> Any:
        """Adapt this render contract into a reporting :class:`ChartSpec`.

        Autochart specs drive rendering; reporting specs bind charts into
        provenance-validated PDF/HTML documents. They are intentionally layered.
        """
        from autocausal.reporting.models import ChartSpec as ReportChartSpec

        resolved_alt = (
            alt_text
            or self.accessibility.alt_text
            or self.accessibility.long_description
            or self.title
        )
        return ReportChartSpec(
            id=self.id,
            chart_type=self.type,
            title=self.title,
            alt_text=str(resolved_alt),
            source_fact_ids=list(source_fact_ids or []),
            image_path=image_path,
            spec=self.to_dict(),
            provenance_ids=[
                str(v)
                for k, v in dict(self.provenance).items()
                if k.endswith("_id") or k == "id"
            ],
            priority=int(priority),
            caption=caption or self.title,
            category=category,
        )

    @classmethod
    def from_recommendation(cls, recommendation: Any) -> "ChartSpec":
        value = (
            recommendation.to_dict()
            if hasattr(recommendation, "to_dict")
            else dict(recommendation)
        )
        hints = dict(value.get("spec_hints") or {})
        required = list(value.get("required_columns") or [])
        x = hints.get("x")
        y = hints.get("y")
        if x is None and required and value.get("chart_type") == "distribution":
            x = required[0]
        return cls(
            id=str(value["id"]),
            type=str(value["chart_type"]),
            title=str(value["title"]),
            x=x,
            y=y,
            color=hints.get("color"),
            facet=hints.get("facet"),
            aggregation=hints.get("aggregation"),
            provenance={
                "planner_source": value.get("source", "rule"),
                "rationale": value.get("rationale"),
            },
            metadata={
                "data_requirements": dict(value.get("data_requirements") or {}),
                "required_columns": required,
                "caveats": list(value.get("caveats") or []),
            },
        )


__all__ = [
    "AccessibilitySpec",
    "ChartAnnotation",
    "ChartFilter",
    "ChartSpec",
    "ChartSpecError",
    "FILTER_OPERATORS",
    "SUPPORTED_CHART_TYPES",
]
