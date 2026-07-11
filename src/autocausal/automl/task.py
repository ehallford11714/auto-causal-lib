"""Tabular prediction task inference with production-safe defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import pandas as pd


TaskType = Literal["regression", "binary_classification", "multiclass_classification"]


@dataclass
class TaskSpec:
    target: str
    task_type: TaskType
    n_rows: int
    n_labeled: int
    n_classes: Optional[int] = None
    class_labels: list[str] = field(default_factory=list)
    class_distribution: dict[str, float] = field(default_factory=dict)
    inferred: bool = True
    notes: list[str] = field(default_factory=list)
    schema: str = "AutoCausalTaskSpec.v1"

    @property
    def is_classification(self) -> bool:
        return self.task_type in (
            "binary_classification",
            "multiclass_classification",
        )

    @property
    def is_binary(self) -> bool:
        return self.task_type == "binary_classification"

    def to_dict(self, *, redact_labels: bool = False) -> dict[str, Any]:
        if redact_labels:
            labels = [f"class_{index + 1}" for index in range(len(self.class_labels))]
            distribution = {
                f"class_{index + 1}": fraction
                for index, fraction in enumerate(self.class_distribution.values())
            }
        else:
            labels = list(self.class_labels)
            distribution = dict(self.class_distribution)
        return {
            "schema": self.schema,
            "target": self.target,
            "task_type": self.task_type,
            "n_rows": self.n_rows,
            "n_labeled": self.n_labeled,
            "n_classes": self.n_classes,
            "class_labels": labels,
            "class_distribution": distribution,
            "inferred": self.inferred,
            "notes": list(self.notes),
        }


def infer_task(
    frame: pd.DataFrame,
    *,
    target: Optional[str],
    task: Optional[TaskType | str] = None,
    production: bool = False,
    max_classes: int = 50,
) -> TaskSpec:
    """Infer regression/binary/multiclass prediction semantics.

    Production mode intentionally refuses target inference.  Selecting a target
    is a modeling decision and must be explicit at that boundary.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("infer_task frame must be a pandas DataFrame")
    if target is None:
        if production:
            raise ValueError("production AutoTabularML requires an explicit target")
        if len(frame.columns) < 2:
            raise ValueError("task inference requires at least two columns")
        target = str(frame.columns[-1])
        target_inferred = True
    else:
        target = str(target)
        target_inferred = False
    if target not in frame.columns:
        raise KeyError(f"target column {target!r} is not in the frame")
    series = frame[target]
    labeled = series.dropna()
    if len(labeled) < 2:
        raise ValueError("target requires at least two non-missing observations")
    unique = int(labeled.nunique(dropna=True))
    allowed: set[str] = {
        "regression",
        "binary_classification",
        "multiclass_classification",
    }
    aliases = {
        "binary": "binary_classification",
        "classification": (
            "binary_classification" if unique == 2 else "multiclass_classification"
        ),
        "multiclass": "multiclass_classification",
    }
    inferred = task is None
    if task is not None:
        resolved = aliases.get(str(task).lower(), str(task).lower())
        if resolved not in allowed:
            raise ValueError(f"unsupported task {task!r}")
        task_type: TaskType = resolved  # type: ignore[assignment]
    elif unique == 2:
        task_type = "binary_classification"
    elif (
        pd.api.types.is_bool_dtype(labeled)
        or isinstance(labeled.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(labeled)
        or pd.api.types.is_string_dtype(labeled)
    ):
        if unique > max_classes:
            raise ValueError(
                f"target {target!r} has {unique} string/category values; "
                "specify task='regression' only if numeric conversion is intended"
            )
        task_type = "multiclass_classification"
    elif (
        pd.api.types.is_integer_dtype(labeled)
        and unique <= max_classes
        and unique / max(len(labeled), 1) <= 0.20
    ):
        task_type = "multiclass_classification"
    else:
        task_type = "regression"

    if task_type == "binary_classification" and unique != 2:
        raise ValueError(
            f"binary classification requires exactly two classes, observed {unique}"
        )
    if task_type == "multiclass_classification" and unique < 3:
        raise ValueError(
            f"multiclass classification requires at least three classes, observed {unique}"
        )
    if task_type == "regression":
        converted = pd.to_numeric(labeled, errors="coerce")
        if converted.notna().sum() != len(labeled):
            raise ValueError("regression target must be numeric")

    labels: list[str] = []
    distribution: dict[str, float] = {}
    notes: list[str] = []
    if task_type != "regression":
        counts = labeled.astype(str).value_counts(dropna=False)
        labels = [str(value) for value in counts.index]
        distribution = {
            str(label): round(float(count) / len(labeled), 8)
            for label, count in counts.items()
        }
        minimum_fraction = min(distribution.values())
        if minimum_fraction < 0.10:
            notes.append(
                f"Class imbalance: minority fraction={minimum_fraction:.3f}; "
                "prefer balanced metrics and inspect subgroup errors."
            )
    if target_inferred:
        notes.append(
            f"Exploratory target inference selected the final column {target!r}; "
            "pass target= explicitly for reviewed runs."
        )
    if series.isna().any():
        notes.append(
            f"Dropped {int(series.isna().sum())} rows with missing target before CV."
        )
    return TaskSpec(
        target=target,
        task_type=task_type,
        n_rows=int(len(frame)),
        n_labeled=int(len(labeled)),
        n_classes=unique if task_type != "regression" else None,
        class_labels=labels,
        class_distribution=distribution,
        inferred=inferred or target_inferred,
        notes=notes,
    )


__all__ = ["TaskSpec", "TaskType", "infer_task"]
