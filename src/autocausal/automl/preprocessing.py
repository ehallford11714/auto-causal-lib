"""Leakage-safe sklearn preprocessing assembled but never pre-fitted."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from autocausal.roles import ColumnRole, infer_column_roles


@dataclass
class FeatureSchema:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    schema: str = "AutoCausalFeatureSchema.v1"

    @property
    def features(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.datetime]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "datetime": list(self.datetime),
            "excluded": list(self.excluded),
            "features": self.features,
            "notes": list(self.notes),
        }


def _datetime_matrix(values: Any) -> np.ndarray:
    frame = pd.DataFrame(values)
    pieces: list[np.ndarray] = []
    for column in frame.columns:
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
        pieces.extend(
            [
                parsed.dt.year.to_numpy(dtype=float),
                parsed.dt.month.to_numpy(dtype=float),
                parsed.dt.day.to_numpy(dtype=float),
                parsed.dt.dayofweek.to_numpy(dtype=float),
            ]
        )
    if not pieces:
        return np.empty((len(frame), 0), dtype=float)
    return np.column_stack(pieces)


def infer_feature_schema(
    frame: pd.DataFrame,
    *,
    target: str,
    exclude_columns: Optional[Sequence[str]] = None,
    max_categories: int = 50,
    include_text: bool = False,
) -> FeatureSchema:
    if target not in frame:
        raise KeyError(f"target column {target!r} is not in the frame")
    excluded_requested = {target, *(str(value) for value in exclude_columns or [])}
    roles = infer_column_roles(frame, max_cat_cardinality=max_categories)
    schema = FeatureSchema()
    for raw_column, role in roles.items():
        column = str(raw_column)
        if column in excluded_requested:
            schema.excluded.append(column)
            continue
        if role == ColumnRole.NUMERIC:
            schema.numeric.append(column)
        elif role in (ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN):
            cardinality = int(frame[column].nunique(dropna=True))
            if cardinality <= max_categories:
                schema.categorical.append(column)
            else:
                schema.excluded.append(column)
                schema.notes.append(
                    f"Excluded high-cardinality categorical feature {column!r} "
                    f"({cardinality} levels)."
                )
        elif role == ColumnRole.DATETIME:
            schema.datetime.append(column)
        elif role == ColumnRole.TEXT and include_text:
            cardinality = int(frame[column].nunique(dropna=True))
            if cardinality <= max_categories:
                schema.categorical.append(column)
            else:
                schema.excluded.append(column)
                schema.notes.append(
                    f"Excluded text feature {column!r}; use AutoNLP fold-safe "
                    "features explicitly rather than one-hot encoding raw text."
                )
        else:
            schema.excluded.append(column)
            if role == ColumnRole.ID:
                schema.notes.append(f"Excluded ID-like feature {column!r}.")
            elif role == ColumnRole.TEXT:
                schema.notes.append(
                    f"Excluded raw text feature {column!r}; use AutoNLP features."
                )
    if not schema.features:
        raise ValueError("no usable predictive features remain after exclusions")
    return schema


def build_preprocessor(schema: FeatureSchema) -> Any:
    """Return an unfitted ``ColumnTransformer`` for use inside each CV pipeline."""

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    transformers: list[tuple[str, Any, list[str]]] = []
    if schema.numeric:
        numeric = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric, list(schema.numeric)))
    if schema.categorical:
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=np.float64,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical, list(schema.categorical)))
    if schema.datetime:
        datetime_pipeline = Pipeline(
            [
                (
                    "components",
                    FunctionTransformer(
                        _datetime_matrix,
                        validate=False,
                        feature_names_out=None,
                    ),
                ),
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("datetime", datetime_pipeline, list(schema.datetime)))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def build_pipeline(schema: FeatureSchema, estimator: Any) -> Any:
    """Return an unfitted pipeline; preprocessing is fitted within each fold."""

    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("preprocess", build_preprocessor(schema)),
            ("model", estimator),
        ]
    )


__all__ = [
    "FeatureSchema",
    "build_pipeline",
    "build_preprocessor",
    "infer_feature_schema",
]
