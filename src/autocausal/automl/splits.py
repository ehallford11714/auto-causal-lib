"""Deterministic random, stratified, group-aware, and time-aware CV splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from autocausal.automl.task import TaskSpec


SplitStrategy = Literal["auto", "random", "random_stratified", "group", "time"]


@dataclass
class SplitPlan:
    strategy: str
    splits: list[tuple[np.ndarray, np.ndarray]]
    n_rows: int
    random_state: int = 0
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    schema: str = "AutoCausalSplitPlan.v1"

    def validate(
        self,
        *,
        groups: Optional[pd.Series] = None,
        time_values: Optional[pd.Series] = None,
    ) -> None:
        if len(self.splits) < 2:
            raise ValueError("cross-validation requires at least two folds")
        for fold, (train, test) in enumerate(self.splits):
            train_set = set(int(value) for value in train)
            test_set = set(int(value) for value in test)
            if not train_set or not test_set:
                raise ValueError(f"fold {fold} has an empty train or validation set")
            if train_set & test_set:
                raise ValueError(f"fold {fold} leaks rows across train and validation")
            if min(train_set | test_set) < 0 or max(train_set | test_set) >= self.n_rows:
                raise ValueError(f"fold {fold} contains out-of-range row indices")
            if self.strategy == "group" and groups is not None:
                train_groups = set(groups.iloc[list(train)].dropna().astype(str))
                test_groups = set(groups.iloc[list(test)].dropna().astype(str))
                if train_groups & test_groups:
                    raise ValueError(f"fold {fold} leaks groups across validation")
            if self.strategy == "time" and time_values is not None:
                train_time = pd.to_datetime(
                    time_values.iloc[list(train)], errors="coerce"
                ).dropna()
                test_time = pd.to_datetime(
                    time_values.iloc[list(test)], errors="coerce"
                ).dropna()
                if (
                    not train_time.empty
                    and not test_time.empty
                    and train_time.max() >= test_time.min()
                ):
                    raise ValueError(f"fold {fold} is not forward-chaining in time")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "strategy": self.strategy,
            "n_splits": len(self.splits),
            "n_rows": self.n_rows,
            "random_state": self.random_state,
            "group_column": self.group_column,
            "time_column": self.time_column,
            "folds": [
                {
                    "fold": index,
                    "n_train": int(len(train)),
                    "n_validation": int(len(test)),
                    "overlap": 0,
                    "preprocessing_fit_scope": "fold_train_only",
                }
                for index, (train, test) in enumerate(self.splits)
            ],
            "notes": list(self.notes),
        }


def make_splits(
    frame: pd.DataFrame,
    task: TaskSpec,
    *,
    y: Optional[pd.Series] = None,
    strategy: SplitStrategy | str = "auto",
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    n_splits: int = 5,
    random_state: int = 0,
) -> SplitPlan:
    """Create and audit CV splits before any preprocessing is fitted."""

    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
    )

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("make_splits frame must be a pandas DataFrame")
    if y is None:
        y = frame[task.target]
    if len(y) != len(frame):
        raise ValueError("split target length must equal frame length")
    requested = str(strategy).lower()
    allowed = {"auto", "random", "random_stratified", "group", "time"}
    if requested not in allowed:
        raise ValueError(f"unsupported split strategy {strategy!r}")
    if group_column is not None and group_column not in frame:
        raise KeyError(f"group column {group_column!r} is not in the frame")
    if time_column is not None and time_column not in frame:
        raise KeyError(f"time column {time_column!r} is not in the frame")

    if requested == "auto":
        if time_column is not None:
            selected = "time"
        elif group_column is not None:
            selected = "group"
        elif task.is_classification:
            selected = "random_stratified"
        else:
            selected = "random"
    else:
        selected = requested
    if selected == "random" and task.is_classification:
        selected = "random_stratified"

    folds = max(2, int(n_splits))
    notes: list[str] = []
    indices = np.arange(len(frame), dtype=int)
    groups = (
        frame[group_column].astype("string").fillna("__MISSING_GROUP__")
        if group_column
        else None
    )
    times = frame[time_column] if time_column else None

    if selected == "group":
        if groups is None:
            raise ValueError("group split requires group_column")
        unique_groups = int(groups.nunique(dropna=False))
        folds = min(folds, unique_groups)
        if folds < 2:
            raise ValueError("group split requires at least two distinct groups")
        splitter = GroupKFold(n_splits=folds)
        splits = [
            (train.astype(int), test.astype(int))
            for train, test in splitter.split(indices, y, groups=groups)
        ]
        notes.append("No group appears in both train and validation within a fold.")
    elif selected == "time":
        if times is None:
            raise ValueError("time split requires time_column")
        parsed = pd.to_datetime(times, errors="coerce")
        if parsed.isna().any():
            raise ValueError("time split requires parseable, non-missing time values")
        unique_times = np.asarray(sorted(parsed.unique()))
        folds = min(folds, len(unique_times) - 1)
        if folds < 2:
            raise ValueError(
                "time split requires at least three distinct timestamps"
            )
        splitter = TimeSeriesSplit(n_splits=folds)
        splits = []
        for train_time_indices, test_time_indices in splitter.split(unique_times):
            train_times = set(unique_times[train_time_indices])
            test_times = set(unique_times[test_time_indices])
            train = np.flatnonzero(parsed.isin(train_times).to_numpy()).astype(int)
            test = np.flatnonzero(parsed.isin(test_times).to_numpy()).astype(int)
            splits.append((train, test))
        notes.append("Forward-chaining folds train only on observations before validation.")
    elif selected == "random_stratified":
        counts = y.value_counts(dropna=False)
        minimum = int(counts.min())
        folds = min(folds, minimum)
        if folds < 2:
            raise ValueError(
                "stratified CV requires at least two observations in every class"
            )
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=int(random_state)
        )
        splits = [
            (train.astype(int), test.astype(int))
            for train, test in splitter.split(indices, y)
        ]
        notes.append("Class proportions are approximately preserved in each fold.")
    else:
        folds = min(folds, len(frame))
        if folds < 2:
            raise ValueError("random CV requires at least two rows")
        splitter = KFold(
            n_splits=folds, shuffle=True, random_state=int(random_state)
        )
        splits = [
            (train.astype(int), test.astype(int))
            for train, test in splitter.split(indices)
        ]

    plan = SplitPlan(
        strategy=selected,
        splits=splits,
        n_rows=len(frame),
        random_state=int(random_state),
        group_column=group_column,
        time_column=time_column,
        notes=notes,
    )
    plan.validate(groups=groups, time_values=times)
    return plan


__all__ = ["SplitPlan", "SplitStrategy", "make_splits"]
