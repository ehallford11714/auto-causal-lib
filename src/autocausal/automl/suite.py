"""Leakage-safe, bounded AutoML for tabular predictive modeling."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Optional

import numpy as np
import pandas as pd

from autocausal.automl.candidates import CandidateSpec, default_candidates
from autocausal.automl.preprocessing import (
    FeatureSchema,
    build_pipeline,
    infer_feature_schema,
)
from autocausal.automl.report import (
    AutoMLReport,
    CandidateEvaluation,
    MetricSummary,
    PREDICTIVE_CAVEAT,
)
from autocausal.automl.splits import SplitPlan, SplitStrategy, make_splits
from autocausal.automl.task import TaskSpec, TaskType, infer_task


@dataclass
class _CandidateRun:
    spec: CandidateSpec
    evaluation: CandidateEvaluation
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    predicted_mask: np.ndarray


class AutoMLGateError(ValueError):
    """Optional fail-closed error carrying the completed predictive report."""

    def __init__(self, report: AutoMLReport) -> None:
        self.report = report
        failed = [
            gate
            for gate in report.gates
            if str(gate.get("status", "")).lower() in ("fail", "escalate")
            or gate.get("ok") is False
        ]
        super().__init__(
            "AutoTabularML policy gates failed: "
            + ", ".join(str(gate.get("id")) for gate in failed)
        )


def _summary(values: Sequence[float], *, direction: str) -> MetricSummary:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return MetricSummary(
            mean=float("nan"),
            std=float("nan"),
            ci95_low=float("nan"),
            ci95_high=float("nan"),
            values=[],
            direction=direction,
        )
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    half_width = 1.96 * std / np.sqrt(max(len(array), 1))
    return MetricSummary(
        mean=round(mean, 10),
        std=round(std, 10),
        ci95_low=round(mean - half_width, 10),
        ci95_high=round(mean + half_width, 10),
        values=[round(float(value), 10) for value in array],
        direction=direction,
    )


def _metric_values(
    task: TaskSpec,
    y_true: pd.Series,
    prediction: np.ndarray,
    probabilities: Optional[np.ndarray],
    classes: Optional[np.ndarray],
) -> dict[str, float]:
    if not task.is_classification:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        truth = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
        predicted = np.asarray(prediction, dtype=float)
        return {
            "mae": float(mean_absolute_error(truth, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
            "r2": (
                float(r2_score(truth, predicted))
                if len(np.unique(truth)) > 1
                else float("nan")
            ),
        }

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    result = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1_weighted": float(
            f1_score(y_true, prediction, average="weighted", zero_division=0)
        ),
    }
    if probabilities is not None and classes is not None:
        try:
            result["log_loss"] = float(
                log_loss(y_true, probabilities, labels=list(classes))
            )
        except Exception:
            result["log_loss"] = float("nan")
        if task.is_binary and probabilities.ndim == 2 and probabilities.shape[1] == 2:
            positive = classes[1]
            binary_truth = (np.asarray(y_true) == positive).astype(int)
            result["brier"] = float(
                brier_score_loss(binary_truth, probabilities[:, 1])
            )
            result["roc_auc"] = (
                float(roc_auc_score(binary_truth, probabilities[:, 1]))
                if len(np.unique(binary_truth)) > 1
                else float("nan")
            )
        elif probabilities.ndim == 2:
            try:
                result["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(
                        y_true,
                        probabilities,
                        labels=list(classes),
                        multi_class="ovr",
                        average="weighted",
                    )
                )
            except Exception:
                result["roc_auc_ovr_weighted"] = float("nan")
    return result


def _evaluate_candidate(
    candidate: CandidateSpec,
    *,
    task: TaskSpec,
    schema: FeatureSchema,
    features: pd.DataFrame,
    target: pd.Series,
    split_plan: SplitPlan,
) -> _CandidateRun:
    from sklearn.base import clone

    metric_folds: dict[str, list[float]] = {}
    fit_seconds: list[float] = []
    predict_seconds: list[float] = []
    predictions = np.empty(len(target), dtype=object if task.is_classification else float)
    predictions[:] = None if task.is_classification else np.nan
    probabilities: Optional[np.ndarray] = None
    predicted_mask = np.zeros(len(target), dtype=bool)
    errors: list[str] = []

    for fold, (train_indices, validation_indices) in enumerate(split_plan.splits):
        pipeline = build_pipeline(schema, clone(candidate.estimator))
        train_x = features.iloc[train_indices]
        validation_x = features.iloc[validation_indices]
        train_y = target.iloc[train_indices]
        validation_y = target.iloc[validation_indices]
        try:
            started = time.perf_counter()
            pipeline.fit(train_x, train_y)
            fit_seconds.append(time.perf_counter() - started)
            started = time.perf_counter()
            fold_prediction = np.asarray(pipeline.predict(validation_x))
            predict_seconds.append(time.perf_counter() - started)
            fold_probability: Optional[np.ndarray] = None
            classes: Optional[np.ndarray] = None
            if task.is_classification and hasattr(pipeline, "predict_proba"):
                fold_probability = np.asarray(pipeline.predict_proba(validation_x))
                classes = np.asarray(getattr(pipeline, "classes_", []))
                if probabilities is None and classes.size:
                    probabilities = np.full(
                        (len(target), len(classes)), np.nan, dtype=float
                    )
                if (
                    probabilities is not None
                    and fold_probability.shape[1] == probabilities.shape[1]
                ):
                    probabilities[validation_indices, :] = fold_probability
            fold_metrics = _metric_values(
                task,
                validation_y,
                fold_prediction,
                fold_probability,
                classes,
            )
            for name, value in fold_metrics.items():
                metric_folds.setdefault(name, []).append(float(value))
            predictions[validation_indices] = fold_prediction
            predicted_mask[validation_indices] = True
        except Exception as exc:
            errors.append(f"fold {fold}: {type(exc).__name__}: {exc}")

    directions = {
        "mae": "lower_is_better",
        "rmse": "lower_is_better",
        "log_loss": "lower_is_better",
        "brier": "lower_is_better",
    }
    metrics = {
        name: _summary(
            values,
            direction=directions.get(name, "higher_is_better"),
        )
        for name, values in metric_folds.items()
    }
    evaluation = CandidateEvaluation(
        name=candidate.name,
        family=candidate.family,
        complexity=candidate.complexity,
        expected_latency=candidate.expected_latency,
        metrics=metrics,
        fit_seconds_mean=round(float(np.mean(fit_seconds)), 8) if fit_seconds else 0.0,
        predict_seconds_mean=(
            round(float(np.mean(predict_seconds)), 8) if predict_seconds else 0.0
        ),
        errors=errors,
        notes=list(candidate.notes),
    )
    return _CandidateRun(
        spec=candidate,
        evaluation=evaluation,
        predictions=predictions,
        probabilities=probabilities,
        predicted_mask=predicted_mask,
    )


def _rank_candidates(runs: list[_CandidateRun], task: TaskSpec) -> list[_CandidateRun]:
    primary = "balanced_accuracy" if task.is_classification else "rmse"
    valid: list[tuple[_CandidateRun, float]] = []
    for run in runs:
        metric = run.evaluation.metrics.get(primary)
        if metric is None or not np.isfinite(metric.mean):
            continue
        performance = metric.mean if task.is_classification else -metric.mean
        valid.append((run, float(performance)))
    if not valid:
        raise RuntimeError("all candidate models failed during cross-validation")
    values = np.asarray([value for _, value in valid], dtype=float)
    low, high = float(values.min()), float(values.max())
    latency_penalty = {"very_low": 0.0, "low": 0.01, "medium": 0.02, "high": 0.04}
    for run, performance in valid:
        normalized = 0.5 if high == low else (performance - low) / (high - low)
        run.evaluation.ranking_score = round(
            normalized
            - 0.02 * run.evaluation.complexity
            - latency_penalty.get(run.evaluation.expected_latency, 0.03),
            8,
        )
    valid_ids = {id(value[0]) for value in valid}
    failed = [run for run in runs if id(run) not in valid_ids]
    ordered = sorted(
        [run for run, _ in valid],
        key=lambda run: (
            -run.evaluation.ranking_score,
            run.evaluation.complexity,
            run.spec.name,
        ),
    ) + failed
    for index, run in enumerate(ordered, start=1):
        run.evaluation.rank = index
        run.evaluation.selected = index == 1
        if index == 1:
            run.evaluation.notes.append(
                "Selected by CV performance with deterministic complexity/latency penalties."
            )
    return ordered


def _subgroup_metrics(
    run: _CandidateRun,
    *,
    task: TaskSpec,
    frame: pd.DataFrame,
    target: pd.Series,
    columns: Sequence[str],
    production: bool,
) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, mean_squared_error

    output: dict[str, Any] = {}
    for column in columns:
        if column not in frame:
            continue
        values = frame[column]
        if values.nunique(dropna=False) > 30:
            output[column] = {
                "status": "skipped",
                "reason": "cardinality_above_30",
            }
            continue
        rows: list[dict[str, Any]] = []
        for index, (group, indices) in enumerate(
            values.groupby(values, dropna=False).groups.items()
        ):
            positions = np.asarray(list(indices), dtype=int)
            positions = positions[run.predicted_mask[positions]]
            if len(positions) < 5:
                continue
            truth = target.iloc[positions]
            prediction = run.predictions[positions]
            if task.is_classification:
                score = float(balanced_accuracy_score(truth, prediction))
                metric_name = "balanced_accuracy"
            else:
                score = float(
                    np.sqrt(
                        mean_squared_error(
                            pd.to_numeric(truth, errors="coerce"),
                            np.asarray(prediction, dtype=float),
                        )
                    )
                )
                metric_name = "rmse"
            rows.append(
                {
                    "group": (
                        f"group_{index + 1}" if production else str(group)
                    ),
                    "n": int(len(positions)),
                    metric_name: round(score, 10),
                }
            )
        scores = [
            float(row["balanced_accuracy" if task.is_classification else "rmse"])
            for row in rows
        ]
        output[column] = {
            "status": "ok" if rows else "insufficient_data",
            "metric": (
                "balanced_accuracy" if task.is_classification else "rmse"
            ),
            "groups": rows,
            "gap": round(max(scores) - min(scores), 10) if len(scores) >= 2 else None,
        }
    return output


def _permutation_importance(
    candidate: CandidateSpec,
    *,
    task: TaskSpec,
    schema: FeatureSchema,
    features: pd.DataFrame,
    target: pd.Series,
    split_plan: SplitPlan,
    random_state: int,
) -> list[dict[str, Any]]:
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance

    train_indices, validation_indices = split_plan.splits[0]
    pipeline = build_pipeline(schema, clone(candidate.estimator))
    pipeline.fit(features.iloc[train_indices], target.iloc[train_indices])
    scoring = "balanced_accuracy" if task.is_classification else "neg_root_mean_squared_error"
    result = permutation_importance(
        pipeline,
        features.iloc[validation_indices],
        target.iloc[validation_indices],
        scoring=scoring,
        n_repeats=3,
        random_state=random_state,
        n_jobs=1,
    )
    rows = [
        {
            "feature": feature,
            "importance_mean": round(float(mean), 10),
            "importance_std": round(float(std), 10),
            "scoring": scoring,
            "evaluation_scope": "held_out_first_cv_fold",
        }
        for feature, mean, std in zip(
            features.columns, result.importances_mean, result.importances_std
        )
    ]
    return sorted(rows, key=lambda item: -abs(item["importance_mean"]))


def _as_gate(value: Any) -> Optional[dict[str, Any]]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        return None
    gate = dict(value)
    if "status" not in gate:
        gate["status"] = "pass" if gate.get("ok") is not False else "fail"
    if "ok" not in gate:
        gate["ok"] = gate["status"] not in ("fail", "escalate")
    return gate


def _policy_values(inputs: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        "min_sample_size": 30,
        "max_cv_coefficient_variation": 0.50,
        "max_brier_score": 0.25,
        "min_class_fraction": 0.05,
        "max_subgroup_gap": 0.25,
        "max_rows": 100_000,
        "max_columns": 200,
        "max_candidates": 4,
    }
    values.update(
        {
            key: value
            for key, value in inputs.items()
            if key in values and value is not None
        }
    )
    policy = inputs.get("policy")
    risk = (
        getattr(policy, "automl_risk", None)
        or getattr(policy, "automl", None)
        or (policy.get("automl_risk") if isinstance(policy, Mapping) else None)
    )
    validity = (
        getattr(policy, "statistical_validity", None)
        or (policy.get("statistical_validity") if isinstance(policy, Mapping) else None)
    )
    operational = (
        getattr(policy, "operational", None)
        or (policy.get("operational") if isinstance(policy, Mapping) else None)
    )
    for source, mappings in (
        (
            risk,
            {
                "max_cv_coefficient_variation": "max_cv_coefficient_variation",
                "max_brier_score": "max_brier_score",
                "min_class_fraction": "min_class_fraction",
            },
        ),
        (validity, {"min_sample_size": "min_sample_size"}),
        (
            operational,
            {"max_rows": "max_rows", "max_columns": "max_columns"},
        ),
    ):
        for source_name, target_name in mappings.items():
            value = (
                source.get(source_name)
                if isinstance(source, Mapping)
                else getattr(source, source_name, None)
                if source is not None
                else None
            )
            if value is not None:
                values[target_name] = value
    return values


def _build_gates(
    *,
    task: TaskSpec,
    schema: FeatureSchema,
    selected: _CandidateRun,
    subgroup: Mapping[str, Any],
    n_rows: int,
    n_columns: int,
    candidate_count: int,
    gate_inputs: Mapping[str, Any],
    external_results: Any,
) -> list[dict[str, Any]]:
    thresholds = _policy_values(gate_inputs)
    gates: list[dict[str, Any]] = []

    def add(
        identifier: str,
        ok: Optional[bool],
        detail: str,
        *,
        metric: Any = None,
        threshold: Any = None,
        status: Optional[str] = None,
    ) -> None:
        gates.append(
            {
                "id": identifier,
                "ok": ok,
                "status": status or ("pass" if ok else "fail"),
                "detail": detail,
                "metric": metric,
                "threshold": threshold,
                "stage": "automl",
            }
        )

    minimum_rows = int(thresholds["min_sample_size"])
    add(
        "automl_sample_size",
        n_rows >= minimum_rows,
        f"n_labeled={n_rows}; minimum={minimum_rows}",
        metric=n_rows,
        threshold=minimum_rows,
    )
    suspicious = list(gate_inputs.get("leakage_columns") or [])
    target_key = task.target.lower().replace("_", "")
    for feature in schema.features:
        key = feature.lower().replace("_", "")
        if key != target_key and (
            key.startswith(target_key + "future")
            or key.startswith(target_key + "prediction")
            or key.startswith("post" + target_key)
        ):
            suspicious.append(feature)
    suspicious = sorted(set(str(value) for value in suspicious))
    add(
        "automl_leakage_scan",
        not suspicious,
        (
            "No explicit/name-based leakage features detected."
            if not suspicious
            else "Potential leakage features: " + ", ".join(suspicious)
        ),
        metric=len(suspicious),
        threshold=0,
    )
    primary_name = "balanced_accuracy" if task.is_classification else "rmse"
    primary = selected.evaluation.metrics.get(primary_name)
    variation = (
        abs(primary.std / primary.mean)
        if primary is not None and np.isfinite(primary.mean) and primary.mean != 0
        else float("inf")
    )
    maximum_variation = float(thresholds["max_cv_coefficient_variation"])
    add(
        "automl_cv_instability",
        variation <= maximum_variation,
        f"{primary_name} coefficient_of_variation={variation:.4g}",
        metric=variation,
        threshold=maximum_variation,
    )
    if task.is_classification:
        brier = selected.evaluation.metrics.get("brier")
        if task.is_binary and brier is not None:
            maximum_brier = float(thresholds["max_brier_score"])
            add(
                "automl_calibration",
                brier.mean <= maximum_brier,
                f"mean Brier score={brier.mean:.4g}",
                metric=brier.mean,
                threshold=maximum_brier,
            )
        else:
            add(
                "automl_calibration",
                None,
                "Brier calibration gate applies to binary probability models.",
                status="skip",
            )
        minimum_fraction = (
            min(task.class_distribution.values())
            if task.class_distribution
            else 0.0
        )
        threshold = float(thresholds["min_class_fraction"])
        add(
            "automl_class_imbalance",
            minimum_fraction >= threshold,
            f"minimum class fraction={minimum_fraction:.4g}",
            metric=minimum_fraction,
            threshold=threshold,
        )
    gaps = [
        float(value["gap"])
        for value in subgroup.values()
        if isinstance(value, Mapping) and value.get("gap") is not None
    ]
    if gaps:
        maximum_gap = float(thresholds["max_subgroup_gap"])
        worst_gap = max(gaps)
        add(
            "automl_subgroup_performance",
            worst_gap <= maximum_gap,
            f"maximum subgroup metric gap={worst_gap:.4g}",
            metric=worst_gap,
            threshold=maximum_gap,
        )
    else:
        add(
            "automl_subgroup_performance",
            None,
            "No eligible subgroup comparison was requested or available.",
            status="skip",
        )
    resource_ok = (
        n_rows <= int(thresholds["max_rows"])
        and n_columns <= int(thresholds["max_columns"])
        and candidate_count <= int(thresholds["max_candidates"])
    )
    add(
        "automl_resource_limits",
        resource_ok,
        (
            f"rows={n_rows}/{int(thresholds['max_rows'])}, "
            f"columns={n_columns}/{int(thresholds['max_columns'])}, "
            f"candidates={candidate_count}/{int(thresholds['max_candidates'])}"
        ),
    )
    values = external_results
    if hasattr(values, "results"):
        values = getattr(values, "results")
    if isinstance(values, Mapping):
        values = values.get("results") or values.get("gates") or [values]
    for value in values or []:
        gate = _as_gate(value)
        if gate is not None:
            gate.setdefault("stage", "external")
            gates.append(gate)
    return gates


class AutoTabularML:
    """Run bounded, deterministic model selection with fold-local preprocessing."""

    def __init__(
        self,
        source: Any = None,
        *,
        policy: Any = None,
        mode: Optional[str] = None,
        random_state: Optional[int] = None,
        cv: Optional[int] = None,
        max_candidates: int = 4,
    ) -> None:
        self.source = source
        self.policy = policy or getattr(source, "policy", None)
        profile = str(getattr(self.policy, "profile", "") or "").lower()
        self.mode = mode or (
            "production" if profile in ("production", "review") else None
        )
        policy_random_state = getattr(self.policy, "random_state", 0)
        self.random_state = int(
            policy_random_state if random_state is None else random_state
        )
        risk = getattr(self.policy, "automl_risk", None)
        policy_folds = getattr(risk, "cv_folds", 5)
        self.cv = max(2, int(policy_folds if cv is None else cv))
        self.max_candidates = max(1, int(max_candidates))
        self.last_report: Optional[AutoMLReport] = None

    def _resolve_frame(self, source: Any) -> tuple[pd.DataFrame, str]:
        active = source if source is not None else self.source
        if active is None:
            raise ValueError("AutoTabularML.run requires a DataFrame or source")
        if isinstance(active, pd.DataFrame):
            return active.copy(), str(self.mode or "exploratory")
        if hasattr(active, "df") and isinstance(active.df, pd.DataFrame):
            return active.df.copy(), str(
                self.mode or getattr(active, "mode", "exploratory")
            )
        from autocausal.suites.base import resolve_frame

        frame, _, ac = resolve_frame(active)
        return frame, str(self.mode or getattr(ac, "mode", "exploratory"))

    def run(
        self,
        frame: Any = None,
        *,
        target: Optional[str] = None,
        task: Optional[TaskType | str] = None,
        split_strategy: SplitStrategy | str = "auto",
        group_column: Optional[str] = None,
        time_column: Optional[str] = None,
        feature_columns: Optional[Sequence[str]] = None,
        exclude_columns: Optional[Sequence[str]] = None,
        candidates: Optional[Sequence[str]] = None,
        subgroup_columns: Optional[Sequence[str]] = None,
        calibrate: bool = False,
        compute_importance: bool = True,
        gate_inputs: Optional[Mapping[str, Any]] = None,
        gate_results: Any = None,
        enforce_gates: bool = False,
    ) -> AutoMLReport:
        source_frame, resolved_mode = self._resolve_frame(frame)
        resolved_mode = resolved_mode.lower()
        if resolved_mode not in ("exploratory", "production"):
            raise ValueError("mode must be 'exploratory' or 'production'")
        production = resolved_mode == "production"
        inputs = dict(gate_inputs or {})
        if self.policy is not None:
            inputs.setdefault("policy", self.policy)
        thresholds = _policy_values(inputs)
        if len(source_frame) > int(thresholds["max_rows"]):
            raise ValueError(
                f"AutoTabularML row limit exceeded: {len(source_frame)} > "
                f"{int(thresholds['max_rows'])}"
            )
        if len(source_frame.columns) > int(thresholds["max_columns"]):
            raise ValueError(
                f"AutoTabularML column limit exceeded: {len(source_frame.columns)} > "
                f"{int(thresholds['max_columns'])}"
            )
        task_spec = infer_task(
            source_frame,
            target=target,
            task=task,
            production=production,
        )
        labeled = source_frame.loc[source_frame[task_spec.target].notna()].copy()
        labeled = labeled.reset_index(drop=True)
        if task_spec.is_classification:
            labeled[task_spec.target] = labeled[task_spec.target].astype(str)
        else:
            labeled[task_spec.target] = pd.to_numeric(
                labeled[task_spec.target], errors="raise"
            )

        if feature_columns is not None:
            requested = [str(column) for column in feature_columns]
            unknown = [column for column in requested if column not in labeled]
            if unknown:
                raise KeyError(f"unknown feature columns: {unknown}")
            keep = list(
                dict.fromkeys(
                    [
                        *requested,
                        task_spec.target,
                        *([group_column] if group_column else []),
                        *([time_column] if time_column else []),
                        *(subgroup_columns or []),
                    ]
                )
            )
            labeled = labeled[keep]
        split_exclusions = [
            value
            for value in (group_column, time_column)
            if value is not None
        ]
        schema = infer_feature_schema(
            labeled,
            target=task_spec.target,
            exclude_columns=[
                *(exclude_columns or []),
                *split_exclusions,
                *(subgroup_columns or []),
            ],
        )
        features = labeled[schema.features].copy()
        target_series = labeled[task_spec.target].copy()
        split_plan = make_splits(
            labeled,
            task_spec,
            y=target_series,
            strategy=split_strategy,
            group_column=group_column,
            time_column=time_column,
            n_splits=self.cv,
            random_state=self.random_state,
        )
        maximum_candidates = min(
            self.max_candidates, int(thresholds["max_candidates"])
        )
        candidate_specs = default_candidates(
            task_spec,
            random_state=self.random_state,
            names=candidates,
            max_candidates=maximum_candidates,
        )
        runs = [
            _evaluate_candidate(
                candidate,
                task=task_spec,
                schema=schema,
                features=features,
                target=target_series,
                split_plan=split_plan,
            )
            for candidate in candidate_specs
        ]
        ordered = _rank_candidates(runs, task_spec)
        selected = ordered[0]

        subgroup = _subgroup_metrics(
            selected,
            task=task_spec,
            frame=labeled,
            target=target_series,
            columns=list(subgroup_columns or []),
            production=production,
        )
        importance: list[dict[str, Any]] = []
        notes = [
            *task_spec.notes,
            *schema.notes,
            PREDICTIVE_CAVEAT,
            "Preprocessing is part of each sklearn Pipeline and is fitted only "
            "on each fold's training rows during CV.",
        ]
        risk = getattr(self.policy, "automl_risk", None)
        if risk is not None and not bool(
            getattr(risk, "allow_feature_importance", True)
        ):
            compute_importance = False
            notes.append(
                "Predictive feature importance was disabled by the active policy."
            )
        if compute_importance:
            try:
                importance = _permutation_importance(
                    selected.spec,
                    task=task_spec,
                    schema=schema,
                    features=features,
                    target=target_series,
                    split_plan=split_plan,
                    random_state=self.random_state,
                )
                notes.append(
                    "Permutation importance used the first held-out CV fold; "
                    "correlated features can split or mask importance."
                )
            except Exception as exc:
                notes.append(
                    f"Permutation importance skipped safely: {type(exc).__name__}: {exc}"
                )

        from sklearn.base import clone

        final_pipeline = build_pipeline(schema, clone(selected.spec.estimator))
        if calibrate and task_spec.is_classification:
            try:
                from sklearn.calibration import CalibratedClassifierCV

                final_pipeline = CalibratedClassifierCV(
                    estimator=final_pipeline,
                    method="sigmoid",
                    cv=split_plan.splits,
                    ensemble=True,
                )
                notes.append(
                    "Selected classifier received sigmoid calibration using "
                    "the audited CV split plan."
                )
            except Exception as exc:
                notes.append(
                    f"Calibration wrapper unavailable; retained uncalibrated "
                    f"pipeline ({type(exc).__name__}: {exc})."
                )
                final_pipeline = build_pipeline(schema, clone(selected.spec.estimator))
        try:
            final_pipeline.fit(features, target_series)
        except Exception as exc:
            if calibrate and task_spec.is_classification:
                notes.append(
                    f"Calibrated final fit failed; fitted the selected base "
                    f"pipeline instead ({type(exc).__name__}: {exc})."
                )
                final_pipeline = build_pipeline(schema, clone(selected.spec.estimator))
                final_pipeline.fit(features, target_series)
            else:
                raise

        gates = _build_gates(
            task=task_spec,
            schema=schema,
            selected=selected,
            subgroup=subgroup,
            n_rows=len(labeled),
            n_columns=len(labeled.columns),
            candidate_count=len(candidate_specs),
            gate_inputs=inputs,
            external_results=gate_results,
        )
        try:
            sklearn_version = metadata.version("scikit-learn")
        except Exception:
            sklearn_version = None
        report = AutoMLReport(
            task=task_spec,
            split_plan=split_plan,
            feature_schema=schema,
            candidates=[run.evaluation for run in ordered],
            selected_name=selected.spec.name,
            selected_pipeline=final_pipeline,
            mode=resolved_mode,
            random_state=self.random_state,
            feature_importance=importance,
            subgroup_performance=subgroup,
            gates=gates,
            manifest={
                "schema": "AutoCausalAutoMLRun.v1",
                "random_state": self.random_state,
                "cv_folds": len(split_plan.splits),
                "split_strategy": split_plan.strategy,
                "candidate_names": [
                    candidate.name for candidate in candidate_specs
                ],
                "sklearn_version": sklearn_version,
                "integration_versions": {
                    candidate.integration_id: candidate.integration_version
                    for candidate in candidate_specs
                    if candidate.integration_id
                    and candidate.integration_version is not None
                },
                "routing_decisions": [
                    candidate.routing_decision
                    for candidate in candidate_specs
                    if candidate.routing_decision is not None
                ],
                "selected_integration": selected.spec.integration_id,
                "contains_raw_predictions": False,
            },
            notes=notes,
        )
        self.last_report = report
        if enforce_gates and any(
            gate.get("ok") is False
            or str(gate.get("status", "")).lower() in ("fail", "escalate")
            for gate in gates
        ):
            raise AutoMLGateError(report)
        return report

    def report(self, frame: Any = None, **kwargs: Any) -> AutoMLReport:
        """Alias for :meth:`run`, matching the other Auto* suites."""
        return self.run(frame, **kwargs)


__all__ = ["AutoMLGateError", "AutoTabularML"]
