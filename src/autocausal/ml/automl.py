"""Small, leakage-safe AutoML portfolio for production-oriented prediction.

Prediction is not causal estimation.  Every preprocessing object is fitted
inside each training fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from autocausal.production import (
    GateReport,
    GateResult,
    ProductionGateError,
    ProductionPolicy,
    resolve_policy,
)

PREDICTION_NOTICE = (
    "Predictive performance and feature importance do not identify causal effects."
)


@dataclass
class AutoMLCandidateResult:
    model: str
    metric: str
    fold_scores: list[float]
    mean_score: float
    std_score: float
    ci_low: float
    ci_high: float
    fit_failures: list[str] = field(default_factory=list)
    calibration: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "metric": self.metric,
            "fold_scores": list(self.fold_scores),
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "fit_failures": list(self.fit_failures),
            "calibration": self.calibration,
        }


@dataclass
class AutoMLReport:
    target: str
    task: str
    metric: str
    selected_model: Optional[str]
    candidates: list[AutoMLCandidateResult]
    cv_plan: dict[str, Any]
    leakage_checks: dict[str, Any]
    class_balance: Optional[dict[str, Any]]
    gates: GateReport
    feature_importance_caveat: str = (
        "Importance reflects predictive model behavior, not intervention effects."
    )
    warnings: list[str] = field(default_factory=list)
    random_state: int = 0
    schema: str = "AutoCausalAutoMLReport.v1"
    epistemic_notice: str = PREDICTION_NOTICE
    raw_predictions_included: bool = False

    @property
    def ok(self) -> bool:
        return self.selected_model is not None and self.gates.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target": self.target,
            "task": self.task,
            "metric": self.metric,
            "selected_model": self.selected_model,
            "candidates": [value.to_dict() for value in self.candidates],
            "cv_plan": dict(self.cv_plan),
            "leakage_checks": dict(self.leakage_checks),
            "class_balance": self.class_balance,
            "gates": self.gates.to_dict(),
            "feature_importance_caveat": self.feature_importance_caveat,
            "warnings": list(self.warnings),
            "random_state": self.random_state,
            "epistemic_notice": self.epistemic_notice,
            "raw_predictions_included": self.raw_predictions_included,
        }

    def report(self) -> str:
        rows = [
            "| model | mean | SD | 95% fold CI |",
            "|---|---:|---:|---:|",
        ]
        for candidate in self.candidates:
            rows.append(
                f"| {candidate.model} | {candidate.mean_score:.5g} | "
                f"{candidate.std_score:.5g} | "
                f"[{candidate.ci_low:.5g}, {candidate.ci_high:.5g}] |"
            )
        return "\n".join(
            [
                "# AutoML report",
                "",
                f"> **{self.epistemic_notice}**",
                "",
                f"- Target: `{self.target}`",
                f"- Task: `{self.task}`",
                f"- Metric: `{self.metric}`",
                f"- Selected model: `{self.selected_model}`",
                f"- Split: `{self.cv_plan.get('strategy')}`",
                "- Preprocessing fit scope: `training fold only`",
                "",
                *rows,
                "",
                self.gates.report(),
            ]
        )


def _gate(
    gate_id: str,
    ok: bool,
    detail: str,
    *,
    policy: ProductionPolicy,
    metric: Any = None,
    threshold: Any = None,
    remediation: Optional[str] = None,
    warning_only: bool = False,
) -> GateResult:
    if ok:
        status = "pass"
    elif warning_only or policy.profile == "exploratory":
        status = "warn"
    elif policy.profile == "review":
        status = "escalate"
    else:
        status = "fail"
    return GateResult(
        id=gate_id,
        ok=status in ("pass", "warn", "skip"),
        status=status,
        detail=detail,
        metric=metric,
        threshold=threshold,
        remediation=remediation,
        stage="automl",
        policy_version=policy.policy_version,
    )


class AutoML:
    """Deterministic baseline/candidate comparison with fold-local pipelines.

    Prefer :class:`autocausal.automl.AutoTabularML` (and
    ``AutoCausal.tabular_ml``) for new production prediction work. This class
    remains as the lighter KPI/legacy portfolio used by older loops.
    """

    def __init__(
        self,
        *,
        policy: Optional[ProductionPolicy | Mapping[str, Any]] = None,
        mode: str = "exploratory",
        random_state: Optional[int] = None,
    ) -> None:
        self.policy = resolve_policy(
            mode,
            policy,
            random_state=random_state,
        )
        self.mode = mode
        self.random_state = self.policy.random_state
        self.last_report: Optional[AutoMLReport] = None

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target: str,
        features: Optional[Sequence[str]] = None,
        task: str = "auto",
        group: Optional[str] = None,
        time: Optional[str] = None,
        candidates: Optional[Sequence[str]] = None,
    ) -> AutoMLReport:
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} not found.")
        risk = self.policy.automl_risk
        statistical = self.policy.statistical_validity
        assert risk is not None and statistical is not None
        feature_columns = (
            [str(value) for value in features]
            if features is not None
            else [
                str(column)
                for column in df.columns
                if str(column) not in {target, group, time}
            ]
        )
        missing = [
            column for column in feature_columns if column not in df.columns
        ]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        duplicate_target = target in feature_columns
        if duplicate_target:
            feature_columns = [
                column for column in feature_columns if column != target
            ]
        if not feature_columns:
            raise ValueError("At least one non-target feature is required.")

        required = [target, *feature_columns]
        if group:
            required.append(group)
        if time:
            required.append(time)
        work = df[list(dict.fromkeys(required))].dropna(subset=[target]).copy()
        if time:
            work = work.sort_values(time).reset_index(drop=True)
        y = work[target]
        inferred_task = self._infer_task(y) if task == "auto" else task
        if inferred_task not in ("binary", "classification", "regression"):
            raise ValueError(
                "task must be auto, binary, classification, or regression"
            )

        exact_leaks = []
        name_leaks = []
        for column in feature_columns:
            pair = work[[column, target]].dropna()
            if len(pair) and pair[column].equals(pair[target]):
                exact_leaks.append(column)
            lowered = column.lower()
            if any(
                token in lowered
                for token in (
                    "target_leak",
                    "ground_truth",
                    "y_true",
                    "future_outcome",
                    "post_outcome",
                )
            ):
                name_leaks.append(column)
        leakage = {
            "target_in_features": duplicate_target,
            "exact_target_copies": exact_leaks,
            "suspicious_feature_names": name_leaks,
            "passed": not (duplicate_target or exact_leaks or name_leaks),
        }

        report_gates = GateReport(
            profile=self.policy.profile,
            policy_version=self.policy.policy_version,
        )
        report_gates.add(
            _gate(
                "automl_sample_size",
                len(work) >= statistical.min_sample_size,
                f"Training rows={len(work)}.",
                policy=self.policy,
                metric=len(work),
                threshold=statistical.min_sample_size,
                remediation="Collect more independent observations.",
            ),
            _gate(
                "automl_leakage",
                leakage["passed"],
                "No direct target leakage detected."
                if leakage["passed"]
                else f"Leakage findings: {leakage}",
                policy=self.policy,
                metric=leakage,
                threshold={"target_copies": 0, "suspicious_names": 0},
                remediation="Remove target/post-outcome features and rerun.",
            ),
        )

        class_balance = None
        if inferred_task in ("binary", "classification"):
            frequencies = y.value_counts(normalize=True, dropna=True)
            minimum = float(frequencies.min()) if len(frequencies) else 0.0
            class_balance = {
                "n_classes": int(len(frequencies)),
                "minimum_class_fraction": minimum,
                "class_labels_redacted": True,
            }
            report_gates.add(
                _gate(
                    "automl_class_balance",
                    minimum >= risk.min_class_fraction,
                    f"Minimum class fraction={minimum:.3f}.",
                    policy=self.policy,
                    metric=minimum,
                    threshold=risk.min_class_fraction,
                    remediation="Collect minority cases or use reviewed imbalance handling.",
                )
            )

        if self.policy.profile == "production" and report_gates.failed:
            partial = AutoMLReport(
                target=target,
                task=inferred_task,
                metric="not_run",
                selected_model=None,
                candidates=[],
                cv_plan={"strategy": "blocked_before_cv"},
                leakage_checks=leakage,
                class_balance=class_balance,
                gates=report_gates,
                random_state=self.random_state,
            )
            self.last_report = partial
            raise ProductionGateError(
                "AutoML blocked by sample/leakage/class-balance gates.",
                code="automl_preflight_failed",
                gates=report_gates.failed,
                partial_result=partial,
            )

        splitter, splits, strategy = self._splits(
            work,
            y,
            task=inferred_task,
            group=group,
            time=time,
            folds=risk.cv_folds,
        )
        model_names = list(
            candidates
            or (
                ["dummy", "logistic", "random_forest"]
                if inferred_task in ("binary", "classification")
                else ["dummy", "ridge", "random_forest"]
            )
        )
        metric = (
            "roc_auc"
            if inferred_task == "binary"
            else "balanced_accuracy"
            if inferred_task == "classification"
            else "rmse"
        )
        candidate_results = []
        for model_name in model_names:
            candidate_results.append(
                self._evaluate_candidate(
                    work,
                    target=target,
                    features=feature_columns,
                    task=inferred_task,
                    metric=metric,
                    model_name=model_name,
                    splits=splits,
                )
            )
        successful = [
            result for result in candidate_results if result.fold_scores
        ]
        selected = None
        if successful:
            selected_result = (
                min(successful, key=lambda value: value.mean_score)
                if metric == "rmse"
                else max(successful, key=lambda value: value.mean_score)
            )
            selected = selected_result.model
            coefficient_variation = selected_result.std_score / max(
                abs(selected_result.mean_score), 1e-12
            )
            report_gates.add(
                _gate(
                    "automl_cv_stability",
                    coefficient_variation
                    <= risk.max_cv_coefficient_variation,
                    f"Selected-model CV coefficient of variation={coefficient_variation:.3f}.",
                    policy=self.policy,
                    metric=coefficient_variation,
                    threshold=risk.max_cv_coefficient_variation,
                    remediation="Collect data, simplify model, or use group/time-aware validation.",
                )
            )
            if inferred_task == "binary":
                brier_values = [
                    float(value.calibration["brier"])
                    for value in successful
                    if value.model == selected
                    and value.calibration
                    and value.calibration.get("brier") is not None
                ]
                brier = brier_values[0] if brier_values else None
                report_gates.add(
                    _gate(
                        "automl_calibration",
                        brier is not None and brier <= risk.max_brier_score,
                        f"Out-of-fold Brier score={brier}.",
                        policy=self.policy,
                        metric=brier,
                        threshold=risk.max_brier_score,
                        remediation="Calibrate on held-out data or defer deployment.",
                    )
                )
        else:
            report_gates.add(
                _gate(
                    "automl_candidates",
                    False,
                    "All model candidates failed.",
                    policy=self.policy,
                    remediation="Inspect candidate failures and feature schema.",
                )
            )

        cv_plan = {
            "strategy": strategy,
            "folds": len(splits),
            "shuffle": bool(not group and not time),
            "group_column": group,
            "time_column": time,
            "preprocessing_fit_scope": "training_fold_only",
            "raw_predictions_retained": False,
        }
        warnings = [PREDICTION_NOTICE]
        if group is None and time is None:
            warnings.append(
                "Rows are treated as independent; pass group= or time= when applicable."
            )
        final_report = AutoMLReport(
            target=target,
            task=inferred_task,
            metric=metric,
            selected_model=selected,
            candidates=candidate_results,
            cv_plan=cv_plan,
            leakage_checks=leakage,
            class_balance=class_balance,
            gates=report_gates,
            warnings=warnings,
            random_state=self.random_state,
            raw_predictions_included=False,
        )
        self.last_report = final_report
        if self.policy.profile == "production" and report_gates.failed:
            raise ProductionGateError(
                "AutoML failed production validation gates.",
                code="automl_validation_failed",
                gates=report_gates.failed,
                partial_result=final_report,
            )
        return final_report

    @staticmethod
    def _infer_task(y: pd.Series) -> str:
        unique = int(y.nunique(dropna=True))
        if unique == 2:
            return "binary"
        if (
            not pd.api.types.is_numeric_dtype(y)
            or unique <= min(20, max(3, len(y) // 20))
        ):
            return "classification"
        return "regression"

    def _splits(
        self,
        work: pd.DataFrame,
        y: pd.Series,
        *,
        task: str,
        group: Optional[str],
        time: Optional[str],
        folds: int,
    ) -> tuple[Any, list[tuple[np.ndarray, np.ndarray]], str]:
        from sklearn.model_selection import (
            GroupKFold,
            KFold,
            StratifiedKFold,
            TimeSeriesSplit,
        )

        if group:
            n_groups = int(work[group].nunique())
            actual = min(folds, n_groups)
            if actual < 2:
                raise ValueError("Group CV requires at least two groups.")
            splitter = GroupKFold(n_splits=actual)
            splits = list(
                splitter.split(work, y, groups=work[group])
            )
            return splitter, splits, "group_kfold"
        if time:
            actual = min(folds, max(2, len(work) // 20))
            splitter = TimeSeriesSplit(n_splits=actual)
            splits = list(splitter.split(work))
            return splitter, splits, "expanding_time_series"
        if task in ("binary", "classification"):
            minimum_class = int(y.value_counts().min())
            actual = min(folds, minimum_class)
            if actual < 2:
                raise ValueError("Stratified CV requires at least two cases per class.")
            splitter = StratifiedKFold(
                n_splits=actual,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = list(splitter.split(work, y))
            return splitter, splits, "stratified_kfold"
        actual = min(folds, max(2, len(work) // 10))
        splitter = KFold(
            n_splits=actual,
            shuffle=True,
            random_state=self.random_state,
        )
        return splitter, list(splitter.split(work)), "kfold"

    def _evaluate_candidate(
        self,
        work: pd.DataFrame,
        *,
        target: str,
        features: Sequence[str],
        task: str,
        metric: str,
        model_name: str,
        splits: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> AutoMLCandidateResult:
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import (
            balanced_accuracy_score,
            brier_score_loss,
            mean_squared_error,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        x = work[list(features)]
        y = work[target]
        numeric = [
            column
            for column in features
            if pd.api.types.is_numeric_dtype(x[column])
        ]
        categorical = [column for column in features if column not in numeric]
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        preprocess = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", encoder),
                        ]
                    ),
                    categorical,
                ),
            ],
            remainder="drop",
        )
        if task in ("binary", "classification"):
            model_map = {
                "dummy": DummyClassifier(strategy="prior"),
                "logistic": LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=self.random_state,
                ),
                "random_forest": RandomForestClassifier(
                    n_estimators=100,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=1,
                ),
            }
        else:
            model_map = {
                "dummy": DummyRegressor(strategy="mean"),
                "ridge": Ridge(alpha=1.0),
                "random_forest": RandomForestRegressor(
                    n_estimators=100,
                    min_samples_leaf=3,
                    random_state=self.random_state,
                    n_jobs=1,
                ),
            }
        if model_name not in model_map:
            raise ValueError(
                f"Unsupported AutoML candidate {model_name!r}; "
                f"choose from {sorted(model_map)}"
            )
        scores: list[float] = []
        failures: list[str] = []
        probabilities: list[float] = []
        binary_truth: list[int] = []
        for train, validation in splits:
            pipeline = Pipeline(
                [
                    ("preprocess", preprocess),
                    ("model", model_map[model_name]),
                ]
            )
            try:
                pipeline.fit(x.iloc[train], y.iloc[train])
                prediction = pipeline.predict(x.iloc[validation])
                if metric == "rmse":
                    score = sqrt(
                        mean_squared_error(y.iloc[validation], prediction)
                    )
                elif metric == "roc_auc":
                    probability = pipeline.predict_proba(x.iloc[validation])[:, 1]
                    score = roc_auc_score(y.iloc[validation], probability)
                    probabilities.extend(float(value) for value in probability)
                    classes = list(pipeline.named_steps["model"].classes_)
                    positive = classes[-1]
                    binary_truth.extend(
                        int(value == positive)
                        for value in y.iloc[validation]
                    )
                else:
                    score = balanced_accuracy_score(
                        y.iloc[validation], prediction
                    )
                scores.append(float(score))
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
        if scores:
            mean = float(np.mean(scores))
            std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            half_width = 1.959964 * std / sqrt(max(len(scores), 1))
        else:
            mean = std = float("nan")
            half_width = float("nan")
        calibration = None
        if metric == "roc_auc" and probabilities and binary_truth:
            calibration = {
                "brier": float(
                    brier_score_loss(binary_truth, probabilities)
                ),
                "scope": "out_of_fold",
                "probability_calibration_model": "none",
            }
        return AutoMLCandidateResult(
            model=model_name,
            metric=metric,
            fold_scores=scores,
            mean_score=mean,
            std_score=std,
            ci_low=float(mean - half_width),
            ci_high=float(mean + half_width),
            fit_failures=failures,
            calibration=calibration,
        )


def run_automl(
    df: pd.DataFrame,
    *,
    target: str,
    policy: Optional[ProductionPolicy | Mapping[str, Any]] = None,
    mode: str = "exploratory",
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> AutoMLReport:
    return AutoML(
        policy=policy,
        mode=mode,
        random_state=random_state,
    ).fit(df, target=target, **kwargs)
