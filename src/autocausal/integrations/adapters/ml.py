"""Curated tabular ML, tuning, resampling, and explanation adapters."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from autocausal.integrations.adapters.base import (
    PREDICTIVE_CAVEAT,
    LazyAdapter,
    bounded_int,
)
from autocausal.integrations.adapters.native import NativeAdapter


class SklearnAdapter(LazyAdapter):
    id = "sklearn.tabular"
    integration_id = "scikit-learn"
    module_name = "sklearn"
    package_name = "scikit-learn"
    capabilities = (
        "ml.preprocessing",
        "ml.tabular_classifier",
        "ml.tabular_regressor",
        "ml.cross_validation",
        "ml.metrics",
        "nlp.embeddings.tfidf",
        "nlp.embeddings",
    )

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability == "ml.preprocessing":
            return self.preprocessing(**kwargs)
        if capability == "ml.tabular_classifier":
            return NativeAdapter.classifier(**kwargs)
        if capability == "ml.tabular_regressor":
            return NativeAdapter.regressor(**kwargs)
        if capability == "ml.cross_validation":
            return self.cross_validate(**kwargs)
        if capability == "ml.metrics":
            return self.metrics(**kwargs)
        if capability in ("nlp.embeddings.tfidf", "nlp.embeddings"):
            return self.tfidf(**kwargs)
        raise KeyError(capability)

    @staticmethod
    def preprocessing(
        *,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
        scale_numeric: bool = True,
        **_: Any,
    ) -> Any:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric_steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median"))
        ]
        if scale_numeric:
            numeric_steps.append(("scale", StandardScaler()))
        return ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(numeric_steps),
                    list(numeric_features),
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(strategy="most_frequent"),
                            ),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    list(categorical_features),
                ),
            ],
            remainder="drop",
        )

    @staticmethod
    def cross_validate(
        *,
        estimator: Any,
        x: Any,
        y: Any,
        scoring: Optional[str] = None,
        cv: int = 5,
        groups: Any = None,
        n_jobs: int = 1,
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np
        from sklearn.model_selection import cross_validate

        folds = bounded_int(cv, default=5, minimum=2, maximum=10, name="cv")
        jobs = bounded_int(
            n_jobs,
            default=1,
            minimum=1,
            maximum=8,
            name="n_jobs",
        )
        result = cross_validate(
            estimator,
            x,
            y,
            scoring=scoring,
            cv=folds,
            groups=groups,
            n_jobs=jobs,
            return_train_score=False,
            error_score="raise",
        )
        return {
            "test_scores": [float(item) for item in result["test_score"]],
            "mean": float(np.mean(result["test_score"])),
            "std": float(np.std(result["test_score"])),
            "fit_time": [float(item) for item in result["fit_time"]],
            "cv": folds,
            "caveat": PREDICTIVE_CAVEAT,
        }

    @staticmethod
    def metrics(
        *,
        task: str,
        y_true: Any,
        y_pred: Any,
        y_score: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np
        from sklearn import metrics

        selected = str(task).lower()
        if selected in ("classification", "classifier", "binary"):
            output: dict[str, Any] = {
                "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(
                    metrics.balanced_accuracy_score(y_true, y_pred)
                ),
                "f1_weighted": float(
                    metrics.f1_score(y_true, y_pred, average="weighted")
                ),
            }
            if y_score is not None:
                output["roc_auc"] = float(metrics.roc_auc_score(y_true, y_score))
        elif selected in ("regression", "regressor"):
            mse = float(metrics.mean_squared_error(y_true, y_pred))
            output = {
                "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mse)),
                "r2": float(metrics.r2_score(y_true, y_pred)),
            }
        else:
            raise ValueError("task must be classification or regression")
        output["caveat"] = PREDICTIVE_CAVEAT
        return output

    @staticmethod
    def tfidf(
        *,
        texts: Sequence[str],
        max_features: int = 4_096,
        ngram_range: tuple[int, int] = (1, 2),
        max_documents: int = 50_000,
        max_total_characters: int = 20_000_000,
        **_: Any,
    ) -> dict[str, Any]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        documents = [str(item) for item in texts]
        if len(documents) > max_documents:
            raise ValueError(f"texts exceeds max_documents={max_documents}")
        if sum(len(item) for item in documents) > max_total_characters:
            raise ValueError("text payload exceeds max_total_characters")
        vectorizer = TfidfVectorizer(
            max_features=bounded_int(
                max_features,
                default=4_096,
                minimum=16,
                maximum=50_000,
                name="max_features",
            ),
            ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform(documents)
        return {
            "embeddings": matrix,
            "vectorizer": vectorizer,
            "shape": tuple(matrix.shape),
            "backend": "sklearn_tfidf",
            "data_egress": False,
        }


class BoostedEstimatorAdapter(LazyAdapter):
    """CPU-only estimator factories for one optional boosted engine."""

    capabilities = (
        "ml.estimator_factory",
        "ml.tabular_classifier",
        "ml.tabular_regressor",
    )

    def __init__(
        self,
        integration_id: str,
        module_name: str,
        package_name: str,
    ) -> None:
        self.integration_id = integration_id
        self.module_name = module_name
        self.package_name = package_name
        self.id = f"{integration_id}.estimator"

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability == "ml.tabular_classifier":
            kwargs.pop("task", None)
            return self.estimator(task="classification", **kwargs)
        if capability == "ml.tabular_regressor":
            kwargs.pop("task", None)
            return self.estimator(task="regression", **kwargs)
        if capability == "ml.estimator_factory":
            return self.estimator(**kwargs)
        raise KeyError(capability)

    def estimator(
        self,
        *,
        task: str,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 0,
        n_jobs: int = 1,
        **params: Any,
    ) -> Any:
        module = self._module()
        estimators = bounded_int(
            n_estimators,
            default=300,
            minimum=10,
            maximum=2_000,
            name="n_estimators",
        )
        depth = bounded_int(
            max_depth,
            default=6,
            minimum=1,
            maximum=16,
            name="max_depth",
        )
        jobs = bounded_int(
            n_jobs,
            default=1,
            minimum=1,
            maximum=8,
            name="n_jobs",
        )
        selected_task = str(task).lower()
        classifier = selected_task in ("classification", "classifier", "binary")
        if not classifier and selected_task not in ("regression", "regressor"):
            raise ValueError("task must be classification or regression")
        common = {
            "n_estimators": estimators,
            "max_depth": depth,
            "learning_rate": float(learning_rate),
            "random_state": int(random_state),
        }
        if self.integration_id == "xgboost":
            cls = module.XGBClassifier if classifier else module.XGBRegressor
            return cls(
                **common,
                n_jobs=jobs,
                tree_method="hist",
                device="cpu",
                verbosity=0,
                **params,
            )
        if self.integration_id == "lightgbm":
            cls = module.LGBMClassifier if classifier else module.LGBMRegressor
            return cls(
                **common,
                n_jobs=jobs,
                device_type="cpu",
                verbosity=-1,
                **params,
            )
        if self.integration_id == "catboost":
            cls = module.CatBoostClassifier if classifier else module.CatBoostRegressor
            return cls(
                **common,
                thread_count=jobs,
                task_type="CPU",
                verbose=False,
                allow_writing_files=False,
                **params,
            )
        raise RuntimeError(f"unsupported boosted engine {self.integration_id!r}")


class OptunaAdapter(LazyAdapter):
    id = "optuna.tuning"
    integration_id = "optuna"
    module_name = "optuna"
    package_name = "optuna"
    capabilities = ("ml.tune",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "ml.tune":
            raise KeyError(capability)
        return self.tune(**kwargs)

    def tune(
        self,
        *,
        objective: Callable[[Any], float],
        direction: str = "minimize",
        n_trials: int = 20,
        timeout: float = 120.0,
        random_state: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        if not callable(objective):
            raise TypeError("objective must be callable")
        trials = bounded_int(
            n_trials,
            default=20,
            minimum=1,
            maximum=100,
            name="n_trials",
        )
        seconds = float(timeout)
        if not 0.1 <= seconds <= 600.0:
            raise ValueError("timeout must be between 0.1 and 600 seconds")
        selected = str(direction).lower()
        if selected not in ("minimize", "maximize"):
            raise ValueError("direction must be minimize or maximize")
        optuna = self._module()
        sampler = optuna.samplers.TPESampler(seed=int(random_state))
        study = optuna.create_study(direction=selected, sampler=sampler)
        study.optimize(
            objective,
            n_trials=trials,
            timeout=seconds,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
        )
        return {
            "best_value": float(study.best_value),
            "best_params": dict(study.best_params),
            "n_trials": len(study.trials),
            "direction": selected,
            "study": study,
            "bounded": True,
        }


class ImbalancedLearnAdapter(LazyAdapter):
    id = "imblearn.pipeline"
    integration_id = "imbalanced-learn"
    module_name = "imblearn"
    package_name = "imbalanced-learn"
    capabilities = ("ml.resampling_pipeline",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "ml.resampling_pipeline":
            raise KeyError(capability)
        return self.pipeline(**kwargs)

    @staticmethod
    def pipeline(
        *,
        estimator: Any,
        preprocessor: Any = None,
        strategy: str = "random_over",
        random_state: int = 0,
        k_neighbors: int = 5,
        **_: Any,
    ) -> Any:
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from imblearn.pipeline import Pipeline

        if estimator is None:
            raise ValueError("estimator is required")
        selected = str(strategy).lower()
        if selected in ("random_over", "randomoversampler"):
            sampler = RandomOverSampler(random_state=int(random_state))
        elif selected == "smote":
            sampler = SMOTE(
                random_state=int(random_state),
                k_neighbors=bounded_int(
                    k_neighbors,
                    default=5,
                    minimum=1,
                    maximum=20,
                    name="k_neighbors",
                ),
            )
        else:
            raise ValueError("strategy must be random_over or smote")
        steps: list[tuple[str, Any]] = []
        if preprocessor is not None:
            steps.append(("preprocess", preprocessor))
        steps.extend([("resample_train_fold_only", sampler), ("model", estimator)])
        return Pipeline(steps)


class ShapAdapter(LazyAdapter):
    id = "shap.explain"
    integration_id = "shap"
    module_name = "shap"
    package_name = "shap"
    capabilities = ("ml.explain",)

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability != "ml.explain":
            raise KeyError(capability)
        return self.explain(**kwargs)

    def explain(
        self,
        *,
        model: Any,
        x: Any,
        background: Any = None,
        max_samples: int = 200,
        max_background: int = 100,
        random_state: int = 0,
        return_explanation: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np

        shap = self._module()
        is_pandas = hasattr(x, "iloc")
        n_rows = len(x)
        if n_rows < 1:
            raise ValueError("x must contain at least one row")
        shape = getattr(x, "shape", ())
        if len(shape) >= 2 and int(shape[1]) > 5_000:
            raise ValueError("SHAP input exceeds 5,000-feature cap")
        sample_cap = bounded_int(
            max_samples,
            default=200,
            minimum=1,
            maximum=500,
            name="max_samples",
        )
        rng = np.random.default_rng(int(random_state))
        sample_indices = np.sort(
            rng.choice(n_rows, size=min(n_rows, sample_cap), replace=False)
        )
        sample = (
            x.iloc[sample_indices]
            if is_pandas
            else np.asarray(x)[sample_indices]
        )
        background_cap = bounded_int(
            max_background,
            default=100,
            minimum=1,
            maximum=200,
            name="max_background",
        )
        if background is None:
            background_indices = np.sort(
                rng.choice(n_rows, size=min(n_rows, background_cap), replace=False)
            )
            background = (
                x.iloc[background_indices]
                if is_pandas
                else np.asarray(x)[background_indices]
            )
        elif len(background) > background_cap:
            background_indices = np.sort(
                rng.choice(
                    len(background),
                    size=background_cap,
                    replace=False,
                )
            )
            background = (
                background.iloc[background_indices]
                if hasattr(background, "iloc")
                else np.asarray(background)[background_indices]
            )
        explainer = shap.Explainer(model, background)
        explanation = explainer(sample)
        output: dict[str, Any] = {
            "values": np.asarray(explanation.values),
            "base_values": np.asarray(explanation.base_values),
            "sample_indices": sample_indices,
            "n_explained": int(len(sample_indices)),
            "caveat": (
                "SHAP explains model predictions; it is not a causal attribution."
            ),
        }
        if return_explanation:
            output["explanation"] = explanation
        return output


def ml_adapters() -> tuple[LazyAdapter, ...]:
    return (
        SklearnAdapter(),
        BoostedEstimatorAdapter("xgboost", "xgboost", "xgboost"),
        BoostedEstimatorAdapter("lightgbm", "lightgbm", "lightgbm"),
        BoostedEstimatorAdapter("catboost", "catboost", "catboost"),
        OptunaAdapter(),
        ImbalancedLearnAdapter(),
        ShapAdapter(),
    )


__all__ = [
    "BoostedEstimatorAdapter",
    "ImbalancedLearnAdapter",
    "OptunaAdapter",
    "ShapAdapter",
    "SklearnAdapter",
    "ml_adapters",
]
