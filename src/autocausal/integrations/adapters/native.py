"""Deterministic local fallbacks and existing AutoCausal engine bridges."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Callable, Mapping, Optional, Sequence

from autocausal.integrations.adapters.base import (
    CAUSAL_CAVEAT,
    LazyAdapter,
    as_2d_controls,
    bounded_int,
    residualize,
)


class NativeAdapter(LazyAdapter):
    id = "autocausal.native"
    integration_id = "autocausal-native"
    module_name = "autocausal"
    package_name = "auto-causal-lib"
    capabilities = (
        "stats.partial_correlation",
        "ml.tabular_classifier",
        "ml.tabular_regressor",
        "nlp.embeddings.tfidf",
        "nlp.embeddings",
        "nlp.vector_search",
        "causal.discovery.pc",
        "causal.estimate.ate",
        "viz.dag",
        "viz.chart",
    )

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        handlers: dict[str, Callable[..., Any]] = {
            "stats.partial_correlation": self.partial_correlation,
            "ml.tabular_classifier": self.classifier,
            "ml.tabular_regressor": self.regressor,
            "nlp.embeddings.tfidf": self.hashed_tfidf,
            "nlp.embeddings": self.hashed_tfidf,
            "nlp.vector_search": self.vector_search,
            "causal.discovery.pc": self.pc_lite,
            "causal.estimate.ate": self.aipw,
            "viz.dag": self.dag_spec,
            "viz.chart": self.chart_spec,
        }
        if capability not in handlers:
            raise KeyError(f"native adapter does not implement {capability!r}")
        return handlers[capability](**kwargs)

    @staticmethod
    def partial_correlation(
        *,
        x: Any,
        y: Any,
        controls: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np

        x_values = np.asarray(x, dtype=float).reshape(-1)
        y_values = np.asarray(y, dtype=float).reshape(-1)
        if len(x_values) != len(y_values):
            raise ValueError("x and y must have equal length")
        matrix = as_2d_controls(controls, len(x_values))
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if matrix.shape[1]:
            finite &= np.isfinite(matrix).all(axis=1)
        rx, _ = residualize(x_values[finite], matrix[finite])
        ry, _ = residualize(y_values[finite], matrix[finite])
        correlation = float(np.corrcoef(rx, ry)[0, 1])
        return {
            "method": "numpy_residual_partial_pearson",
            "correlation": correlation,
            "pvalue": None,
            "n": int(len(rx)),
            "n_controls": int(matrix.shape[1]),
            "caveat": "Association only; no causal identification.",
        }

    @staticmethod
    def classifier(
        *,
        model: str = "logistic",
        random_state: int = 0,
        max_iter: int = 1_000,
        **params: Any,
    ) -> Any:
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression

        name = str(model).lower()
        if name in ("logistic", "logistic_regression"):
            return LogisticRegression(
                max_iter=bounded_int(
                    max_iter,
                    default=1_000,
                    minimum=50,
                    maximum=5_000,
                    name="max_iter",
                ),
                random_state=int(random_state),
                class_weight=params.get("class_weight"),
            )
        if name in ("random_forest", "rf"):
            return RandomForestClassifier(
                n_estimators=bounded_int(
                    params.get("n_estimators"),
                    default=200,
                    minimum=10,
                    maximum=1_000,
                    name="n_estimators",
                ),
                max_depth=params.get("max_depth"),
                n_jobs=bounded_int(
                    params.get("n_jobs"),
                    default=1,
                    minimum=1,
                    maximum=8,
                    name="n_jobs",
                ),
                random_state=int(random_state),
                class_weight=params.get("class_weight"),
            )
        if name in ("hist_gradient_boosting", "histgb"):
            return HistGradientBoostingClassifier(
                max_iter=bounded_int(
                    params.get("n_estimators"),
                    default=100,
                    minimum=10,
                    maximum=1_000,
                    name="n_estimators",
                ),
                max_depth=params.get("max_depth"),
                random_state=int(random_state),
            )
        raise ValueError(
            "model must be logistic, random_forest, or hist_gradient_boosting"
        )

    @staticmethod
    def regressor(
        *,
        model: str = "ridge",
        random_state: int = 0,
        **params: Any,
    ) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge

        name = str(model).lower()
        if name == "ridge":
            return Ridge(alpha=float(params.get("alpha", 1.0)))
        if name in ("random_forest", "rf"):
            return RandomForestRegressor(
                n_estimators=bounded_int(
                    params.get("n_estimators"),
                    default=200,
                    minimum=10,
                    maximum=1_000,
                    name="n_estimators",
                ),
                max_depth=params.get("max_depth"),
                n_jobs=bounded_int(
                    params.get("n_jobs"),
                    default=1,
                    minimum=1,
                    maximum=8,
                    name="n_jobs",
                ),
                random_state=int(random_state),
            )
        if name in ("hist_gradient_boosting", "histgb"):
            return HistGradientBoostingRegressor(
                max_iter=bounded_int(
                    params.get("n_estimators"),
                    default=100,
                    minimum=10,
                    maximum=1_000,
                    name="n_estimators",
                ),
                max_depth=params.get("max_depth"),
                random_state=int(random_state),
            )
        raise ValueError(
            "model must be ridge, random_forest, or hist_gradient_boosting"
        )

    @staticmethod
    def hashed_tfidf(
        *,
        texts: Sequence[str],
        max_features: int = 1_024,
        max_documents: int = 10_000,
        max_total_characters: int = 5_000_000,
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np

        documents = [str(item) for item in texts]
        if len(documents) > max_documents:
            raise ValueError(f"texts exceeds max_documents={max_documents}")
        if sum(len(item) for item in documents) > max_total_characters:
            raise ValueError("text payload exceeds max_total_characters")
        dimensions = bounded_int(
            max_features,
            default=1_024,
            minimum=16,
            maximum=8_192,
            name="max_features",
        )
        tokenized = [
            re.findall(r"(?u)\b\w\w+\b", document.lower())
            for document in documents
        ]
        document_frequencies: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequencies.update(set(tokens))
        matrix = np.zeros((len(documents), dimensions), dtype=float)
        n_documents = max(len(documents), 1)
        for row, tokens in enumerate(tokenized):
            counts = Counter(tokens)
            for token, count in counts.items():
                digest = hashlib.sha256(
                    ("autocausal:" + token).encode("utf-8")
                ).digest()
                column = int.from_bytes(digest[:8], "big") % dimensions
                inverse_document_frequency = math.log(
                    (1 + n_documents) / (1 + document_frequencies[token])
                ) + 1.0
                matrix[row, column] += float(count) * inverse_document_frequency
            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0:
                matrix[row] /= norm
        return {
            "embeddings": matrix,
            "shape": tuple(matrix.shape),
            "backend": "autocausal_hashed_tfidf",
            "deterministic": True,
            "data_egress": False,
        }

    @staticmethod
    def vector_search(
        *,
        vectors: Any,
        queries: Any,
        k: int = 5,
        metric: str = "cosine",
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np

        base = np.asarray(vectors, dtype=float)
        query = np.asarray(queries, dtype=float)
        if base.ndim != 2 or query.ndim != 2 or base.shape[1] != query.shape[1]:
            raise ValueError("vectors and queries must be compatible 2D matrices")
        if len(base) > 1_000_000 or len(query) > 10_000:
            raise ValueError("vector search exceeds bounded in-memory limits")
        count = bounded_int(k, default=5, minimum=1, maximum=100, name="k")
        if metric == "cosine":
            base_norm = np.linalg.norm(base, axis=1, keepdims=True)
            query_norm = np.linalg.norm(query, axis=1, keepdims=True)
            scores = (query / np.maximum(query_norm, 1e-12)) @ (
                base / np.maximum(base_norm, 1e-12)
            ).T
        elif metric == "dot":
            scores = query @ base.T
        else:
            raise ValueError("metric must be cosine or dot")
        count = min(count, len(base))
        indices = np.argsort(-scores, axis=1)[:, :count]
        ordered_scores = np.take_along_axis(scores, indices, axis=1)
        return {
            "indices": indices,
            "scores": ordered_scores,
            "metric": metric,
            "backend": "numpy",
        }

    @staticmethod
    def pc_lite(
        *,
        frame: Any,
        columns: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Any:
        from autocausal.engines import discover_with

        return discover_with(
            frame,
            method="score_pc_lite",
            columns=columns,
            **kwargs,
        )

    @staticmethod
    def aipw(
        *,
        frame: Any,
        outcome: str,
        treatment: str,
        controls: Optional[Sequence[str]] = None,
        random_state: int = 0,
        n_splits: int = 5,
        propensity_clip: float = 0.02,
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        control_columns = [str(item) for item in (controls or ())]
        required = [outcome, treatment, *control_columns]
        missing = [item for item in required if item not in frame.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        work = frame[required].copy()
        work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
        work[treatment] = pd.to_numeric(work[treatment], errors="coerce")
        work = work.dropna(subset=[outcome, treatment])
        d = work[treatment].to_numpy(dtype=int)
        y = work[outcome].to_numpy(dtype=float)
        if set(np.unique(d)) != {0, 1}:
            raise ValueError("AIPW adapter requires a binary 0/1 treatment")
        if len(work) < 30 or min(np.bincount(d)) < 5:
            raise ValueError(
                "AIPW adapter requires at least 30 rows and 5 per treatment arm"
            )
        if control_columns:
            encoded = pd.get_dummies(work[control_columns], dummy_na=True)
            x = encoded.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        else:
            x = np.ones((len(work), 1), dtype=float)
        folds = min(
            bounded_int(
                n_splits,
                default=5,
                minimum=2,
                maximum=10,
                name="n_splits",
            ),
            int(min(np.bincount(d))),
        )
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=int(random_state),
        )
        propensity = np.zeros(len(work), dtype=float)
        mu0 = np.zeros(len(work), dtype=float)
        mu1 = np.zeros(len(work), dtype=float)
        for train, test in splitter.split(x, d):
            propensity_model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=1_000, random_state=int(random_state)),
            )
            propensity_model.fit(x[train], d[train])
            propensity[test] = propensity_model.predict_proba(x[test])[:, 1]
            for arm, target in ((0, mu0), (1, mu1)):
                arm_train = train[d[train] == arm]
                outcome_model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    RandomForestRegressor(
                        n_estimators=200,
                        min_samples_leaf=5,
                        n_jobs=1,
                        random_state=int(random_state) + arm,
                    ),
                )
                outcome_model.fit(x[arm_train], y[arm_train])
                target[test] = outcome_model.predict(x[test])
        clip = float(propensity_clip)
        if not 0.001 <= clip <= 0.2:
            raise ValueError("propensity_clip must be between 0.001 and 0.2")
        propensity = np.clip(propensity, clip, 1.0 - clip)
        influence = (
            mu1
            - mu0
            + d * (y - mu1) / propensity
            - (1 - d) * (y - mu0) / (1.0 - propensity)
        )
        ate = float(np.mean(influence))
        standard_error = float(np.std(influence, ddof=1) / math.sqrt(len(influence)))
        return {
            "ok": True,
            "method": "cross_fitted_aipw",
            "backend": "autocausal-native/sklearn",
            "estimate": {
                "ate": ate,
                "standard_error": standard_error,
                "ci95": [
                    ate - 1.96 * standard_error,
                    ate + 1.96 * standard_error,
                ],
                "n": int(len(work)),
                "outcome": outcome,
                "treatment": treatment,
                "controls": control_columns,
                "propensity_min": float(propensity.min()),
                "propensity_max": float(propensity.max()),
            },
            "notes": [CAUSAL_CAVEAT],
        }

    @staticmethod
    def dag_spec(
        *,
        edges: Sequence[Any],
        nodes: Optional[Sequence[str]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        normalized: list[dict[str, Any]] = []
        inferred = set(str(item) for item in (nodes or ()))
        for edge in edges:
            if isinstance(edge, Mapping):
                source = str(edge.get("source"))
                target = str(edge.get("target"))
                attributes = {
                    str(key): value
                    for key, value in edge.items()
                    if key not in ("source", "target")
                }
            else:
                source, target = str(edge[0]), str(edge[1])
                attributes = {}
            inferred.update((source, target))
            normalized.append({"source": source, "target": target, **attributes})
        return {
            "backend": "spec-only",
            "nodes": sorted(inferred),
            "edges": normalized,
            "epistemic_caveat": (
                "Rendered edges are hypotheses, not identified effects."
            ),
        }

    @staticmethod
    def chart_spec(*, frame: Any, spec: Any, **_: Any) -> Any:
        from autocausal.autochart import AutoChart

        return AutoChart(spec=spec, backend="data").render(frame)


__all__ = ["NativeAdapter"]
