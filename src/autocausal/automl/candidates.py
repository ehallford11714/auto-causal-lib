"""Bounded, deterministic candidate model registry for tabular prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from autocausal.automl.task import TaskSpec


@dataclass
class CandidateSpec:
    name: str
    estimator: Any
    family: str
    complexity: int
    expected_latency: str
    notes: tuple[str, ...] = ()
    integration_id: Optional[str] = None
    integration_version: Optional[str] = None
    routing_decision: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "complexity": self.complexity,
            "expected_latency": self.expected_latency,
            "estimator": type(self.estimator).__name__,
            "notes": list(self.notes),
            "integration_id": self.integration_id,
            "integration_version": self.integration_version,
            "routing_decision": self.routing_decision,
        }


def _optional_integration_candidate(
    task: TaskSpec,
    *,
    integration_id: str,
    random_state: int,
) -> Optional[CandidateSpec]:
    """Build one installed optional estimator through the capability registry."""

    from autocausal.integrations import (
        CapabilityRouter,
        RoutingPolicy,
        get_default_registry,
    )

    registry = get_default_registry()
    capability = (
        "ml.tabular_classifier"
        if task.is_classification
        else "ml.tabular_regressor"
    )
    decision = CapabilityRouter(registry).route(
        capability,
        policy=RoutingPolicy(explicit_integration=integration_id),
    )
    if decision.selected_integration != integration_id:
        return None
    adapter = registry.get_capability(
        capability,
        integration_id=integration_id,
    )
    estimator = adapter.invoke(
        capability,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=1,
        random_state=int(random_state),
    )
    version = registry.status(integration_id).spec.version_detected
    return CandidateSpec(
        name=integration_id,
        estimator=estimator,
        family="boosted_tree",
        complexity=5,
        expected_latency="high",
        notes=(
            "Optional CPU-only integration candidate; bounded to 200 estimators "
            "and one worker by AutoCausal.",
        ),
        integration_id=integration_id,
        integration_version=version,
        routing_decision=decision.to_dict(),
    )


def default_candidates(
    task: TaskSpec,
    *,
    random_state: int = 0,
    names: Optional[Sequence[str]] = None,
    max_candidates: int = 4,
) -> list[CandidateSpec]:
    """Return a small candidate set; no unbounded hyperparameter search."""

    if task.is_classification:
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import LogisticRegression

        candidates = [
            CandidateSpec(
                "dummy",
                DummyClassifier(strategy="prior", random_state=random_state),
                "baseline",
                0,
                "very_low",
                ("Sanity-check baseline; never selected on accuracy alone.",),
            ),
            CandidateSpec(
                "logistic",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
                "linear",
                1,
                "low",
                ("Probabilities are inspectable but still require calibration checks.",),
            ),
            CandidateSpec(
                "random_forest",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=random_state,
                ),
                "bagged_tree",
                3,
                "medium",
            ),
            CandidateSpec(
                "hist_gradient_boosting",
                HistGradientBoostingClassifier(
                    max_iter=100,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=random_state,
                ),
                "boosted_tree",
                4,
                "medium",
            ),
        ]
    else:
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import (
            HistGradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.linear_model import Ridge

        candidates = [
            CandidateSpec(
                "dummy",
                DummyRegressor(strategy="mean"),
                "baseline",
                0,
                "very_low",
            ),
            CandidateSpec(
                "ridge",
                Ridge(alpha=1.0),
                "linear",
                1,
                "low",
            ),
            CandidateSpec(
                "random_forest",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_leaf=2,
                    n_jobs=1,
                    random_state=random_state,
                ),
                "bagged_tree",
                3,
                "medium",
            ),
            CandidateSpec(
                "hist_gradient_boosting",
                HistGradientBoostingRegressor(
                    max_iter=100,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=random_state,
                ),
                "boosted_tree",
                4,
                "medium",
            ),
        ]
    aliases = {
        "linear": "logistic" if task.is_classification else "ridge",
        "logistic_regression": "logistic",
        "rf": "random_forest",
        "gradient_boosting": "hist_gradient_boosting",
        "hist_gb": "hist_gradient_boosting",
        "baseline": "dummy",
        "xgb": "xgboost",
        "lgbm": "lightgbm",
    }
    selected = (
        {aliases.get(str(name).lower(), str(name).lower()) for name in names}
        if names
        else set()
    )
    optional_names = ("xgboost", "lightgbm", "catboost")
    known = {candidate.name for candidate in candidates} | set(optional_names)
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown candidate model(s): {sorted(unknown)}")
    requested_optional = (
        [name for name in optional_names if name in selected]
        if names
        else list(optional_names)
    )
    unavailable: list[str] = []
    for integration_id in requested_optional:
        try:
            candidate = _optional_integration_candidate(
                task,
                integration_id=integration_id,
                random_state=random_state,
            )
        except Exception as exc:
            if names:
                raise RuntimeError(
                    f"optional candidate {integration_id!r} failed bounded "
                    f"construction: {type(exc).__name__}: {exc}"
                ) from exc
            candidate = None
        if candidate is None:
            unavailable.append(integration_id)
        else:
            candidates.append(candidate)
            if not names and len(candidates) >= max(1, int(max_candidates)):
                break
    if names:
        missing_requested = selected & set(unavailable)
        if missing_requested:
            raise ValueError(
                "requested optional candidate(s) are not installed/policy-eligible: "
                f"{sorted(missing_requested)}"
            )
        candidates = [
            candidate for candidate in candidates if candidate.name in selected
        ]
    bounded = candidates[: max(1, int(max_candidates))]
    if not bounded:
        raise ValueError("candidate set cannot be empty")
    return bounded


__all__ = ["CandidateSpec", "default_candidates"]
