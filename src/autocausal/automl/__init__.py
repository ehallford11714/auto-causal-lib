"""Leakage-safe, bounded AutoTabularML.

This package complements the existing ``autocausal.ml`` KPI loop without
changing that package's APIs.
"""

from autocausal.automl.candidates import CandidateSpec, default_candidates
from autocausal.automl.preprocessing import (
    FeatureSchema,
    build_pipeline,
    build_preprocessor,
    infer_feature_schema,
)
from autocausal.automl.report import (
    AutoMLReport,
    CandidateEvaluation,
    MetricSummary,
    PREDICTIVE_CAVEAT,
    load_trusted_model,
)
from autocausal.automl.splits import SplitPlan, SplitStrategy, make_splits
from autocausal.automl.suite import AutoMLGateError, AutoTabularML
from autocausal.automl.task import TaskSpec, TaskType, infer_task

__all__ = [
    "AutoMLGateError",
    "AutoMLReport",
    "AutoTabularML",
    "CandidateEvaluation",
    "CandidateSpec",
    "FeatureSchema",
    "MetricSummary",
    "PREDICTIVE_CAVEAT",
    "SplitPlan",
    "SplitStrategy",
    "TaskSpec",
    "TaskType",
    "build_pipeline",
    "build_preprocessor",
    "default_candidates",
    "infer_feature_schema",
    "infer_task",
    "load_trusted_model",
    "make_splits",
]
