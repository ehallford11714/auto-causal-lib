"""Explicit-design causal effect estimation.

Discovery suggests structures, identification requires assumptions, estimation
quantifies an estimand, and refutation probes sensitivity. This package does
not treat correlation or discovery scores as causal identification evidence.
"""

from .core import (
    METHOD_SUPPORT,
    AutoInference,
    AutoInferencePlanner,
    method_support_matrix,
)
from .types import (
    CausalInferenceResult,
    CausalSpec,
    InferenceResult,
    MethodRecommendation,
)

__all__ = [
    "METHOD_SUPPORT",
    "AutoInference",
    "AutoInferencePlanner",
    "CausalInferenceResult",
    "CausalSpec",
    "InferenceResult",
    "MethodRecommendation",
    "method_support_matrix",
]
