"""Association analysis with typed, honest results.

Association is descriptive evidence, not causal identification.
"""

from .core import (
    ASSOCIATION_NOTICE,
    CorrelationMatrixResult,
    CorrelationResult,
    CorrelationSuite,
    correlation,
    correlation_matrix,
)

__all__ = [
    "ASSOCIATION_NOTICE",
    "CorrelationMatrixResult",
    "CorrelationResult",
    "CorrelationSuite",
    "correlation",
    "correlation_matrix",
]
