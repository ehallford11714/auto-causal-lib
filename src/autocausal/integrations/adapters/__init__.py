"""Built-in adapter set.  Importing this module does not import third parties."""

from __future__ import annotations

from autocausal.integrations.adapters.base import LazyAdapter
from autocausal.integrations.adapters.causal import causal_adapters
from autocausal.integrations.adapters.data import data_adapters
from autocausal.integrations.adapters.ml import ml_adapters
from autocausal.integrations.adapters.native import NativeAdapter
from autocausal.integrations.adapters.nlp import nlp_adapters
from autocausal.integrations.adapters.statistics import (
    ScipyAdapter,
    StatsmodelsAdapter,
)
from autocausal.integrations.adapters.visualization import visualization_adapters
from autocausal.integrations.types import IntegrationAdapter


def builtin_adapters() -> tuple[IntegrationAdapter, ...]:
    return (
        NativeAdapter(),
        ScipyAdapter(),
        StatsmodelsAdapter(),
        *ml_adapters(),
        *nlp_adapters(),
        *causal_adapters(),
        *visualization_adapters(),
        *data_adapters(),
    )


__all__ = [
    "IntegrationAdapter",
    "LazyAdapter",
    "NativeAdapter",
    "ScipyAdapter",
    "StatsmodelsAdapter",
    "builtin_adapters",
]
