"""autocausal — auto-impute and exploratory causal discovery for tabular data."""

from __future__ import annotations

from autocausal.api import AutoCausal
from autocausal.results import AutoResult, DiscoveryResult
from autocausal.__version__ import __version__

__all__ = [
    "AutoCausal",
    "DiscoveryResult",
    "AutoResult",
    "__version__",
    "create_from_context",
    "infer_from_results",
    "list_tools",
    "validate_pipeline",
    "slm_status",
    "list_guides",
    "direct",
    "KPIMinedCausalLoop",
    "ModelConstructPlan",
]


def __getattr__(name: str):
    if name in ("create_from_context", "infer_from_results", "slm_status"):
        from autocausal import slm as _slm

        return getattr(_slm, name)
    if name in ("list_tools", "validate_pipeline"):
        from autocausal import suite_tools as _st

        return getattr(_st, name)
    if name in ("list_guides", "direct"):
        from autocausal import guides as _guides

        return getattr(_guides, name)
    if name in ("KPIMinedCausalLoop", "ModelConstructPlan", "FitReport", "construct_model_plan"):
        from autocausal import ml as _ml

        return getattr(_ml, name)
    raise AttributeError(f"module 'autocausal' has no attribute {name!r}")
