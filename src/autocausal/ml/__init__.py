"""ML Model Hub slice: KPI-mined loops, SLM model construction, optional PyTorch.

Public API::

    from autocausal.ml import KPIMinedCausalLoop, ModelConstructPlan

    result = KPIMinedCausalLoop.from_csv("data.csv").run(
        text="what drives Y?",
        use_slm=False,
        use_torch=True,
        horizon=5,
    )
"""

from __future__ import annotations

from autocausal.ml.construct import (
    ModelConstructPlan,
    construct_model_plan,
    torch_available,
    torch_preferred,
)
from autocausal.ml.loop import KPIMinedCausalLoop, KPILoopResult
from autocausal.ml.fit_report import FitReport
from autocausal.ml.automl import (
    AutoML,
    AutoMLCandidateResult,
    AutoMLReport,
    run_automl,
)

__all__ = [
    "KPIMinedCausalLoop",
    "KPILoopResult",
    "ModelConstructPlan",
    "FitReport",
    "construct_model_plan",
    "torch_available",
    "torch_preferred",
    "AutoML",
    "AutoMLCandidateResult",
    "AutoMLReport",
    "run_automl",
]
