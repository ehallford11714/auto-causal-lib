"""
Physics predictive engine + physical insight grounding + autocausal loop.

Public API::

    from autocausal.physics import PhysicsEngine, PhysicsCausalSuite, ground_physical

    suite = PhysicsCausalSuite.from_csv("data.csv")
    result = suite.loop(horizon=5, text="what drives outcome?")
"""

from __future__ import annotations

from autocausal.physics.engine import (
    PhysicsEngine,
    state_from_dataframe,
    state_from_kpis,
    try_nextframeseq_npe,
)
from autocausal.physics.grounding import (
    PHYSICS_DOMAIN_GLOSSARIES,
    ground_physical,
    merge_with_domain_grounding,
)
from autocausal.physics.suite import PhysicsCausalSuite
from autocausal.physics.types import (
    PhysicalGroundingReport,
    PhysicalInsight,
    PhysicsLoopResult,
    PhysicsState,
    Trajectory,
    TrajectoryPoint,
)

__all__ = [
    "PhysicsEngine",
    "PhysicsCausalSuite",
    "PhysicsState",
    "Trajectory",
    "TrajectoryPoint",
    "PhysicalInsight",
    "PhysicalGroundingReport",
    "PhysicsLoopResult",
    "ground_physical",
    "merge_with_domain_grounding",
    "PHYSICS_DOMAIN_GLOSSARIES",
    "state_from_dataframe",
    "state_from_kpis",
    "try_nextframeseq_npe",
]
