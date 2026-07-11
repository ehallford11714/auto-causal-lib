"""AutoEDA — dedicated EDA actions + SLM-directed suite.

Library-first::

    from autocausal.suites.autoeda import AutoEDASuite, EDAActions
    EDAActions.suggest_roles(df)
    eda = AutoEDASuite(df, use_slm=True).run()
"""

from __future__ import annotations

from autocausal.suites.autoeda.actions import (
    EDA_REGISTRY,
    EDAActions,
    cardinality_report,
    correlation_matrix,
    leakage_hints,
    mining_profile,
    qc_snapshot,
    suggest_roles,
    summarize_distributions,
)
from autocausal.suites.autoeda.report import EDAReport, RoleProposal
from autocausal.suites.autoeda.suite import AutoEDASuite, auto_eda

__all__ = [
    "AutoEDASuite",
    "EDAActions",
    "EDA_REGISTRY",
    "EDAReport",
    "RoleProposal",
    "auto_eda",
    "summarize_distributions",
    "correlation_matrix",
    "cardinality_report",
    "suggest_roles",
    "qc_snapshot",
    "leakage_hints",
    "mining_profile",
]
