"""AutoMine — dedicated mine actions + SLM-directed suite.

Library-first::

    from autocausal.suites.automine import AutoMineSuite, MineActions
    MineActions.mine_associations(df)
    mine = AutoMineSuite(df, use_slm=True).run()
"""

from __future__ import annotations

from autocausal.suites.automine.actions import (
    MINE_REGISTRY,
    MineActions,
    join_public_sources,
    mine_associations,
    mine_behavioral,
    mine_kpi_hints,
    rank_candidates,
    to_mine_report,
)
from autocausal.suites.automine.report import MineReport
from autocausal.suites.automine.suite import AutoMineSuite, auto_mine

__all__ = [
    "AutoMineSuite",
    "MineActions",
    "MINE_REGISTRY",
    "MineReport",
    "auto_mine",
    "mine_associations",
    "mine_kpi_hints",
    "join_public_sources",
    "mine_behavioral",
    "rank_candidates",
    "to_mine_report",
]
