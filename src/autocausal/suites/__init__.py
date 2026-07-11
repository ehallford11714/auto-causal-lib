"""First-class auto suites: AutoCleanse, AutoEDA, AutoMine (SLM-directed).

Library-first API::

    from autocausal.suites.autocleanse import AutoCleanseSuite, CleanseActions
    from autocausal.suites.autoeda import AutoEDASuite, EDAActions
    from autocausal.suites.automine import AutoMineSuite, MineActions
    from autocausal.skilling import suite_tool_surface, SLMToolBroker

    CleanseActions.impute(df, method="auto")
    clean = AutoCleanseSuite(df, use_slm=True).run()

Every ``auto*`` path is directed by :class:`~autocausal.suites.director.SLMAutoDirector`
via the skilling ToolSurface when available; deterministic rules always work offline.

Env: ``AUTOCAUSAL_SLM=1``. Docs: ``docs/SUITES.md``, ``docs/SLM_SKILLING.md``.
"""

from __future__ import annotations

from autocausal.suites.autocleanse import AutoCleanseSuite, CleanseActions, CleanseReport, auto_cleanse
from autocausal.suites.autoeda import AutoEDASuite, EDAActions, EDAReport, auto_eda
from autocausal.suites.automine import AutoMineSuite, MineActions, MineReport, auto_mine
from autocausal.suites.director import (
    EPISTEMIC_NOTE,
    SLMAutoDirector,
    SLMDirectives,
    frame_profile,
    resolve_suite_slm,
)

__all__ = [
    "AutoCleanseSuite",
    "AutoEDASuite",
    "AutoMineSuite",
    "CleanseActions",
    "EDAActions",
    "MineActions",
    "CleanseReport",
    "EDAReport",
    "MineReport",
    "SLMAutoDirector",
    "SLMDirectives",
    "EPISTEMIC_NOTE",
    "auto_cleanse",
    "auto_eda",
    "auto_mine",
    "frame_profile",
    "resolve_suite_slm",
]
