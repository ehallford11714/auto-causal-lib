"""First-class auto suites: AutoCleanse, AutoEDA, AutoMine (SLM-directed).

Library-first API::

    from autocausal import AutoCleanseSuite, AutoEDASuite, AutoMineSuite, AutoCausal

    clean = AutoCleanseSuite(df, use_slm=True).run()
    eda = AutoEDASuite(clean.frame, use_slm=True).run()
    mine = AutoMineSuite(clean.frame, use_slm=True).run()
    ac = AutoCausal.from_dataframe(clean.frame).cleanse().eda().automine().discover()

Every ``auto*`` path is directed by :class:`~autocausal.suites.director.SLMAutoDirector`
when available; deterministic rules always work offline (never hard-crash).

Env: ``AUTOCAUSAL_SLM=1``. Docs: ``docs/SUITES.md``.
"""

from __future__ import annotations

from autocausal.suites.autocleanse import AutoCleanseSuite, CleanseReport, auto_cleanse
from autocausal.suites.autoeda import AutoEDASuite, EDAReport, auto_eda
from autocausal.suites.automine import AutoMineSuite, MineReport, auto_mine
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
