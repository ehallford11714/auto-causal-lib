"""AutoCleanse — dedicated cleanse actions + SLM-directed suite.

Library-first::

    from autocausal.suites.autocleanse import AutoCleanseSuite, CleanseActions
    CleanseActions.impute(df, method="auto")
    clean = AutoCleanseSuite(df, use_slm=True).run()
"""

from __future__ import annotations

from autocausal.suites.autocleanse.actions import (
    CLEANSE_REGISTRY,
    CleanseActions,
    coerce_types,
    drop_constant_cols,
    drop_duplicates,
    drop_high_null_cols,
    flag_outliers,
    impute,
    profile_missingness,
    qc_snapshot,
    strip_id_leakage,
)
from autocausal.suites.autocleanse.report import CleanseOp, CleanseReport
from autocausal.suites.autocleanse.suite import AutoCleanseSuite, auto_cleanse

__all__ = [
    "AutoCleanseSuite",
    "CleanseActions",
    "CLEANSE_REGISTRY",
    "CleanseOp",
    "CleanseReport",
    "auto_cleanse",
    "profile_missingness",
    "coerce_types",
    "drop_duplicates",
    "drop_high_null_cols",
    "drop_constant_cols",
    "flag_outliers",
    "impute",
    "strip_id_leakage",
    "qc_snapshot",
]
