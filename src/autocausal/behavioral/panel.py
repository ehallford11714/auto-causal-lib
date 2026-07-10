"""Join behavioral traces → tabular panel for mine / discover."""

from __future__ import annotations

from typing import Any, Optional, Union

import pandas as pd

from autocausal.behavioral.features import (
    engineer_trace_features,
    subject_panel,
    traces_to_frame,
)
from autocausal.behavioral.loaders import load_demo, load_traces_csv, load_traces_json
from autocausal.behavioral.schema import TraceCollection


def collection_to_panel(
    collection: TraceCollection,
    *,
    level: str = "subject",
    engineer: bool = True,
) -> pd.DataFrame:
    """Convert traces to a tabular panel.

    Parameters
    ----------
    level:
        ``subject`` → one row per subject (default, best for mine/discover).
        ``event`` → engineered event-level frame.
    """
    df = traces_to_frame(collection)
    if engineer:
        df = engineer_trace_features(df)
    if level == "event":
        return df
    return subject_panel(df)


def load_panel(
    source: str,
    *,
    level: str = "subject",
    is_demo: bool = False,
) -> tuple[pd.DataFrame, TraceCollection]:
    """Load demo id / CSV / JSON path into a panel + collection."""
    if is_demo or source in ("habit_loop", "nudge_ab", "reinforcement_schedule"):
        coll = load_demo(source)
    elif str(source).lower().endswith(".json"):
        coll = load_traces_json(source)
    else:
        coll = load_traces_csv(source)
    panel = collection_to_panel(coll, level=level)
    return panel, coll


def join_traces_to_frame(
    df: pd.DataFrame,
    collection: TraceCollection,
    *,
    on: str = "subject_id",
    how: str = "left",
    level: str = "subject",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Left/right/inner join an existing frame with a behavioral panel."""
    panel = collection_to_panel(collection, level=level)
    log: list[dict[str, Any]] = [
        {
            "op": "join_behavioral",
            "trace": collection.name,
            "on": on,
            "how": how,
            "level": level,
            "panel_rows": len(panel),
            "panel_cols": list(panel.columns),
        }
    ]
    if on not in df.columns:
        # If no subject key, concatenate horizontally with a synthetic index note
        log[0]["warning"] = f"{on!r} missing in left frame — returning panel only concat attempt"
        if on not in panel.columns:
            return panel.copy(), log
        # attach with row alignment if same length
        if len(df) == len(panel):
            joined = pd.concat(
                [df.reset_index(drop=True), panel.drop(columns=[on], errors="ignore")],
                axis=1,
            )
            return joined, log
        return panel, log
    joined = df.merge(panel, on=on, how=how, suffixes=("", "_beh"))
    log[0]["joined_rows"] = len(joined)
    return joined, log


def mineable_columns(panel: pd.DataFrame) -> list[str]:
    """Numeric / coded columns suitable for mine/discover focus."""
    prefer = [
        "habit_strength",
        "compliance_rate",
        "exposure_count",
        "mean_response",
        "mean_reward",
        "outcome",
        "action_code",
        "response_positive",
        "reward_filled",
    ]
    cols = [c for c in prefer if c in panel.columns]
    # add numeric ctx_
    for c in panel.columns:
        if c.startswith("ctx_") and c not in cols:
            if pd.api.types.is_numeric_dtype(panel[c]):
                cols.append(c)
    return cols


__all__ = [
    "collection_to_panel",
    "load_panel",
    "join_traces_to_frame",
    "mineable_columns",
]
