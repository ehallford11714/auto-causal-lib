"""Feature engineering on behavioral traces: lags, habit, compliance, exposure."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from autocausal.behavioral.schema import TraceCollection


def traces_to_frame(collection: TraceCollection) -> pd.DataFrame:
    """Convert a TraceCollection to a flat event-level DataFrame."""
    rows = collection.to_records()
    if not rows:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "timestamp",
                "action",
                "response",
                "reward",
                "outcome",
                "trial",
            ]
        )
    return pd.DataFrame(rows)


def engineer_trace_features(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    lag: int = 1,
) -> pd.DataFrame:
    """Add lag effects, habit proxies, compliance, and exposure counts.

    Expects event-level rows with at least subject_id, action, response.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    # Sort within subject
    if "trial" in out.columns:
        out = out.sort_values([subject_col, "trial"], kind="mergesort")
    else:
        out = out.sort_values([subject_col, "timestamp"], kind="mergesort")

    # Binary response helpers
    resp = out["response"].astype(str).str.lower()
    out["response_positive"] = resp.isin(
        {"routine", "comply", "peck", "yes", "1", "true", "engage"}
    ).astype(int)
    if "reward" in out.columns:
        out["reward_filled"] = pd.to_numeric(out["reward"], errors="coerce").fillna(0.0)
    else:
        out["reward_filled"] = 0.0

    # Exposure counts (cumulative actions per subject)
    out["exposure_count"] = out.groupby(subject_col).cumcount() + 1
    out["action_exposure"] = out.groupby([subject_col, "action"]).cumcount() + 1

    # Compliance rate (expanding mean of positive responses)
    out["compliance_rate"] = (
        out.groupby(subject_col)["response_positive"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Habit strength proxy: EWMA of positive responses
    out["habit_strength"] = (
        out.groupby(subject_col)["response_positive"]
        .transform(lambda s: s.ewm(alpha=0.3, adjust=False).mean())
    )

    # Lag features
    for col in ("response_positive", "reward_filled", "habit_strength", "outcome"):
        if col not in out.columns:
            continue
        out[f"{col}_lag{lag}"] = out.groupby(subject_col)[col].shift(lag)

    # Action one-hot (compact: top actions only via codes)
    out["action_code"] = pd.Categorical(out["action"]).codes
    out["response_code"] = pd.Categorical(out["response"]).codes

    # Pull habit from context if present
    if "ctx_habit_strength" in out.columns:
        out["ctx_habit_strength"] = pd.to_numeric(
            out["ctx_habit_strength"], errors="coerce"
        )

    return out.reset_index(drop=True)


def subject_panel(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """Aggregate engineered event features to one row per subject."""
    if df.empty:
        return pd.DataFrame()
    work = df if "habit_strength" in df.columns else engineer_trace_features(df)
    agg: dict[str, Any] = {
        "exposure_count": "max",
        "compliance_rate": "last",
        "habit_strength": "last",
        "response_positive": "mean",
        "reward_filled": "mean",
    }
    if "outcome" in work.columns:
        agg["outcome"] = "last"
    if "action_code" in work.columns:
        agg["action_code"] = "first"
    # Arm / schedule from context if present
    for c in work.columns:
        if c.startswith("ctx_") and c not in agg:
            agg[c] = "first"

    panel = work.groupby(subject_col, as_index=False).agg(agg)
    # Rename means for clarity
    panel = panel.rename(
        columns={
            "response_positive": "mean_response",
            "reward_filled": "mean_reward",
        }
    )
    return panel


def feature_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compact summary of engineered features for reports."""
    if df.empty:
        return {"n_rows": 0}
    cols = [
        c
        for c in (
            "habit_strength",
            "compliance_rate",
            "exposure_count",
            "response_positive",
            "outcome",
        )
        if c in df.columns
    ]
    summary: dict[str, Any] = {"n_rows": int(len(df)), "columns": list(df.columns)}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        summary[c] = {
            "mean": float(s.mean()) if s.notna().any() else None,
            "std": float(s.std()) if s.notna().any() else None,
        }
    return summary


__all__ = [
    "traces_to_frame",
    "engineer_trace_features",
    "subject_panel",
    "feature_summary",
]
