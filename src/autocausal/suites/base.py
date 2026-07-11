"""Shared helpers for AutoCleanse / AutoEDA / AutoMine suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

__all__ = ["resolve_frame", "write_report"]


def resolve_frame(
    source: Any,
    *,
    table: Optional[str] = None,
    query: Optional[str] = None,
) -> tuple[pd.DataFrame, str, Any]:
    """Accept DataFrame / path / AutoCausal → (df, source_label, optional AutoCausal).

    Returns
    -------
    df, source_label, ac_or_none
    """
    from autocausal.api import AutoCausal

    if isinstance(source, AutoCausal):
        return source.df.copy(), source.source, source
    if isinstance(source, pd.DataFrame):
        return source.copy(), "dataframe", None
    if isinstance(source, (str, Path)):
        p = str(source)
        lower = p.lower()
        if lower.endswith(".csv"):
            ac = AutoCausal.from_csv(p)
            return ac.df.copy(), ac.source, ac
        if lower.endswith(".parquet"):
            ac = AutoCausal.from_parquet(p)
            return ac.df.copy(), ac.source, ac
        if "://" in p or p.startswith("sqlite:"):
            ac = AutoCausal.from_sqlalchemy(p, table=table, query=query)
            return ac.df.copy(), ac.source, ac
        ac = AutoCausal.from_csv(p)
        return ac.df.copy(), ac.source, ac
    raise TypeError(
        "source must be a pandas DataFrame, CSV/Parquet path, SQL URL, or AutoCausal"
    )


def write_report(report: Any, path: Union[str, Path], *, fmt: str = "auto") -> Path:
    """Write report via ``to_markdown`` / ``to_dict`` / ``to_json``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kind = fmt
    if kind == "auto":
        suf = out.suffix.lower()
        if suf == ".json":
            kind = "json"
        elif suf in (".md", ".markdown"):
            kind = "markdown"
        else:
            kind = "markdown"
    if kind == "json":
        if hasattr(report, "to_json"):
            text = report.to_json()
        else:
            text = json.dumps(report.to_dict(), indent=2, default=str)
    else:
        text = report.to_markdown() if hasattr(report, "to_markdown") else str(report)
    out.write_text(text, encoding="utf-8")
    return out
