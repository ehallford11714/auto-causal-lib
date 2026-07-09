"""Optional DataMine adapter — soft import; AutoCausal remains primary miner."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("datamine") is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


def mine_via_datamine(
    df: pd.DataFrame,
    *,
    prefer_autocausal: bool = True,
    min_score: float = 0.15,
) -> Optional[dict[str, Any]]:
    """
    Run the shared DataMine facade on ``df``.

    When DataMine is installed it prefers AutoCausal mining internally
    (``prefer_autocausal=True``), so this is a thin cross-product entrypoint
    rather than a duplicate of ``autocausal.mining``.
    """
    if not available():
        return None
    try:
        from datamine import DataMiner

        report = DataMiner.from_df(df, prefer_autocausal=prefer_autocausal).run(
            min_score=min_score
        )
        return report.to_dict()
    except Exception as e:
        return {"ok": False, "error": str(e)}
