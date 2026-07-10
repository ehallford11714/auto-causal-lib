"""Minimal v1 envelope helpers for Causal Fabric contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA_MINE_REPORT = "MineReport.v1"
SCHEMA_CAUSAL_EDGE = "CausalEdge.v1"
SCHEMA_INSIGHT_PACK = "InsightPack.v1"
SCHEMA_FABRIC_BUNDLE = "FabricBundle.v1"
SCHEMA_SEARCH_DAG = "SearchDAG.v1"

PRODUCED_BY = "autocausal"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def envelope(
    schema: str,
    payload: dict[str, Any],
    *,
    produced_by: str = PRODUCED_BY,
    produced_at: Optional[str] = None,
) -> dict[str, Any]:
    """Build ``{schema, produced_by, produced_at, payload}`` envelope."""
    return {
        "schema": schema,
        "produced_by": produced_by,
        "produced_at": produced_at or now_iso(),
        "payload": payload,
    }
