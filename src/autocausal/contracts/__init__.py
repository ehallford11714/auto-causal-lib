"""Fabric contract envelopes (MineReport / CausalEdge / InsightPack / bundles).

Aligned with ``research/shared_contracts`` when present; otherwise defines
compatible v1 envelopes in-repo for Bridge / CausalSearch / AutoCausalOS.
"""

from __future__ import annotations

from autocausal.contracts.envelope import (
    SCHEMA_CAUSAL_EDGE,
    SCHEMA_FABRIC_BUNDLE,
    SCHEMA_INSIGHT_PACK,
    SCHEMA_MINE_REPORT,
    SCHEMA_SEARCH_DAG,
    envelope,
    now_iso,
)
from autocausal.contracts.fabric import (
    edges_to_causal_edge_envelopes,
    fabric_bundle,
    mining_to_mine_report,
    discovery_to_search_dag,
)

__all__ = [
    "SCHEMA_CAUSAL_EDGE",
    "SCHEMA_FABRIC_BUNDLE",
    "SCHEMA_INSIGHT_PACK",
    "SCHEMA_MINE_REPORT",
    "SCHEMA_SEARCH_DAG",
    "envelope",
    "now_iso",
    "edges_to_causal_edge_envelopes",
    "fabric_bundle",
    "mining_to_mine_report",
    "discovery_to_search_dag",
]
