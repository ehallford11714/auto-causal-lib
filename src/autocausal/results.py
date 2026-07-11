"""Result dataclasses (kept separate to avoid circular imports)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from autocausal.impute import ImputationReport
from autocausal.roles import ColumnRole


@dataclass
class DiscoveryResult:
    """Structured output from causal relationship discovery."""

    edges: list[dict[str, Any]]
    graph: dict[str, Any]
    roles: dict[str, ColumnRole]
    candidates: dict[str, list[str]]
    imputation: Optional[ImputationReport] = None
    method: str = "score_pc_lite"
    notes: list[str] = field(default_factory=list)
    mining: Optional[dict[str, Any]] = None
    guide: Optional[dict[str, Any]] = None
    grounding: Optional[dict[str, Any]] = None
    stability_enabled: bool = False
    bootstrap_n: int = 0
    ensemble_methods: list[str] = field(default_factory=list)
    method_edges: Optional[dict[str, list[dict[str, Any]]]] = None
    sensitivity: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        roles = {k: (v.value if hasattr(v, "value") else str(v)) for k, v in self.roles.items()}
        out: dict[str, Any] = {
            "method": self.method,
            "edges": self.edges,
            "graph": self.graph,
            "roles": roles,
            "candidates": self.candidates,
            "notes": self.notes,
            "stability_enabled": self.stability_enabled,
            "bootstrap_n": self.bootstrap_n,
        }
        if self.ensemble_methods:
            out["ensemble_methods"] = list(self.ensemble_methods)
        if self.method_edges is not None:
            out["method_edges"] = self.method_edges
        if self.imputation is not None:
            out["imputation"] = (
                self.imputation.to_dict()
                if hasattr(self.imputation, "to_dict")
                else asdict(self.imputation)
            )
        if self.mining is not None:
            out["mining"] = self.mining
        if self.guide is not None:
            out["guide"] = self.guide
        if self.grounding is not None:
            out["grounding"] = self.grounding
        if self.sensitivity is not None:
            out["sensitivity"] = self.sensitivity
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        from autocausal.report import render_markdown_report

        return render_markdown_report(self)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``.

        Prefer ``result.report()`` or ``ac.report()`` after ``discover()``.
        """
        if as_markdown:
            return self.to_markdown()
        return self.to_json()

    def to_causal_edges(self) -> list[dict[str, Any]]:
        """Export edges as CausalEdge.v1 envelopes (shared Fabric contract)."""
        from autocausal.contracts import edges_to_causal_edge_envelopes

        return edges_to_causal_edge_envelopes(self.edges)

    def to_mine_report(
        self,
        *,
        n_rows: int = 0,
        n_cols: int = 0,
        backend: str = "autocausal.mine",
    ) -> dict[str, Any]:
        """Export attached mining (if any) as a MineReport.v1 envelope."""
        from autocausal.contracts import mining_to_mine_report

        return mining_to_mine_report(
            self.mining,
            n_rows=n_rows,
            n_cols=n_cols,
            backend=backend,
        )

    def to_search_dag(self, *, soft: bool = True) -> dict[str, Any]:
        """Soft-optional CausalSearch DAG export (SearchDAG.v1 envelope)."""
        from autocausal.contracts import discovery_to_search_dag

        return discovery_to_search_dag(self, soft=soft)

    def to_fabric_bundle(
        self,
        *,
        n_rows: int = 0,
        n_cols: int = 0,
        insight: Any = None,
        source: str = "",
        notes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Assemble FabricBundle.v1 from this discovery (+ attached mining).

        Prefer ``ac.to_fabric_bundle()`` when you still have the ``AutoCausal``
        session (includes frame shape / QC). This method lets callers export
        directly from a ``DiscoveryResult``::

            result = ac.discover()
            bundle = result.to_fabric_bundle()
        """
        from autocausal.contracts import fabric_bundle

        return fabric_bundle(
            mining=self.mining,
            discovery=self,
            insight=insight,
            n_rows=n_rows,
            n_cols=n_cols,
            source=source,
            notes=list(notes or []) + list(self.notes or []),
            sensitivity=self.sensitivity,
        )


@dataclass
class AutoResult:
    """Full orchestrated auto() pipeline output."""

    discovery: DiscoveryResult
    mining: Optional[dict[str, Any]] = None
    guide: Optional[dict[str, Any]] = None
    direction_plan: Optional[dict[str, Any]] = None
    grounding: Optional[dict[str, Any]] = None
    physics: Optional[dict[str, Any]] = None
    join_log: list[dict[str, Any]] = field(default_factory=list)
    ping: Optional[dict[str, Any]] = None
    source: str = ""
    notes: list[str] = field(default_factory=list)
    sensitivity: Optional[dict[str, Any]] = None
    qc: Optional[dict[str, Any]] = None
    nlp_hints: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ping": self.ping,
            "join_log": self.join_log,
            "mining": self.mining,
            "discovery": self.discovery.to_dict(),
            "guide": self.guide,
            "direction_plan": self.direction_plan,
            "grounding": self.grounding,
            "physics": self.physics,
            "sensitivity": self.sensitivity,
            "qc": self.qc,
            "nlp_hints": self.nlp_hints,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        from autocausal.report import render_auto_markdown

        return render_auto_markdown(self)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()

    def to_causal_edges(self) -> list[dict[str, Any]]:
        """Export discovery edges as CausalEdge.v1 envelopes."""
        return self.discovery.to_causal_edges()

    def to_mine_report(
        self,
        *,
        n_rows: int = 0,
        n_cols: int = 0,
        backend: str = "autocausal.mine",
    ) -> dict[str, Any]:
        """Export mining as a MineReport.v1 envelope."""
        from autocausal.contracts import mining_to_mine_report

        return mining_to_mine_report(
            self.mining if self.mining is not None else self.discovery.mining,
            n_rows=n_rows,
            n_cols=n_cols,
            backend=backend,
        )

    def to_search_dag(self, *, soft: bool = True) -> dict[str, Any]:
        """Soft-optional CausalSearch DAG export (SearchDAG.v1 envelope)."""
        return self.discovery.to_search_dag(soft=soft)

    def to_fabric_bundle(
        self,
        *,
        n_rows: int = 0,
        n_cols: int = 0,
        insight: Any = None,
    ) -> dict[str, Any]:
        """Assemble FabricBundle.v1 (MineReport + CausalEdges + optional InsightPack)."""
        from autocausal.contracts import fabric_bundle

        return fabric_bundle(
            mining=self.mining if self.mining is not None else self.discovery.mining,
            discovery=self.discovery,
            insight=insight,
            n_rows=n_rows,
            n_cols=n_cols,
            source=self.source,
            notes=list(self.notes),
            sensitivity=self.sensitivity,
            extra={"qc": self.qc, "nlp_hints": self.nlp_hints},
        )
