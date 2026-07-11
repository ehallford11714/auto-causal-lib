"""Convert AutoCausal artifacts into shared Fabric contract envelopes."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from autocausal.contracts.envelope import (
    SCHEMA_CAUSAL_EDGE,
    SCHEMA_FABRIC_BUNDLE,
    SCHEMA_INSIGHT_PACK,
    SCHEMA_MINE_REPORT,
    SCHEMA_SEARCH_DAG,
    envelope,
)


def mining_to_mine_report(
    mining: Any,
    *,
    n_rows: int = 0,
    n_cols: int = 0,
    backend: str = "autocausal.mine",
    extra_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a MineReport.v1 envelope from a MiningReport or dict."""
    if mining is None:
        payload: dict[str, Any] = {
            "backend": backend,
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "columns": [],
            "associations": [],
            "suggestions": [],
            "kpis": [],
            "kpi_names": [],
            "insights": [],
            "notes": ["No mining report available."],
            "meta": dict(extra_meta or {}),
        }
        return envelope(SCHEMA_MINE_REPORT, payload)

    data = mining.to_dict() if hasattr(mining, "to_dict") else dict(mining)
    columns = list(data.get("columns") or [])
    associations = list(data.get("associations") or [])
    suggestions = list(data.get("suggestions") or [])
    kpi_names = list(data.get("kpis") or [])
    kpis = [{"id": str(k), "kind": "suggested_kpi"} for k in kpi_names]
    notes = list(data.get("notes") or [])
    if n_rows <= 0 and columns:
        # best-effort: mining profile may not carry n_rows
        pass
    payload = {
        "backend": backend,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols or len(columns)),
        "columns": columns,
        "associations": associations,
        "suggestions": suggestions,
        "kpis": kpis,
        "kpi_names": [str(k) for k in kpi_names],
        "morphemes": [],
        "insights": [str(s.get("reason", "")) for s in suggestions[:10] if isinstance(s, dict)],
        "notes": notes,
        "meta": dict(extra_meta or {}),
    }
    return envelope(SCHEMA_MINE_REPORT, payload)


def edges_to_causal_edge_envelopes(
    edges: Sequence[dict[str, Any]],
    *,
    default_relation: str = "causes",
) -> list[dict[str, Any]]:
    """Map discovery edges → list of CausalEdge.v1 envelopes."""
    out: list[dict[str, Any]] = []
    for i, e in enumerate(edges or []):
        src = str(e.get("source") or e.get("a") or "")
        tgt = str(e.get("target") or e.get("b") or "")
        if not src or not tgt:
            continue
        conf = e.get("confidence", e.get("score"))
        try:
            conf_f = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf_f = None
        if conf_f is not None:
            conf_f = max(0.0, min(1.0, conf_f))
        relation = str(e.get("relation") or e.get("type") or default_relation)
        if relation in (
            "association",
            "score_pc_lite",
            "corr_skeleton",
            "mi",
            "mi_binned",
            "mi_stub",
            "grail_boost",
        ):
            relation = default_relation
        if relation == "iv_2sls":
            relation = "instruments"
        payload: dict[str, Any] = {
            "source": src,
            "target": tgt,
            "relation": relation,
            "rationale": str(e.get("rationale") or e.get("orientation") or e.get("type") or ""),
            "evidence_ids": list(e.get("evidence_ids") or [f"edge:{i}"]),
            "meta": {
                k: v
                for k, v in e.items()
                if k
                not in (
                    "source",
                    "target",
                    "a",
                    "b",
                    "confidence",
                    "relation",
                    "rationale",
                    "evidence_ids",
                )
            },
        }
        if conf_f is not None:
            payload["confidence"] = conf_f
        out.append(envelope(SCHEMA_CAUSAL_EDGE, payload))
    return out


def discovery_to_search_dag(
    discovery: Any,
    *,
    soft: bool = True,
) -> dict[str, Any]:
    """Export DiscoveryResult as a CausalSearch-compatible SearchDAG.v1 envelope.

    Soft-optional: never requires ``causalsearch`` installed. When soft=True
    (default), always returns an in-repo envelope. Callers may try importing
    causalsearch separately.
    """
    if discovery is None:
        payload: dict[str, Any] = {
            "nodes": [],
            "edges": [],
            "method": "none",
            "notes": ["No discovery result."],
            "soft": soft,
        }
        return envelope(SCHEMA_SEARCH_DAG, payload)

    data = discovery.to_dict() if hasattr(discovery, "to_dict") else dict(discovery)
    graph = data.get("graph") or {}
    nodes_raw = graph.get("nodes") or []
    nodes: list[dict[str, Any]] = []
    for n in nodes_raw:
        if isinstance(n, dict):
            nodes.append({"id": str(n.get("id", "")), "role": n.get("role"), "meta": n})
        else:
            nodes.append({"id": str(n)})
    edges_out: list[dict[str, Any]] = []
    for e in data.get("edges") or []:
        edges_out.append(
            {
                "source": str(e.get("source", "")),
                "target": str(e.get("target", "")),
                "score": e.get("score"),
                "confidence": e.get("confidence"),
                "stability": e.get("stability"),
                "type": e.get("type"),
                "methods": e.get("methods"),
            }
        )
    # Soft try: annotate if causalsearch is importable (no hard dep)
    causalsearch_available = False
    try:
        from importlib.util import find_spec

        causalsearch_available = find_spec("causalsearch") is not None
    except Exception:
        causalsearch_available = False

    payload = {
        "nodes": nodes,
        "edges": edges_out,
        "candidates": data.get("candidates") or {},
        "method": data.get("method") or "score_pc_lite",
        "notes": list(data.get("notes") or [])
        + [
            "SearchDAG.v1 is a soft export for CausalSearch; not a formal ID graph.",
        ],
        "soft": soft,
        "causalsearch_available": causalsearch_available,
    }
    return envelope(SCHEMA_SEARCH_DAG, payload)


def fabric_bundle(
    *,
    mining: Any = None,
    discovery: Any = None,
    insight: Any = None,
    n_rows: int = 0,
    n_cols: int = 0,
    source: str = "",
    notes: Optional[list[str]] = None,
    sensitivity: Any = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble a FabricBundle.v1 with MineReport + CausalEdges + optional InsightPack."""
    mine_env = mining_to_mine_report(mining, n_rows=n_rows, n_cols=n_cols)
    edges = []
    if discovery is not None:
        edge_list = discovery.edges if hasattr(discovery, "edges") else (discovery.get("edges") or [])
        edges = edges_to_causal_edge_envelopes(edge_list)
    insight_env = None
    if insight is not None:
        if isinstance(insight, dict) and insight.get("schema") == SCHEMA_INSIGHT_PACK:
            insight_env = insight
        else:
            idata = insight.to_dict() if hasattr(insight, "to_dict") else dict(insight)
            insight_env = envelope(
                SCHEMA_INSIGHT_PACK,
                {
                    "ok": bool(idata.get("ok", True)),
                    "usecase": idata.get("usecase"),
                    "text_hint": idata.get("text_hint") or idata.get("text"),
                    "backend": str(idata.get("backend") or "autocausal"),
                    "insights": idata.get("insights") or idata,
                    "markdown": idata.get("markdown") or "",
                    "meta": idata.get("meta") or {},
                },
            )
    sens = None
    if sensitivity is not None:
        sens = sensitivity.to_dict() if hasattr(sensitivity, "to_dict") else sensitivity

    payload: dict[str, Any] = {
        "source": source,
        "mine_report": mine_env,
        "causal_edges": edges,
        "insight_pack": insight_env,
        "search_dag": discovery_to_search_dag(discovery) if discovery is not None else None,
        "sensitivity": sens,
        "notes": list(notes or [])
        + [
            "FabricBundle.v1 is an interoperability envelope — not a claim of identification.",
        ],
        "meta": dict(extra or {}),
    }
    return envelope(SCHEMA_FABRIC_BUNDLE, payload)
