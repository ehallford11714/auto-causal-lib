"""Dedicated AutoMine actions — callable + SLM-selectable registry.

Library-first::

    from autocausal.suites.automine import MineActions
    MineActions.mine_associations(df)
    print(MineActions.list())
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import pandas as pd

from autocausal.suites.action_protocol import ActionRegistry, ActionResult

__all__ = [
    "MINE_REGISTRY",
    "MineActions",
    "mine_associations",
    "mine_kpi_hints",
    "join_public_sources",
    "mine_behavioral",
    "to_mine_report",
    "rank_candidates",
]

MINE_REGISTRY = ActionRegistry("automine")


def mine_associations(
    df: pd.DataFrame,
    *,
    min_score: float = 0.15,
) -> ActionResult:
    """Run ``autocausal.mining.mine`` for associations + column profiles."""
    from autocausal.mining import mine

    report = mine(df, min_score=min_score)
    return ActionResult(
        name="mine_associations",
        payload={
            "columns": list(getattr(report, "columns", None) or []),
            "associations": list(getattr(report, "associations", None) or []),
            "suggestions": list(getattr(report, "suggestions", None) or []),
            "kpis": list(getattr(report, "kpis", None) or []),
            "n_rows": int(getattr(report, "n_rows", len(df)) or len(df)),
            "n_cols": int(getattr(report, "n_cols", df.shape[1]) or df.shape[1]),
            "_mining": report,
        },
        notes=["Associations are exploratory — not causal effects."],
        n_affected=len(getattr(report, "associations", None) or []),
    )


MINE_REGISTRY.register("mine_associations", mine_associations)


def mine_kpi_hints(
    df: pd.DataFrame,
    *,
    kpi_focus: Optional[Sequence[str]] = None,
    mining: Optional[Any] = None,
) -> ActionResult:
    """Suggest KPI-like columns (name heuristics + mining KPIs)."""
    kpis: list[str] = []
    if mining is not None:
        kpis.extend(list(getattr(mining, "kpis", None) or []))
    hints = (
        "revenue",
        "sales",
        "conversion",
        "churn",
        "retention",
        "ltv",
        "ctr",
        "roi",
        "profit",
        "outcome",
    )
    for c in df.columns:
        cl = str(c).lower()
        if any(h in cl for h in hints) and str(c) not in kpis:
            kpis.append(str(c))
    for k in kpi_focus or []:
        if k in df.columns and k not in kpis:
            kpis.insert(0, k)
    return ActionResult(
        name="mine_kpi_hints",
        payload={"kpis": kpis},
        notes=["KPI hints are name/heuristic based."],
        n_affected=len(kpis),
    )


MINE_REGISTRY.register("mine_kpi_hints", mine_kpi_hints)


def join_public_sources(
    df: pd.DataFrame,
    *,
    sources: Optional[Union[str, Sequence[str]]] = None,
    allow_network: bool = False,
) -> ActionResult:
    """Optional public-suite join (soft-fail)."""
    if not sources:
        return ActionResult(
            name="join_public_sources",
            frame=df.copy(),
            notes=["No public sources requested."],
        )
    if isinstance(sources, str):
        ids = [x.strip() for x in sources.split(",") if x.strip()]
    else:
        ids = list(sources)
    try:
        from autocausal.public_suite import join_public_frames

        joined, log = join_public_frames(
            df,
            ids if len(ids) > 1 else ids[0],
            allow_network=allow_network,
        )
        return ActionResult(
            name="join_public_sources",
            frame=joined,
            payload={"join_log": log if isinstance(log, list) else [log], "sources": ids},
            notes=[f"Joined public: {ids}"],
            n_affected=len(ids),
        )
    except Exception as e:
        return ActionResult(
            name="join_public_sources",
            frame=df.copy(),
            warnings=[f"Public join soft-fail: {type(e).__name__}: {e}"],
        )


MINE_REGISTRY.register("join_public_sources", join_public_sources)


def mine_behavioral(
    df: pd.DataFrame,
    *,
    discover: bool = False,
) -> ActionResult:
    """Soft behavioral-trace mining (no-op if module/data unavailable)."""
    try:
        from autocausal.behavioral import BehavioralTraceStore

        # Prefer demo store when frame lacks behavioral schema
        store = BehavioralTraceStore.demo() if hasattr(BehavioralTraceStore, "demo") else None
        if store is None:
            return ActionResult(
                name="mine_behavioral",
                notes=["BehavioralTraceStore.demo unavailable — skipped."],
            )
        result = store.mine(discover=discover) if hasattr(store, "mine") else None
        payload = result.to_dict() if result is not None and hasattr(result, "to_dict") else {"ok": True}
        return ActionResult(
            name="mine_behavioral",
            payload={"behavioral": payload},
            notes=["Behavioral mining is soft-optional and exploratory."],
            n_affected=1,
        )
    except Exception as e:
        return ActionResult(
            name="mine_behavioral",
            warnings=[f"Behavioral soft-fail: {type(e).__name__}: {e}"],
            notes=["Behavioral module not available — skipped."],
        )


MINE_REGISTRY.register("mine_behavioral", mine_behavioral)


def rank_candidates(
    df: pd.DataFrame,
    *,
    associations: Optional[Sequence[dict[str, Any]]] = None,
    suggestions: Optional[Sequence[dict[str, Any]]] = None,
    focus: Optional[Sequence[str]] = None,
    kpis: Optional[Sequence[str]] = None,
) -> ActionResult:
    """Rank association / relationship candidates by score + focus overlap."""
    focus_set = set(focus or []) | set(kpis or [])
    assocs = list(associations or [])
    ranked = sorted(
        assocs,
        key=lambda a: (
            0 if {a.get("a"), a.get("b")} & focus_set else 1,
            -float(a.get("score") or 0),
        ),
    )
    sugg = list(suggestions or [])
    sugg_ranked = sorted(
        sugg,
        key=lambda s: -float(s.get("score") or 0),
    )
    return ActionResult(
        name="rank_candidates",
        payload={
            "associations": ranked,
            "suggestions": sugg_ranked,
            "ranked_candidates": ranked[:20] + sugg_ranked[:10],
        },
        n_affected=len(ranked) + len(sugg_ranked),
    )


MINE_REGISTRY.register("rank_candidates", rank_candidates)


def to_mine_report(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[dict[str, Any]]] = None,
    associations: Optional[Sequence[dict[str, Any]]] = None,
    suggestions: Optional[Sequence[dict[str, Any]]] = None,
    kpis: Optional[Sequence[str]] = None,
    backend: str = "autocausal.mining",
) -> ActionResult:
    """Build Fabric MineReport.v1 envelope from mining artifacts."""
    from autocausal.contracts import mining_to_mine_report

    class _Tmp:
        def to_dict(self) -> dict[str, Any]:
            return {
                "columns": list(columns or []),
                "associations": list(associations or []),
                "suggestions": list(suggestions or []),
                "kpis": list(kpis or []),
                "notes": [],
            }

    env = mining_to_mine_report(
        _Tmp(),
        n_rows=len(df),
        n_cols=df.shape[1],
        backend=backend,
        extra_meta={"suite": "AutoMineSuite"},
    )
    return ActionResult(
        name="to_mine_report",
        payload={"fabric_envelope": env},
        n_affected=1,
    )


MINE_REGISTRY.register("to_mine_report", to_mine_report)


class MineActions:
    """Namespace for dedicated mine actions + registry."""

    registry = MINE_REGISTRY
    mine_associations = staticmethod(mine_associations)
    mine_kpi_hints = staticmethod(mine_kpi_hints)
    join_public_sources = staticmethod(join_public_sources)
    mine_behavioral = staticmethod(mine_behavioral)
    rank_candidates = staticmethod(rank_candidates)
    to_mine_report = staticmethod(to_mine_report)

    @classmethod
    def list(cls) -> list[str]:
        return cls.registry.list()

    @classmethod
    def run(cls, name: str, df: pd.DataFrame, **kwargs: Any) -> ActionResult:
        return cls.registry.run(name, df, **kwargs)

    @classmethod
    def default_sequence(cls) -> list[str]:
        return [
            "mine_associations",
            "mine_kpi_hints",
            "rank_candidates",
            "to_mine_report",
        ]
