"""BehavioralTraceStore — programmatic store + mine_behavioral_traces pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from autocausal.behavioral.bridge import soft_isolates_annotate, soft_reason_trace_hook
from autocausal.behavioral.features import engineer_trace_features, traces_to_frame
from autocausal.behavioral.loaders import (
    DEMO_IDS,
    generate_demo,
    list_demos,
    load_demo,
    load_traces_csv,
    load_traces_json,
)
from autocausal.behavioral.panel import collection_to_panel, mineable_columns
from autocausal.behavioral.report import BehavioralReport, build_behavioral_report
from autocausal.behavioral.schema import CAVEAT, TraceCollection, TraceEvent


@dataclass
class BehavioralMineResult:
    """Result of :func:`mine_behavioral_traces` / :meth:`BehavioralTraceStore.mine`."""

    panel: pd.DataFrame
    collection: TraceCollection
    report: BehavioralReport
    autocausal: Any = None  # Optional[AutoCausal]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_shape": list(self.panel.shape),
            "panel_columns": list(self.panel.columns),
            "collection": {
                "name": self.collection.name,
                "n_events": len(self.collection),
                "n_subjects": self.collection.n_subjects,
            },
            "report": self.report.to_dict(),
            "notes": list(self.notes),
            "caveat": CAVEAT,
        }

    def to_markdown(self) -> str:
        return self.report.to_markdown()


class BehavioralTraceStore:
    """In-memory / file-backed store for behavioral science traces.

    Designed for embedding in apps and notebooks::

        from autocausal.behavioral import BehavioralTraceStore

        store = BehavioralTraceStore.from_demo("habit_loop")
        panel = store.to_panel()
        result = store.mine(discover=True)
        print(result.report.to_markdown())
    """

    def __init__(self, collection: TraceCollection) -> None:
        self.collection = collection
        self._panel: Optional[pd.DataFrame] = None
        self.last_result: Optional[BehavioralMineResult] = None

    @classmethod
    def from_demo(cls, demo_id: str = "habit_loop") -> "BehavioralTraceStore":
        return cls(load_demo(demo_id))

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "BehavioralTraceStore":
        return cls(load_traces_csv(path))

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BehavioralTraceStore":
        return cls(load_traces_json(path))

    @classmethod
    def from_events(cls, events: list[TraceEvent], *, name: str = "custom") -> "BehavioralTraceStore":
        return cls(TraceCollection(name=name, events=list(events)))

    @classmethod
    def generate(cls, demo_id: str, **kwargs: Any) -> "BehavioralTraceStore":
        return cls(generate_demo(demo_id, **kwargs))

    @property
    def name(self) -> str:
        return self.collection.name

    def events_frame(self, *, engineer: bool = False) -> pd.DataFrame:
        df = traces_to_frame(self.collection)
        if engineer:
            return engineer_trace_features(df)
        return df

    def to_panel(self, *, level: str = "subject", refresh: bool = False) -> pd.DataFrame:
        if self._panel is None or refresh or level != "subject":
            panel = collection_to_panel(self.collection, level=level)
            if level == "subject":
                self._panel = panel
            return panel
        return self._panel

    def mineable_columns(self) -> list[str]:
        return mineable_columns(self.to_panel())

    def mine(
        self,
        *,
        discover: bool = False,
        min_score: float = 0.15,
        use_isolates_text: Optional[str] = None,
        **discover_kwargs: Any,
    ) -> BehavioralMineResult:
        """Build panel → AutoCausal.mine [→ discover] → BehavioralReport."""
        return mine_behavioral_traces(
            self.collection,
            discover=discover,
            min_score=min_score,
            use_isolates_text=use_isolates_text,
            **discover_kwargs,
        )

    def soft_bridges(self, text: str = "") -> dict[str, Any]:
        """Optional IntentIsolates / ReasonTrace hooks (never hard-fail)."""
        return {
            "isolates": soft_isolates_annotate(text) if text else {"ok": False, "notes": ["no text"]},
            "reason_trace": soft_reason_trace_hook(self.collection.to_records()),
        }


def mine_behavioral_traces(
    source: Union[str, TraceCollection, BehavioralTraceStore, Path],
    *,
    discover: bool = False,
    min_score: float = 0.15,
    level: str = "subject",
    use_isolates_text: Optional[str] = None,
    **discover_kwargs: Any,
) -> BehavioralMineResult:
    """End-to-end: traces → panel → mine [→ discover] → hypothesized edges.

    ``source`` may be a demo id, CSV/JSON path, TraceCollection, or store.
    """
    notes: list[str] = [CAVEAT]
    if isinstance(source, BehavioralTraceStore):
        collection = source.collection
    elif isinstance(source, TraceCollection):
        collection = source
    elif isinstance(source, Path) or (
        isinstance(source, str)
        and (source.endswith(".csv") or source.endswith(".json") or "/" in source or "\\" in source)
    ):
        path = Path(source)
        if path.suffix.lower() == ".json":
            collection = load_traces_json(path)
        else:
            collection = load_traces_csv(path)
    elif isinstance(source, str) and source in DEMO_IDS:
        collection = load_demo(source)
    else:
        # try demo id first, then csv
        try:
            collection = load_demo(str(source))
        except KeyError:
            collection = load_traces_csv(str(source))

    panel = collection_to_panel(collection, level=level)
    focus = mineable_columns(panel)
    work = panel.copy()
    if "subject_id" in work.columns and not pd.api.types.is_numeric_dtype(work["subject_id"]):
        work = work.drop(columns=["subject_id"])
    # Keep numeric / coded columns for stable mine/discover
    keep = [
        c
        for c in work.columns
        if pd.api.types.is_numeric_dtype(work[c]) or c in focus
    ]
    if len(keep) >= 2:
        work = work[keep]

    from autocausal.api import AutoCausal

    ac = AutoCausal.from_dataframe(work, source=f"behavioral:{collection.name}")
    ac.mine(min_score=min_score)
    discovery_edges: list[dict[str, Any]] = []
    if discover and len(work.columns) >= 2:
        # Prefer numeric focus columns
        numeric = [
            c
            for c in (focus or list(work.columns))
            if c in work.columns and pd.api.types.is_numeric_dtype(work[c])
        ]
        if len(numeric) >= 2:
            result = ac.discover(focus_columns=numeric, **discover_kwargs)
        else:
            result = ac.discover(**discover_kwargs)
        discovery_edges = list(result.edges or [])
        notes.append(f"discover edges={len(discovery_edges)}")

    if use_isolates_text:
        iso = soft_isolates_annotate(use_isolates_text)
        notes.append(f"isolates ok={iso.get('ok')} backend={iso.get('backend')}")

    mining_dict = ac.mining.to_dict() if ac.mining is not None and hasattr(ac.mining, "to_dict") else None
    report = build_behavioral_report(
        panel,
        collection=collection,
        discovery_edges=discovery_edges,
        mining=mining_dict,
        notes=notes,
    )
    result = BehavioralMineResult(
        panel=panel,
        collection=collection,
        report=report,
        autocausal=ac,
        notes=notes,
    )
    if isinstance(source, BehavioralTraceStore):
        source.last_result = result
        source._panel = panel
    return result


__all__ = [
    "BehavioralMineResult",
    "BehavioralTraceStore",
    "mine_behavioral_traces",
    "list_demos",
    "DEMO_IDS",
]
