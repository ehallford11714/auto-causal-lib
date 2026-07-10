"""Behavioral science traces as structured event sequences for causal discovery.

Library-first usage (apps / notebooks)::

    from autocausal.behavioral import (
        BehavioralTraceStore,
        mine_behavioral_traces,
        list_demos,
    )

    store = BehavioralTraceStore.from_demo("habit_loop")
    panel = store.to_panel()
    result = mine_behavioral_traces("habit_loop", discover=True)
    print(result.report.to_markdown())

Epistemic honesty: hypothesized stimulus→response / habit→outcome edges are
exploratory — not identified causation.
"""

from __future__ import annotations

from autocausal.behavioral.bridge import (
    isolates_available,
    reason_trace_available,
    soft_isolates_annotate,
    soft_reason_trace_hook,
)
from autocausal.behavioral.features import (
    engineer_trace_features,
    feature_summary,
    subject_panel,
    traces_to_frame,
)
from autocausal.behavioral.loaders import (
    DEMO_IDS,
    generate_demo,
    list_demos,
    load_demo,
    load_traces_csv,
    load_traces_json,
)
from autocausal.behavioral.panel import (
    collection_to_panel,
    join_traces_to_frame,
    load_panel,
    mineable_columns,
)
from autocausal.behavioral.report import (
    BehavioralEdge,
    BehavioralReport,
    build_behavioral_report,
    hypothesize_edges,
)
from autocausal.behavioral.schema import CAVEAT, TraceCollection, TraceEvent
from autocausal.behavioral.store import (
    BehavioralMineResult,
    BehavioralTraceStore,
    mine_behavioral_traces,
)

__all__ = [
    "CAVEAT",
    "DEMO_IDS",
    "BehavioralEdge",
    "BehavioralMineResult",
    "BehavioralReport",
    "BehavioralTraceStore",
    "TraceCollection",
    "TraceEvent",
    "build_behavioral_report",
    "collection_to_panel",
    "engineer_trace_features",
    "feature_summary",
    "generate_demo",
    "hypothesize_edges",
    "isolates_available",
    "join_traces_to_frame",
    "list_demos",
    "load_demo",
    "load_panel",
    "load_traces_csv",
    "load_traces_json",
    "mine_behavioral_traces",
    "mineable_columns",
    "reason_trace_available",
    "soft_isolates_annotate",
    "soft_reason_trace_hook",
    "subject_panel",
    "traces_to_frame",
]
