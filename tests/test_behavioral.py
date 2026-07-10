"""Library API tests for autocausal.behavioral (offline demos)."""

from __future__ import annotations

import pytest

from autocausal.behavioral import (
    BehavioralTraceStore,
    TraceEvent,
    list_demos,
    load_demo,
    mine_behavioral_traces,
)


def test_list_demos():
    demos = list_demos()
    ids = {d["id"] for d in demos}
    assert "habit_loop" in ids
    assert "nudge_ab" in ids
    assert "reinforcement_schedule" in ids


def test_load_demo_and_panel():
    coll = load_demo("habit_loop")
    assert len(coll) > 0
    assert coll.n_subjects >= 2
    store = BehavioralTraceStore.from_demo("habit_loop")
    panel = store.to_panel()
    assert len(panel) == coll.n_subjects
    assert "habit_strength" in panel.columns or "compliance_rate" in panel.columns
    events = store.events_frame(engineer=True)
    assert "exposure_count" in events.columns
    assert "response_positive_lag1" in events.columns or "habit_strength" in events.columns


def test_mine_behavioral_traces_habit_loop():
    result = mine_behavioral_traces("habit_loop", discover=True)
    assert result.panel is not None
    assert len(result.report.edges) >= 1
    kinds = {e.kind for e in result.report.edges}
    assert kinds & {"habit_outcome", "stimulus_response", "exposure_compliance", "reward_response"}
    assert "exploratory" in result.report.caveat.lower() or "not identified" in result.report.caveat.lower()
    md = result.to_markdown()
    assert "habit_loop" in md or "Behavioral" in md


def test_mine_nudge_ab_without_discover():
    result = mine_behavioral_traces("nudge_ab", discover=False)
    assert result.collection.name == "nudge_ab"
    assert result.report.edges  # priors + corr edges


def test_custom_events_store():
    store = BehavioralTraceStore.from_events(
        [
            TraceEvent("S1", "0", "cue", "routine", reward=1.0, outcome=0.9, trial=0),
            TraceEvent("S1", "1", "cue", "routine", reward=1.0, outcome=0.95, trial=1),
            TraceEvent("S2", "0", "cue", "skip", reward=0.0, outcome=0.1, trial=0),
            TraceEvent("S2", "1", "cue", "skip", reward=0.0, outcome=0.15, trial=1),
        ],
        name="tiny",
    )
    panel = store.to_panel()
    assert len(panel) == 2
    result = store.mine(discover=False)
    assert result.report.trace_name == "tiny"


def test_top_level_lazy_exports():
    import autocausal as ac

    assert ac.BehavioralTraceStore is BehavioralTraceStore
    assert ac.mine_behavioral_traces is mine_behavioral_traces


def test_autocausal_mine_behavioral_traces_facade():
    from autocausal import AutoCausal

    result = AutoCausal.mine_behavioral_traces("reinforcement_schedule", discover=False)
    assert result.report.edges
    assert result.autocausal is not None


def test_cli_behavioral_list():
    from autocausal.cli import main

    assert main(["behavioral", "list"]) == 0


def test_cli_nlp_extract():
    from autocausal.cli import main

    assert main(["nlp", "extract", "--text", "treatment leads to outcome"]) == 0
