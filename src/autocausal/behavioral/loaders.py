"""Loaders / generators for bundled behavioral demo traces (offline)."""

from __future__ import annotations

import csv
import json
from importlib import resources
from pathlib import Path
from typing import Any, Optional, Union

from autocausal.behavioral.schema import TraceCollection, TraceEvent


DEMO_IDS = ("habit_loop", "nudge_ab", "reinforcement_schedule")

_DEMO_META: dict[str, dict[str, str]] = {
    "habit_loop": {
        "description": "Cue → routine → reward habit loop stub (Duhigg/Wood-style).",
        "filename": "habit_loop.csv",
    },
    "nudge_ab": {
        "description": "Nudge A/B exposure with compliance and distal outcome.",
        "filename": "nudge_ab.csv",
    },
    "reinforcement_schedule": {
        "description": "Fixed/variable reinforcement schedule stub with response rates.",
        "filename": "reinforcement_schedule.csv",
    },
}


def list_demos() -> list[dict[str, Any]]:
    """List bundled behavioral demo trace ids."""
    return [
        {
            "id": did,
            "description": _DEMO_META[did]["description"],
            "filename": _DEMO_META[did]["filename"],
            "access": "bundled",
        }
        for did in DEMO_IDS
    ]


def _package_data_path(filename: str) -> Path:
    """Resolve a file under autocausal.data.behavioral."""
    try:
        root = resources.files("autocausal.data.behavioral")
        return Path(str(root.joinpath(filename)))
    except Exception:
        # Fallback relative to this file
        return (
            Path(__file__).resolve().parents[1]
            / "data"
            / "behavioral"
            / filename
        )


def _parse_context(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    s = str(raw).strip()
    if not s:
        return {}
    if s.startswith("{"):
        try:
            return dict(json.loads(s))
        except json.JSONDecodeError:
            return {"raw": s}
    # key=value;key2=value2
    out: dict[str, Any] = {}
    for part in s.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = _coerce(v.strip())
    return out


def _coerce(v: str) -> Any:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _float_or_none(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int_or_none(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def load_traces_csv(path: Union[str, Path]) -> TraceCollection:
    """Load TraceEvents from a CSV with standard columns."""
    path = Path(path)
    events: list[TraceEvent] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(
                TraceEvent(
                    subject_id=str(row.get("subject_id", "")),
                    timestamp=str(row.get("timestamp", "")),
                    action=str(row.get("action", "")),
                    response=str(row.get("response", "")),
                    context=_parse_context(row.get("context")),
                    reward=_float_or_none(row.get("reward")),
                    outcome=_float_or_none(row.get("outcome")),
                    trial=_int_or_none(row.get("trial")),
                )
            )
    return TraceCollection(
        name=path.stem,
        events=events,
        description=f"Loaded from {path.name}",
        notes=["offline CSV load"],
    )


def load_traces_json(path: Union[str, Path]) -> TraceCollection:
    """Load TraceCollection from JSON (list of events or {name, events})."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        name = path.stem
        description = ""
        raw_events = payload
        notes: list[str] = []
    else:
        name = str(payload.get("name", path.stem))
        description = str(payload.get("description", ""))
        raw_events = payload.get("events") or []
        notes = list(payload.get("notes") or [])
    events: list[TraceEvent] = []
    for row in raw_events:
        events.append(
            TraceEvent(
                subject_id=str(row.get("subject_id", "")),
                timestamp=str(row.get("timestamp", "")),
                action=str(row.get("action", "")),
                response=str(row.get("response", "")),
                context=_parse_context(row.get("context")),
                reward=_float_or_none(row.get("reward")),
                outcome=_float_or_none(row.get("outcome")),
                trial=_int_or_none(row.get("trial")),
            )
        )
    return TraceCollection(name=name, events=events, description=description, notes=notes)


def load_demo(demo_id: str) -> TraceCollection:
    """Load a bundled offline demo by id (habit_loop, nudge_ab, …)."""
    if demo_id not in _DEMO_META:
        known = ", ".join(DEMO_IDS)
        raise KeyError(f"Unknown demo {demo_id!r}. Known: {known}")
    meta = _DEMO_META[demo_id]
    path = _package_data_path(meta["filename"])
    if not path.exists():
        # Generate in-memory if file missing (dev safety)
        return generate_demo(demo_id)
    coll = load_traces_csv(path)
    coll.name = demo_id
    coll.description = meta["description"]
    coll.notes = list(coll.notes) + ["bundled offline demo"]
    return coll


def generate_demo(demo_id: str, *, n_subjects: int = 12, seed: int = 17) -> TraceCollection:
    """Deterministic offline generator for demos (no network)."""
    import random

    rng = random.Random(seed)
    if demo_id == "habit_loop":
        events: list[TraceEvent] = []
        for s in range(n_subjects):
            sid = f"S{s:03d}"
            habit = 0.2
            for t in range(8):
                cue = "cue_morning" if t % 2 == 0 else "cue_evening"
                # habit strength raises routine probability
                do_routine = rng.random() < min(0.95, 0.3 + habit)
                response = "routine" if do_routine else "skip"
                reward = 1.0 if do_routine and rng.random() < 0.7 else 0.0
                if do_routine:
                    habit = min(1.0, habit + 0.08)
                else:
                    habit = max(0.0, habit - 0.03)
                outcome = round(0.3 + 0.6 * habit + rng.uniform(-0.05, 0.05), 3)
                events.append(
                    TraceEvent(
                        subject_id=sid,
                        timestamp=f"2024-01-{(t + 1):02d}T08:00:00",
                        action=cue,
                        response=response,
                        context={"habit_strength": round(habit, 3), "day": t},
                        reward=reward,
                        outcome=outcome,
                        trial=t,
                    )
                )
        return TraceCollection(
            name="habit_loop",
            events=events,
            description=_DEMO_META["habit_loop"]["description"],
            notes=["generated offline"],
        )

    if demo_id == "nudge_ab":
        events = []
        for s in range(n_subjects):
            sid = f"N{s:03d}"
            arm = "nudge_A" if s % 2 == 0 else "nudge_B"
            base = 0.35 if arm == "nudge_A" else 0.55
            for t in range(5):
                comply = 1 if rng.random() < base else 0
                response = "comply" if comply else "ignore"
                reward = float(comply)
                outcome = round(base * 0.8 + 0.2 * comply + rng.uniform(-0.05, 0.05), 3)
                events.append(
                    TraceEvent(
                        subject_id=sid,
                        timestamp=f"2024-02-{(t + 1):02d}T12:00:00",
                        action=arm,
                        response=response,
                        context={"arm": arm, "prior_exposure": t},
                        reward=reward,
                        outcome=outcome,
                        trial=t,
                    )
                )
        return TraceCollection(
            name="nudge_ab",
            events=events,
            description=_DEMO_META["nudge_ab"]["description"],
            notes=["generated offline"],
        )

    if demo_id == "reinforcement_schedule":
        events = []
        for s in range(n_subjects):
            sid = f"R{s:03d}"
            schedule = "fixed_ratio" if s % 2 == 0 else "variable_ratio"
            resp_rate = 0.4
            for t in range(10):
                action = f"stim_{schedule}"
                respond = rng.random() < resp_rate
                response = "peck" if respond else "omit"
                # VR slightly higher asymptotic rate
                if schedule == "variable_ratio":
                    rewarded = respond and rng.random() < 0.5
                    resp_rate = min(0.95, resp_rate + (0.06 if rewarded else -0.01))
                else:
                    rewarded = respond and (t % 3 == 0)
                    resp_rate = min(0.9, resp_rate + (0.04 if rewarded else -0.02))
                reward = 1.0 if rewarded else 0.0
                outcome = round(resp_rate + rng.uniform(-0.05, 0.05), 3)
                events.append(
                    TraceEvent(
                        subject_id=sid,
                        timestamp=str(t),
                        action=action,
                        response=response,
                        context={"schedule": schedule, "resp_rate": round(resp_rate, 3)},
                        reward=reward,
                        outcome=outcome,
                        trial=t,
                    )
                )
        return TraceCollection(
            name="reinforcement_schedule",
            events=events,
            description=_DEMO_META["reinforcement_schedule"]["description"],
            notes=["generated offline"],
        )

    raise KeyError(f"Unknown demo {demo_id!r}")


__all__ = [
    "DEMO_IDS",
    "list_demos",
    "load_demo",
    "load_traces_csv",
    "load_traces_json",
    "generate_demo",
]
