"""Behavioral science trace schema — structured event sequences for causal mining.

Traces are exploratory behavioral records, not identified causal evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


CAVEAT = (
    "Behavioral traces and hypothesized edges are exploratory. "
    "Stimulus→response / habit→outcome links are not identified causation."
)


@dataclass
class TraceEvent:
    """One behavioral event in a subject timeline.

    Fields
    ------
    subject_id:
        Subject / unit identifier.
    timestamp:
        ISO-8601 string or ordinal time index (stored as string for CSV friendliness).
    action:
        Action or stimulus label (e.g. cue, nudge_A, reward).
    response:
        Observed response (e.g. habit_act, comply, ignore).
    context:
        Optional context covariates (dict serialized in loaders).
    reward:
        Optional reward / reinforcement signal.
    outcome:
        Optional distal outcome (e.g. retention, wellbeing).
    """

    subject_id: str
    timestamp: str
    action: str
    response: str
    context: dict[str, Any] = field(default_factory=dict)
    reward: Optional[float] = None
    outcome: Optional[float] = None
    trial: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class TraceCollection:
    """A named collection of behavioral events plus metadata."""

    name: str
    events: list[TraceEvent]
    description: str = ""
    domain: str = "behavioral"
    notes: list[str] = field(default_factory=list)
    caveat: str = CAVEAT

    def __len__(self) -> int:
        return len(self.events)

    @property
    def n_subjects(self) -> int:
        return len({e.subject_id for e in self.events})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "n_events": len(self.events),
            "n_subjects": self.n_subjects,
            "events": [e.to_dict() for e in self.events],
            "notes": list(self.notes),
            "caveat": self.caveat,
        }

    def to_records(self) -> list[dict[str, Any]]:
        """Flat records suitable for DataFrame construction."""
        rows: list[dict[str, Any]] = []
        for e in self.events:
            row: dict[str, Any] = {
                "subject_id": e.subject_id,
                "timestamp": e.timestamp,
                "action": e.action,
                "response": e.response,
                "reward": e.reward,
                "outcome": e.outcome,
                "trial": e.trial,
            }
            for k, v in (e.context or {}).items():
                row[f"ctx_{k}"] = v
            rows.append(row)
        return rows


__all__ = ["CAVEAT", "TraceEvent", "TraceCollection"]
