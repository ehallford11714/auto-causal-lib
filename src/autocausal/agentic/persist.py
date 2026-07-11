"""Persist compacted episodes / insights to disk as JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

__all__ = ["EpisodeStore", "persist_episode", "load_episodes"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EpisodeStore:
    """Append-only JSONL store for agentic loop episodes."""

    path: Path
    n_written: int = 0
    notes: list[str] = field(default_factory=list)

    @classmethod
    def open(cls, path: Union[str, Path]) -> "EpisodeStore":
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return cls(path=p)

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        row = dict(record)
        row.setdefault("ts", _utc_now())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        self.n_written += 1
        return row

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    self.notes.append("Skipped corrupt JSONL line")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "n_written": self.n_written,
            "notes": list(self.notes),
        }


def persist_episode(
    path: Union[str, Path],
    *,
    narrative: str,
    handles: Optional[dict[str, Any]] = None,
    round: int = 0,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Convenience: append one episode record to ``path``."""
    store = EpisodeStore.open(path)
    record = {
        "kind": "episode",
        "round": round,
        "narrative": narrative,
        "handles": dict(handles or {}),
        **(extra or {}),
    }
    return store.append(record)


def load_episodes(path: Union[str, Path]) -> list[dict[str, Any]]:
    return EpisodeStore.open(path).read_all()
