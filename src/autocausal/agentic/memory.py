"""Working + episodic memory with a constant token/item budget (MEM1-inspired).

Design inspiration (not a reimplementation):
- MEM1 (arXiv:2506.15841) — constant-size agent memory under long horizons
- A-MEM (arXiv:2502.12110) — evolving linked memory notes

Working memory holds the current round scratchpad; episodic memory keeps a
bounded ring of compacted episodes with optional link pointers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from uuid import uuid4

__all__ = [
    "MemoryItem",
    "WorkingMemory",
    "EpisodicMemory",
    "AgentMemory",
]


@dataclass
class MemoryItem:
    """One memory note (working or episodic)."""

    id: str
    kind: str  # working | episodic | insight | experiment | handle
    content: str
    handles: dict[str, Any] = field(default_factory=dict)
    links: list[str] = field(default_factory=list)  # A-MEM-style note links
    round: int = 0
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def make(
        cls,
        content: str,
        *,
        kind: str = "episodic",
        handles: Optional[dict[str, Any]] = None,
        links: Optional[list[str]] = None,
        round: int = 0,
        score: float = 0.0,
        meta: Optional[dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> "MemoryItem":
        return cls(
            id=id or f"mem-{uuid4().hex[:10]}",
            kind=kind,
            content=content,
            handles=dict(handles or {}),
            links=list(links or []),
            round=round,
            score=float(score),
            meta=dict(meta or {}),
        )


class WorkingMemory:
    """Scratchpad for the active round — cleared or compacted each cycle."""

    def __init__(self, *, max_items: int = 32) -> None:
        self.max_items = max(1, int(max_items))
        self.items: list[MemoryItem] = []

    def add(self, item: MemoryItem) -> MemoryItem:
        self.items.append(item)
        if len(self.items) > self.max_items:
            # Drop lowest-score oldest first
            self.items.sort(key=lambda x: (x.score, x.round))
            self.items = self.items[-self.max_items :]
        return item

    def clear(self) -> None:
        self.items.clear()

    def summary(self, *, max_chars: int = 800) -> str:
        parts = [f"[{it.kind}] {it.content}" for it in self.items]
        text = " | ".join(parts)
        return text[:max_chars]

    def to_dict(self) -> dict[str, Any]:
        return {"max_items": self.max_items, "items": [i.to_dict() for i in self.items]}


class EpisodicMemory:
    """Constant-budget episodic store (MEM1-inspired).

    Keeps at most ``max_episodes`` compacted notes. Eviction prefers lowest
    score; newest high-score episodes are retained. Optional ``links`` connect
    related notes (A-MEM-inspired evolving graph of memories).
    """

    def __init__(self, *, max_episodes: int = 16, max_chars_total: int = 12000) -> None:
        self.max_episodes = max(1, int(max_episodes))
        self.max_chars_total = max(500, int(max_chars_total))
        self.episodes: deque[MemoryItem] = deque()

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, item: MemoryItem) -> MemoryItem:
        # Link to most recent episode when present
        if self.episodes:
            prev = self.episodes[-1]
            if prev.id not in item.links:
                item.links.append(prev.id)
            if item.id not in prev.links:
                prev.links.append(item.id)
        self.episodes.append(item)
        self._enforce_budget()
        return item

    def _char_total(self) -> int:
        return sum(len(e.content) for e in self.episodes)

    def _enforce_budget(self) -> None:
        while len(self.episodes) > self.max_episodes:
            self._evict_one()
        while self._char_total() > self.max_chars_total and len(self.episodes) > 1:
            self._evict_one()

    def _evict_one(self) -> None:
        if not self.episodes:
            return
        # Evict lowest score; ties → oldest
        idx = min(range(len(self.episodes)), key=lambda i: (self.episodes[i].score, i))
        # Rotate so idx is left, then popleft
        self.episodes.rotate(-idx)
        self.episodes.popleft()
        self.episodes.rotate(idx)

    def recent(self, n: int = 5) -> list[MemoryItem]:
        items = list(self.episodes)
        return items[-max(1, n) :]

    def linked(self, item_id: str) -> list[MemoryItem]:
        by_id = {e.id: e for e in self.episodes}
        root = by_id.get(item_id)
        if root is None:
            return []
        return [by_id[i] for i in root.links if i in by_id]

    def as_context(self, *, max_chars: int = 2000) -> str:
        parts = []
        used = 0
        for ep in reversed(list(self.episodes)):
            chunk = f"[r{ep.round}/{ep.kind}] {ep.content}"
            if used + len(chunk) > max_chars:
                break
            parts.append(chunk)
            used += len(chunk)
        return "\n".join(reversed(parts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_episodes": self.max_episodes,
            "max_chars_total": self.max_chars_total,
            "n": len(self.episodes),
            "episodes": [e.to_dict() for e in self.episodes],
        }


class AgentMemory:
    """Facade combining working + episodic stores."""

    def __init__(
        self,
        *,
        max_working: int = 32,
        max_episodes: int = 16,
        max_chars_total: int = 12000,
    ) -> None:
        self.working = WorkingMemory(max_items=max_working)
        self.episodic = EpisodicMemory(
            max_episodes=max_episodes, max_chars_total=max_chars_total
        )

    def promote_working_to_episodic(
        self,
        *,
        narrative: str,
        handles: Optional[dict[str, Any]] = None,
        round: int = 0,
        score: float = 0.5,
    ) -> MemoryItem:
        item = MemoryItem.make(
            narrative or self.working.summary(),
            kind="episodic",
            handles=handles,
            round=round,
            score=score,
        )
        self.episodic.add(item)
        self.working.clear()
        return item

    def to_dict(self) -> dict[str, Any]:
        return {
            "working": self.working.to_dict(),
            "episodic": self.episodic.to_dict(),
        }
