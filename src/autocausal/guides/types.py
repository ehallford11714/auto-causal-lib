"""Shared types for direction-steering guide backends."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class GuideSuggestion:
    action: str
    detail: str
    priority: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuideResult:
    """Per-backend guide output (compatible with autocausal.slm.GuideResult)."""

    backend: str
    suggestions: list[GuideSuggestion] = field(default_factory=list)
    focus_columns: list[str] = field(default_factory=list)
    drop_edges: list[dict[str, str]] = field(default_factory=list)
    validate_edges: list[dict[str, str]] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    confounders: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    raw_text: str = ""
    notes: list[str] = field(default_factory=list)
    # Extended fields used by DirectionPlan merge
    treatment: list[str] = field(default_factory=list)
    outcome: list[str] = field(default_factory=list)
    boost_edges: list[dict[str, Any]] = field(default_factory=list)
    suppress_edges: list[dict[str, Any]] = field(default_factory=list)
    next_questions: list[str] = field(default_factory=list)
    related_variables: list[str] = field(default_factory=list)
    lag_hints: list[dict[str, Any]] = field(default_factory=list)
    available: bool = True
    # ML Model Hub construction hints (optional; used by autocausal.ml)
    imputer: Optional[str] = None  # torch_mlp | iterative | median
    predictor: Optional[str] = None  # torch_mlp | sklearn_rf | none
    kpi_focus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "focus_columns": self.focus_columns,
            "drop_edges": self.drop_edges,
            "validate_edges": self.validate_edges,
            "instruments": self.instruments,
            "confounders": self.confounders,
            "search_queries": self.search_queries,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "boost_edges": self.boost_edges,
            "suppress_edges": self.suppress_edges,
            "next_questions": self.next_questions,
            "related_variables": self.related_variables,
            "lag_hints": self.lag_hints,
            "raw_text": self.raw_text,
            "notes": self.notes,
            "imputer": self.imputer,
            "predictor": self.predictor,
            "kpi_focus": self.kpi_focus,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = ["# Guide suggestions", "", f"**Backend:** `{self.backend}`", ""]
        if not self.available:
            lines.append("_Backend unavailable (soft-fail)._")
            lines.append("")
        if self.focus_columns:
            lines += ["## Focus columns", ""] + [f"- `{c}`" for c in self.focus_columns] + [""]
        if self.treatment or self.outcome:
            lines.append("## Roles")
            lines.append("")
            if self.treatment:
                lines.append(f"- **Treatment:** {', '.join(f'`{t}`' for t in self.treatment)}")
            if self.outcome:
                lines.append(f"- **Outcome:** {', '.join(f'`{o}`' for o in self.outcome)}")
            lines.append("")
        if self.validate_edges or self.boost_edges:
            lines += ["## Validate / boost edges", ""]
            for e in (self.validate_edges or []) + [
                {"source": b.get("source"), "target": b.get("target")} for b in self.boost_edges
            ]:
                lines.append(f"- `{e.get('source')}` → `{e.get('target')}`")
            lines.append("")
        if self.drop_edges or self.suppress_edges:
            lines += ["## Suppress / drop", ""]
            for e in (self.drop_edges or []) + [
                {"source": s.get("source"), "target": s.get("target")} for s in self.suppress_edges
            ]:
                lines.append(f"- `{e.get('source')}` → `{e.get('target')}`")
            lines.append("")
        if self.instruments:
            lines.append(f"**Instruments (Z):** {', '.join(f'`{i}`' for i in self.instruments)}")
            lines.append("")
        if self.confounders:
            lines.append(f"**Confounders:** {', '.join(f'`{c}`' for c in self.confounders)}")
            lines.append("")
        if self.next_questions:
            lines += ["## Next questions", ""] + [f"- {q}" for q in self.next_questions] + [""]
        if self.search_queries:
            lines += ["## Search queries", ""] + [f"- {q}" for q in self.search_queries] + [""]
        if self.lag_hints:
            lines += ["## Lag / temporal hints", ""]
            for h in self.lag_hints:
                lines.append(f"- {h}")
            lines.append("")
        if self.suggestions:
            lines += ["## Actions", ""]
            for s in self.suggestions:
                lines.append(f"- [{s.priority:.2f}] **{s.action}**: {s.detail}")
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


@dataclass
class DirectionPlan:
    """Merged multi-backend plan that steers mine/discover second pass."""

    backends: list[str] = field(default_factory=list)
    focus_columns: list[str] = field(default_factory=list)
    candidate_z: list[str] = field(default_factory=list)
    treatment: list[str] = field(default_factory=list)
    outcome: list[str] = field(default_factory=list)
    confounders: list[str] = field(default_factory=list)
    boost_edges: list[dict[str, Any]] = field(default_factory=list)
    suppress_edges: list[dict[str, Any]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    next_questions: list[str] = field(default_factory=list)
    related_variables: list[str] = field(default_factory=list)
    lag_hints: list[dict[str, Any]] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    contributions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    # ML Model Hub fields (merged from guide backends)
    imputer: Optional[str] = None
    predictor: Optional[str] = None
    kpi_focus: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Direction plan",
            "",
            f"**Backends:** {', '.join(f'`{b}`' for b in self.backends) or '_none_'}",
            "",
        ]
        if self.unavailable:
            lines.append(
                f"_Soft-unavailable:_ {', '.join(f'`{u}`' for u in self.unavailable)}"
            )
            lines.append("")
        if self.focus_columns:
            lines += ["## Focus columns", ""] + [f"- `{c}`" for c in self.focus_columns] + [""]
        if self.treatment or self.outcome or self.candidate_z:
            lines.append("## Causal roles")
            lines.append("")
            if self.treatment:
                lines.append(f"- **Treatment:** {', '.join(f'`{t}`' for t in self.treatment)}")
            if self.outcome:
                lines.append(f"- **Outcome:** {', '.join(f'`{o}`' for o in self.outcome)}")
            if self.candidate_z:
                lines.append(f"- **Candidate Z:** {', '.join(f'`{z}`' for z in self.candidate_z)}")
            if self.confounders:
                lines.append(
                    f"- **Confounders:** {', '.join(f'`{c}`' for c in self.confounders)}"
                )
            lines.append("")
        if self.boost_edges:
            lines += ["## Boost edges", ""]
            for e in self.boost_edges:
                lines.append(
                    f"- `{e.get('source')}` → `{e.get('target')}` "
                    f"({e.get('reason', e.get('backend', ''))})"
                )
            lines.append("")
        if self.suppress_edges:
            lines += ["## Suppress edges", ""]
            for e in self.suppress_edges:
                lines.append(
                    f"- `{e.get('source')}` → `{e.get('target')}` "
                    f"({e.get('reason', e.get('backend', ''))})"
                )
            lines.append("")
        if self.related_variables:
            lines += ["## Related variables", ""] + [
                f"- `{v}`" for v in self.related_variables
            ] + [""]
        if self.lag_hints:
            lines += ["## Lag / temporal hints", ""]
            for h in self.lag_hints:
                lines.append(f"- `{h}`" if isinstance(h, str) else f"- {h}")
            lines.append("")
        if self.next_questions:
            lines += ["## Next questions", ""] + [f"- {q}" for q in self.next_questions] + [""]
        if self.search_queries:
            lines += ["## Search queries", ""] + [f"- {q}" for q in self.search_queries] + [""]
        if self.rationale:
            lines += ["## Rationale", ""] + [f"- {r}" for r in self.rationale] + [""]
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()

    def as_guide_result(self) -> GuideResult:
        """Project to GuideResult for second-pass discover compatibility."""
        return GuideResult(
            backend="+".join(self.backends) if self.backends else "direction_plan",
            suggestions=[
                GuideSuggestion(action="inspect_columns", detail=r, priority=0.6)
                for r in self.rationale[:12]
            ],
            focus_columns=list(self.focus_columns),
            drop_edges=[
                {"source": str(e.get("source")), "target": str(e.get("target"))}
                for e in self.suppress_edges
                if e.get("source") and e.get("target")
            ],
            validate_edges=[
                {"source": str(e.get("source")), "target": str(e.get("target"))}
                for e in self.boost_edges
                if e.get("source") and e.get("target")
            ],
            instruments=list(self.candidate_z),
            confounders=list(self.confounders),
            search_queries=list(self.search_queries),
            treatment=list(self.treatment),
            outcome=list(self.outcome),
            boost_edges=list(self.boost_edges),
            suppress_edges=list(self.suppress_edges),
            next_questions=list(self.next_questions),
            related_variables=list(self.related_variables),
            lag_hints=list(self.lag_hints),
            notes=list(self.notes) + list(self.rationale[:5]),
            imputer=self.imputer,
            predictor=self.predictor,
            kpi_focus=list(self.kpi_focus),
        )


class DirectionGuide(Protocol):
    """Protocol for direction-steering backends."""

    name: str

    def available(self) -> bool: ...

    def guide(self, context: dict[str, Any]) -> GuideResult: ...


def col_names(context: dict[str, Any]) -> list[str]:
    columns = context.get("columns") or []
    return [c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in columns]


def uniq(items: list[Any], *, limit: Optional[int] = None) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for x in items:
        key = json.dumps(x, sort_keys=True, default=str) if isinstance(x, dict) else str(x)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(x)
        if limit is not None and len(out) >= limit:
            break
    return out
