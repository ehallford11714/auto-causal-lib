"""InsightReport — first-class structured insight + research-loop output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


CAVEATS = [
    "Exploratory discovery ≠ identification — edges are candidate relationships, not causal effects.",
    "Correlation is not causation; joins and missingness can bias associations.",
    "Role hypotheses (X/Y/Z) are heuristic labels for follow-up design, not proven assignments.",
    "Optional SLM narrative / experiment ideas are generative assistance only — verify before acting.",
    "Auto research-loop joins/re-mines are exploratory; A/B and IV recommendations need human design review.",
]


@dataclass
class RoleHypotheses:
    """X (treatment) / Y (outcome) / Z (instrument) / W (confounder) hypotheses."""

    treatment: list[str] = field(default_factory=list)  # X
    outcome: list[str] = field(default_factory=list)  # Y
    instrument: list[str] = field(default_factory=list)  # Z
    confounder: list[str] = field(default_factory=list)  # W

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "treatment_X": list(self.treatment),
            "outcome_Y": list(self.outcome),
            "instrument_Z": list(self.instrument),
            "confounder_W": list(self.confounder),
        }

    @classmethod
    def from_candidates(cls, candidates: Optional[dict[str, Any]]) -> "RoleHypotheses":
        c = candidates or {}
        return cls(
            treatment=list(c.get("treatment") or c.get("treatments") or []),
            outcome=list(c.get("outcome") or c.get("outcomes") or []),
            instrument=list(c.get("instrument") or c.get("instruments") or []),
            confounder=list(c.get("confounder") or c.get("confounders") or []),
        )


@dataclass
class InsightReport:
    """Structured insight report from the AutoCausal insight / research loop."""

    summary: str = ""
    key_edges: list[dict[str, Any]] = field(default_factory=list)
    role_hypotheses: RoleHypotheses = field(default_factory=RoleHypotheses)
    data_sources: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=lambda: list(CAVEATS))
    guide_backend: str = "rule"
    guide: Optional[dict[str, Any]] = None
    slm_narrative: Optional[str] = None
    slm_used: bool = False
    slm_label: str = "generative assistance (not identification)"
    nlp_hints: Optional[dict[str, Any]] = None
    mining: Optional[dict[str, Any]] = None
    discovery: Optional[dict[str, Any]] = None
    stages: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = ""
    n_rows: int = 0
    n_cols: int = 0
    experiments_recommended: list[dict[str, Any]] = field(default_factory=list)
    relationships_mined_further: list[dict[str, Any]] = field(default_factory=list)
    round_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "key_edges": list(self.key_edges),
            "role_hypotheses": self.role_hypotheses.to_dict(),
            "data_sources": list(self.data_sources),
            "caveats": list(self.caveats),
            "guide_backend": self.guide_backend,
            "guide": self.guide,
            "slm_narrative": self.slm_narrative,
            "slm_used": self.slm_used,
            "slm_label": self.slm_label if self.slm_narrative else None,
            "nlp_hints": self.nlp_hints,
            "mining": self.mining,
            "discovery": self.discovery,
            "stages": list(self.stages),
            "notes": list(self.notes),
            "source": self.source,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "experiments_recommended": list(self.experiments_recommended),
            "relationships_mined_further": list(self.relationships_mined_further),
            "round_history": list(self.round_history),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines: list[str] = [
            "# AutoCausal insight report",
            "",
            f"**Source:** `{self.source or '—'}`",
        ]
        if self.n_rows or self.n_cols:
            lines.append(f"**Shape:** {self.n_rows} rows × {self.n_cols} cols")
        lines.append(f"**Guide backend:** `{self.guide_backend}`")
        if self.stages:
            lines.append(f"**Stages:** {' → '.join(self.stages)}")
        if self.round_history:
            lines.append(f"**Research rounds:** {len(self.round_history)}")
        lines.extend(["", "> Exploratory discovery ≠ identification. " + CAVEATS[0], ""])

        lines.extend(["## Summary", "", self.summary or "_No summary._", ""])

        if self.data_sources:
            lines.extend(["## Data sources", ""])
            for s in self.data_sources:
                lines.append(f"- `{s}`")
            lines.append("")

        rh = self.role_hypotheses.to_dict()
        lines.extend(["## Role hypotheses (X / Y / Z / W)", ""])
        for label, key in (
            ("X (treatment)", "treatment_X"),
            ("Y (outcome)", "outcome_Y"),
            ("Z (instrument)", "instrument_Z"),
            ("W (confounder)", "confounder_W"),
        ):
            cols = rh.get(key) or []
            joined = ", ".join(f"`{c}`" for c in cols) if cols else "—"
            lines.append(f"- **{label}:** {joined}")
        lines.append("")

        lines.extend(["## Key edges (exploratory)", ""])
        if not self.key_edges:
            lines.append("_No edges above threshold._")
        else:
            lines.append("| source | target | type | score | confidence |")
            lines.append("|---|---|---|---:|---:|")
            for e in self.key_edges:
                lines.append(
                    f"| `{e.get('source')}` | `{e.get('target')}` | "
                    f"{e.get('type', '')} | {e.get('score', '')} | "
                    f"{e.get('confidence', '')} |"
                )
        lines.append("")

        if self.relationships_mined_further:
            lines.extend(["## Relationships mined further", ""])
            lines.append("| round | change | source | target | detail |")
            lines.append("|---:|---|---|---|---|")
            for r in self.relationships_mined_further:
                lines.append(
                    f"| {r.get('round', '')} | {r.get('change', '')} | "
                    f"`{r.get('source', '')}` | `{r.get('target', '')}` | "
                    f"{r.get('detail', '')} |"
                )
            lines.append("")

        if self.experiments_recommended:
            lines.extend(["## Experiments recommended", ""])
            for i, ex in enumerate(self.experiments_recommended, 1):
                lines.append(
                    f"{i}. **[{ex.get('priority', '')}]** `{ex.get('kind')}` — "
                    f"{ex.get('title')}"
                )
                if ex.get("rationale"):
                    lines.append(f"   - {ex['rationale']}")
                he = ex.get("hypothesized_edge") or {}
                if he:
                    lines.append(
                        f"   - Hypothesized: `{he.get('source')}` → `{he.get('target')}`"
                    )
                cols = ex.get("columns_to_collect") or []
                if cols:
                    lines.append(
                        "   - Collect: " + ", ".join(f"`{c}`" for c in cols[:8])
                    )
                pubs = ex.get("public_sources") or []
                if pubs:
                    lines.append(
                        "   - Join: " + ", ".join(f"`{p}`" for p in pubs)
                    )
            lines.append("")

        if self.round_history:
            lines.extend(["## Round history", ""])
            for rh_row in self.round_history:
                lines.append(
                    f"- **Round {rh_row.get('round')}:** "
                    f"edges={rh_row.get('n_edges')} "
                    f"(+{rh_row.get('n_new_edges', 0)} / "
                    f"-{rh_row.get('n_dropped_edges', 0)}); "
                    f"sources={rh_row.get('sources_joined') or []}; "
                    f"stop={rh_row.get('stop', False)}"
                )
                if rh_row.get("notes"):
                    for n in rh_row["notes"][:4]:
                        lines.append(f"  - {n}")
            lines.append("")

        if self.guide:
            lines.extend(["## Guide", ""])
            focus = self.guide.get("focus_columns") or []
            if focus:
                lines.append("- Focus: " + ", ".join(f"`{c}`" for c in focus[:12]))
            validate = self.guide.get("validate_edges") or []
            for e in validate[:8]:
                lines.append(
                    f"- Validate: `{e.get('source')}` → `{e.get('target')}`"
                )
            gnotes = self.guide.get("notes") or []
            for n in gnotes[:6]:
                lines.append(f"- {n}")
            lines.append("")

        if self.slm_narrative:
            lines.extend(
                [
                    "## SLM narrative",
                    "",
                    f"> Label: **{self.slm_label}** — not a substitute for identification.",
                    "",
                    self.slm_narrative,
                    "",
                ]
            )

        if self.nlp_hints:
            lines.extend(["## NLP hints (optional)", ""])
            roles = (
                (self.nlp_hints.get("roles") or {})
                if isinstance(self.nlp_hints, dict)
                else {}
            )
            for role, items in roles.items():
                if items:
                    lines.append(
                        f"- **{role}:** {', '.join(str(x) for x in items[:8])}"
                    )
            caveat = (
                self.nlp_hints.get("caveat")
                if isinstance(self.nlp_hints, dict)
                else None
            )
            if caveat:
                lines.append(f"- _{caveat}_")
            lines.append("")

        if self.notes:
            lines.extend(["## Pipeline notes", ""])
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")

        lines.extend(["## Caveats", ""])
        for c in self.caveats:
            lines.append(f"- {c}")
        lines.append("")
        return "\n".join(lines)

    def write(self, path: Union[str, Path], *, fmt: Optional[str] = None) -> Path:
        """Write report to ``path``. Format from suffix or ``fmt`` (``md`` / ``json``)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        kind = (fmt or p.suffix.lstrip(".")).lower()
        if kind in ("json",):
            p.write_text(self.to_json(), encoding="utf-8")
        else:
            if not p.suffix:
                p = p.with_suffix(".md")
            p.write_text(self.to_markdown(), encoding="utf-8")
        return p
