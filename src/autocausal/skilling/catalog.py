"""Skill catalog + offline SkillDrill for docs and eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from autocausal.skilling.broker import SLMToolBroker
from autocausal.skilling.registry import SkillRegistry, default_skill_registry
from autocausal.skilling.surface import ToolSurface, suite_tool_surface
from autocausal.skilling.trace import SkillTrace
from autocausal.suites.director import EPISTEMIC_NOTE

__all__ = ["skill_catalog", "SkillDrill"]


def skill_catalog(
    surface: Optional[ToolSurface] = None,
    skills: Optional[SkillRegistry] = None,
) -> dict[str, Any]:
    """Enumerate skills and tools for docs / offline drills."""
    surface = surface or suite_tool_surface()
    skills = skills or default_skill_registry(surface)
    return {
        "epistemic": EPISTEMIC_NOTE,
        "skills": [s.to_dict() for s in skills.list_skills()],
        "tools": surface.schemas(),
        "n_skills": len(skills.list_ids()),
        "n_tools": len(surface.list_names()),
    }


@dataclass
class SkillDrill:
    """Run an offline rule-path drill for a skill (no HF required)."""

    skill: str = "skill:autocleanse"
    use_slm: bool = False
    broker: Optional[SLMToolBroker] = None
    last_trace: Optional[SkillTrace] = None
    notes: list[str] = field(default_factory=list)

    def run(self, df: Optional[pd.DataFrame] = None) -> SkillTrace:
        broker = self.broker or SLMToolBroker(use_slm=self.use_slm, record_traces=True)
        if df is None:
            df = pd.DataFrame(
                {
                    "treat_x": [0, 1, 0, 1, 1, 0],
                    "outcome_y": [1.0, 2.0, 1.5, 2.2, None, 1.1],
                    "instrument_z": [0.1, 0.2, 0.15, 0.3, 0.25, 0.12],
                    "const_col": [1, 1, 1, 1, 1, 1],
                }
            )
        _, results, trace = broker.run_skill(self.skill, df, text="offline skill drill")
        self.last_trace = trace
        self.notes = [
            EPISTEMIC_NOTE,
            f"skill={self.skill}",
            f"tools_ok={sum(1 for r in results if r.ok)}/{len(results)}",
        ]
        return trace

    def to_markdown(self) -> str:
        cat = skill_catalog()
        lines = [
            "# Skill drill / catalog",
            "",
            f"> {EPISTEMIC_NOTE}",
            "",
            f"**Skills:** {cat['n_skills']}  **Tools:** {cat['n_tools']}",
            "",
            "## Skills",
            "",
        ]
        for s in cat["skills"]:
            lines.append(f"- `{s['id']}` — {s['description']} ({len(s['tools'])} tools)")
        lines += ["", "## Sample tools", ""]
        for t in cat["tools"][:15]:
            lines.append(f"- `{t['name']}` ({t['suite']}): {t['description']}")
        if self.last_trace:
            lines += ["", "## Last drill trace", ""]
            lines.append(f"- skill: `{self.last_trace.skill}`")
            lines.append(f"- backend: `{self.last_trace.backend}`")
            lines.append(f"- tool_calls: {len(self.last_trace.tool_calls)}")
            lines.append(f"- outcomes: {len(self.last_trace.outcomes)}")
        lines.append("")
        return "\n".join(lines)
