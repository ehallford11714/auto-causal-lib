"""SkillRegistry — named skills bundling tools + director prompt snippets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from autocausal.skilling.surface import ToolSurface, suite_tool_surface

__all__ = ["SkillDef", "SkillRegistry", "default_skill_registry"]


@dataclass
class SkillDef:
    """A named skill: allowed tools + system prompt for the director."""

    id: str
    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    system_prompt: str = ""
    suite: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillRegistry:
    """Registry of skills that constrain which tools the broker may run."""

    def __init__(self, skills: Optional[list[SkillDef]] = None) -> None:
        self._skills: dict[str, SkillDef] = {}
        for s in skills or []:
            self.register(s)

    def register(self, skill: SkillDef) -> None:
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> SkillDef:
        if skill_id not in self._skills:
            # allow short form without skill: prefix
            alt = skill_id if skill_id.startswith("skill:") else f"skill:{skill_id}"
            if alt in self._skills:
                return self._skills[alt]
            raise KeyError(f"Unknown skill: {skill_id!r}. Known: {self.list_ids()}")
        return self._skills[skill_id]

    def list_ids(self) -> list[str]:
        return sorted(self._skills.keys())

    def list_skills(self) -> list[SkillDef]:
        return [self._skills[k] for k in self.list_ids()]

    def tools_for(self, skill_id: str) -> list[str]:
        return list(self.get(skill_id).tools)

    def prompt_for(self, skill_id: str) -> str:
        return self.get(skill_id).system_prompt


def default_skill_registry(surface: Optional[ToolSurface] = None) -> SkillRegistry:
    """Built-in skills for cleanse / eda / mine / full autocausal loop."""
    surface = surface or suite_tool_surface()
    by_suite: dict[str, list[str]] = {}
    for t in surface.list_tools():
        by_suite.setdefault(t.suite, []).append(t.name)

    reg = SkillRegistry()
    reg.register(
        SkillDef(
            id="skill:autocleanse",
            name="AutoCleanse",
            description="Hygiene before causal work — missingness, coerce, impute, outliers.",
            tools=sorted(by_suite.get("autocleanse") or []),
            suite="autocleanse",
            system_prompt=(
                "You are directing AutoCleanse. Prefer structured tools over free text. "
                "Call autocleanse.* tools in a sensible order: profile → coerce → drop → "
                "outliers → impute → qc. Do not claim causal identification."
            ),
            notes=["Wraps CleanseActions; rule fallback always available."],
        )
    )
    reg.register(
        SkillDef(
            id="skill:autoeda",
            name="AutoEDA",
            description="Causal-readiness EDA — distributions, corr, roles, leakage, QC.",
            tools=sorted(by_suite.get("autoeda") or []),
            suite="autoeda",
            system_prompt=(
                "You are directing AutoEDA. Prefer autoeda.* tools. Propose role hypotheses "
                "as hypotheses only. Flag leakage; never assert identification."
            ),
        )
    )
    reg.register(
        SkillDef(
            id="skill:automine",
            name="AutoMine",
            description="Association / KPI mining + optional public join.",
            tools=sorted(by_suite.get("automine") or []),
            suite="automine",
            system_prompt=(
                "You are directing AutoMine. Prefer automine.* tools. Associations are "
                "exploratory. Optional join_public_sources only when enriching covariates."
            ),
        )
    )
    loop_tools = (
        sorted(by_suite.get("autocleanse") or [])
        + sorted(by_suite.get("autoeda") or [])
        + sorted(by_suite.get("automine") or [])
        + sorted(by_suite.get("autocausal") or [])
        + sorted(by_suite.get("insight") or [])
    )
    reg.register(
        SkillDef(
            id="skill:autocausal_loop",
            name="AutoCausal loop",
            description="Full cleanse → eda → mine → discover / insight tool bundle.",
            tools=loop_tools,
            suite="autocausal",
            system_prompt=(
                "You direct the AutoCausal auto* loop. Prefer structured tools: "
                "autocleanse.* then autoeda.* then automine.* then autocausal.discover / "
                "insight.*. Label all SLM text as generative assistance."
            ),
        )
    )
    return reg
