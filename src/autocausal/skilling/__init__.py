"""SLM skilling / tooling surface — structured tools wrapping suite actions.

Library-first::

    from autocausal.skilling import ToolSurface, SkillRegistry, SLMToolBroker, suite_tool_surface

    surface = suite_tool_surface()
    broker = SLMToolBroker(surface)
    result = broker.invoke("autocleanse.impute", {"method": "auto"}, df=df)

Skilling **wraps** dedicated suite actions — it does not replace them.
SLM tool use is generative assistance; rule broker always works offline.
"""

from __future__ import annotations

from autocausal.skilling.broker import SLMToolBroker, ToolResult
from autocausal.skilling.catalog import SkillDrill, skill_catalog
from autocausal.skilling.registry import SkillDef, SkillRegistry
from autocausal.skilling.surface import ToolDef, ToolSurface, suite_tool_surface
from autocausal.skilling.trace import SkillTrace

__all__ = [
    "ToolSurface",
    "ToolDef",
    "SkillRegistry",
    "SkillDef",
    "SLMToolBroker",
    "ToolResult",
    "suite_tool_surface",
    "skill_catalog",
    "SkillDrill",
    "SkillTrace",
]
