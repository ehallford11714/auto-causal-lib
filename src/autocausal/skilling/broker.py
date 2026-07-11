"""SLMToolBroker — list/invoke tools; soft SLM tool_calls or rule selection."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import pandas as pd

from autocausal.skilling.registry import SkillRegistry, default_skill_registry
from autocausal.skilling.surface import ToolSurface, suite_tool_surface
from autocausal.skilling.trace import SkillTrace
from autocausal.suites.action_protocol import ActionResult
from autocausal.suites.director import EPISTEMIC_NOTE, resolve_suite_slm

__all__ = ["ToolResult", "SLMToolBroker"]


@dataclass
class ToolResult:
    name: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    n_affected: int = 0
    frame_mutated: bool = False
    epistemic: str = EPISTEMIC_NOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SLMToolBroker:
    """Execute tool calls from SLM/director against a ToolSurface.

    Soft path: if SLM returns tool_calls, run them; else rule director picks tools.
    Never hard-crashes when HF is missing.
    """

    def __init__(
        self,
        surface: Optional[ToolSurface] = None,
        *,
        skills: Optional[SkillRegistry] = None,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        record_traces: bool = True,
    ) -> None:
        self.surface = surface or suite_tool_surface()
        self.skills = skills or default_skill_registry(self.surface)
        self.use_slm = resolve_suite_slm(use_slm)
        self.model_name = model_name
        self.record_traces = record_traces
        self.traces: list[SkillTrace] = []
        self.last_invocations: list[dict[str, Any]] = []

    def list_tools(self, skill: Optional[str] = None) -> list[dict[str, Any]]:
        if skill:
            allowed = set(self.skills.tools_for(skill))
            return [t.schema() for t in self.surface.list_tools() if t.name in allowed]
        return self.surface.schemas()

    def invoke(
        self,
        name: str,
        args: Optional[dict[str, Any]] = None,
        *,
        df: Optional[pd.DataFrame] = None,
        skill: Optional[str] = None,
    ) -> ToolResult:
        args = dict(args or {})
        if skill:
            allowed = set(self.skills.tools_for(skill))
            if name not in allowed:
                return ToolResult(
                    name=name,
                    ok=False,
                    warnings=[f"Tool `{name}` not allowed by skill `{skill}`."],
                )
        try:
            tool = self.surface.get(name)
        except KeyError as e:
            return ToolResult(name=name, ok=False, warnings=[str(e)])

        if tool.handler is None:
            return ToolResult(name=name, ok=False, warnings=["Tool has no handler."])
        if df is None:
            return ToolResult(name=name, ok=False, warnings=["df is required to invoke tools."])

        try:
            result: ActionResult = tool.handler(df, **args)
        except TypeError:
            # retry without unexpected kwargs
            try:
                result = tool.handler(df)
            except Exception as e:
                return ToolResult(
                    name=name,
                    ok=False,
                    warnings=[f"Invoke soft-fail: {type(e).__name__}: {e}"],
                )
        except Exception as e:
            return ToolResult(
                name=name,
                ok=False,
                warnings=[f"Invoke soft-fail: {type(e).__name__}: {e}"],
            )

        # Drop non-serializable
        payload = dict(result.payload or {})
        payload.pop("_mining", None)
        tr = ToolResult(
            name=name,
            ok=True,
            payload=payload,
            warnings=list(result.warnings),
            notes=list(result.notes) + [EPISTEMIC_NOTE],
            n_affected=result.n_affected,
            frame_mutated=result.frame is not None,
        )
        self.last_invocations.append(tr.to_dict())
        return tr

    def select_tools(
        self,
        skill: str,
        *,
        text: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Return ordered tool call dicts ``{name, arguments}``.

        Prefers SLM tool_calls when available; else rule sequence from skill tools.
        """
        allowed = self.skills.tools_for(skill)
        rule_calls = [{"name": n, "arguments": {}} for n in allowed]
        # Prefer default suite order when skill maps to a suite
        skill_def = self.skills.get(skill)
        suite = skill_def.suite
        if suite == "autocleanse":
            from autocausal.suites.autocleanse import CleanseActions

            order = [f"autocleanse.{a}" for a in CleanseActions.default_sequence()]
            rule_calls = [{"name": n, "arguments": {}} for n in order if n in allowed]
        elif suite == "autoeda":
            from autocausal.suites.autoeda import EDAActions

            order = [f"autoeda.{a}" for a in EDAActions.default_sequence()]
            rule_calls = [{"name": n, "arguments": {}} for n in order if n in allowed]
        elif suite == "automine":
            from autocausal.suites.automine import MineActions

            order = [f"automine.{a}" for a in MineActions.default_sequence()]
            rule_calls = [{"name": n, "arguments": {}} for n in order if n in allowed]

        if not self.use_slm:
            return rule_calls

        slm_calls = self._slm_tool_calls(skill, text=text, context=context, allowed=allowed)
        if slm_calls:
            return slm_calls
        return rule_calls

    def run_skill(
        self,
        skill: str,
        df: pd.DataFrame,
        *,
        text: str = "",
        context: Optional[dict[str, Any]] = None,
        tool_calls: Optional[Sequence[dict[str, Any]]] = None,
    ) -> tuple[pd.DataFrame, list[ToolResult], SkillTrace]:
        """Select (or accept) tool calls, invoke them, optionally mutate frame."""
        calls = list(tool_calls) if tool_calls is not None else self.select_tools(
            skill, text=text, context=context
        )
        out = df
        results: list[ToolResult] = []
        for call in calls:
            name = str(call.get("name") or call.get("tool") or "")
            args = dict(call.get("arguments") or call.get("args") or {})
            tr = self.invoke(name, args, df=out, skill=skill)
            results.append(tr)
            if tr.ok and tr.frame_mutated:
                # Re-invoke to get frame — handlers return ActionResult with frame;
                # ToolResult doesn't carry frame. Re-run handler for mutation.
                tool = self.surface.get(name)
                if tool.handler is not None:
                    try:
                        ar = tool.handler(out, **args)
                        if ar.frame is not None:
                            out = ar.frame
                    except Exception:
                        pass

        trace = SkillTrace(
            skill=skill,
            backend="huggingface" if self.use_slm else "rule",
            context={"text": text, **(context or {})},
            tool_calls=list(calls),
            outcomes=[r.to_dict() for r in results],
            notes=[EPISTEMIC_NOTE, self.skills.prompt_for(skill)[:200]],
        )
        if self.record_traces:
            self.traces.append(trace)
        return out, results, trace

    def _slm_tool_calls(
        self,
        skill: str,
        *,
        text: str,
        context: Optional[dict[str, Any]],
        allowed: list[str],
    ) -> list[dict[str, Any]]:
        try:
            from autocausal.slm import get_backend, slm_available
        except Exception:
            return []

        if not slm_available() and not self.use_slm:
            return []

        try:
            backend = get_backend(use_slm=True, model_name=self.model_name)
        except Exception:
            return []

        prompt_bits = [
            self.skills.prompt_for(skill),
            "Available tools:",
            ", ".join(allowed[:40]),
            f"User text: {text or '(none)'}",
            "List tool names to call, one per line, using exact names.",
        ]
        raw = ""
        try:
            if hasattr(backend, "guide"):
                gres = backend.guide(
                    {
                        "text": "\n".join(prompt_bits),
                        "columns": (context or {}).get("columns") or [],
                    }
                )
                raw = getattr(gres, "raw_text", "") or ""
                # Also scan suggestion details
                for s in getattr(gres, "suggestions", None) or []:
                    detail = getattr(s, "detail", "") or ""
                    raw = raw + "\n" + detail
        except Exception:
            return []

        if not raw:
            return []

        found: list[dict[str, Any]] = []
        allowed_set = set(allowed)
        for line in re.split(r"[\n,;]+", raw):
            line = line.strip(" -*`\t")
            for name in allowed:
                if name in line or name.split(".", 1)[-1] in line.lower():
                    if name in allowed_set and name not in {c["name"] for c in found}:
                        found.append({"name": name, "arguments": {}})
        return found
