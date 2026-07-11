"""Soft adapter: prefer live Kineteq GRAIL MCP/module; else offline stub.

Never hard-requires a Kineteq install. Live path reuses the same env flags as
``autocausal.guides.kineteq_guide`` (``KINETEQ_MCP_URL`` + ``AUTOCAUSAL_KINETEQ_MCP``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from autocausal.grail.stub import GrailStub
from autocausal.grail.types import (
    Assumption,
    ExpertChain,
    ExpertStep,
    GrailReport,
    ImputationAudit,
)

__all__ = [
    "GrailEngine",
    "grail_backend_status",
    "kineteq_grail_available",
    "try_kineteq_grail_module",
]


def _env_flag(*names: str) -> bool:
    for n in names:
        if os.environ.get(n, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def kineteq_mcp_url() -> str:
    return (
        os.environ.get("AUTOCAUSAL_KINETEQ_MCP_URL")
        or os.environ.get("KINETEQ_MCP_URL")
        or os.environ.get("EMOTIVEVISION_MCP_URL")
        or ""
    ).strip()


def kineteq_mcp_configured() -> bool:
    return bool(kineteq_mcp_url()) and _env_flag(
        "AUTOCAUSAL_KINETEQ_MCP",
        "AUTOCAUSAL_GRAIL_MCP",
        "EMOTIVEVISION_LIVE_MCP",
        "KINETEQ_LIVE_MCP",
    )


def try_kineteq_grail_module() -> tuple[bool, str, Any]:
    """Try optional local modules that may expose GRAIL helpers."""
    for mod_name in ("kineteq", "kineteq_grail", "grail"):
        try:
            mod = __import__(mod_name)
        except Exception:
            continue
        for attr in ("grail_run", "Grail", "GRAIL", "run_grail", "impute"):
            if hasattr(mod, attr):
                return True, f"{mod_name}.{attr}", mod
        # package with submodules
        if hasattr(mod, "grail"):
            return True, f"{mod_name}.grail", mod
    return False, "", None


def kineteq_grail_available() -> bool:
    ok, _, _ = try_kineteq_grail_module()
    return ok or kineteq_mcp_configured()


def grail_backend_status() -> dict[str, Any]:
    mod_ok, mod_label, _ = try_kineteq_grail_module()
    return {
        "stub": True,
        "kineteq_module": mod_ok,
        "kineteq_module_label": mod_label or None,
        "kineteq_mcp": kineteq_mcp_configured(),
        "mcp_url_set": bool(kineteq_mcp_url()),
        "preferred": (
            "kineteq_module"
            if mod_ok
            else ("kineteq_mcp" if kineteq_mcp_configured() else "grail_stub")
        ),
        "epistemic": (
            "Live Kineteq GRAIL only when module/MCP succeeds; "
            "otherwise offline stub (not full GRAIL)."
        ),
    }


class GrailEngine:
    """Public AutoCausal GRAIL facade.

    Example::

        from autocausal.grail import GrailEngine

        eng = GrailEngine()
        report = eng.run("Does spend cause revenue?", context={"columns": [...]})
        print(report.to_markdown())
    """

    def __init__(self, *, domain: str = "causal", prefer_live: bool = True) -> None:
        self.domain = domain
        self.prefer_live = prefer_live
        self.stub = GrailStub(domain=domain)
        self.last_report: Optional[GrailReport] = None

    def available(self) -> bool:
        """Always True — stub path is always available."""
        return True

    def live_available(self) -> bool:
        return kineteq_grail_available()

    def impute(
        self,
        goal: str,
        *,
        context: Optional[dict[str, Any]] = None,
        domain: Optional[str] = None,
    ) -> ImputationAudit:
        if self.prefer_live:
            live = self._live_tool("grail_impute", {"goal": goal, "domain": domain or self.domain})
            if live is not None:
                return self._coerce_imputation(goal, live, domain or self.domain)
        return self.stub.impute(goal, context=context, domain=domain)

    def compose(
        self,
        goal: str,
        *,
        context: Optional[dict[str, Any]] = None,
        chain_length: int = 3,
        domain: Optional[str] = None,
    ) -> ExpertChain:
        if self.prefer_live:
            live = self._live_tool(
                "grail_compose",
                {
                    "goal": goal,
                    "domain": domain or self.domain,
                    "chain_length": chain_length,
                },
            )
            if live is not None:
                return self._coerce_chain(goal, live, chain_length)
        return self.stub.compose(
            goal, context=context, chain_length=chain_length, domain=domain
        )

    def fold(self, chain: ExpertChain):
        return self.stub.fold(chain)

    def memory_step(
        self,
        query: str,
        *,
        context: Optional[dict[str, Any]] = None,
        top_k: int = 8,
    ):
        return self.stub.memory_step(query, context=context, top_k=top_k)

    def graph_retrieve(
        self,
        *,
        context: Optional[dict[str, Any]] = None,
        focus: Optional[list[str]] = None,
        top_k: int = 10,
    ):
        return self.stub.graph_retrieve(context=context, focus=focus, top_k=top_k)

    def run(
        self,
        goal: str,
        *,
        context: Optional[dict[str, Any]] = None,
        max_cycles: int = 2,
        chain_length: int = 3,
        domain: Optional[str] = None,
    ) -> GrailReport:
        domain = domain or self.domain
        if self.prefer_live:
            live = self._live_tool(
                "grail_run",
                {
                    "goal": goal,
                    "domain": domain,
                    "max_cycles": max_cycles,
                    "chain_length": chain_length,
                },
            )
            if live is not None:
                report = self._coerce_report(goal, live, domain=domain)
                # Enrich with local memory/graph for AutoCausal hooks
                report.memory = self.stub.memory_step(goal, context=context, top_k=10)
                if not report.boost_edges:
                    report.boost_edges = self.stub.graph_retrieve(
                        context=context, focus=report.focus_columns, top_k=8
                    )
                if not report.focus_columns and context:
                    stub_r = self.stub.run(
                        goal,
                        context=context,
                        max_cycles=1,
                        chain_length=chain_length,
                        domain=domain,
                    )
                    report.focus_columns = stub_r.focus_columns
                    report.next_questions = report.next_questions or stub_r.next_questions
                self.last_report = report
                return report

        report = self.stub.run(
            goal,
            context=context,
            max_cycles=max_cycles,
            chain_length=chain_length,
            domain=domain,
        )
        self.last_report = report
        return report

    # --- live helpers -----------------------------------------------------

    def _live_tool(self, tool: str, args: dict[str, Any]) -> Optional[Any]:
        mod_ok, label, mod = try_kineteq_grail_module()
        if mod_ok and mod is not None:
            try:
                return self._call_module(mod, tool, args, label)
            except Exception:
                pass
        if kineteq_mcp_configured():
            try:
                return self._call_mcp(tool, args)
            except Exception:
                return None
        return None

    def _call_module(
        self, mod: Any, tool: str, args: dict[str, Any], label: str
    ) -> Any:
        # Direct function
        fn = getattr(mod, tool, None) or getattr(mod, tool.replace("grail_", ""), None)
        if callable(fn):
            return fn(**args)
        # Nested grail submodule
        sub = getattr(mod, "grail", None)
        if sub is not None:
            fn2 = getattr(sub, tool, None) or getattr(sub, "run", None)
            if callable(fn2):
                return fn2(**args)
            if hasattr(sub, "GrailEngine"):
                eng = sub.GrailEngine()
                method = tool.replace("grail_", "")
                if hasattr(eng, method):
                    return getattr(eng, method)(**args)
        # Object API
        for cls_name in ("Grail", "GRAIL", "GrailEngine"):
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            inst = cls() if callable(cls) else cls
            method = tool.replace("grail_", "")
            if hasattr(inst, method):
                return getattr(inst, method)(**args)
            if hasattr(inst, "run") and tool == "grail_run":
                return inst.run(**args)
        raise AttributeError(f"{label} has no callable for {tool}")

    def _call_mcp(self, tool: str, args: dict[str, Any]) -> Any:
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "Kineteq GRAIL MCP needs httpx (pip install autocausal[web])"
            ) from e

        url = kineteq_mcp_url()
        token = (
            os.environ.get("AUTOCAUSAL_KINETEQ_TOKEN")
            or os.environ.get("KINETEQ_AUTH_TOKEN")
            or os.environ.get("EMOTIVEVISION_MCP_TOKEN")
            or ""
        ).strip()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if token:
            headers["x-api-key"] = token

        with httpx.Client(timeout=60.0) as client:
            try:
                client.post(
                    url,
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "autocausal.grail", "version": "0.9.2"},
                        },
                    },
                )
            except Exception:
                pass

            resp = client.post(
                url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": tool, "arguments": args},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            result = data.get("result")
            if isinstance(result, dict) and "content" in result:
                parts = result["content"]
                if parts and isinstance(parts[0], dict) and "text" in parts[0]:
                    try:
                        return json.loads(parts[0]["text"])
                    except Exception:
                        return {"raw_text": parts[0]["text"]}
            return result

    @staticmethod
    def _coerce_imputation(goal: str, raw: Any, domain: str) -> ImputationAudit:
        if isinstance(raw, ImputationAudit):
            return raw
        if not isinstance(raw, dict):
            return ImputationAudit(
                original_goal=goal,
                enriched_goal=str(raw),
                domain=domain,
                backend="kineteq_grail",
                notes=["Coerced non-dict live result."],
            )
        assumptions = []
        for a in raw.get("assumptions") or []:
            if isinstance(a, dict):
                assumptions.append(
                    Assumption(
                        parameter=str(a.get("parameter") or a.get("name") or "slot"),
                        value=a.get("value"),
                        confidence=float(a.get("confidence") or 0.5),
                        rationale=str(a.get("rationale") or ""),
                    )
                )
        return ImputationAudit(
            original_goal=str(raw.get("original_goal") or goal),
            enriched_goal=str(raw.get("enriched_goal") or raw.get("goal") or goal),
            assumptions=assumptions,
            underspecified=list(raw.get("underspecified") or []),
            domain=str(raw.get("domain") or domain),
            backend="kineteq_grail",
            notes=list(raw.get("notes") or ["Live Kineteq grail_impute."]),
        )

    @staticmethod
    def _coerce_chain(goal: str, raw: Any, chain_length: int) -> ExpertChain:
        if isinstance(raw, ExpertChain):
            return raw
        if not isinstance(raw, dict):
            return ExpertChain(
                goal=goal,
                steps=[],
                mutation_prompt=str(raw)[:400],
                chain_length=chain_length,
                backend="kineteq_grail",
            )
        steps = []
        for i, s in enumerate(raw.get("steps") or raw.get("chain") or [], start=1):
            if isinstance(s, dict):
                steps.append(
                    ExpertStep(
                        step=int(s.get("step") or i),
                        role=str(s.get("role") or f"expert_{i}"),
                        prompt=str(s.get("prompt") or ""),
                        charges=dict(s.get("charges") or {}),
                    )
                )
        return ExpertChain(
            goal=str(raw.get("goal") or goal),
            steps=steps,
            mutation_prompt=str(raw.get("mutation_prompt") or ""),
            chain_length=len(steps) or chain_length,
            backend="kineteq_grail",
            notes=list(raw.get("notes") or ["Live Kineteq grail_compose."]),
        )

    @staticmethod
    def _coerce_report(goal: str, raw: Any, *, domain: str) -> GrailReport:
        if isinstance(raw, GrailReport):
            raw.live_kineteq = True
            raw.backend = raw.backend or "kineteq_grail"
            return raw
        if not isinstance(raw, dict):
            return GrailReport(
                goal=goal,
                domain=domain,
                backend="kineteq_grail",
                live_kineteq=True,
                final_answer=str(raw)[:2000],
                notes=["Live Kineteq grail_run (opaque result)."],
            )
        return GrailReport(
            goal=str(raw.get("goal") or goal),
            domain=str(raw.get("domain") or domain),
            backend="kineteq_grail",
            live_kineteq=True,
            final_answer=str(
                raw.get("final_answer") or raw.get("answer") or raw.get("summary") or ""
            ),
            genome_id=str(raw.get("genome_id") or ""),
            focus_columns=list(raw.get("focus_columns") or []),
            next_questions=list(raw.get("next_questions") or []),
            search_queries=list(raw.get("search_queries") or []),
            boost_edges=list(raw.get("boost_edges") or []),
            notes=list(raw.get("notes") or ["Live Kineteq grail_run."]),
        )
