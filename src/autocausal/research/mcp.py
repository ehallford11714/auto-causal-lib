"""MCP/AgentHook registration adapter for deep-research tools."""

from __future__ import annotations

from typing import Any, Mapping

from autocausal.research.handoff import to_research_handoff
from autocausal.research.models import ResearchHandoff, ResearchPolicy
from autocausal.research.suite import DeepResearchSuite


_RUNS: dict[str, Any] = {}
_SUITES: dict[str, DeepResearchSuite] = {}
_HANDOFFS: dict[str, ResearchHandoff] = {}


def _sid(args: Mapping[str, Any]) -> str:
    return str(args.get("session_id") or "default")


def _policy(args: Mapping[str, Any]) -> ResearchPolicy:
    raw = args.get("policy")
    if isinstance(raw, ResearchPolicy):
        policy = ResearchPolicy.from_dict(raw.to_dict())
    elif isinstance(raw, Mapping):
        policy = ResearchPolicy.from_dict(raw)
    else:
        allowed = tuple(str(item) for item in args.get("providers") or ("local",))
        policy = ResearchPolicy(allowed_providers=allowed)
    payload = policy.to_dict()
    for key in (
        "allow_network",
        "external_network_consent",
        "allow_generic_web",
        "approval_granted",
        "production_mode",
    ):
        if key in args and args[key] is not None:
            payload[key] = bool(args[key])
    if args.get("providers"):
        payload["allowed_providers"] = [str(item) for item in args["providers"]]
    return ResearchPolicy.from_dict(payload)


def _handoff(args: Mapping[str, Any], store: Any) -> ResearchHandoff:
    sid = _sid(args)
    if hasattr(store, "has") and store.has(sid):
        ac = store.get(sid)
        method = getattr(ac, "to_research_handoff", None)
        if callable(method):
            return method(
                domain=args.get("domain"),
                context=args.get("context"),
                policy=_policy(args),
            )
        result = getattr(ac, "result", None)
        if result is None:
            raise ValueError("session has no discovery result")
        return to_research_handoff(
            result,
            domain=args.get("domain"),
            context=args.get("context"),
            policy=_policy(args),
        )
    raw = args.get("handoff")
    if isinstance(raw, ResearchHandoff):
        return raw
    if isinstance(raw, Mapping):
        return ResearchHandoff.from_dict(raw)
    raise ValueError(
        "session_id with discovery result or a ResearchHandoff payload is required"
    )


def _suite(args: Mapping[str, Any], key: str) -> DeepResearchSuite:
    existing = _SUITES.get(key)
    if existing is not None:
        return existing
    suite = DeepResearchSuite(
        policy=_policy(args),
        use_slm=bool(args.get("use_slm") if args.get("use_slm") is not None else True),
        model_name=args.get("model_name"),
        local_records=args.get("sources") or [],
    )
    _SUITES[key] = suite
    return suite


def _plan(args: dict[str, Any], store: Any) -> dict[str, Any]:
    handoff = _handoff(args, store)
    key = str(args.get("research_id") or handoff.run_id)
    suite = _suite(args, key)
    plan = suite.plan(
        handoff,
        intensity=str(args.get("intensity") or "standard"),
        budget_overrides=args.get("budget_overrides"),
    )
    _HANDOFFS[key] = handoff
    return {
        "ok": True,
        "tool": "autocausal_research_plan",
        "research_id": key,
        "plan": plan,
    }


def _run(args: dict[str, Any], store: Any) -> dict[str, Any]:
    handoff = _handoff(args, store)
    key = str(args.get("research_id") or handoff.run_id)
    suite = _suite(args, key)
    prior = _RUNS.get(key) if bool(args.get("deepen")) else None
    report = suite.run(
        handoff,
        intensity=str(args.get("intensity") or "standard"),
        budget_overrides=args.get("budget_overrides"),
        resume_from=prior,
        approval_granted=(
            bool(args["approval_granted"])
            if args.get("approval_granted") is not None
            else None
        ),
    )
    _RUNS[key] = report
    _HANDOFFS[key] = handoff
    return {
        "ok": True,
        "tool": "autocausal_deep_research",
        "research_id": key,
        "status": report.status,
        "report": report.to_dict(),
    }


def _status(args: dict[str, Any], store: Any) -> dict[str, Any]:
    key = str(args.get("research_id") or "")
    if key:
        report = _RUNS.get(key)
        return {
            "ok": True,
            "tool": "autocausal_research_status",
            "research_id": key,
            "found": report is not None,
            "status": report.status if report else "unknown",
            "budget_used": report.budget_used.to_dict() if report else None,
            "stop_reason": report.stop_reason if report else "",
        }
    return {
        "ok": True,
        "tool": "autocausal_research_status",
        "runs": [
            {
                "research_id": run_id,
                "status": report.status,
                "intensity": report.selected_intensity.value,
                "sources": len(report.sources),
            }
            for run_id, report in sorted(_RUNS.items())
        ],
    }


def _report(args: dict[str, Any], store: Any) -> dict[str, Any]:
    key = str(args.get("research_id") or "")
    report = _RUNS.get(key)
    if report is None:
        return {
            "ok": False,
            "tool": "autocausal_research_report",
            "error": f"unknown research_id {key!r}",
        }
    fmt = str(args.get("format") or "markdown").lower()
    if fmt == "json":
        return {
            "ok": True,
            "tool": "autocausal_research_report",
            "research_id": key,
            "format": "json",
            "report": report.to_dict(),
        }
    return {
        "ok": True,
        "tool": "autocausal_research_report",
        "research_id": key,
        "format": "markdown",
        "markdown": report.to_markdown(),
    }


def register_research_tools(
    registry: Any,
    *,
    tool_spec_cls: Any,
    props_fn: Any,
) -> list[str]:
    """Register tools without importing registry internals (avoids a cycle)."""

    common = {
        "session_id": {"type": "string"},
        "research_id": {"type": "string"},
        "handoff": {"type": "object"},
        "intensity": {
            "type": "string",
            "enum": ["quick", "standard", "deep", "exhaustive"],
            "default": "standard",
        },
        "budget_overrides": {"type": "object"},
        "providers": {"type": "array", "items": {"type": "string"}},
        "sources": {
            "type": "array",
            "items": {"type": "object"},
            "description": "User-supplied SourceRecord payloads for offline retrieval",
        },
        "allow_network": {"type": "boolean", "default": False},
        "external_network_consent": {"type": "boolean", "default": False},
        "approval_granted": {"type": "boolean", "default": False},
        "use_slm": {"type": "boolean", "default": True},
        "domain": {"type": "string"},
        "context": {"type": "object"},
    }
    specs = [
        (
            "autocausal_research_plan",
            "Build a rule-first, privacy-safe research agenda and bounded query plan.",
            common,
            _plan,
        ),
        (
            "autocausal_deep_research",
            "Run or deepen citation-grounded research over fetched/user sources.",
            {**common, "deepen": {"type": "boolean", "default": False}},
            _run,
        ),
        (
            "autocausal_research_status",
            "Return research run status and actual budget consumption.",
            {"research_id": {"type": "string"}},
            _status,
        ),
        (
            "autocausal_research_report",
            "Return a verified deep-research report as markdown or JSON.",
            {
                "research_id": {"type": "string"},
                "format": {
                    "type": "string",
                    "enum": ["markdown", "json"],
                    "default": "markdown",
                },
            },
            _report,
        ),
    ]
    names: list[str] = []
    for name, description, properties, handler in specs:
        registry.register(
            tool_spec_cls(
                name=name,
                description=description,
                parameters=props_fn(properties),
                handler=handler,
                optional_module="autocausal.research",
                epistemic=(
                    "Literature is external context. Citations require fetched or "
                    "user-supplied SourceRecords and never upgrade identification."
                ),
            )
        )
        names.append(name)
    return names


__all__ = ["register_research_tools"]
