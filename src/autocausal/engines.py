"""Unified engine connectivity surface for AutoCausalLib.

Library-first APIs so insight / MCP / skilling / CLI can reach every soft engine::

    from autocausal.engines import list_engines, engine_status, estimate, refute, discover_with

Fabric-compatible dict outputs; heavy deps soft-skip.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import pandas as pd

from autocausal.backends import (
    DISCOVERY_BACKENDS,
    ESTIMATE_BACKENDS,
    REFUTE_BACKENDS,
    backend_status,
)

__all__ = [
    "EngineSpec",
    "EstimateResult",
    "list_engines",
    "engine_status",
    "discover_with",
    "estimate",
    "refute",
    "connectivity_map",
]


@dataclass
class EngineSpec:
    id: str
    kind: str  # discovery | estimate | refute | package
    available: bool
    builtin: bool = False
    description: str = ""
    extra: Optional[str] = None
    install: str = ""
    soft_skip: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EstimateResult:
    ok: bool
    method: str
    backend: str
    estimate: Optional[dict[str, Any]] = None
    data: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None
    soft_skip: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Package-level "engines" (insight, mcp, skilling, …) for connectivity map
PACKAGE_ENGINES: dict[str, dict[str, Any]] = {
    "insight": {
        "module": "autocausal.insight",
        "description": "InsightSuite / research loop / ExperimentRecommender",
        "builtin": True,
    },
    "mcp": {
        "module": "autocausal.mcp",
        "description": "MCP stdio server + AgentHook connective tools",
        "extra": "mcp",
        "builtin": True,  # package code always present; mcp SDK optional
    },
    "skilling": {
        "module": "autocausal.skilling",
        "description": "SLM tool surface / SkillRegistry / suite ToolSurface",
        "builtin": True,
    },
    "cli": {
        "module": "autocausal.cli",
        "description": "python -m autocausal CLI",
        "builtin": True,
    },
    "agentic": {
        "module": "autocausal.agentic",
        "description": "Agentic causal loop (FSM + soft LangGraph)",
        "builtin": True,
    },
    "grail": {
        "module": "autocausal.grail",
        "description": "Kineteq GRAIL soft loop",
        "builtin": True,
    },
    "suites": {
        "module": "autocausal.suites",
        "description": "AutoCleanse / AutoEDA / AutoMine",
        "builtin": True,
    },
    "connective": {
        "module": "autocausal.connective",
        "description": "In-process AgentHook (same tools as MCP)",
        "builtin": True,
    },
}


def _probe_module(path: str) -> bool:
    try:
        from importlib.util import find_spec

        return find_spec(path) is not None
    except Exception:
        return False


def list_engines(*, kind: Optional[str] = None) -> list[EngineSpec]:
    """List discovery / estimate / refute / package engines with availability."""
    out: list[EngineSpec] = []
    status = backend_status()

    def _add(bucket: str, items: dict[str, Any]) -> None:
        for eid, meta in items.items():
            st = status.get(bucket, {}).get(eid) or backend_status(eid)
            out.append(
                EngineSpec(
                    id=eid,
                    kind=bucket if bucket != "package" else "package",
                    available=bool(st.get("available")),
                    builtin=bool(st.get("builtin")),
                    description=str(st.get("description") or meta.get("description") or ""),
                    extra=st.get("extra") or meta.get("extra"),
                    install=str(st.get("install") or meta.get("install") or ""),
                    soft_skip=bool(st.get("soft_skip")),
                )
            )

    if kind is None or kind == "discovery":
        _add("discovery", DISCOVERY_BACKENDS)
    if kind is None or kind == "estimate":
        _add("estimate", ESTIMATE_BACKENDS)
    if kind is None or kind == "refute":
        _add("refute", REFUTE_BACKENDS)
    if kind is None or kind == "package":
        for eid, meta in PACKAGE_ENGINES.items():
            avail = bool(meta.get("builtin")) or _probe_module(str(meta["module"]))
            # mcp SDK soft
            if eid == "mcp":
                sdk = _probe_module("mcp")
                out.append(
                    EngineSpec(
                        id=eid,
                        kind="package",
                        available=avail,
                        builtin=True,
                        description=meta["description"]
                        + ("" if sdk else " (mcp SDK missing — AgentHook still works)"),
                        extra=meta.get("extra"),
                        install="pip install 'auto-causal-lib[mcp]'",
                        soft_skip=False,
                    )
                )
            else:
                out.append(
                    EngineSpec(
                        id=eid,
                        kind="package",
                        available=_probe_module(str(meta["module"])),
                        builtin=bool(meta.get("builtin")),
                        description=str(meta.get("description") or ""),
                        extra=meta.get("extra"),
                    )
                )
    return out


def engine_status(name: Optional[str] = None) -> dict[str, Any]:
    """Full or single-engine status dict (Fabric-friendly)."""
    if name:
        for e in list_engines():
            if e.id == name:
                return e.to_dict()
        st = backend_status(name)
        if st.get("error") != "unknown backend":
            return st
        if name in PACKAGE_ENGINES:
            meta = PACKAGE_ENGINES[name]
            return {
                "id": name,
                "kind": "package",
                "available": _probe_module(str(meta["module"])),
                **meta,
            }
        return {"id": name, "available": False, "error": "unknown engine"}
    engines = list_engines()
    return {
        "schema": "AutoCausalEngineStatus.v1",
        "n": len(engines),
        "engines": [e.to_dict() for e in engines],
        "by_kind": {
            k: [e.to_dict() for e in engines if e.kind == k]
            for k in ("discovery", "estimate", "refute", "package")
        },
        "notes": [
            "Soft-optional heavy libs; core path never requires them.",
            "Availability ≠ causal identification.",
        ],
        "connectivity": connectivity_map(),
    }


def connectivity_map() -> dict[str, Any]:
    """How insight / MCP / skilling / CLI reach engines."""
    return {
        "pipeline": "mine → discover(+causal-learn/lingam/gcastle) → estimate(DoubleML|EconML) → refute(DoWhy)",
        "library": {
            "discover": "autocausal.engines.discover_with / AutoCausal.discover(methods=[...])",
            "estimate": "autocausal.engines.estimate / AutoCausal.estimate / DiscoveryResult.estimate",
            "refute": "autocausal.engines.refute / AutoCausal.refute / DiscoveryResult.refute",
            "status": "autocausal.engines.engine_status / list_engines",
        },
        "cli": {
            "status": "python -m autocausal engines status",
            "list": "python -m autocausal engines list",
            "estimate": "python -m autocausal estimate --csv … --backend doubleml",
            "refute": "python -m autocausal refute --csv … --method dowhy",
            "insight": "python -m autocausal insight …",
            "skilling": "python -m autocausal skilling list",
            "mcp": "python -m autocausal.mcp",
        },
        "mcp_tools": [
            "autocausal_list_engines",
            "autocausal_estimate",
            "autocausal_refute",
            "autocausal_discover",
            "autocausal_insight_loop",
            "autocausal_skilling_list",
        ],
        "skilling": "suite ToolSurface + SkillRegistry expose cleanse/eda/mine/loop; engines via suite_tools",
        "insight": "InsightSuite can call discover/estimate/refute via AutoCausal session",
        "connective": "AgentHook.call_tool mirrors MCP tool names in-process",
    }


def discover_with(
    df: pd.DataFrame,
    *,
    method: str,
    columns: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a single soft discovery backend; returns edge dict payload."""
    m = (method or "").lower().strip()
    if m in ("causal_learn_pc", "causal_learn_ges", "causal_learn_fci", "pc", "ges", "fci"):
        from autocausal.backends import causal_learn

        alias = {
            "pc": "causal_learn_pc",
            "ges": "causal_learn_ges",
            "fci": "causal_learn_fci",
        }.get(m, m)
        return causal_learn.discover(df, method=alias, columns=columns, **kwargs)
    if m in ("lingam", "direct_lingam"):
        from autocausal.backends import lingam_backend

        return lingam_backend.discover(df, method=m, columns=columns, **kwargs)
    if m in ("gcastle_notears", "notears", "gcastle"):
        from autocausal.backends import gcastle_backend

        return gcastle_backend.discover(df, columns=columns, **kwargs)
    # Builtin path via discovery module
    if m in ("score_pc_lite", "corr_skeleton", "mi", "mi_binned", "mi_stub"):
        from autocausal.discovery import discover_relationships
        from autocausal.roles import infer_column_roles

        work = df[columns] if columns else df
        roles = infer_column_roles(work)
        # Normalize MI aliases to the canonical method id for discover_relationships
        method_id = "mi_binned" if m in ("mi", "mi_binned", "mi_stub") else m
        res = discover_relationships(
            work, roles=roles, method=method_id, use_iv=False, **kwargs
        )  # type: ignore[arg-type]
        return {
            "ok": True,
            "soft_skip": False,
            "method": method_id if m in ("mi", "mi_binned", "mi_stub") else m,
            "backend": "builtin",
            "edges": list(res.edges),
            "data": {"n_edges": len(res.edges)},
            "notes": list(res.notes)
            + (
                ["mi_stub is an alias of mi_binned"]
                if m == "mi_stub"
                else (["mi is an alias of mi_binned"] if m == "mi" else [])
            ),
            "error": None,
        }
    return {
        "ok": True,
        "soft_skip": True,
        "method": m,
        "backend": "unknown",
        "edges": [],
        "data": {},
        "notes": [f"Unknown discovery method {m!r} — soft-skip."],
        "error": None,
    }


_INFERENCE_BACKEND_MAP: dict[str, str] = {
    "builtin_ols": "regression",
    "regression": "regression",
    "ols": "regression",
    "builtin_2sls": "iv_2sls",
    "2sls": "iv_2sls",
    "iv": "iv_2sls",
    "iv_2sls": "iv_2sls",
    "aipw": "aipw",
    "iptw": "iptw",
    "matching": "matching",
    "propensity_score": "propensity_score",
    "difference_in_differences": "difference_in_differences",
    "did": "difference_in_differences",
    "panel_fixed_effects": "panel_fixed_effects",
    "panel_fe": "panel_fixed_effects",
    "regression_discontinuity": "regression_discontinuity",
    "rdd": "regression_discontinuity",
    "interrupted_time_series": "interrupted_time_series",
    "its": "interrupted_time_series",
    "doubleml": "doubleml",
    "dml": "doubleml",
    "plr": "doubleml",
    "econml": "econml_linear_dml",
    "linear_dml": "econml_linear_dml",
    "econml_linear_dml": "econml_linear_dml",
    "causal_forest": "econml_causal_forest",
    "econml_causal_forest": "econml_causal_forest",
    "cate": "econml_causal_forest",
}


def _estimate_via_inference(
    df: pd.DataFrame,
    *,
    method: str,
    y: str,
    d: str,
    x: Optional[list[str]] = None,
    z: Optional[str] = None,
    mode: str = "exploratory",
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> EstimateResult:
    """Delegate to the unified AutoInference runtime and adapt the result."""
    from autocausal.inference import AutoInference, CausalSpec

    notes_base = ["Estimates are exploratory unless design assumptions hold."]
    spec = CausalSpec(
        treatment=d,
        outcome=y,
        confounders=list(x or []),
        instrument=z,
        instrument_provenance="observed" if z else "unknown",
        unit=kwargs.pop("unit", None),
        time=kwargs.pop("time", None),
        post=kwargs.pop("post", None),
        running=kwargs.pop("running", None),
        cutoff=kwargs.pop("cutoff", None),
        bandwidth=kwargs.pop("bandwidth", None),
        cluster=kwargs.pop("cluster", None),
    )
    try:
        fitted = AutoInference(
            spec,
            mode=mode,
            random_state=random_state,
        ).fit(df, method=method, **kwargs)
    except Exception as exc:  # soft path for engines surface
        return EstimateResult(
            ok=False,
            method=method,
            backend="autocausal.inference",
            soft_skip=True,
            notes=notes_base + [f"AutoInference soft-skip: {exc}"],
            error=str(exc),
        )

    payload = fitted.to_dict() if hasattr(fitted, "to_dict") else {}
    diagnostics = dict(payload.get("diagnostics") or {})
    estimate_payload = {
        "ate": payload.get("estimate"),
        "standard_error": payload.get("standard_error"),
        "ci_low": payload.get("ci_low"),
        "ci_high": payload.get("ci_high"),
        "p_value": payload.get("p_value"),
        "y": y,
        "d": d,
        "x": list(x or []),
        "z": z,
        "n": payload.get("n"),
        "diagnostics": diagnostics,
        "evidence_grade": payload.get("evidence_grade"),
        "first_stage_f": diagnostics.get("first_stage_f"),
    }
    return EstimateResult(
        ok=bool(payload.get("ok", True)),
        method=str(payload.get("method") or method),
        backend="autocausal.inference",
        estimate=estimate_payload,
        data={
            "spec": payload.get("spec"),
            "assumptions": payload.get("assumptions"),
            "gates": payload.get("gates"),
            "manifest": payload.get("manifest"),
        },
        notes=list(payload.get("notes") or []) + notes_base + [
            "Routed through autocausal.inference (unified runtime)."
        ],
        soft_skip=bool(payload.get("soft_skip", False)),
        error=payload.get("error"),
    )


def estimate(
    df: pd.DataFrame,
    *,
    backend: str = "builtin_ols",
    y: Optional[str] = None,
    d: Optional[str] = None,
    x: Optional[list[str]] = None,
    candidates: Optional[dict[str, list[str]]] = None,
    z: Optional[str] = None,
    mode: str = "exploratory",
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> EstimateResult:
    """Estimate ATE/CATE via unified inference / DoubleML / EconML soft backends.

    Prefer explicit ``y`` / ``d``. When roles resolve, native backends route
    through :class:`autocausal.inference.AutoInference` so statistical gates and
    provenance stay consistent with ``AutoCausal.infer()``.
    """
    b = (backend or "builtin_ols").lower().strip()
    notes_base = ["Estimates are exploratory unless design assumptions hold."]
    mapped = _INFERENCE_BACKEND_MAP.get(b)

    # Resolve roles for inference routing / OLS fallback.
    from autocausal.backends._common import resolve_roles

    roles = resolve_roles(df, y=y, d=d, x=x, candidates=candidates)
    yy, dd, xx = roles["y"], roles["d"], roles["x"]
    zz = z
    if zz is None and candidates:
        instruments = list(candidates.get("instrument") or [])
        zz = instruments[0] if instruments else None

    if mapped and yy and dd and yy in df.columns and dd in df.columns:
        # Keep optional DoubleML/EconML on their existing soft adapters, but still
        # prefer AutoInference when those methods are registered there.
        if mapped in (
            "regression",
            "iv_2sls",
            "aipw",
            "iptw",
            "matching",
            "propensity_score",
            "difference_in_differences",
            "panel_fixed_effects",
            "regression_discontinuity",
            "interrupted_time_series",
            "doubleml",
            "econml_linear_dml",
            "econml_causal_forest",
        ):
            if mapped == "iv_2sls" and not zz:
                return EstimateResult(
                    ok=True,
                    method="iv_2sls",
                    backend="autocausal.inference",
                    soft_skip=True,
                    notes=notes_base + ["IV estimate needs an observed z= instrument."],
                )
            return _estimate_via_inference(
                df,
                method=mapped,
                y=yy,
                d=dd,
                x=list(xx or []),
                z=zz,
                mode=mode,
                random_state=random_state,
                **kwargs,
            )

    if b in ("doubleml", "dml", "plr"):
        from autocausal.backends import doubleml_backend

        raw = doubleml_backend.estimate(df, y=y, d=d, x=x, candidates=candidates, **kwargs)
        return EstimateResult(
            ok=bool(raw.get("ok")),
            method=str(raw.get("method") or b),
            backend=str(raw.get("backend") or b),
            estimate=raw.get("estimate"),
            data=dict(raw.get("data") or {}),
            notes=list(raw.get("notes") or notes_base),
            error=raw.get("error"),
            soft_skip=bool(raw.get("soft_skip")),
        )

    if b.startswith("econml") or b in ("linear_dml", "causal_forest", "cate"):
        from autocausal.backends import econml_backend

        raw = econml_backend.estimate(
            df, y=y, d=d, x=x, candidates=candidates, method=b, **kwargs
        )
        return EstimateResult(
            ok=bool(raw.get("ok")),
            method=str(raw.get("method") or b),
            backend=str(raw.get("backend") or b),
            estimate=raw.get("estimate"),
            data=dict(raw.get("data") or {}),
            notes=list(raw.get("notes") or notes_base),
            error=raw.get("error"),
            soft_skip=bool(raw.get("soft_skip")),
        )

    # Last-resort legacy OLS when roles cannot be resolved for AutoInference.
    if not yy or not dd or yy not in df.columns or dd not in df.columns:
        return EstimateResult(
            ok=True,
            method="builtin_ols",
            backend="builtin",
            soft_skip=True,
            notes=notes_base + ["Could not resolve Y/D columns."],
        )
    cols = [yy, dd] + list(xx)
    work = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 8:
        return EstimateResult(
            ok=True,
            method="builtin_ols",
            backend="builtin",
            soft_skip=True,
            notes=notes_base + ["Too few rows for OLS."],
        )
    import numpy as np
    from numpy.linalg import lstsq

    yv = work[yy].to_numpy(dtype=float)
    dv = work[dd].to_numpy(dtype=float)
    if xx:
        X = np.column_stack([np.ones(len(work)), dv, work[xx].to_numpy(dtype=float)])
    else:
        X = np.column_stack([np.ones(len(work)), dv])
    beta, *_ = lstsq(X, yv, rcond=None)
    return EstimateResult(
        ok=True,
        method="builtin_ols",
        backend="numpy",
        estimate={
            "ate": round(float(beta[1]), 6),
            "intercept": round(float(beta[0]), 6),
            "y": yy,
            "d": dd,
            "x": xx,
            "n": len(work),
        },
        notes=notes_base
        + [
            "Legacy OLS fallback — prefer AutoCausal.infer()/engines.estimate with explicit roles.",
            "Built-in OLS association — not an identified causal effect.",
        ],
    )


def refute(
    edge: Optional[dict[str, Any]] = None,
    *,
    method: str = "placebo",
    df: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    d: Optional[str] = None,
    x: Optional[list[str]] = None,
    candidates: Optional[dict[str, list[str]]] = None,
    **kwargs: Any,
) -> Any:
    """Delegate to suite_tools.refute (DoWhy real path + builtins)."""
    from autocausal.suite_tools import refute as suite_refute

    return suite_refute(
        edge,
        method=method,
        df=df,
        y=y,
        d=d,
        x=x,
        candidates=candidates,
        **kwargs,
    )


EXTERNAL_DISCOVERY_METHODS = frozenset(
    {
        "causal_learn_pc",
        "causal_learn_ges",
        "causal_learn_fci",
        "lingam",
        "direct_lingam",
        "gcastle_notears",
    }
)


def optional_ensemble_methods(installed_only: bool = True) -> list[str]:
    """Default soft methods to append to discover_ensemble when installed."""
    out: list[str] = []
    for mid in ("causal_learn_pc", "causal_learn_ges", "lingam", "gcastle_notears"):
        st = backend_status(mid)
        if not installed_only or st.get("available"):
            out.append(mid)
    return out
