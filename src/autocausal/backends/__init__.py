"""Soft-optional causal discovery / estimation / refutation backends.

Heavy libraries (causal-learn, DoWhy, DoubleML, EconML, lingam, gCastle) are
never hard-required. Import probes and soft-skip notes keep the core
numpy/pandas path intact.

Prefer the unified surface::

    from autocausal.engines import list_engines, engine_status, estimate, discover_with
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "backend_status",
    "DISCOVERY_BACKENDS",
    "ESTIMATE_BACKENDS",
    "REFUTE_BACKENDS",
]


DISCOVERY_BACKENDS: dict[str, dict[str, Any]] = {
    "score_pc_lite": {
        "module": None,
        "extra": None,
        "kind": "discovery",
        "builtin": True,
        "description": "Built-in PC-lite partial-correlation skeleton",
    },
    "corr_skeleton": {
        "module": None,
        "extra": None,
        "kind": "discovery",
        "builtin": True,
        "description": "Built-in correlation-threshold skeleton",
    },
    "mi_stub": {
        "module": None,
        "extra": None,
        "kind": "discovery",
        "builtin": True,
        "description": "Built-in binned mutual-information stub",
    },
    "causal_learn_pc": {
        "module": "causallearn",
        "extra": "causal-extra",
        "kind": "discovery",
        "builtin": False,
        "description": "causal-learn PC algorithm",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
    "causal_learn_ges": {
        "module": "causallearn",
        "extra": "causal-extra",
        "kind": "discovery",
        "builtin": False,
        "description": "causal-learn GES algorithm",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
    "causal_learn_fci": {
        "module": "causallearn",
        "extra": "causal-extra",
        "kind": "discovery",
        "builtin": False,
        "description": "causal-learn FCI (partial ancestral graph)",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
    "lingam": {
        "module": "lingam",
        "extra": "causal-extra",
        "kind": "discovery",
        "builtin": False,
        "description": "DirectLiNGAM linear non-Gaussian discovery",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
    "direct_lingam": {
        "module": "lingam",
        "extra": "causal-extra",
        "kind": "discovery",
        "builtin": False,
        "description": "Alias for DirectLiNGAM",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
    "gcastle_notears": {
        "module": "castle",
        "extra": "causal-extra",
        "kind": "discovery",
        "builtin": False,
        "description": "gCastle NOTEARS continuous DAG learning",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
}

ESTIMATE_BACKENDS: dict[str, dict[str, Any]] = {
    "builtin_ols": {
        "module": None,
        "extra": None,
        "kind": "estimate",
        "builtin": True,
        "description": "Built-in OLS ATE-style association (exploratory)",
    },
    "builtin_2sls": {
        "module": None,
        "extra": None,
        "kind": "estimate",
        "builtin": True,
        "description": "Built-in numpy 2SLS",
    },
    "doubleml": {
        "module": "doubleml",
        "extra": "causal-extra",
        "kind": "estimate",
        "builtin": False,
        "description": "DoubleML partially linear regression (ATE/PLR)",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
    "econml": {
        "module": "econml",
        "extra": "causal-extra",
        "kind": "estimate",
        "builtin": False,
        "description": "EconML LinearDML / CausalForestDML (CATE)",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
    "econml_linear_dml": {
        "module": "econml",
        "extra": "causal-extra",
        "kind": "estimate",
        "builtin": False,
        "description": "EconML LinearDML",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
    "econml_causal_forest": {
        "module": "econml",
        "extra": "causal-extra",
        "kind": "estimate",
        "builtin": False,
        "description": "EconML CausalForestDML",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
}

REFUTE_BACKENDS: dict[str, dict[str, Any]] = {
    "placebo": {
        "module": None,
        "extra": None,
        "kind": "refute",
        "builtin": True,
        "description": "Built-in placebo treatment shuffle",
    },
    "random_common_cause": {
        "module": None,
        "extra": None,
        "kind": "refute",
        "builtin": True,
        "description": "Built-in random common-cause stub",
    },
    "dowhy": {
        "module": "dowhy",
        "extra": "causal-extra",
        "kind": "refute",
        "builtin": False,
        "description": "DoWhy CausalModel.refute_estimate variants",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
    "dowhy_placebo": {
        "module": "dowhy",
        "extra": "causal-extra",
        "kind": "refute",
        "builtin": False,
        "description": "DoWhy placebo_treatment_refuter",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
    "dowhy_random_common_cause": {
        "module": "dowhy",
        "extra": "causal-extra",
        "kind": "refute",
        "builtin": False,
        "description": "DoWhy random_common_cause",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
    "dowhy_data_subset": {
        "module": "dowhy",
        "extra": "causal-extra",
        "kind": "refute",
        "builtin": False,
        "description": "DoWhy data_subset_refuter",
        "install": "pip install 'auto-causal-lib[causal-extra]'",
    },
}


def _probe(module: str | None) -> bool:
    if not module:
        return True
    try:
        from importlib.util import find_spec

        return find_spec(module) is not None
    except Exception:
        return False


def backend_status(name: str | None = None) -> dict[str, Any]:
    """Return availability for one backend or the full catalog."""
    catalog = {**DISCOVERY_BACKENDS, **ESTIMATE_BACKENDS, **REFUTE_BACKENDS}
    if name:
        meta = catalog.get(name)
        if meta is None:
            return {"id": name, "available": False, "error": "unknown backend"}
        available = bool(meta.get("builtin")) or _probe(meta.get("module"))
        return {
            "id": name,
            "available": available,
            "soft_skip": not available and not meta.get("builtin"),
            **{k: v for k, v in meta.items() if k != "module"},
            "module": meta.get("module"),
        }
    out: dict[str, Any] = {
        "discovery": {},
        "estimate": {},
        "refute": {},
        "notes": [
            "Heavy engines are soft-optional; core numpy/pandas path always works.",
            "Availability ≠ causal identification.",
        ],
    }
    for nid, meta in DISCOVERY_BACKENDS.items():
        out["discovery"][nid] = backend_status(nid)
    for nid, meta in ESTIMATE_BACKENDS.items():
        out["estimate"][nid] = backend_status(nid)
    for nid, meta in REFUTE_BACKENDS.items():
        out["refute"][nid] = backend_status(nid)
    return out
