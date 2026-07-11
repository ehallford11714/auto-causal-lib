"""Real DoWhy refute_estimate wiring (soft-optional)."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from autocausal.backends._common import resolve_roles, soft_import, soft_skip_result

INSTALL = "pip install 'auto-causal-lib[causal-extra]'  # includes dowhy"


def available() -> bool:
    return soft_import("dowhy") is not None


_REFUTER_MAP = {
    "dowhy": "placebo_treatment_refuter",
    "dowhy_refute": "placebo_treatment_refuter",
    "dowhy_placebo": "placebo_treatment_refuter",
    "placebo_treatment_refuter": "placebo_treatment_refuter",
    "dowhy_random_common_cause": "random_common_cause",
    "random_common_cause": "random_common_cause",
    "dowhy_data_subset": "data_subset_refuter",
    "data_subset_refuter": "data_subset_refuter",
}


def refute(
    df: pd.DataFrame,
    *,
    method: str = "dowhy",
    y: Optional[str] = None,
    d: Optional[str] = None,
    x: Optional[list[str]] = None,
    candidates: Optional[dict[str, list[str]]] = None,
    edge: Optional[dict[str, Any]] = None,
    estimate_method: str = "backdoor.linear_regression",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build DoWhy CausalModel from roles and run refute_estimate variants."""
    notes = [
        "DoWhy refutation is exploratory — passing a refuter does not prove causation.",
    ]
    method_l = (method or "dowhy").lower().strip()
    refuter = _REFUTER_MAP.get(method_l, method_l)
    if not available():
        out = soft_skip_result(method=method_l, module="dowhy", install=INSTALL, notes=notes)
        out["edge"] = dict(edge or {})
        return out

    edge = dict(edge or {})
    # Prefer edge endpoints when roles missing
    if edge.get("source") and not d:
        d = str(edge["source"])
    if edge.get("target") and not y:
        y = str(edge["target"])

    roles = resolve_roles(df, y=y, d=d, x=x, candidates=candidates)
    yy, dd, xx = roles["y"], roles["d"], roles["x"]
    if not yy or not dd or yy not in df.columns or dd not in df.columns:
        return {
            "ok": True,
            "soft_skip": True,
            "method": method_l,
            "backend": "dowhy",
            "edge": edge,
            "data": {},
            "notes": notes + ["Could not resolve treatment/outcome for DoWhy."],
            "error": None,
        }

    cols = [c for c in [yy, dd, *xx] if c in df.columns]
    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()
    if len(work) < 20:
        return {
            "ok": True,
            "soft_skip": True,
            "method": method_l,
            "backend": "dowhy",
            "edge": edge,
            "data": {"n": len(work)},
            "notes": notes + ["Need ≥20 complete rows for DoWhy refute."],
            "error": None,
        }

    try:
        from dowhy import CausalModel

        # Simple graph: common causes → treatment & outcome; treatment → outcome
        lines = [f"{dd} -> {yy};"]
        for c in xx:
            lines.append(f"{c} -> {dd};")
            lines.append(f"{c} -> {yy};")
        graph = "digraph {" + " ".join(lines) + "}"

        model = CausalModel(
            data=work,
            treatment=dd,
            outcome=yy,
            common_causes=list(xx) if xx else None,
            graph=graph if xx else None,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name=estimate_method,
        )
        refute_kwargs: dict[str, Any] = {}
        if refuter == "data_subset_refuter":
            refute_kwargs["subset_fraction"] = float(kwargs.get("subset_fraction", 0.8))
        if refuter == "placebo_treatment_refuter":
            refute_kwargs["placebo_type"] = kwargs.get("placebo_type", "permute")

        ref = model.refute_estimate(identified, estimate, method_name=refuter, **refute_kwargs)

        new_effect = getattr(ref, "new_effect", None)
        estimated = getattr(estimate, "value", None)
        try:
            new_val = float(new_effect) if new_effect is not None else None
        except Exception:
            new_val = None
        try:
            est_val = float(estimated) if estimated is not None else None
        except Exception:
            est_val = None

        # Heuristic: placebo / random cause should shrink effect
        refute_passed = None
        if est_val is not None and new_val is not None:
            if refuter == "placebo_treatment_refuter":
                refute_passed = abs(new_val) < abs(est_val) * 0.5 + 0.05
            elif refuter == "random_common_cause":
                refute_passed = abs(new_val - est_val) < abs(est_val) * 0.3 + 0.05
            elif refuter == "data_subset_refuter":
                refute_passed = abs(new_val - est_val) < abs(est_val) * 0.5 + 0.1

        return {
            "ok": True,
            "soft_skip": False,
            "method": method_l,
            "backend": f"dowhy.{refuter}",
            "edge": edge or {"source": dd, "target": yy},
            "data": {
                "refuter": refuter,
                "estimated_effect": round(est_val, 6) if est_val is not None else None,
                "new_effect": round(new_val, 6) if new_val is not None else None,
                "refute_passed": refute_passed,
                "refutation_text": str(ref)[:800],
                "y": yy,
                "d": dd,
                "common_causes": xx,
                "n": len(work),
            },
            "notes": notes
            + [
                f"DoWhy {refuter}: estimated={est_val}, new={new_val}, passed={refute_passed}",
            ],
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "soft_skip": False,
            "method": method_l,
            "backend": "dowhy",
            "edge": edge,
            "data": {},
            "notes": notes,
            "error": f"{type(e).__name__}: {e}",
        }
