"""SLM/Rule construction of ModelConstructPlan for KPI-mined loops.

RuleGuide always works offline. Optional HuggingFace / LLMIntent / Kineteq
backends refine kpi_focus and may prefer torch when AUTOCAUSAL_TORCH=1.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Sequence

ImputerKind = Literal["torch_mlp", "iterative", "median", "sklearn"]
PredictorKind = Literal["torch_mlp", "sklearn_rf", "none"]


def _env_flag(*names: str) -> bool:
    for n in names:
        if os.environ.get(n, "").strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


def torch_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


def torch_preferred() -> bool:
    """True when torch is installed and AUTOCAUSAL_TORCH prefers it."""
    return torch_available() and _env_flag("AUTOCAUSAL_TORCH")


def sklearn_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("sklearn") is not None
    except (ModuleNotFoundError, ValueError, ImportError):
        return False


@dataclass
class ModelConstructPlan:
    """Plan emitted by SLM/Rule for impute + predict on KPI-mined features."""

    imputer: ImputerKind = "median"
    predictor: PredictorKind = "none"
    kpi_focus: list[str] = field(default_factory=list)
    outcome: Optional[str] = None
    treatment: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    use_physics: bool = True
    backend: str = "rule"
    rationale: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Model construct plan",
            "",
            f"**Backend:** `{self.backend}`",
            f"**Imputer:** `{self.imputer}`",
            f"**Predictor:** `{self.predictor}`",
            "",
        ]
        if self.kpi_focus:
            lines += ["## KPI focus", ""] + [f"- `{k}`" for k in self.kpi_focus] + [""]
        if self.outcome:
            lines.append(f"**Outcome:** `{self.outcome}`")
            lines.append("")
        if self.treatment:
            lines.append(f"**Treatment:** {', '.join(f'`{t}`' for t in self.treatment)}")
            lines.append("")
        if self.feature_columns:
            lines += ["## Features", ""] + [f"- `{c}`" for c in self.feature_columns[:24]] + [""]
        if self.rationale:
            lines += ["## Rationale", ""] + [f"- {r}" for r in self.rationale] + [""]
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


def _col_names(context: dict[str, Any]) -> list[str]:
    columns = context.get("columns") or []
    return [c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in columns]


def _extract_kpis(context: dict[str, Any], columns: list[str]) -> list[str]:
    kpis = list(context.get("kpis") or [])
    # DataMine-style nested
    mining = context.get("mining") or {}
    if isinstance(mining, dict):
        kpis.extend(mining.get("kpis") or [])
        kpis.extend(mining.get("kpi_names") or [])
    # hint from text
    text = (context.get("text") or "").lower()
    for c in columns:
        cl = c.lower()
        if any(h in cl for h in ("y", "outcome", "revenue", "sales", "kpi", "target", "score")):
            kpis.append(c)
        if text and cl in text:
            kpis.append(c)
    # uniq preserve order
    seen: set[str] = set()
    out: list[str] = []
    for k in kpis:
        ks = str(k)
        if ks in columns and ks not in seen:
            seen.add(ks)
            out.append(ks)
    if not out:
        # fallback: first numeric-looking names or first few columns
        out = [c for c in columns if re.search(r"(y|score|rev|sales|kpi)", c, re.I)][:6]
    if not out:
        out = list(columns[:6])
    return out


def _choose_imputer(
    *,
    prefer_torch: bool,
    force_torch: bool,
    force_median: bool,
    force_sklearn: bool,
) -> ImputerKind:
    if force_median:
        return "median"
    if force_torch and torch_available():
        return "torch_mlp"
    if force_sklearn and sklearn_available():
        return "sklearn"
    if prefer_torch and torch_available():
        return "torch_mlp"
    if sklearn_available():
        return "iterative"  # iterative falls back to median if sklearn missing pieces
    return "median"


def _choose_predictor(
    *,
    prefer_torch: bool,
    force_torch: bool,
    outcome: Optional[str],
) -> PredictorKind:
    if not outcome:
        return "none"
    if force_torch and torch_available():
        return "torch_mlp"
    if prefer_torch and torch_available():
        return "torch_mlp"
    if sklearn_available():
        return "sklearn_rf"
    return "none"


def construct_model_plan(
    context: dict[str, Any],
    *,
    use_slm: bool = False,
    use_torch: Optional[bool] = None,
    guides: Optional[Sequence[str]] = None,
    prefer_physics: bool = True,
) -> ModelConstructPlan:
    """
    Build a ModelConstructPlan from mining/KPI context via Rule (+ optional guides).

    ``use_torch``:
      - True  → prefer torch_mlp when installed
      - False → never torch
      - None  → follow AUTOCAUSAL_TORCH env when torch installed
    """
    columns = _col_names(context)
    kpis = _extract_kpis(context, columns)
    text = context.get("text") or ""

    # Direction / guide merge for roles
    treatment: list[str] = []
    outcome: Optional[str] = None
    focus: list[str] = list(kpis)
    backend_names = ["rule"]
    rationale: list[str] = []
    notes: list[str] = []
    guide_imputer: Optional[str] = None
    guide_predictor: Optional[str] = None

    try:
        from autocausal.guides import direct as run_direct

        backends = list(guides) if guides else (["rule", "huggingface"] if use_slm else ["rule"])
        # Soft-wire llmintent/kineteq when requested via guides
        plan = run_direct(
            {
                "text": text,
                "columns": context.get("columns") or [{"name": c} for c in columns],
                "associations": context.get("associations") or [],
                "edges": context.get("edges") or [],
                "candidates": context.get("candidates") or {},
                "kpis": kpis,
            },
            backends=backends,
            use_slm=use_slm,
        )
        backend_names = list(plan.backends) or backend_names
        if plan.focus_columns:
            focus = [c for c in plan.focus_columns if c in columns] or focus
        if plan.outcome:
            outcome = plan.outcome[0]
        if plan.treatment:
            treatment = [t for t in plan.treatment if t in columns]
        rationale.extend(plan.rationale[:8])
        notes.extend(plan.notes[:6])
        # merge KPI focus with guide focus
        for c in list(getattr(plan, "kpi_focus", None) or []) + focus:
            if c not in kpis and c in columns:
                kpis.append(c)
        # Honor explicit guide imputer/predictor when present (unless use_torch overrides)
        guide_imputer = getattr(plan, "imputer", None)
        guide_predictor = getattr(plan, "predictor", None)
        if guide_imputer:
            notes.append(f"guide suggested imputer={guide_imputer}")
        if guide_predictor:
            notes.append(f"guide suggested predictor={guide_predictor}")
    except Exception as e:
        notes.append(f"guide soft-skip: {type(e).__name__}: {e}")

    # Outcome from KPIs / text if still unset
    if outcome is None:
        for k in kpis:
            if re.search(r"(^y$|outcome|target|revenue|sales)", k, re.I):
                outcome = k
                break
        if outcome is None and kpis:
            outcome = kpis[0]

    # Explicit torch preference
    if use_torch is True:
        prefer = True
        force_torch = True
        force_median = False
    elif use_torch is False:
        prefer = False
        force_torch = False
        force_median = False
    else:
        prefer = torch_preferred() or (guide_imputer == "torch_mlp")
        force_torch = prefer
        force_median = False

    imputer = _choose_imputer(
        prefer_torch=prefer,
        force_torch=force_torch,
        force_median=force_median,
        force_sklearn=False,
    )
    predictor = _choose_predictor(
        prefer_torch=prefer,
        force_torch=force_torch,
        outcome=outcome,
    )

    # Apply guide hints when caller did not force torch on/off
    if use_torch is None and guide_imputer in ("torch_mlp", "iterative", "median", "sklearn"):
        if guide_imputer == "torch_mlp" and not torch_available():
            imputer = "iterative" if sklearn_available() else "median"
        else:
            imputer = guide_imputer  # type: ignore[assignment]
    if use_torch is None and guide_predictor in ("torch_mlp", "sklearn_rf", "none"):
        if guide_predictor == "torch_mlp" and not torch_available():
            predictor = "sklearn_rf" if sklearn_available() else "none"
        else:
            predictor = guide_predictor  # type: ignore[assignment]
    if use_torch is False:
        if imputer == "torch_mlp":
            imputer = "median"
        if predictor == "torch_mlp":
            predictor = "sklearn_rf" if sklearn_available() else "none"

    # Feature columns = KPI focus ∪ treatment ∪ related, excluding pure IDs
    features = list(dict.fromkeys([*kpis, *treatment, *[c for c in focus if c in columns]]))
    if outcome and outcome not in features:
        features.append(outcome)

    if imputer == "torch_mlp":
        rationale.append("SLM/Rule selected torch_mlp imputer (mask→reconstruct MLP).")
    elif imputer == "iterative":
        rationale.append("Selected iterative/sklearn-style imputer; torch not preferred or unavailable.")
    else:
        rationale.append("Selected median/mode imputer (always-available baseline).")

    if predictor == "torch_mlp":
        rationale.append(f"Selected torch_mlp predictor for outcome `{outcome}`.")
    elif predictor == "sklearn_rf":
        rationale.append(f"Selected sklearn_rf predictor for outcome `{outcome}`.")

    if not torch_available() and (use_torch or prefer):
        notes.append("torch not installed; fell back from torch preference (pip install -e '.[torch]').")
        if imputer == "torch_mlp":
            imputer = "iterative" if sklearn_available() else "median"
        if predictor == "torch_mlp":
            predictor = "sklearn_rf" if sklearn_available() else "none"

    return ModelConstructPlan(
        imputer=imputer,
        predictor=predictor,
        kpi_focus=kpis[:16],
        outcome=outcome,
        treatment=treatment[:8],
        feature_columns=features[:32],
        use_physics=prefer_physics,
        backend="+".join(backend_names),
        rationale=rationale,
        notes=notes,
        meta={
            "use_slm": use_slm,
            "use_torch": use_torch,
            "torch_available": torch_available(),
            "torch_preferred_env": _env_flag("AUTOCAUSAL_TORCH"),
            "sklearn_available": sklearn_available(),
            "text": text[:200] if text else "",
        },
    )


def plan_from_backend_override(
    plan: ModelConstructPlan,
    *,
    backend: Optional[str] = None,
) -> ModelConstructPlan:
    """CLI helper: force imputer backend string onto an existing plan."""
    if not backend:
        return plan
    b = backend.strip().lower()
    if b in ("torch", "torch_mlp"):
        plan.imputer = "torch_mlp" if torch_available() else "median"
        if plan.imputer != "torch_mlp":
            plan.notes.append("Requested torch but unavailable; using median.")
    elif b in ("sklearn", "iterative"):
        plan.imputer = "iterative" if sklearn_available() else "median"
    elif b in ("median", "median_mode"):
        plan.imputer = "median"
    else:
        plan.notes.append(f"Unknown backend override {backend!r}; left as {plan.imputer}")
    return plan
