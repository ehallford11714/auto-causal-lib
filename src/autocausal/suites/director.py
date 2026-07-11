"""SLMAutoDirector — shared SLM/rule director for AutoCleanse / AutoEDA / AutoMine.

Every ``auto*`` suite path is SLM-directed when available; deterministic rules
always run as the offline fallback. Never hard-crashes on missing HF/torch.

Env: ``AUTOCAUSAL_SLM=1`` (also ``EMOTIVEVISION_SLM`` / ``CAUSALIV_SLM``).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

import pandas as pd

__all__ = [
    "SuiteStage",
    "SLMDirectives",
    "SLMAutoDirector",
    "resolve_suite_slm",
    "frame_profile",
    "EPISTEMIC_NOTE",
]

SuiteStage = Literal["cleanse", "eda", "mine", "auto"]

EPISTEMIC_NOTE = (
    "SLM text is generative assistance only — not causal identification. "
    "Directives are proposals; applied steps are those feasible under offline rules."
)


def resolve_suite_slm(use_slm: Optional[bool] = None) -> bool:
    """Auto suites default to *trying* SLM (soft). Explicit ``False`` disables.

    When ``use_slm`` is ``None``, prefer SLM (soft-fail to rules). Env flags
    ``AUTOCAUSAL_SLM`` / related also force try-on when set.
    """
    if use_slm is False:
        return False
    if use_slm is True:
        return True
    for n in ("AUTOCAUSAL_SLM", "EMOTIVEVISION_SLM", "CAUSALIV_SLM"):
        if os.environ.get(n, "").strip().lower() in ("1", "true", "yes"):
            return True
    # Default for auto* paths: try SLM, fall back to rules
    return True


def frame_profile(df: pd.DataFrame, *, max_cols: int = 40) -> dict[str, Any]:
    """Compact frame profile for director context (offline, no heavy deps)."""
    cols = [str(c) for c in list(df.columns)[:max_cols]]
    missing: dict[str, float] = {}
    dtypes: dict[str, str] = {}
    nunique: dict[str, int] = {}
    for c in cols:
        s = df[c]
        missing[c] = float(s.isna().mean()) if len(df) else 0.0
        dtypes[c] = str(s.dtype)
        nunique[c] = int(s.nunique(dropna=True))
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": cols,
        "dtypes": dtypes,
        "missingness": missing,
        "nunique": nunique,
    }


@dataclass
class SLMDirectives:
    """Structured directives for one auto-suite stage."""

    stage: SuiteStage
    backend: str = "rule"
    continue_: bool = True
    stop_reason: str = ""
    # cleanse
    drop_columns: list[str] = field(default_factory=list)
    impute_columns: list[str] = field(default_factory=list)
    flag_outliers: list[str] = field(default_factory=list)
    coerce_numeric: list[str] = field(default_factory=list)
    drop_duplicates: bool = True
    drop_constant: bool = True
    # eda
    focus_columns: list[str] = field(default_factory=list)
    role_hypotheses: dict[str, list[str]] = field(default_factory=dict)
    analyses: list[str] = field(default_factory=list)
    # mine
    kpi_focus: list[str] = field(default_factory=list)
    association_pairs: list[dict[str, str]] = field(default_factory=list)
    join_sources: list[str] = field(default_factory=list)
    mine_actions: list[str] = field(default_factory=list)
    # ordered dedicated action names for the suite registry
    actions: list[str] = field(default_factory=list)
    # shared
    notes: list[str] = field(default_factory=list)
    raw_text: str = ""
    generative: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["continue"] = d.pop("continue_", True)
        tools = getattr(self, "_tools_invoked", None)
        if tools:
            d["tools_invoked"] = list(tools)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            f"# SLM directives ({self.stage})",
            "",
            f"**Backend:** `{self.backend}`",
            f"**Continue:** {self.continue_}",
            "",
            f"> {EPISTEMIC_NOTE}",
            "",
        ]
        if self.drop_columns:
            lines.append(f"- Drop columns: {', '.join(f'`{c}`' for c in self.drop_columns)}")
        if self.impute_columns:
            lines.append(f"- Impute: {', '.join(f'`{c}`' for c in self.impute_columns)}")
        if self.focus_columns:
            lines.append(f"- Focus: {', '.join(f'`{c}`' for c in self.focus_columns)}")
        if self.role_hypotheses:
            lines.append("- Role hypotheses:")
            for role, cols in self.role_hypotheses.items():
                lines.append(f"  - {role}: {', '.join(f'`{c}`' for c in cols)}")
        if self.kpi_focus:
            lines.append(f"- KPI focus: {', '.join(f'`{k}`' for k in self.kpi_focus)}")
        if self.join_sources:
            lines.append(f"- Join sources: {', '.join(self.join_sources)}")
        if self.actions:
            lines.append(f"- Actions: {', '.join(f'`{a}`' for a in self.actions)}")
        if self.notes:
            lines.append("")
            lines.append("## Notes")
            for n in self.notes:
                lines.append(f"- {n}")
        lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


class SLMAutoDirector:
    """Propose structured directives for cleanse / eda / mine stages.

    Always computes a rule plan first; optionally enriches via HuggingFace SLM
    through ``autocausal.slm.get_backend`` (soft-fail).
    """

    def __init__(
        self,
        *,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.use_slm = resolve_suite_slm(use_slm)
        self.model_name = model_name

    def direct(
        self,
        stage: SuiteStage,
        df: pd.DataFrame,
        *,
        text: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> SLMDirectives:
        profile = frame_profile(df)
        ctx = dict(context or {})
        ctx.setdefault("text", text)
        ctx["profile"] = profile
        ctx["stage"] = stage
        ctx["columns"] = profile["columns"]

        plan = self._rule_direct(stage, df, profile=profile, text=text, context=ctx)

        # Prefer tool-surface action sequence via skilling broker (rule or SLM)
        tools_invoked: list[dict[str, Any]] = []
        try:
            from autocausal.skilling import SLMToolBroker

            skill_map = {
                "cleanse": "skill:autocleanse",
                "eda": "skill:autoeda",
                "mine": "skill:automine",
                "auto": "skill:autocausal_loop",
            }
            skill_id = skill_map.get(stage)
            if skill_id:
                broker = SLMToolBroker(use_slm=self.use_slm, model_name=self.model_name)
                calls = broker.select_tools(
                    skill_id, text=text, context={"columns": profile.get("columns")}
                )
                # Map tool names → short action names for suite registries
                actions: list[str] = []
                for call in calls:
                    name = str(call.get("name") or "")
                    short = name.split(".", 1)[-1] if "." in name else name
                    if short and short not in actions:
                        actions.append(short)
                    tools_invoked.append({"tool": name, "arguments": call.get("arguments") or {}})
                if actions:
                    plan.actions = actions
                plan.notes.append("Director preferred ToolSurface / SLMToolBroker action sequence.")
        except Exception as e:
            plan.notes.append(f"Skilling broker soft-skip: {type(e).__name__}: {e}")

        if not self.use_slm:
            plan.notes.append("use_slm=False — rule director only.")
            if tools_invoked:
                # stash on notes via dict later — attach after enrich
                pass
            # Attach tools_invoked onto a transient attribute for suites
            plan._tools_invoked = tools_invoked  # type: ignore[attr-defined]
            return plan

        enriched = self._slm_enrich(plan, stage=stage, profile=profile, text=text, context=ctx)
        if enriched is not None:
            enriched._tools_invoked = tools_invoked  # type: ignore[attr-defined]
            return enriched
        plan.notes.append(
            "SLM requested but unavailable/failed — using rule SLMAutoDirector."
        )
        plan.backend = "rule"
        plan._tools_invoked = tools_invoked  # type: ignore[attr-defined]
        return plan

    def _rule_direct(
        self,
        stage: SuiteStage,
        df: pd.DataFrame,
        *,
        profile: dict[str, Any],
        text: str,
        context: dict[str, Any],
    ) -> SLMDirectives:
        cols = list(profile.get("columns") or [])
        missing = dict(profile.get("missingness") or {})
        nunique = dict(profile.get("nunique") or {})
        n_rows = int(profile.get("n_rows") or 0)

        d = SLMDirectives(
            stage=stage,
            backend="rule",
            notes=[EPISTEMIC_NOTE, "Rule director: deterministic heuristics."],
        )

        # High-missing / constant / ID-like
        for c in cols:
            miss = float(missing.get(c, 0.0))
            nunq = int(nunique.get(c, 0))
            cl = c.lower()
            if miss >= 0.95:
                d.drop_columns.append(c)
            elif nunq <= 1 and n_rows > 1:
                d.drop_columns.append(c)
            elif miss > 0 and c not in d.drop_columns:
                d.impute_columns.append(c)
            if any(h in cl for h in ("id", "uuid", "guid")) and nunq >= max(0.9 * max(n_rows, 1), 1):
                if c not in d.drop_columns:
                    d.notes.append(f"ID-like column `{c}` — consider excluding from discovery.")

        # Numeric coercion candidates (object with majority numeric)
        for c in cols:
            if c in d.drop_columns:
                continue
            s = df[c]
            if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
                continue
            if s.dtype == object or str(s.dtype) == "string":
                converted = pd.to_numeric(s, errors="coerce")
                if converted.notna().mean() >= 0.5:
                    d.coerce_numeric.append(c)

        # Outlier flags for numeric with high missing-imputed later
        for c in cols:
            if c in d.drop_columns:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                d.flag_outliers.append(c)

        # Role / focus from name hints + text
        roles: dict[str, list[str]] = {
            "treatment": [],
            "outcome": [],
            "instrument": [],
            "confounder": [],
        }
        _hints = {
            "treatment": ("treat", "exposure", "campaign", "spend", "x_"),
            "outcome": ("outcome", "y_", "revenue", "sales", "target", "churn"),
            "instrument": ("instrument", "z_", "iv_", "lottery", "assign"),
            "confounder": ("age", "gender", "region", "control", "covar"),
        }
        text_l = (text or "").lower()
        for c in cols:
            cl = c.lower()
            for role, hints in _hints.items():
                if any(h in cl for h in hints) or (text_l and cl in text_l):
                    roles[role].append(c)
        d.role_hypotheses = {k: v for k, v in roles.items() if v}
        d.focus_columns = list(
            dict.fromkeys(
                (roles.get("outcome") or [])
                + (roles.get("treatment") or [])
                + (roles.get("instrument") or [])
                + cols[:8]
            )
        )[:12]

        if stage == "cleanse":
            d.analyses = ["missingness", "coerce", "duplicates", "outliers", "impute"]
            d.mine_actions = []
            d.actions = [
                "profile_missingness",
                "coerce_types",
                "drop_high_null_cols",
                "drop_constant_cols",
                "drop_duplicates",
                "strip_id_leakage",
                "flag_outliers",
                "impute",
                "qc_snapshot",
            ]
        elif stage == "eda":
            d.analyses = [
                "distributions",
                "correlations",
                "cardinality",
                "role_hypotheses",
                "qc",
                "leakage_hints",
            ]
            d.actions = [
                "summarize_distributions",
                "correlation_matrix",
                "cardinality_report",
                "suggest_roles",
                "qc_snapshot",
                "leakage_hints",
                "mining_profile",
            ]
            if d.focus_columns:
                d.notes.append(f"EDA focus prioritized: {d.focus_columns[:6]}")
        elif stage in ("mine", "auto"):
            d.analyses = ["associations", "kpis", "public_join"]
            kpi_hints = (
                "revenue",
                "sales",
                "conversion",
                "churn",
                "retention",
                "ltv",
                "ctr",
                "roi",
                "profit",
                "outcome",
            )
            d.kpi_focus = [c for c in cols if any(h in c.lower() for h in kpi_hints)][:8]
            d.mine_actions = ["profile", "associate", "suggest_kpis"]
            d.actions = [
                "mine_associations",
                "mine_kpi_hints",
                "rank_candidates",
                "to_mine_report",
            ]
            # Soft public join suggestions (offline ids)
            if not context.get("skip_join"):
                d.join_sources = ["demographics_demo", "finance_demo"][:1]
                d.mine_actions.append("join_public")
                d.actions.insert(0, "join_public_sources")
            if context.get("include_behavioral"):
                d.actions.append("mine_behavioral")

        if n_rows < 5:
            d.continue_ = False
            d.stop_reason = "Too few rows for stable auto pipeline."
            d.notes.append(d.stop_reason)

        return d

    def _slm_enrich(
        self,
        plan: SLMDirectives,
        *,
        stage: SuiteStage,
        profile: dict[str, Any],
        text: str,
        context: dict[str, Any],
    ) -> Optional[SLMDirectives]:
        try:
            from autocausal.slm import get_backend, slm_available
        except Exception:
            return None

        try:
            backend = get_backend(use_slm=True, model_name=self.model_name)
        except Exception:
            return None

        # Prefer create() for structured role/question proposals; guide for actions
        slm_ctx = {
            "text": text or f"Direct the {stage} stage for causal readiness.",
            "columns": [{"name": c} for c in profile.get("columns") or []],
            "profile": profile,
            "stage": stage,
            "candidates": plan.role_hypotheses,
            "associations": context.get("associations") or [],
            "edges": context.get("edges") or [],
        }

        raw = ""
        backend_name = getattr(backend, "name", "huggingface")
        try:
            if hasattr(backend, "create"):
                cres = backend.create(slm_ctx)
                raw = getattr(cres, "raw_text", "") or ""
                backend_name = getattr(cres, "backend", backend_name)
                # Merge role proposals from creation
                roles = getattr(cres, "roles", None) or {}
                if isinstance(roles, dict):
                    for k, v in roles.items():
                        key = str(k)
                        if key not in plan.role_hypotheses:
                            plan.role_hypotheses[key] = []
                        for col in v or []:
                            if col not in plan.role_hypotheses[key]:
                                plan.role_hypotheses[key].append(str(col))
            if hasattr(backend, "guide") and stage in ("eda", "mine", "auto"):
                gres = backend.guide(slm_ctx)
                graw = getattr(gres, "raw_text", "") or ""
                if graw:
                    raw = (raw + "\n" + graw).strip()
                backend_name = getattr(gres, "backend", backend_name)
                for c in getattr(gres, "focus_columns", None) or []:
                    if c not in plan.focus_columns:
                        plan.focus_columns.append(str(c))
                for c in getattr(gres, "instruments", None) or []:
                    plan.role_hypotheses.setdefault("instrument", [])
                    if c not in plan.role_hypotheses["instrument"]:
                        plan.role_hypotheses["instrument"].append(str(c))
                for c in getattr(gres, "confounders", None) or []:
                    plan.role_hypotheses.setdefault("confounder", [])
                    if c not in plan.role_hypotheses["confounder"]:
                        plan.role_hypotheses["confounder"].append(str(c))
        except Exception as e:
            plan.notes.append(f"SLM enrich soft-fail: {type(e).__name__}: {e}")
            return None

        # Heuristic parse of generative text for drop/impute/kpi/join tokens
        if raw:
            plan.raw_text = raw[:2000]
            plan.generative = True
            cols = set(profile.get("columns") or [])
            for line in re.split(r"[\n;]+", raw):
                line = line.strip(" -*\t")
                if len(line) < 4:
                    continue
                low = line.lower()
                for c in cols:
                    cl = c.lower()
                    if cl not in low:
                        continue
                    if "drop" in low and c not in plan.drop_columns:
                        plan.drop_columns.append(c)
                    if "imput" in low and c not in plan.impute_columns:
                        plan.impute_columns.append(c)
                    if any(k in low for k in ("kpi", "metric", "outcome")) and c not in plan.kpi_focus:
                        plan.kpi_focus.append(c)
                    if c not in plan.focus_columns:
                        plan.focus_columns.append(c)
                if "join" in low:
                    for sid in ("demographics_demo", "finance_demo", "marketing_demo", "policy_demo"):
                        if sid in low and sid not in plan.join_sources:
                            plan.join_sources.append(sid)
                if "stop" in low and "don't stop" not in low:
                    plan.continue_ = False
                    plan.stop_reason = line[:200]

        # If HF unavailable, backend name often contains rule+hf
        if "rule" in str(backend_name) and "hf" in str(backend_name):
            return None  # signal caller to keep rule notes

        if not slm_available() and backend_name == "rule":
            return None

        plan.backend = str(backend_name)
        plan.notes.append(EPISTEMIC_NOTE)
        if plan.generative:
            plan.notes.append("SLM generative text attached; parse is heuristic.")
        # Avoid dumping huge unused JSON
        _ = json.dumps({"stage": stage, "n_cols": profile.get("n_cols")}, default=str)
        return plan
