"""SLM / rule backends for causal *creation* and *inference*.

Creation: propose questions, instruments, morphemes from context.
Inference: interpret discovery/IV results with causal caveats.
Guide: search over intermediate pipeline outputs (legacy path).

Never blocks import on torch/transformers — HuggingFace loads lazily.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol

__all__ = [
    "GuideSuggestion",
    "GuideResult",
    "CreationResult",
    "InferenceResult",
    "RuleBackend",
    "RuleGuide",
    "HuggingFaceSLM",
    "get_backend",
    "get_guide",
    "guide_pipeline",
    "create_from_context",
    "infer_from_results",
    "slm_available",
    "slm_status",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GuideSuggestion:
    action: str  # inspect_columns | drop_edge | validate_edge | instrument | confounder | search_query | focus_table
    detail: str
    priority: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GuideResult:
    backend: str
    suggestions: list[GuideSuggestion]
    focus_columns: list[str] = field(default_factory=list)
    drop_edges: list[dict[str, str]] = field(default_factory=list)
    validate_edges: list[dict[str, str]] = field(default_factory=list)
    instruments: list[str] = field(default_factory=list)
    confounders: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    raw_text: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "focus_columns": self.focus_columns,
            "drop_edges": self.drop_edges,
            "validate_edges": self.validate_edges,
            "instruments": self.instruments,
            "confounders": self.confounders,
            "search_queries": self.search_queries,
            "raw_text": self.raw_text,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = ["# Guide suggestions", "", f"**Backend:** `{self.backend}`", ""]
        if self.focus_columns:
            lines.append("## Focus columns")
            lines.append("")
            for c in self.focus_columns:
                lines.append(f"- `{c}`")
            lines.append("")
        if self.validate_edges:
            lines.append("## Validate edges")
            lines.append("")
            for e in self.validate_edges:
                lines.append(f"- `{e.get('source')}` → `{e.get('target')}`")
            lines.append("")
        if self.drop_edges:
            lines.append("## Consider dropping")
            lines.append("")
            for e in self.drop_edges:
                lines.append(f"- `{e.get('source')}` → `{e.get('target')}`")
            lines.append("")
        if self.instruments:
            lines.append(f"**Instruments:** {', '.join(f'`{i}`' for i in self.instruments)}")
            lines.append("")
        if self.confounders:
            lines.append(f"**Confounders:** {', '.join(f'`{c}`' for c in self.confounders)}")
            lines.append("")
        if self.search_queries:
            lines.append("## Search queries")
            lines.append("")
            for q in self.search_queries:
                lines.append(f"- {q}")
            lines.append("")
        if self.suggestions:
            lines.append("## Actions")
            lines.append("")
            for s in self.suggestions:
                lines.append(f"- [{s.priority:.2f}] **{s.action}**: {s.detail}")
            lines.append("")
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


@dataclass
class CreationResult:
    """SLM/rule proposals for causal *creation* (questions, Z, morphemes)."""

    backend: str
    questions: list[str] = field(default_factory=list)
    instruments: list[dict[str, Any]] = field(default_factory=list)
    morphemes: list[dict[str, Any]] = field(default_factory=list)
    roles: dict[str, list[str]] = field(default_factory=dict)
    raw_text: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = ["# Causal creation proposals", "", f"**Backend:** `{self.backend}`", ""]
        if self.questions:
            lines += ["## Questions", ""]
            for q in self.questions:
                lines.append(f"- {q}")
            lines.append("")
        if self.instruments:
            lines += ["## Instruments", ""]
            for z in self.instruments:
                lines.append(f"- `{z.get('name')}` — {z.get('rationale', '')} (score={z.get('score', '')})")
            lines.append("")
        if self.morphemes:
            lines += ["## Morphemes / role tags", ""]
            for m in self.morphemes:
                lines.append(f"- `{m.get('token')}` → {m.get('role')} ({m.get('detail', '')})")
            lines.append("")
        if self.roles:
            lines += ["## Role buckets", ""]
            for role, cols in self.roles.items():
                if cols:
                    lines.append(f"- **{role}:** {', '.join(f'`{c}`' for c in cols)}")
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


@dataclass
class InferenceResult:
    """SLM/rule interpretation of causal *inference* outputs."""

    backend: str
    narrative: str = ""
    caveats: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    raw_text: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Causal inference interpretation",
            "",
            f"**Backend:** `{self.backend}`",
            f"**Confidence:** {self.confidence:.2f}",
            "",
            "## Narrative",
            "",
            self.narrative or "_No narrative._",
            "",
        ]
        if self.claims:
            lines += ["## Claims", ""]
            for c in self.claims:
                lines.append(
                    f"- `{c.get('source')}` → `{c.get('target')}` "
                    f"(effect={c.get('effect', '—')}; {c.get('note', '')})"
                )
            lines.append("")
        if self.caveats:
            lines += ["## Caveats", ""] + [f"- {c}" for c in self.caveats] + [""]
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        """Ergonomic alias for ``to_markdown()`` / ``to_json()``."""
        if as_markdown:
            return self.to_markdown()
        return self.to_json()


class GuideBackend(Protocol):
    def guide(self, context: dict[str, Any]) -> GuideResult: ...


class CausalSLMBackend(Protocol):
    def guide(self, context: dict[str, Any]) -> GuideResult: ...
    def create(self, context: dict[str, Any]) -> CreationResult: ...
    def infer(self, context: dict[str, Any]) -> InferenceResult: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_flag(*names: str) -> bool:
    for n in names:
        if os.environ.get(n, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _context_summary(context: dict[str, Any]) -> str:
    parts = []
    if context.get("text"):
        parts.append(f"User question: {context['text']}")
    if context.get("emotion") or context.get("intent"):
        parts.append(
            f"Affect/intent: emotion={context.get('emotion')} intent={context.get('intent')} "
            f"valence={context.get('valence')} arousal={context.get('arousal')}"
        )
    cols = context.get("columns") or []
    if cols:
        names = [c.get("name", c) if isinstance(c, dict) else str(c) for c in cols[:20]]
        parts.append("Columns: " + ", ".join(names))
    assocs = context.get("associations") or []
    if assocs:
        top = assocs[:5]
        parts.append(
            "Top associations: "
            + "; ".join(f"{a.get('a')}~{a.get('b')}({a.get('score')})" for a in top)
        )
    edges = context.get("edges") or []
    if edges:
        parts.append(
            "Draft edges: "
            + "; ".join(f"{e.get('source')}->{e.get('target')}" for e in edges[:8])
        )
    cands = context.get("candidates") or {}
    if cands:
        parts.append(f"Candidates: {json.dumps(cands, default=str)[:400]}")
    iv = context.get("iv") or context.get("estimates")
    if iv:
        parts.append(f"Estimates: {json.dumps(iv, default=str)[:400]}")
    return "\n".join(parts)


def _col_names(context: dict[str, Any]) -> list[str]:
    columns = context.get("columns") or []
    return [c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in columns]


_IV_NAME_HINTS = ("z", "iv", "instrument", "assign", "lottery", "shock", "exog", "random")
_TREAT_HINTS = ("treat", "d_", "exposure", "campaign", "spend", "policy", "dose")
_OUT_HINTS = ("y_", "outcome", "revenue", "sales", "churn", "conversion", "kpi", "score")
_CONF_HINTS = ("age", "income", "region", "segment", "gender", "cohort", "baseline")


def _bucket_roles(col_names: list[str]) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {
        "instrument": [],
        "treatment": [],
        "outcome": [],
        "confounder": [],
    }
    for c in col_names:
        cl = c.lower()
        if any(h in cl for h in _IV_NAME_HINTS):
            roles["instrument"].append(c)
        if any(h in cl for h in _TREAT_HINTS):
            roles["treatment"].append(c)
        if any(h in cl for h in _OUT_HINTS):
            roles["outcome"].append(c)
        if any(h in cl for h in _CONF_HINTS):
            roles["confounder"].append(c)
    return roles


def slm_available() -> bool:
    """True if transformers can be imported (torch may still fail at load)."""
    try:
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def slm_status() -> dict[str, Any]:
    """Backend availability snapshot for status UIs (e.g. CausalBridge)."""
    env_on = _env_flag("AUTOCAUSAL_SLM", "EMOTIVEVISION_SLM", "CAUSALIV_SLM")
    transformers_ok = False
    torch_ok = False
    err = None
    try:
        import transformers  # noqa: F401

        transformers_ok = True
    except Exception as e:
        err = f"transformers: {type(e).__name__}: {e}"
    try:
        import torch  # noqa: F401

        torch_ok = True
    except Exception as e:
        if err is None:
            err = f"torch: {type(e).__name__}: {e}"
    return {
        "rule_backend": True,
        "env_slm_enabled": env_on,
        "transformers_installed": transformers_ok,
        "torch_installed": torch_ok,
        "huggingface_ready": transformers_ok and torch_ok,
        "default_model": os.environ.get("AUTOCAUSAL_SLM_MODEL") or "sshleifer/tiny-gpt2",
        "recommended_instruct": [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
        ],
        "error": err,
    }


# ---------------------------------------------------------------------------
# Rule backend (always available)
# ---------------------------------------------------------------------------


class RuleBackend:
    """Deterministic offline backend — always available (alias: RuleGuide)."""

    name = "rule"

    def guide(self, context: dict[str, Any]) -> GuideResult:
        edges = list(context.get("edges") or [])
        assocs = list(context.get("associations") or [])
        candidates = dict(context.get("candidates") or {})
        text = (context.get("text") or "").lower()
        col_names = _col_names(context)

        suggestions: list[GuideSuggestion] = []
        focus: list[str] = []
        validate: list[dict[str, str]] = []
        drop: list[dict[str, str]] = []
        instruments = list(candidates.get("instrument") or [])
        confounders = list(candidates.get("confounder") or [])
        queries: list[str] = []

        ranked = sorted(
            edges,
            key=lambda e: float(e.get("confidence") or e.get("score") or 0),
            reverse=True,
        )
        for e in ranked[:5]:
            validate.append({"source": str(e["source"]), "target": str(e["target"])})
            focus.extend([str(e["source"]), str(e["target"])])
            suggestions.append(
                GuideSuggestion(
                    action="validate_edge",
                    detail=f"Validate {e['source']} → {e['target']} (conf={e.get('confidence', e.get('score'))})",
                    priority=float(e.get("confidence") or 0.5),
                    meta=e,
                )
            )

        for e in edges:
            names = f"{e.get('source', '')} {e.get('target', '')}".lower()
            score = float(e.get("confidence") or e.get("score") or 0)
            if "noise" in names or score < 0.12:
                drop.append({"source": str(e["source"]), "target": str(e["target"])})
                suggestions.append(
                    GuideSuggestion(
                        action="drop_edge",
                        detail=f"Consider dropping weak/noise edge {e['source']}→{e['target']}",
                        priority=0.3,
                    )
                )

        for c in col_names:
            cl = c.lower()
            if any(h in cl for h in _IV_NAME_HINTS):
                if c not in instruments:
                    instruments.append(c)
                suggestions.append(
                    GuideSuggestion(
                        action="instrument",
                        detail=f"Column `{c}` looks instrument-like",
                        priority=0.7,
                    )
                )
            if any(h in cl for h in _CONF_HINTS):
                if c not in confounders:
                    confounders.append(c)

        if text:
            for c in col_names:
                if c.lower() in text or any(
                    tok and tok in text for tok in re.split(r"\W+", c.lower()) if len(tok) > 2
                ):
                    focus.append(c)
                    suggestions.append(
                        GuideSuggestion(
                            action="inspect_columns",
                            detail=f"User question mentions `{c}`",
                            priority=0.85,
                        )
                    )
            m = re.search(r"caus(?:e|es|ing)\s+(\w+)", text)
            if m:
                y = m.group(1)
                queries.append(f"what causes {y}")
                for c in col_names:
                    if y in c.lower():
                        focus.append(c)

        for a in assocs[:5]:
            focus.extend([str(a.get("a")), str(a.get("b"))])
            suggestions.append(
                GuideSuggestion(
                    action="inspect_columns",
                    detail=f"Strong association {a.get('a')}~{a.get('b')} ({a.get('metric')}={a.get('score')})",
                    priority=float(a.get("score") or 0.4),
                )
            )

        for e in validate[:3]:
            queries.append(f"causal evidence {e['source']} affects {e['target']}")

        seen: set[str] = set()
        focus_u = []
        for c in focus:
            if c and c not in seen:
                seen.add(c)
                focus_u.append(c)

        return GuideResult(
            backend=self.name,
            suggestions=suggestions[:40],
            focus_columns=focus_u[:20],
            drop_edges=drop[:20],
            validate_edges=validate[:20],
            instruments=instruments[:10],
            confounders=confounders[:10],
            search_queries=queries[:10],
            notes=["RuleBackend is offline and deterministic."],
        )

    def create(self, context: dict[str, Any]) -> CreationResult:
        col_names = _col_names(context)
        roles = _bucket_roles(col_names)
        text = (context.get("text") or "").strip()
        emotion = context.get("emotion")
        intent = context.get("intent")
        candidates = dict(context.get("candidates") or {})

        # Prefer candidate lists from discovery when present
        for key in ("instrument", "treatment", "outcome", "confounder"):
            for c in candidates.get(key) or []:
                if c not in roles[key]:
                    roles[key].append(str(c))

        questions: list[str] = []
        treats = roles["treatment"] or [c for c in col_names if "x" in c.lower()][:2]
        outs = roles["outcome"] or [c for c in col_names if "y" in c.lower()][:2]
        zs = roles["instrument"]

        if treats and outs:
            questions.append(f"Does `{treats[0]}` cause `{outs[0]}`?")
        if zs and treats and outs:
            questions.append(
                f"Using `{zs[0]}` as an instrument, what is the effect of `{treats[0]}` on `{outs[0]}`?"
            )
        if emotion or intent:
            questions.append(
                f"How does affect ({emotion or 'unknown'}) / intent ({intent or 'unknown'}) "
                f"relate to outcomes {', '.join(f'`{o}`' for o in outs[:2]) or 'KPIs'}?"
            )
        if text:
            questions.append(f"Refine: {text}")
        if not questions:
            questions.append("Which columns are plausible treatments vs outcomes?")

        instruments: list[dict[str, Any]] = []
        for z in zs[:8]:
            instruments.append(
                {
                    "name": z,
                    "rationale": "Name/heuristic suggests exogenous or assignment-like variable",
                    "score": 0.7,
                }
            )
        # text cues
        lower = text.lower()
        for cue, name, score in (
            ("lottery", "lottery_assignment", 0.85),
            ("random", "randomized_assignment", 0.9),
            ("rainfall", "weather_rainfall", 0.75),
            ("judge", "judge_leniency", 0.8),
            ("shift-share", "shift_share", 0.8),
            ("bartik", "bartik_shift_share", 0.85),
        ):
            if cue in lower:
                instruments.append(
                    {"name": name, "rationale": f"Text cue '{cue}'", "score": score}
                )

        morphemes: list[dict[str, Any]] = []
        for role, cols in roles.items():
            for c in cols[:6]:
                morphemes.append(
                    {
                        "token": c,
                        "role": role,
                        "detail": f"Heuristic role tag from column name ({role})",
                    }
                )
        if emotion:
            morphemes.append(
                {
                    "token": str(emotion),
                    "role": "affect_context",
                    "detail": "Emotive context — not a causal instrument by itself",
                }
            )
        if intent:
            morphemes.append(
                {
                    "token": str(intent),
                    "role": "intent_context",
                    "detail": "Communicative intent — use as covariate/context, not Z",
                }
            )

        return CreationResult(
            backend=self.name,
            questions=questions[:12],
            instruments=instruments[:12],
            morphemes=morphemes[:30],
            roles=roles,
            notes=[
                "RuleBackend creation is heuristic; validate exclusion/relevance before IV.",
                "Affect/intent are contextual — do not treat as instruments.",
            ],
        )

    def infer(self, context: dict[str, Any]) -> InferenceResult:
        edges = list(context.get("edges") or [])
        iv = context.get("iv") or context.get("estimates") or {}
        emotion = context.get("emotion")
        intent = context.get("intent")
        caveats = [
            "Exploratory associations are not identified causal effects.",
            "Check instrument relevance (first-stage F) and exclusion before claiming IV effects.",
            "Placebo and sensitivity checks are recommended before decisions.",
        ]
        claims: list[dict[str, Any]] = []
        for e in edges[:8]:
            claims.append(
                {
                    "source": e.get("source"),
                    "target": e.get("target"),
                    "effect": e.get("confidence", e.get("score")),
                    "note": "exploratory edge",
                }
            )

        narrative_parts = []
        if claims:
            top = claims[0]
            narrative_parts.append(
                f"Top exploratory link: `{top['source']}` → `{top['target']}` "
                f"(score={top['effect']}). Treat as a hypothesis, not proof."
            )
        if isinstance(iv, dict) and iv:
            fstat = iv.get("first_stage_f") or iv.get("f")
            coef = iv.get("coef") or iv.get("ate") or iv.get("effect")
            if coef is not None:
                narrative_parts.append(f"Point estimate ≈ {coef}.")
            if fstat is not None:
                try:
                    fval = float(fstat)
                    if fval < 10:
                        caveats.append(f"Weak instrument warning: first-stage F≈{fval:.2f} (<10).")
                    narrative_parts.append(f"First-stage F≈{fval:.2f}.")
                except (TypeError, ValueError):
                    pass
        if emotion or intent:
            narrative_parts.append(
                f"Affect/intent context ({emotion}/{intent}) may correlate with outcomes; "
                "do not interpret as a causal mechanism without design."
            )
        if not narrative_parts:
            narrative_parts.append("Insufficient structure for a strong causal narrative.")

        conf = 0.35
        if claims:
            try:
                conf = min(0.75, 0.35 + float(claims[0].get("effect") or 0) * 0.4)
            except (TypeError, ValueError):
                pass

        return InferenceResult(
            backend=self.name,
            narrative=" ".join(narrative_parts),
            caveats=caveats,
            claims=claims,
            confidence=round(conf, 3),
            notes=["RuleBackend inference is template-based."],
        )


# Back-compat alias
RuleGuide = RuleBackend


# ---------------------------------------------------------------------------
# HuggingFace SLM (lazy)
# ---------------------------------------------------------------------------


class HuggingFaceSLM:
    """Lazy Hugging Face transformers backend (optional heavy deps)."""

    name = "huggingface"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = (
            model_name
            or os.environ.get("AUTOCAUSAL_SLM_MODEL")
            or "sshleifer/tiny-gpt2"
        )
        self._pipe = None
        self._error: Optional[str] = None
        self._rule = RuleBackend()

    def _ensure(self) -> bool:
        if self._pipe is not None:
            return True
        if self._error:
            return False
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as e:
            self._error = (
                "transformers not installed; pip install 'autocausal[slm]'. "
                f"({e})"
            )
            return False
        try:
            self._pipe = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=120,
            )
            return True
        except Exception as e:
            self._error = f"SLM load failed (soft-fail): {type(e).__name__}: {e}"
            return False

    def _generate(self, prompt: str) -> str:
        if not self._ensure():
            return ""
        try:
            assert self._pipe is not None
            out = self._pipe(prompt, do_sample=False, truncation=True)
            text = out[0]["generated_text"] if out else ""
            if text.startswith(prompt):
                text = text[len(prompt) :]
            return text.strip()[:2000]
        except Exception as e:
            self._error = f"SLM generate soft-fail: {type(e).__name__}: {e}"
            return ""

    def guide(self, context: dict[str, Any]) -> GuideResult:
        base = self._rule.guide(context)
        if not self._ensure():
            base.backend = "rule+hf_unavailable"
            base.notes.append(self._error or "HF SLM unavailable")
            return base

        prompt = (
            "You are a causal analysis assistant. Given this summary, list next steps "
            "as short bullets: columns to inspect, edges to validate/drop, instruments, "
            "confounders, search queries.\n\n"
            + _context_summary(context)
            + "\n\nSuggestions:\n-"
        )
        text = self._generate(prompt)
        if not text:
            base.backend = "rule+hf_error"
            base.notes.append(self._error or "SLM generate failed")
            return base

        base.raw_text = text
        base.backend = f"huggingface:{self.model_name}"
        for line in re.split(r"[\n;]+", base.raw_text):
            line = line.strip(" -*\t")
            if len(line) < 8:
                continue
            action = "inspect_columns"
            low = line.lower()
            if "drop" in low:
                action = "drop_edge"
            elif "valid" in low or "confirm" in low:
                action = "validate_edge"
            elif "instrument" in low:
                action = "instrument"
            elif "confound" in low:
                action = "confounder"
            elif "search" in low or "query" in low:
                action = "search_query"
                base.search_queries.append(line[:200])
            base.suggestions.append(
                GuideSuggestion(action=action, detail=line[:300], priority=0.55)
            )
        base.notes.append("HuggingFace SLM used; parse is heuristic.")
        return base

    def create(self, context: dict[str, Any]) -> CreationResult:
        base = self._rule.create(context)
        prompt = (
            "Propose causal questions, instruments (Z), and role tags from this context. "
            "Short bullets only.\n\n"
            + _context_summary(context)
            + "\n\nProposals:\n-"
        )
        text = self._generate(prompt)
        if not text:
            base.backend = "rule+hf_unavailable" if self._error else base.backend
            if self._error:
                base.notes.append(self._error)
            return base
        base.raw_text = text
        base.backend = f"huggingface:{self.model_name}"
        for line in re.split(r"[\n;]+", text):
            line = line.strip(" -*\t")
            if len(line) < 8:
                continue
            low = line.lower()
            if "?" in line or "does " in low or "what " in low:
                base.questions.append(line[:240])
            elif "instrument" in low or " z " in f" {low} ":
                base.instruments.append(
                    {"name": line[:80], "rationale": "SLM proposal", "score": 0.55}
                )
            else:
                base.morphemes.append(
                    {"token": line[:60], "role": "slm_tag", "detail": line[:200]}
                )
        base.notes.append("HuggingFace SLM creation; merge with rule heuristics.")
        return base

    def infer(self, context: dict[str, Any]) -> InferenceResult:
        base = self._rule.infer(context)
        prompt = (
            "Interpret these causal results cautiously. Give a short narrative and caveats.\n\n"
            + _context_summary(context)
            + "\n\nInterpretation:\n"
        )
        text = self._generate(prompt)
        if not text:
            base.backend = "rule+hf_unavailable" if self._error else base.backend
            if self._error:
                base.notes.append(self._error)
            return base
        base.raw_text = text
        base.backend = f"huggingface:{self.model_name}"
        # Prefer SLM narrative but keep rule caveats
        base.narrative = (text.split("\n")[0][:500] or base.narrative)
        for line in re.split(r"[\n;]+", text):
            line = line.strip(" -*\t")
            if any(k in line.lower() for k in ("caveat", "warning", "not causal", "confound", "weak")):
                if line not in base.caveats:
                    base.caveats.append(line[:240])
        base.notes.append("HuggingFace SLM inference; caveats still apply.")
        return base


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def get_backend(
    *,
    use_slm: bool = False,
    model_name: Optional[str] = None,
) -> CausalSLMBackend:
    """Return RuleBackend by default; HF when use_slm or AUTOCAUSAL_SLM=1."""
    env_on = _env_flag("AUTOCAUSAL_SLM", "EMOTIVEVISION_SLM", "CAUSALIV_SLM")
    if use_slm or env_on:
        return HuggingFaceSLM(model_name=model_name)
    return RuleBackend()


def get_guide(*, use_slm: bool = False, model_name: Optional[str] = None) -> GuideBackend:
    """Back-compat: same as get_backend."""
    return get_backend(use_slm=use_slm, model_name=model_name)


def guide_pipeline(
    context: dict[str, Any],
    *,
    use_slm: bool = False,
    model_name: Optional[str] = None,
) -> GuideResult:
    return get_backend(use_slm=use_slm, model_name=model_name).guide(context)


def create_from_context(
    context: dict[str, Any],
    *,
    use_slm: bool = False,
    model_name: Optional[str] = None,
) -> CreationResult:
    """Creation API: questions / instruments / morphemes."""
    return get_backend(use_slm=use_slm, model_name=model_name).create(context)


def infer_from_results(
    context: dict[str, Any],
    *,
    use_slm: bool = False,
    model_name: Optional[str] = None,
) -> InferenceResult:
    """Inference API: narrative + caveats over edges/estimates."""
    return get_backend(use_slm=use_slm, model_name=model_name).infer(context)
