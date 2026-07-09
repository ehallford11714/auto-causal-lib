"""LLMIntentGuide — soft-optional adapter over llmintent intent/morpheme/report APIs."""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from autocausal.guides.types import GuideResult, GuideSuggestion, col_names, uniq

_IV_HINTS = ("z", "iv", "instrument", "assign", "lottery", "shock", "exog", "random")
_TREAT_HINTS = ("treat", "d_", "exposure", "campaign", "spend", "policy", "dose")
_OUT_HINTS = ("y_", "outcome", "revenue", "sales", "churn", "conversion", "kpi", "score")


def llmintent_importable() -> bool:
    try:
        import llmintent  # noqa: F401

        return True
    except Exception:
        return False


def _token_overlap(text: str, col: str) -> float:
    toks = {t for t in re.split(r"\W+", text.lower()) if len(t) > 2}
    ctoks = {t for t in re.split(r"\W+", col.lower()) if len(t) > 2}
    if not toks or not ctoks:
        return 1.0 if col.lower() in text.lower() else 0.0
    return len(toks & ctoks) / max(1, len(ctoks))


def _heuristic_roles(names: list[str], text: str) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {
        "treatment": [],
        "outcome": [],
        "instrument": [],
        "confounder": [],
    }
    lower = text.lower()
    for c in names:
        cl = c.lower()
        score = _token_overlap(lower, c)
        if any(h in cl for h in _IV_HINTS):
            roles["instrument"].append(c)
        if any(h in cl for h in _TREAT_HINTS) or ("cause" in lower and score > 0):
            roles["treatment"].append(c)
        if any(h in cl for h in _OUT_HINTS) or score > 0.4:
            if any(h in cl for h in _OUT_HINTS) or ("cause" in lower and score > 0.3):
                roles["outcome"].append(c)
        if any(h in cl for h in ("age", "income", "region", "segment", "gender", "cohort")):
            roles["confounder"].append(c)
        if score > 0.5 and c not in roles["treatment"] and c not in roles["outcome"]:
            roles["treatment" if "treat" in lower or "effect of" in lower else "outcome"].append(c)
    return roles


class LLMIntentGuide:
    """
    Use llmintent (when installed) to propose causal direction.

    Soft modes:
    - Full: LLMIntentAnalyzer.analyze_prompt + heighten/query when AUTOCAUSAL_LLMINTENT_MODEL set
    - Light: import llmintent APIs / heighten.retrace templates without loading a transformer
    - Offline stub: column/text heuristics labeled as llmintent_stub
    """

    name = "llmintent"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = (
            model_name
            or os.environ.get("AUTOCAUSAL_LLMINTENT_MODEL")
            or os.environ.get("LLMINTENT_MODEL")
            or ""
        ).strip()

    def available(self) -> bool:
        return llmintent_importable()

    def guide(self, context: dict[str, Any]) -> GuideResult:
        names = col_names(context)
        text = (context.get("text") or "").strip()
        edges = list(context.get("edges") or [])
        assocs = list(context.get("associations") or [])
        candidates = dict(context.get("candidates") or {})

        if not llmintent_importable():
            return self._stub(
                names,
                text,
                edges,
                assocs,
                candidates,
                note="llmintent not installed — soft-fail stub. "
                "Install: pip install -e ../LLMIntent",
            )

        notes: list[str] = []
        raw_bits: list[str] = []
        next_q: list[str] = []
        morpheme_tokens: list[str] = []

        # Prefer light APIs that do not require torch model load
        try:
            from llmintent.heighten.retrace import build_retrace_prompt, build_focused_prompt

            concepts = names[:6] or ["treatment", "outcome"]
            if text:
                next_q.append(
                    build_focused_prompt(text, concepts=concepts)[:240]
                )
                next_q.append(
                    build_retrace_prompt(text, concepts=concepts)[:240]
                )
            notes.append("Used llmintent.heighten.retrace prompt scaffolding.")
        except Exception as e:
            notes.append(f"heighten.retrace soft-skip: {type(e).__name__}: {e}")

        try:
            from llmintent.morphemes import MorphemeExtractor

            ext = MorphemeExtractor("lemma")
            corpus = " ".join([text] + names + [str(a.get("a", "")) for a in assocs[:5]])
            morpheme_tokens = list(dict.fromkeys(ext.extract(corpus.split())[:30]))
            if morpheme_tokens:
                raw_bits.append("morphemes=" + ",".join(morpheme_tokens[:12]))
            notes.append("Used llmintent.morphemes.MorphemeExtractor.")
        except Exception as e:
            notes.append(f"morphemes soft-skip: {type(e).__name__}: {e}")

        # Optional heavy analyzer when model env is set
        if self.model_name and os.environ.get("AUTOCAUSAL_LLMINTENT_HEAVY", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            try:
                from llmintent import LLMIntentAnalyzer

                analyzer = LLMIntentAnalyzer(
                    self.model_name, load_glove=False, fit_jspace_transport=False
                )
                prompt = text or f"Causal roles among: {', '.join(names[:12])}"
                report = analyzer.analyze_prompt(prompt, include_jspace=False)
                raw_bits.append(f"inference_pivot={report.inference_pivot}")
                raw_bits.append(f"activation_layers={report.activation_layers}")
                notes.append(f"LLMIntentAnalyzer({self.model_name}) ran (heavy).")
                try:
                    analyzer.cleanup()
                except Exception:
                    pass
            except Exception as e:
                notes.append(f"LLMIntentAnalyzer soft-fail: {type(e).__name__}: {e}")

        roles = _heuristic_roles(names, text)
        for key in ("instrument", "treatment", "outcome", "confounder"):
            for c in candidates.get(key) or []:
                if str(c) not in roles[key]:
                    roles[key].append(str(c))

        # Morpheme tokens that match columns boost focus
        focus: list[str] = []
        for c in names:
            cl = c.lower()
            if any(m.lower() in cl or cl in m.lower() for m in morpheme_tokens if len(m) > 2):
                focus.append(c)
            if _token_overlap(text, c) > 0.3:
                focus.append(c)
        focus.extend(roles["treatment"])
        focus.extend(roles["outcome"])
        focus.extend(roles["instrument"])

        boost: list[dict[str, Any]] = []
        suppress: list[dict[str, Any]] = []
        for e in edges:
            src, tgt = str(e.get("source")), str(e.get("target"))
            conf = float(e.get("confidence") or e.get("score") or 0)
            # Prefer treatment→outcome; suppress reverse if both present
            if src in roles["treatment"] and tgt in roles["outcome"]:
                boost.append(
                    {
                        "source": src,
                        "target": tgt,
                        "reason": "llmintent_role_align",
                        "backend": self.name,
                    }
                )
            elif src in roles["outcome"] and tgt in roles["treatment"]:
                suppress.append(
                    {
                        "source": src,
                        "target": tgt,
                        "reason": "llmintent_reverse_path",
                        "backend": self.name,
                    }
                )
            elif conf < 0.12 or "noise" in f"{src} {tgt}".lower():
                suppress.append(
                    {
                        "source": src,
                        "target": tgt,
                        "reason": "llmintent_weak",
                        "backend": self.name,
                    }
                )

        if roles["treatment"] and roles["outcome"]:
            next_q.append(
                f"Does `{roles['treatment'][0]}` cause `{roles['outcome'][0]}`?"
            )
        if roles["instrument"] and roles["treatment"] and roles["outcome"]:
            next_q.append(
                f"Instrument `{roles['instrument'][0]}` for "
                f"`{roles['treatment'][0]}` → `{roles['outcome'][0]}`?"
            )

        suggestions = [
            GuideSuggestion(
                action="inspect_columns",
                detail=f"LLMIntent focus on `{c}`",
                priority=0.75,
            )
            for c in uniq(focus, limit=8)
        ]
        for e in boost[:5]:
            suggestions.append(
                GuideSuggestion(
                    action="validate_edge",
                    detail=f"Keep {e['source']}→{e['target']} ({e['reason']})",
                    priority=0.8,
                )
            )
        for e in suppress[:5]:
            suggestions.append(
                GuideSuggestion(
                    action="drop_edge",
                    detail=f"Drop/suppress {e['source']}→{e['target']} ({e['reason']})",
                    priority=0.55,
                )
            )

        queries = [
            f"causal evidence {a} affects {b}"
            for a, b in zip(roles["treatment"][:2], roles["outcome"][:2])
        ]
        if text:
            queries.append(text[:160])

        return GuideResult(
            backend=self.name,
            available=True,
            suggestions=suggestions[:40],
            focus_columns=uniq(focus, limit=20),
            instruments=uniq(roles["instrument"] + list(candidates.get("instrument") or []), limit=10),
            confounders=uniq(roles["confounder"] + list(candidates.get("confounder") or []), limit=10),
            treatment=uniq(roles["treatment"], limit=8),
            outcome=uniq(roles["outcome"], limit=8),
            boost_edges=boost[:20],
            suppress_edges=suppress[:20],
            drop_edges=[
                {"source": e["source"], "target": e["target"]} for e in suppress[:20]
            ],
            validate_edges=[
                {"source": e["source"], "target": e["target"]} for e in boost[:20]
            ],
            next_questions=uniq(next_q, limit=10),
            search_queries=uniq(queries, limit=10),
            related_variables=uniq(morpheme_tokens, limit=15),
            raw_text="; ".join(raw_bits),
            notes=notes
            + [
                "LLMIntentGuide maps intent/morpheme signals onto tabular causal roles.",
                "Heavy analyzer only runs when AUTOCAUSAL_LLMINTENT_HEAVY=1 and a model is set.",
            ],
        )

    def _stub(
        self,
        names: list[str],
        text: str,
        edges: list[dict[str, Any]],
        assocs: list[dict[str, Any]],
        candidates: dict[str, Any],
        *,
        note: str,
    ) -> GuideResult:
        roles = _heuristic_roles(names, text)
        focus = uniq(
            roles["treatment"] + roles["outcome"] + roles["instrument"] + [
                str(a.get("a")) for a in assocs[:3]
            ] + [str(a.get("b")) for a in assocs[:3]],
            limit=20,
        )
        return GuideResult(
            backend="llmintent_stub",
            available=False,
            focus_columns=focus,
            treatment=roles["treatment"][:8],
            outcome=roles["outcome"][:8],
            instruments=uniq(roles["instrument"] + list(candidates.get("instrument") or []), limit=10),
            confounders=uniq(roles["confounder"] + list(candidates.get("confounder") or []), limit=10),
            next_questions=[text] if text else [],
            notes=[note],
            suggestions=[
                GuideSuggestion(
                    action="inspect_columns",
                    detail="Stub LLMIntent roles from column/text heuristics",
                    priority=0.4,
                )
            ],
        )
