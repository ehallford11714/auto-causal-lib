"""RetracementGuide — soft-optional adapter over llmintent.retracement (+ stub).

Standalone `retracement` packages are not present on disk; when
`llmintent.retracement` is importable we use its config/modes and reverse-path
heuristics. Otherwise a documented stub biases temporal/causal direction from
association order and edge reverse checks.
"""

from __future__ import annotations

from typing import Any, Optional

from autocausal.guides.types import GuideResult, GuideSuggestion, col_names, uniq


def retracement_importable() -> tuple[bool, str]:
    """Return (ok, source_label). Prefer llmintent.retracement; else bare retracement."""
    try:
        import llmintent.retracement  # noqa: F401

        return True, "llmintent.retracement"
    except Exception:
        pass
    try:
        import retracement  # noqa: F401

        return True, "retracement"
    except Exception:
        return False, ""


class RetracementGuide:
    """
    Bias temporal/causal direction using retracement signals.

    - Lag structure hints from column name suffixes (_t, _lag, lag_)
    - Reverse-path checks: if A→B and B→A both appear, prefer higher-score direction
    - "Retracement" of effects: suppress edges that reverse a strong association path
    """

    name = "retracement"

    def available(self) -> bool:
        ok, _ = retracement_importable()
        return ok

    def guide(self, context: dict[str, Any]) -> GuideResult:
        names = col_names(context)
        text = (context.get("text") or "").strip()
        edges = list(context.get("edges") or [])
        assocs = list(context.get("associations") or [])
        ok, source = retracement_importable()

        notes: list[str] = []
        mode_label = "stub"
        if ok:
            mode_label = source
            try:
                from llmintent.retracement import RetracementConfig, RetracementMode

                cfg = RetracementConfig(mode=RetracementMode.FOCUS_GATE)
                notes.append(
                    f"Bound llmintent.retracement ({cfg.mode.value}); "
                    "using reverse-path + lag heuristics (no model load)."
                )
            except Exception as e:
                notes.append(f"retracement config soft-skip: {type(e).__name__}: {e}")
                notes.append(f"Using {source} presence + local reverse-path heuristics.")
        else:
            notes.append(
                "retracement package not found — documented stub adapter. "
                "Will bind to llmintent.retracement or a standalone retracement "
                "package when installed (pip install -e ../LLMIntent)."
            )

        lag_hints = self._lag_hints(names)
        boost, suppress, focus = self._path_bias(edges, assocs, names)

        # Temporal columns preferred as instruments / treatments upstream
        instruments: list[str] = []
        treatment: list[str] = []
        for h in lag_hints:
            col = h.get("column")
            kind = h.get("kind")
            if not col:
                continue
            focus.append(str(col))
            if kind == "lag":
                instruments.append(str(col))
                treatment.append(str(col))
            elif kind == "lead":
                # lead outcomes — prefer as Y
                pass

        next_q: list[str] = []
        if boost:
            e = boost[0]
            next_q.append(
                f"Retrace: confirm `{e['source']}` → `{e['target']}` vs reverse path?"
            )
        if lag_hints:
            next_q.append(
                "Check lag structure: do lagged covariates precede treatment/outcome?"
            )
        if text:
            next_q.append(f"Retrace effect path for: {text[:160]}")

        suggestions = [
            GuideSuggestion(
                action="validate_edge",
                detail=f"Retracement boost {e['source']}→{e['target']}",
                priority=0.78,
                meta=e,
            )
            for e in boost[:6]
        ]
        suggestions += [
            GuideSuggestion(
                action="drop_edge",
                detail=f"Retracement suppress reverse {e['source']}→{e['target']}",
                priority=0.6,
                meta=e,
            )
            for e in suppress[:6]
        ]
        for h in lag_hints[:6]:
            suggestions.append(
                GuideSuggestion(
                    action="inspect_columns",
                    detail=f"Lag hint `{h.get('column')}` ({h.get('kind')})",
                    priority=0.65,
                    meta=h,
                )
            )

        backend = self.name if ok else "retracement_stub"
        return GuideResult(
            backend=backend,
            available=ok,
            suggestions=suggestions[:40],
            focus_columns=uniq(focus, limit=20),
            instruments=uniq(instruments, limit=10),
            treatment=uniq(treatment, limit=8),
            boost_edges=boost[:20],
            suppress_edges=suppress[:20],
            drop_edges=[
                {"source": e["source"], "target": e["target"]} for e in suppress[:20]
            ],
            validate_edges=[
                {"source": e["source"], "target": e["target"]} for e in boost[:20]
            ],
            lag_hints=lag_hints[:20],
            next_questions=uniq(next_q, limit=8),
            search_queries=[
                f"reverse causality {e['target']} causes {e['source']}"
                for e in suppress[:3]
            ],
            notes=notes
            + [
                f"mode={mode_label}",
                "RetracementGuide biases direction via reverse-path checks and lag tokens.",
            ],
        )

    def _lag_hints(self, names: list[str]) -> list[dict[str, Any]]:
        hints: list[dict[str, Any]] = []
        for c in names:
            cl = c.lower()
            if any(tok in cl for tok in ("_lag", "lag_", "_t1", "_t2", "_tm1", "lagged")):
                hints.append({"column": c, "kind": "lag", "reason": "name_token"})
            elif any(tok in cl for tok in ("_lead", "lead_", "_tp1", "forward_")):
                hints.append({"column": c, "kind": "lead", "reason": "name_token"})
            elif cl.endswith("_t") or "_t_" in cl:
                hints.append({"column": c, "kind": "time_indexed", "reason": "name_token"})
        return hints

    def _path_bias(
        self,
        edges: list[dict[str, Any]],
        assocs: list[dict[str, Any]],
        names: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        boost: list[dict[str, Any]] = []
        suppress: list[dict[str, Any]] = []
        focus: list[str] = []

        # Index edges by undirected pair
        pair_best: dict[tuple[str, str], dict[str, Any]] = {}
        for e in edges:
            src, tgt = str(e.get("source")), str(e.get("target"))
            if not src or not tgt:
                continue
            score = float(e.get("confidence") or e.get("score") or 0)
            key = tuple(sorted([src, tgt]))
            prev = pair_best.get(key)
            directed = {
                "source": src,
                "target": tgt,
                "score": score,
                "reason": "retracement_forward",
                "backend": self.name,
            }
            if prev is None or score > float(prev.get("score") or 0):
                # If reversing a previous winner, suppress the weaker reverse
                if prev is not None and (
                    prev.get("source") != src or prev.get("target") != tgt
                ):
                    suppress.append(
                        {
                            "source": prev["source"],
                            "target": prev["target"],
                            "reason": "retracement_weaker_direction",
                            "backend": self.name,
                            "score": prev.get("score"),
                        }
                    )
                pair_best[key] = directed
            else:
                suppress.append(
                    {
                        "source": src,
                        "target": tgt,
                        "reason": "retracement_weaker_direction",
                        "backend": self.name,
                        "score": score,
                    }
                )

        for directed in pair_best.values():
            boost.append(directed)
            focus.extend([directed["source"], directed["target"]])

        # Association-based reverse check: strong A~B with existing B→A edge → boost A→B
        assoc_score = {
            tuple(sorted([str(a.get("a")), str(a.get("b"))])): float(a.get("score") or 0)
            for a in assocs
            if a.get("a") and a.get("b")
        }
        for e in edges:
            src, tgt = str(e.get("source")), str(e.get("target"))
            key = tuple(sorted([src, tgt]))
            if assoc_score.get(key, 0) >= 0.5:
                # Prefer alphabetical? No — prefer treatment-like → outcome-like by name
                if self._looks_outcome(src) and self._looks_treatment(tgt):
                    suppress.append(
                        {
                            "source": src,
                            "target": tgt,
                            "reason": "retracement_assoc_reverse",
                            "backend": self.name,
                        }
                    )
                    boost.append(
                        {
                            "source": tgt,
                            "target": src,
                            "reason": "retracement_assoc_forward",
                            "backend": self.name,
                        }
                    )
                    focus.extend([src, tgt])

        # Ensure we mention columns even with empty edges
        if not focus and names:
            focus = names[:6]

        return uniq(boost, limit=20), uniq(suppress, limit=20), focus

    @staticmethod
    def _looks_outcome(name: str) -> bool:
        cl = name.lower()
        return any(h in cl for h in ("y_", "outcome", "revenue", "sales", "churn", "kpi"))

    @staticmethod
    def _looks_treatment(name: str) -> bool:
        cl = name.lower()
        return any(h in cl for h in ("treat", "d_", "exposure", "campaign", "spend", "policy"))
