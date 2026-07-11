"""KineteqGrailGuide — direction backend wrapping ``autocausal.grail``."""

from __future__ import annotations

from typing import Any, Optional

from autocausal.guides.types import GuideResult, GuideSuggestion, col_names, uniq
from autocausal.grail.adapter import GrailEngine, grail_backend_status

__all__ = ["KineteqGrailGuide", "GrailGuide"]


class KineteqGrailGuide:
    """Soft guide backend: GRAIL impute/compose/run → DirectionPlan fields.

    Backend ids: ``grail``, ``kineteq_grail``.
    Always contributes (stub path); ``available`` is True when live Kineteq
    is reachable, else stub still returns usable suggestions with clear notes.
    """

    name = "kineteq_grail"

    def __init__(
        self,
        *,
        max_cycles: int = 2,
        chain_length: int = 3,
        prefer_live: bool = True,
    ) -> None:
        self.max_cycles = max_cycles
        self.chain_length = chain_length
        self.prefer_live = prefer_live
        self.engine = GrailEngine(prefer_live=prefer_live)

    def available(self) -> bool:
        # Soft: always runnable via stub; report live separately in notes.
        return True

    def live_available(self) -> bool:
        return self.engine.live_available()

    def guide(self, context: dict[str, Any]) -> GuideResult:
        text = (context.get("text") or "").strip()
        columns = col_names(context)
        goal = text or (
            f"Explore causal structure among {', '.join(columns[:8])}"
            if columns
            else "Explore causal relationships in the table"
        )
        status = grail_backend_status()
        report = self.engine.run(
            goal,
            context=context,
            max_cycles=self.max_cycles,
            chain_length=self.chain_length,
        )

        treat: list[str] = []
        outcome: list[str] = []
        instruments: list[str] = []
        confounders: list[str] = []
        if report.imputation:
            for a in report.imputation.assumptions:
                if a.value is None:
                    continue
                val = str(a.value)
                if a.parameter == "treatment":
                    treat.append(val)
                elif a.parameter == "outcome":
                    outcome.append(val)
                elif a.parameter == "instrument":
                    instruments.append(val)
                elif a.parameter == "confounder":
                    confounders.append(val)

        focus = uniq(list(report.focus_columns) + treat + outcome, limit=20)
        suggestions = [
            GuideSuggestion(
                action="grail_impute",
                detail=(report.imputation.enriched_goal[:200] if report.imputation else goal),
                priority=0.75,
            ),
            GuideSuggestion(
                action="grail_compose",
                detail=(
                    f"Expert chain length={report.chain.chain_length if report.chain else 0}"
                ),
                priority=0.7,
            ),
        ]
        for q in report.next_questions[:4]:
            suggestions.append(
                GuideSuggestion(action="ask", detail=q, priority=0.65)
            )
        for e in report.boost_edges[:6]:
            suggestions.append(
                GuideSuggestion(
                    action="validate_edge",
                    detail=f"`{e.get('source')}` → `{e.get('target')}` ({e.get('reason')})",
                    priority=0.68,
                    meta=dict(e),
                )
            )

        live = bool(report.live_kineteq)
        backend_label = report.backend if live else "grail_stub"
        notes = list(report.notes) + [
            f"grail_status preferred={status.get('preferred')}",
            report.epistemic,
        ]
        if not live:
            notes.append(
                "Used offline GRAIL stub (NOT live Kineteq). "
                "Set KINETEQ_MCP_URL + AUTOCAUSAL_KINETEQ_MCP=1 or install kineteq for live."
            )

        return GuideResult(
            backend=backend_label,
            available=True,  # stub always usable; live flagged via backend name
            suggestions=suggestions[:40],
            focus_columns=focus,
            instruments=uniq(instruments, limit=10),
            confounders=uniq(confounders, limit=10),
            treatment=uniq(treat, limit=8),
            outcome=uniq(outcome, limit=8),
            related_variables=uniq(focus, limit=15),
            boost_edges=list(report.boost_edges)[:20],
            validate_edges=[
                {"source": str(e.get("source")), "target": str(e.get("target"))}
                for e in report.boost_edges[:20]
                if e.get("source") and e.get("target")
            ],
            search_queries=uniq(report.search_queries, limit=10),
            next_questions=uniq(report.next_questions, limit=10),
            raw_text=report.final_answer,
            notes=notes,
        )


# Alias for clearer imports
GrailGuide = KineteqGrailGuide
