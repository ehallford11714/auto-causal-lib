"""ExperimentRecommender — rule / SLM suggestions for next measurements & mining."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from autocausal.insight.report import InsightReport, RoleHypotheses


__all__ = [
    "ExperimentRecommendation",
    "ExperimentPlan",
    "ExperimentRecommender",
]


def _env_slm() -> bool:
    for n in ("AUTOCAUSAL_SLM", "EMOTIVEVISION_SLM", "CAUSALIV_SLM"):
        if os.environ.get(n, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


@dataclass
class ExperimentRecommendation:
    """One ranked next-step experiment / mining action."""

    kind: str
    # measure | join | mine | ab_test | iv | confounder | stop | nlp | behavioral
    title: str
    rationale: str
    priority: float = 0.5
    hypothesized_edge: Optional[dict[str, str]] = None  # {source, target}
    columns_to_collect: list[str] = field(default_factory=list)
    public_sources: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentPlan:
    """Ranked experiment recommendations + stop signal."""

    backend: str
    recommendations: list[ExperimentRecommendation] = field(default_factory=list)
    stop: bool = False
    stop_reason: str = ""
    notes: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "stop": self.stop,
            "stop_reason": self.stop_reason,
            "notes": list(self.notes),
            "raw_text": self.raw_text,
        }

    def feasible_actions(self) -> list[ExperimentRecommendation]:
        """Actions the research loop can apply automatically (join/mine/nlp/behavioral)."""
        applyable = {"join", "mine", "nlp", "behavioral"}
        return [r for r in self.recommendations if r.kind in applyable and not self.stop]


class ExperimentRecommender:
    """Recommend next experiments from discovery/mining context.

    Soft-optional HuggingFace SLM via ``autocausal.slm``; always falls back to
    deterministic rules when SLM is unavailable or ``use_slm=False``.
    """

    def __init__(
        self,
        *,
        use_slm: bool = False,
        model_name: Optional[str] = None,
    ) -> None:
        self.use_slm = bool(use_slm) or _env_slm()
        self.model_name = model_name

    def recommend(
        self,
        *,
        edges: Optional[list[dict[str, Any]]] = None,
        mining: Optional[dict[str, Any]] = None,
        candidates: Optional[dict[str, Any]] = None,
        text: str = "",
        prior_report: Optional[InsightReport] = None,
        joined_sources: Optional[list[str]] = None,
        available_public: Optional[list[str]] = None,
        round_index: int = 0,
        max_rounds: int = 3,
    ) -> ExperimentPlan:
        edges = list(edges or [])
        mining = dict(mining or {})
        candidates = dict(candidates or {})
        joined = list(joined_sources or [])
        available = list(available_public or [])

        plan = self._rule_recommend(
            edges=edges,
            mining=mining,
            candidates=candidates,
            text=text,
            prior_report=prior_report,
            joined=joined,
            available=available,
            round_index=round_index,
            max_rounds=max_rounds,
        )

        if self.use_slm:
            slm_plan = self._slm_enrich(plan, edges=edges, mining=mining, text=text)
            if slm_plan is not None:
                return slm_plan
            plan.notes.append(
                "SLM requested but unavailable/failed — using rule ExperimentRecommender."
            )
            plan.backend = "rule"
        return plan

    def _rule_recommend(
        self,
        *,
        edges: list[dict[str, Any]],
        mining: dict[str, Any],
        candidates: dict[str, Any],
        text: str,
        prior_report: Optional[InsightReport],
        joined: list[str],
        available: list[str],
        round_index: int,
        max_rounds: int,
    ) -> ExperimentPlan:
        recs: list[ExperimentRecommendation] = []
        notes: list[str] = []
        roles = RoleHypotheses.from_candidates(candidates)
        if prior_report is not None and not any(
            roles.treatment or roles.outcome or roles.instrument or roles.confounder
        ):
            roles = prior_report.role_hypotheses

        ranked = sorted(
            edges,
            key=lambda e: float(e.get("confidence") or e.get("score") or 0),
            reverse=True,
        )

        # Stop heuristics
        stop = False
        stop_reason = ""
        if round_index >= max_rounds - 1 and round_index > 0:
            # last round — still recommend but may stop after
            notes.append(f"Approaching max_rounds={max_rounds}.")
        if prior_report is not None and round_index > 0:
            prev = {
                (e.get("source"), e.get("target"))
                for e in (prior_report.key_edges or [])
            }
            cur = {(e.get("source"), e.get("target")) for e in edges}
            if cur and cur <= prev and not (set(available) - set(joined)):
                stop = True
                stop_reason = "No new edges and no remaining public sources to join."
                recs.append(
                    ExperimentRecommendation(
                        kind="stop",
                        title="Stop iterative mining",
                        rationale=stop_reason,
                        priority=1.0,
                    )
                )

        # Join unused public sources
        remaining = [s for s in available if s not in joined]
        for i, sid in enumerate(remaining[:3]):
            recs.append(
                ExperimentRecommendation(
                    kind="join",
                    title=f"Join public source `{sid}`",
                    rationale=(
                        "Enrich covariates / instruments via offline public suite join, "
                        "then re-mine associations."
                    ),
                    priority=0.85 - 0.05 * i,
                    public_sources=[sid],
                    meta={"action": "join_public", "source_id": sid},
                )
            )

        # Measure confounders / instruments missing from frame
        if roles.outcome and not roles.confounder:
            recs.append(
                ExperimentRecommendation(
                    kind="confounder",
                    title="Collect confounders (W)",
                    rationale=(
                        "Outcomes hypothesized but few confounder candidates — "
                        "measure demographics, baseline KPIs, or region/time fixed effects."
                    ),
                    priority=0.8,
                    columns_to_collect=["age", "region", "baseline_y", "segment"],
                    hypothesized_edge=None,
                )
            )
        if roles.treatment and roles.outcome and not roles.instrument:
            x = roles.treatment[0]
            y = roles.outcome[0]
            recs.append(
                ExperimentRecommendation(
                    kind="iv",
                    title=f"Design IV / encouragement for `{x}` → `{y}`",
                    rationale=(
                        "Treatment–outcome pair without instrument candidates — "
                        "consider lottery, eligibility cutoff, or leave-out share Z."
                    ),
                    priority=0.78,
                    hypothesized_edge={"source": x, "target": y},
                    columns_to_collect=["z_assign", "eligibility", "distance_to_cutoff"],
                    public_sources=["instruments_demo"]
                    if "instruments_demo" in remaining or "instruments_demo" not in joined
                    else [],
                )
            )

        # A/B from top edge
        if ranked:
            top = ranked[0]
            src, tgt = top.get("source"), top.get("target")
            recs.append(
                ExperimentRecommendation(
                    kind="ab_test",
                    title=f"A/B or RCT on `{src}` → `{tgt}`",
                    rationale=(
                        f"Top exploratory edge (score={top.get('score')}, "
                        f"conf={top.get('confidence')}). Randomize `{src}` if feasible."
                    ),
                    priority=0.75,
                    hypothesized_edge={"source": str(src), "target": str(tgt)},
                    columns_to_collect=[str(src), str(tgt), "unit_id", "assignment"],
                )
            )

        # Mine further associations / KPIs
        assocs = mining.get("associations") or []
        kpis = mining.get("kpis") or []
        if len(assocs) < 3 or not kpis:
            recs.append(
                ExperimentRecommendation(
                    kind="mine",
                    title="Re-mine associations with lower threshold / more KPIs",
                    rationale="Sparse mining report — expand association search before next discover.",
                    priority=0.7,
                    meta={"action": "remine", "min_score": 0.08},
                )
            )

        # Hypothesized X→Y not yet in edges
        edge_pairs = {(e.get("source"), e.get("target")) for e in edges}
        for x in roles.treatment[:2]:
            for y in roles.outcome[:2]:
                if (x, y) not in edge_pairs and (y, x) not in edge_pairs:
                    recs.append(
                        ExperimentRecommendation(
                            kind="measure",
                            title=f"Measure / validate hypothesized `{x}` → `{y}`",
                            rationale="Role hypotheses suggest this path; not yet in discovery edges.",
                            priority=0.72,
                            hypothesized_edge={"source": x, "target": y},
                            columns_to_collect=[x, y],
                        )
                    )

        # Optional NLP / behavioral expansion when text present
        if text.strip():
            recs.append(
                ExperimentRecommendation(
                    kind="nlp",
                    title="Inject NLP causal role hints into guide context",
                    rationale="Free-text context available — extract linguistic X/Y/Z hints (soft).",
                    priority=0.65,
                    meta={"action": "nlp_hints", "text": text[:500]},
                )
            )
            if "habit" in text.lower() or "behavior" in text.lower() or "nudge" in text.lower():
                recs.append(
                    ExperimentRecommendation(
                        kind="behavioral",
                        title="Join behavioral demo traces",
                        rationale="Text mentions behavior/habit/nudge — soft-join behavioral panel.",
                        priority=0.62,
                        meta={"action": "behavioral", "demo": "habit_loop"},
                    )
                )

        # Prefer instruments_demo join if IV recommended
        if any(r.kind == "iv" for r in recs) and "instruments_demo" in remaining:
            # bump join for instruments
            for r in recs:
                if r.kind == "join" and r.public_sources == ["instruments_demo"]:
                    r.priority = max(r.priority, 0.9)

        recs.sort(key=lambda r: r.priority, reverse=True)
        # Deduplicate by title
        seen: set[str] = set()
        uniq: list[ExperimentRecommendation] = []
        for r in recs:
            if r.title in seen:
                continue
            seen.add(r.title)
            uniq.append(r)

        if not uniq and not stop:
            uniq.append(
                ExperimentRecommendation(
                    kind="mine",
                    title="Continue exploratory mining",
                    rationale="No strong gaps detected — re-run mine/discover for stability.",
                    priority=0.4,
                    meta={"action": "remine"},
                )
            )

        return ExperimentPlan(
            backend="rule",
            recommendations=uniq[:12],
            stop=stop,
            stop_reason=stop_reason,
            notes=notes
            + [
                "Rule-based experiment recommendations — exploratory only.",
                "A/B and IV ideas require design review; auto-loop only applies join/mine/nlp/behavioral.",
            ],
        )

    def _slm_enrich(
        self,
        base: ExperimentPlan,
        *,
        edges: list[dict[str, Any]],
        mining: dict[str, Any],
        text: str,
    ) -> Optional[ExperimentPlan]:
        try:
            from autocausal.slm import get_backend, slm_available

            if not slm_available():
                return None
            backend = get_backend(use_slm=True, model_name=self.model_name)
            # Prefer create() for experiment-like proposals; fall back to infer narrative
            ctx = {
                "text": text
                or "Recommend next causal experiments, measurements, and data joins.",
                "edges": edges[:20],
                "columns": [{"name": a.get("a")} for a in (mining.get("associations") or [])[:10]],
                "candidates": {},
            }
            created = backend.create(ctx)
            inferred = backend.infer({"edges": edges[:15], "text": text})
            notes = list(base.notes) + list(getattr(created, "notes", []) or [])
            notes.append("SLM generative assistance — verify before running experiments.")
            # Keep rule ranking; prepend SLM questions as measure recs
            extra: list[ExperimentRecommendation] = []
            for q in (getattr(created, "questions", None) or [])[:4]:
                extra.append(
                    ExperimentRecommendation(
                        kind="measure",
                        title=str(q)[:120],
                        rationale="SLM-proposed research question (generative assistance).",
                        priority=0.68,
                        meta={"slm": True},
                    )
                )
            for z in (getattr(created, "instruments", None) or [])[:3]:
                name = z.get("name") if isinstance(z, dict) else str(z)
                extra.append(
                    ExperimentRecommendation(
                        kind="iv",
                        title=f"SLM instrument idea: `{name}`",
                        rationale=str(z.get("rationale", "") if isinstance(z, dict) else ""),
                        priority=0.7,
                        columns_to_collect=[str(name)],
                        meta={"slm": True},
                    )
                )
            merged = extra + list(base.recommendations)
            merged.sort(key=lambda r: r.priority, reverse=True)
            narrative = getattr(inferred, "narrative", "") or getattr(created, "raw_text", "")
            return ExperimentPlan(
                backend=getattr(created, "backend", "huggingface"),
                recommendations=merged[:14],
                stop=base.stop,
                stop_reason=base.stop_reason,
                notes=notes,
                raw_text=str(narrative or "")[:2000],
            )
        except Exception as e:
            base.notes.append(f"SLM enrich soft-failed: {type(e).__name__}: {e}")
            return None
