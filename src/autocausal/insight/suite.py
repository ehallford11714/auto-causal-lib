"""InsightSuite — compose AutoCausal mine/discover/guide + experiment research loop."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from autocausal.insight.experiments import ExperimentPlan, ExperimentRecommender
from autocausal.insight.narrator import build_insight_report
from autocausal.insight.report import InsightReport


__all__ = [
    "InsightSuite",
    "run_insight_loop",
    "run_slm_research_loop",
    "mine_further",
    "demo_insight",
    "edge_key",
    "edge_delta",
]


def _env_slm() -> bool:
    for n in ("AUTOCAUSAL_SLM", "EMOTIVEVISION_SLM", "CAUSALIV_SLM"):
        if os.environ.get(n, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


def edge_key(e: dict[str, Any]) -> tuple[Any, Any]:
    return (e.get("source"), e.get("target"))


def edge_delta(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (new_edges, dropped_edges)."""
    b = {edge_key(e): e for e in before}
    a = {edge_key(e): e for e in after}
    new = [a[k] for k in a.keys() - b.keys()]
    dropped = [b[k] for k in b.keys() - a.keys()]
    return new, dropped


def _as_autocausal(
    source: Any,
    *,
    existing: Any = None,
    table: Optional[str] = None,
    query: Optional[str] = None,
) -> Any:
    from autocausal import AutoCausal

    if existing is not None:
        return existing
    if isinstance(source, AutoCausal):
        return source
    if isinstance(source, pd.DataFrame):
        return AutoCausal.from_dataframe(source)
    if isinstance(source, (str, Path)):
        p = str(source)
        lower = p.lower()
        if lower.endswith(".csv"):
            return AutoCausal.from_csv(p)
        if lower.endswith(".parquet"):
            return AutoCausal.from_parquet(p)
        if "://" in p or p.startswith("sqlite:"):
            return AutoCausal.from_sqlalchemy(p, table=table, query=query)
        return AutoCausal.from_csv(p)
    raise TypeError(
        "source must be AutoCausal, DataFrame, CSV/Parquet path, or pass discovery=..."
    )


def _candidates_from_ac(ac: Any) -> dict[str, Any]:
    if ac.result is not None:
        return dict(ac.result.candidates or {})
    return {}


def _edges_from_ac(ac: Any) -> list[dict[str, Any]]:
    if ac.result is not None:
        return list(ac.result.edges or [])
    return []


def _mining_dict(ac: Any) -> Optional[dict[str, Any]]:
    if ac.mining is None:
        return None
    if hasattr(ac.mining, "to_dict"):
        return ac.mining.to_dict()
    if isinstance(ac.mining, dict):
        return ac.mining
    return {"repr": str(ac.mining)}


def _bundled_public_ids() -> list[str]:
    try:
        from autocausal.public_suite import BUNDLED_IDS, list_public

        offline = [s.id for s in list_public(offline_only=True) if s.access == "bundled"]
        return offline or list(BUNDLED_IDS)
    except Exception:
        return [
            "finance_demo",
            "marketing_demo",
            "demographics_demo",
            "instruments_demo",
            "policy_demo",
        ]


def mine_further(
    ac: Any,
    plan: ExperimentPlan,
    *,
    joined_sources: list[str],
    allow_network: bool = False,
    text: str = "",
) -> tuple[list[str], list[str]]:
    """Apply feasible experiment actions (join / remine / nlp / behavioral).

    Returns (actions_applied, newly_joined_sources).
    """
    applied: list[str] = []
    newly: list[str] = []

    for rec in plan.feasible_actions():
        meta = rec.meta or {}
        action = meta.get("action") or rec.kind

        if action in ("join_public", "join") or rec.kind == "join":
            for sid in rec.public_sources or [meta.get("source_id")]:
                if not sid or sid in joined_sources:
                    continue
                try:
                    ac.join_public(sid, allow_network=allow_network)
                    joined_sources.append(sid)
                    newly.append(sid)
                    applied.append(f"join:{sid}")
                except Exception as e:
                    applied.append(f"join_failed:{sid}:{type(e).__name__}")
            continue

        if action == "remine" or rec.kind == "mine":
            try:
                min_score = float(meta.get("min_score", 0.1))
                ac.mine(min_score=min_score)
                applied.append("remine")
            except TypeError:
                ac.mine()
                applied.append("remine")
            except Exception as e:
                applied.append(f"remine_failed:{type(e).__name__}")
            continue

        if action == "nlp_hints" or rec.kind == "nlp":
            try:
                from autocausal.nlp import extract_causal_hints_from_text

                t = meta.get("text") or text
                if t:
                    hints = extract_causal_hints_from_text(t)
                    ac.nlp_hints = hints
                    applied.append("nlp_hints")
            except Exception as e:
                applied.append(f"nlp_soft_fail:{type(e).__name__}")
            continue

        if action == "behavioral" or rec.kind == "behavioral":
            try:
                demo = meta.get("demo") or "habit_loop"
                if hasattr(ac, "attach_behavioral"):
                    ac.attach_behavioral(demo, discover=False)
                    applied.append(f"behavioral:{demo}")
                else:
                    applied.append("behavioral_skip")
            except Exception as e:
                applied.append(f"behavioral_soft_fail:{type(e).__name__}")
            continue

    return applied, newly


class InsightSuite:
    """Library-first insight + closed SLM/rule research loop.

    Example::

        from autocausal.insight import InsightSuite

        suite = InsightSuite(use_slm=False)  # soft-fail to rules if True without HF
        report = suite.run_loop("data.csv", max_rounds=3, join_sources=["demographics_demo"])
        print(report.to_markdown())
    """

    def __init__(
        self,
        *,
        use_slm: bool = False,
        model_name: Optional[str] = None,
        guide_backends: Optional[list[str]] = None,
    ) -> None:
        # Prefer SLM for auto* research loops when env set; constructor default
        # remains False for backward compat — pass use_slm=True (recommended).
        self.use_slm = bool(use_slm) or _env_slm()
        self.model_name = model_name
        self.guide_backends = guide_backends
        self.recommender = ExperimentRecommender(
            use_slm=self.use_slm, model_name=model_name
        )
        self.last_report: Optional[InsightReport] = None
        self.ac: Any = None
        # Documented recommendation: auto* paths should pass use_slm=True
        self._slm_recommended = True

    @classmethod
    def from_autocausal(
        cls,
        ac: Any,
        *,
        use_slm: bool = False,
        model_name: Optional[str] = None,
        guide_backends: Optional[list[str]] = None,
    ) -> "InsightSuite":
        suite = cls(
            use_slm=use_slm, model_name=model_name, guide_backends=guide_backends
        )
        suite.ac = ac
        return suite

    def run(
        self,
        source: Any = None,
        *,
        ac: Any = None,
        discovery: Any = None,
        text: str = "",
        join: Optional[Union[str, list[str]]] = None,
        join_on: Optional[Union[str, list[str]]] = None,
        impute_method: str = "auto",
        skip_guide: bool = False,
        use_slm: Optional[bool] = None,
        **discover_kwargs: Any,
    ) -> InsightReport:
        """Single-pass: load/join → mine → impute → discover → guide → synthesize."""
        if use_slm is not None:
            self.use_slm = bool(use_slm) or _env_slm()
            self.recommender = ExperimentRecommender(
                use_slm=self.use_slm, model_name=self.model_name
            )
        return run_insight_loop(
            source,
            ac=ac if ac is not None else self.ac,
            discovery=discovery,
            text=text,
            use_slm=self.use_slm,
            model_name=self.model_name,
            guide_backends=self.guide_backends,
            join=join,
            join_on=join_on,
            impute_method=impute_method,
            skip_guide=skip_guide,
            suite=self,
            **discover_kwargs,
        )

    def run_loop(
        self,
        source: Any = None,
        *,
        ac: Any = None,
        max_rounds: int = 3,
        join_sources: Optional[list[str]] = None,
        text: str = "",
        impute_method: str = "auto",
        allow_network: bool = False,
        **discover_kwargs: Any,
    ) -> InsightReport:
        """Closed iterative research loop with experiment recommendations."""
        return run_slm_research_loop(
            source,
            ac=ac if ac is not None else self.ac,
            max_rounds=max_rounds,
            join_sources=join_sources,
            text=text,
            use_slm=self.use_slm,
            model_name=self.model_name,
            guide_backends=self.guide_backends,
            impute_method=impute_method,
            allow_network=allow_network,
            suite=self,
            **discover_kwargs,
        )

    def recommend_experiments(
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
        return self.recommender.recommend(
            edges=edges,
            mining=mining,
            candidates=candidates,
            text=text,
            prior_report=prior_report,
            joined_sources=joined_sources,
            available_public=available_public,
            round_index=round_index,
            max_rounds=max_rounds,
        )


def demo_insight(
    *,
    use_slm: bool = False,
    max_rounds: int = 2,
    research_loop: bool = True,
    dataset: Optional[str] = None,
) -> InsightReport:
    """Offline demo → insight report (rule path by default).

    Without ``dataset``, uses a synthetic IV-style panel. With ``dataset``
    (e.g. ``iris``, ``wine``, ``titanic``), loads a bundled real example CSV.
    """
    import numpy as np

    if dataset:
        from autocausal.datasets import get_dataset, load_dataset

        meta = get_dataset(dataset)
        df = load_dataset(dataset, allow_network=False)
        text = (
            f"Exploratory associations in {meta.name} "
            f"(outcome hint: {meta.suggested_outcome or 'n/a'}). "
            f"{meta.epistemic_note}"
        )
        suite = InsightSuite(use_slm=use_slm)
        if research_loop:
            return suite.run_loop(df, max_rounds=max_rounds, join_sources=None, text=text)
        return suite.run(df, text=text)

    rng = np.random.default_rng(42)
    n = 80
    z = rng.normal(size=n)
    treat_x = 0.7 * z + rng.normal(0, 0.35, size=n)
    outcome_y = 1.1 * treat_x + rng.normal(0, 0.5, size=n)
    df = pd.DataFrame(
        {
            "z_iv": z,
            "treat_x": treat_x,
            "outcome_y": outcome_y,
            "age": rng.normal(40, 10, n),
            "region": rng.choice(["US", "EU", "APAC"], size=n),
        }
    )
    suite = InsightSuite(use_slm=use_slm)
    if research_loop:
        return suite.run_loop(
            df,
            max_rounds=max_rounds,
            join_sources=["demographics_demo", "instruments_demo"],
            text="Does treat_x cause outcome_y?",
        )
    return suite.run(df, text="Does treat_x cause outcome_y?")


def run_insight_loop(
    source: Any = None,
    *,
    ac: Any = None,
    discovery: Any = None,
    text: str = "",
    use_slm: bool = False,
    model_name: Optional[str] = None,
    guide_backends: Optional[list[str]] = None,
    join: Optional[Union[str, list[str]]] = None,
    join_on: Optional[Union[str, list[str]]] = None,
    impute_method: str = "auto",
    skip_guide: bool = False,
    suite: Optional[InsightSuite] = None,
    recommend: bool = True,
    table: Optional[str] = None,
    query: Optional[str] = None,
    out: Any = None,  # accepted for CLI compat; write via report.write separately
    **discover_kwargs: Any,
) -> InsightReport:
    """One-shot insight pipeline composing existing AutoCausal steps."""
    use_slm = bool(use_slm) or _env_slm()
    stages: list[str] = []
    notes: list[str] = []
    data_sources: list[str] = []
    nlp_hints_dict: Optional[dict[str, Any]] = None
    _ = out  # CLI may pass; callers should use InsightReport.write

    if discovery is not None and ac is None and source is None:
        # Build minimal report from DiscoveryResult alone
        edges = list(getattr(discovery, "edges", None) or [])
        candidates = dict(getattr(discovery, "candidates", None) or {})
        disc_dict = discovery.to_dict() if hasattr(discovery, "to_dict") else None
        stages = ["discover"]
        recommender = ExperimentRecommender(use_slm=use_slm, model_name=model_name)
        plan = (
            recommender.recommend(
                edges=edges,
                candidates=candidates,
                text=text,
                available_public=_bundled_public_ids(),
            )
            if recommend
            else ExperimentPlan(backend="rule")
        )
        report = build_insight_report(
            edges=edges,
            candidates=candidates,
            source="discovery_result",
            n_rows=0,
            n_cols=0,
            stages=stages,
            data_sources=[],
            guide=getattr(discovery, "guide", None),
            mining=getattr(discovery, "mining", None),
            discovery=disc_dict,
            notes=notes + ["Built from precomputed DiscoveryResult."],
            use_slm=use_slm,
            model_name=model_name,
            experiments_recommended=[r.to_dict() for r in plan.recommendations],
            text=text,
        )
        if suite is not None:
            suite.last_report = report
        return report

    ac = _as_autocausal(source, existing=ac, table=table, query=query)
    data_sources.append(getattr(ac, "source", "memory") or "memory")
    stages.append("load")

    if join:
        ids = [join] if isinstance(join, str) else list(join)
        on = None
        if isinstance(join_on, str):
            on = [x.strip() for x in join_on.split(",") if x.strip()]
        elif join_on:
            on = list(join_on)
        for sid in ids:
            try:
                ac.join_public(sid, on=on, allow_network=False)
                data_sources.append(f"public:{sid}")
                stages.append(f"join:{sid}")
            except Exception as e:
                notes.append(f"join {sid} soft-failed: {type(e).__name__}: {e}")

    if text.strip():
        try:
            from autocausal.nlp import extract_causal_hints_from_text

            hints = extract_causal_hints_from_text(text)
            ac.nlp_hints = hints
            nlp_hints_dict = hints.to_dict()
            stages.append("nlp_hints")
        except Exception as e:
            notes.append(f"nlp hints soft-failed: {type(e).__name__}")

    if ac.mining is None:
        ac.mine()
        stages.append("mine")
    if ac.imputation is None and ac.df.isna().any().any():
        ac.impute(method=impute_method)  # type: ignore[arg-type]
        stages.append("impute")
    elif ac.imputation is None:
        stages.append("impute_skip")

    if ac.result is None:
        ac.discover(**discover_kwargs)
        stages.append("discover")

    guide_dict = None
    grail_dict = None
    if not skip_guide:
        try:
            gres = ac.guide(
                text=text or None,
                use_slm=use_slm,
                model_name=model_name,
                backends=guide_backends,
            )
            guide_dict = gres.to_dict() if hasattr(gres, "to_dict") else dict(gres or {})
            stages.append("guide")
        except Exception as e:
            notes.append(f"guide soft-failed: {type(e).__name__}: {e}")
            stages.append("guide_skip")

    # Soft GRAIL memory/graph step (always offline-safe)
    try:
        from autocausal.grail import insight_grail_step

        grail_ctx = {
            "columns": [{"name": str(c)} for c in list(ac.df.columns)],
            "edges": _edges_from_ac(ac),
            "text": text,
            "candidates": _candidates_from_ac(ac),
        }
        grail_dict = insight_grail_step(text=text, context=grail_ctx, max_cycles=1)
        stages.append("grail")
        if grail_dict.get("notes"):
            notes.extend(list(grail_dict["notes"])[:3])
    except Exception as e:
        notes.append(f"grail soft-failed: {type(e).__name__}: {e}")
        stages.append("grail_skip")

    stages.append("insight_synthesize")
    edges = _edges_from_ac(ac)
    candidates = _candidates_from_ac(ac)
    mining = _mining_dict(ac)
    disc = ac.result.to_dict() if ac.result is not None else None

    recommender = (
        suite.recommender
        if suite is not None
        else ExperimentRecommender(use_slm=use_slm, model_name=model_name)
    )
    plan = (
        recommender.recommend(
            edges=edges,
            mining=mining or {},
            candidates=candidates,
            text=text,
            joined_sources=[s.replace("public:", "") for s in data_sources if s.startswith("public:")],
            available_public=_bundled_public_ids(),
        )
        if recommend
        else ExperimentPlan(backend="rule")
    )

    report = build_insight_report(
        edges=edges,
        candidates=candidates,
        source=str(getattr(ac, "source", "")),
        n_rows=int(len(ac.df)),
        n_cols=int(len(ac.df.columns)),
        stages=stages,
        data_sources=data_sources,
        guide=guide_dict,
        grail=grail_dict,
        mining=mining,
        discovery=disc,
        nlp_hints=nlp_hints_dict,
        notes=notes,
        use_slm=use_slm,
        model_name=model_name,
        experiments_recommended=[r.to_dict() for r in plan.recommendations],
        text=text,
    )
    if suite is not None:
        suite.last_report = report
        suite.ac = ac
    return report


def run_slm_research_loop(
    source: Any = None,
    *,
    ac: Any = None,
    max_rounds: int = 3,
    join_sources: Optional[list[str]] = None,
    text: str = "",
    use_slm: bool = False,
    model_name: Optional[str] = None,
    guide_backends: Optional[list[str]] = None,
    impute_method: str = "auto",
    allow_network: bool = False,
    suite: Optional[InsightSuite] = None,
    **discover_kwargs: Any,
) -> InsightReport:
    """Closed loop: mine/discover → guide/SLM → recommend → apply → re-discover.

    Stops on ``max_rounds``, no new edges + no remaining joins, or recommender stop.
    """
    use_slm = bool(use_slm) or _env_slm()
    max_rounds = max(1, int(max_rounds))
    ac = _as_autocausal(source, existing=ac)

    available = list(join_sources) if join_sources else _bundled_public_ids()
    # Prefer a small default set for auto-loop if caller passed nothing explicit
    if join_sources is None:
        available = [
            s
            for s in (
                "demographics_demo",
                "instruments_demo",
                "marketing_demo",
                "finance_demo",
                "policy_demo",
            )
            if s in set(_bundled_public_ids())
        ] or _bundled_public_ids()[:5]

    joined: list[str] = []
    # seed joins from join_log if present
    for j in getattr(ac, "join_log", None) or []:
        if isinstance(j, dict) and j.get("id") and j.get("ok", True):
            joined.append(str(j["id"]))

    stages: list[str] = ["load"]
    notes: list[str] = [
        f"SLM research loop max_rounds={max_rounds}; use_slm={use_slm} (soft-optional)."
    ]
    data_sources: list[str] = [str(getattr(ac, "source", "memory"))]
    round_history: list[dict[str, Any]] = []
    further: list[dict[str, Any]] = []
    all_experiments: list[dict[str, Any]] = []
    prior_edges: list[dict[str, Any]] = []
    prior_report: Optional[InsightReport] = None
    nlp_hints_dict: Optional[dict[str, Any]] = None
    guide_dict: Optional[dict[str, Any]] = None
    last_grail: Optional[dict[str, Any]] = None

    recommender = (
        suite.recommender
        if suite is not None
        else ExperimentRecommender(use_slm=use_slm, model_name=model_name)
    )

    for r in range(max_rounds):
        round_notes: list[str] = []
        # mine / impute / discover
        ac.mine()
        if ac.df.isna().any().any():
            ac.impute(method=impute_method)  # type: ignore[arg-type]
        ac.discover(**discover_kwargs)
        stages.append(f"round{r}:mine_discover")

        try:
            gres = ac.guide(
                text=text or None,
                use_slm=use_slm,
                model_name=model_name,
                backends=guide_backends,
            )
            guide_dict = gres.to_dict() if hasattr(gres, "to_dict") else dict(gres or {})
            stages.append(f"round{r}:guide")
        except Exception as e:
            round_notes.append(f"guide soft-failed: {type(e).__name__}")

        # GRAIL memory/graph step each research round
        grail_round: Optional[dict[str, Any]] = None
        try:
            from autocausal.grail import insight_grail_step

            grail_round = insight_grail_step(
                text=text,
                context={
                    "columns": [{"name": str(c)} for c in list(ac.df.columns)],
                    "edges": _edges_from_ac(ac),
                    "text": text,
                    "candidates": _candidates_from_ac(ac),
                },
                max_cycles=1,
            )
            last_grail = grail_round
            stages.append(f"round{r}:grail")
            round_notes.append(
                f"grail backend={grail_round.get('backend')} "
                f"focus={grail_round.get('focus_columns', [])[:4]}"
            )
        except Exception as e:
            round_notes.append(f"grail soft-failed: {type(e).__name__}")

        edges = _edges_from_ac(ac)
        new_e, drop_e = edge_delta(prior_edges, edges)
        for e in new_e:
            further.append(
                {
                    "round": r,
                    "change": "new",
                    "source": e.get("source"),
                    "target": e.get("target"),
                    "detail": f"score={e.get('score')}",
                }
            )
        for e in drop_e:
            further.append(
                {
                    "round": r,
                    "change": "dropped",
                    "source": e.get("source"),
                    "target": e.get("target"),
                    "detail": "absent after re-discover",
                }
            )

        plan = recommender.recommend(
            edges=edges,
            mining=_mining_dict(ac) or {},
            candidates=_candidates_from_ac(ac),
            text=text,
            prior_report=prior_report,
            joined_sources=joined,
            available_public=available,
            round_index=r,
            max_rounds=max_rounds,
        )
        all_experiments.extend([rec.to_dict() for rec in plan.recommendations])
        stages.append(f"round{r}:recommend")

        stop = bool(plan.stop)
        if r > 0 and not new_e and not (set(available) - set(joined)):
            stop = True
            round_notes.append("Stop: no new edges and no remaining join sources.")

        applied, newly = ([], [])
        if not stop and r < max_rounds - 1:
            applied, newly = mine_further(
                ac,
                plan,
                joined_sources=joined,
                allow_network=allow_network,
                text=text,
            )
            for sid in newly:
                data_sources.append(f"public:{sid}")
                further.append(
                    {
                        "round": r,
                        "change": "source_joined",
                        "source": sid,
                        "target": "",
                        "detail": "public suite join",
                    }
                )
            if applied:
                stages.append(f"round{r}:apply")
                round_notes.append("Applied: " + ", ".join(applied))
            if getattr(ac, "nlp_hints", None) is not None and hasattr(ac.nlp_hints, "to_dict"):
                nlp_hints_dict = ac.nlp_hints.to_dict()

        hist = {
            "round": r,
            "n_edges": len(edges),
            "n_new_edges": len(new_e),
            "n_dropped_edges": len(drop_e),
            "sources_joined": list(newly),
            "experiments": [rec.to_dict() for rec in plan.recommendations[:6]],
            "stop": stop,
            "stop_reason": plan.stop_reason,
            "backend": plan.backend,
            "grail": {
                "backend": (grail_round or {}).get("backend"),
                "focus_columns": (grail_round or {}).get("focus_columns"),
                "memory_keys": (grail_round or {}).get("memory_keys"),
            }
            if grail_round
            else None,
            "notes": round_notes + list(plan.notes[:3]),
        }
        round_history.append(hist)
        prior_edges = edges
        # lightweight prior for next recommend
        prior_report = InsightReport(
            key_edges=edges,
            role_hypotheses=build_insight_report(
                edges=edges,
                candidates=_candidates_from_ac(ac),
                source="",
                n_rows=0,
                n_cols=0,
                stages=[],
                data_sources=[],
            ).role_hypotheses,
        )

        if stop:
            notes.append(plan.stop_reason or f"Stopped after round {r}.")
            break

    stages.append("insight_synthesize")
    edges = _edges_from_ac(ac)
    candidates = _candidates_from_ac(ac)
    mining = _mining_dict(ac)
    disc = ac.result.to_dict() if ac.result is not None else None

    # Deduplicate experiments by title keeping highest priority
    by_title: dict[str, dict[str, Any]] = {}
    for ex in all_experiments:
        t = ex.get("title") or ""
        if t not in by_title or float(ex.get("priority") or 0) > float(
            by_title[t].get("priority") or 0
        ):
            by_title[t] = ex
    uniq_ex = sorted(by_title.values(), key=lambda x: float(x.get("priority") or 0), reverse=True)

    report = build_insight_report(
        edges=edges,
        candidates=candidates,
        source=str(getattr(ac, "source", "")),
        n_rows=int(len(ac.df)),
        n_cols=int(len(ac.df.columns)),
        stages=stages,
        data_sources=data_sources,
        guide=guide_dict,
        grail=last_grail,
        mining=mining,
        discovery=disc,
        nlp_hints=nlp_hints_dict,
        notes=notes,
        use_slm=use_slm,
        model_name=model_name,
        experiments_recommended=uniq_ex[:20],
        relationships_mined_further=further,
        round_history=round_history,
        text=text,
    )
    if suite is not None:
        suite.last_report = report
        suite.ac = ac
    return report
