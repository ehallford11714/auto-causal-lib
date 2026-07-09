"""KPI-mined autocausal loop: mine → SLM plan → impute → discover → physics → report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pandas as pd

from autocausal.ml.construct import ModelConstructPlan, construct_model_plan, plan_from_backend_override
from autocausal.ml.fit_report import FitReport
from autocausal.ml.imputers import apply_imputer
from autocausal.ml.predictors import fit_predictor, predict_outcome


@dataclass
class KPILoopResult:
    """Bundle of FitReport + causal edges + KPI trajectory + construct plan."""

    plan: ModelConstructPlan
    fit: FitReport
    edges: list[dict[str, Any]] = field(default_factory=list)
    candidates: dict[str, Any] = field(default_factory=dict)
    mining: Optional[dict[str, Any]] = None
    physics: Optional[dict[str, Any]] = None
    kpi_trajectory: Optional[dict[str, Any]] = None
    predictions: Optional[dict[str, Any]] = None
    source: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "KPILoopResult.v1",
            "source": self.source,
            "plan": self.plan.to_dict(),
            "fit": self.fit.to_dict(),
            "edges": self.edges,
            "candidates": self.candidates,
            "mining": self.mining,
            "physics": self.physics,
            "kpi_trajectory": self.kpi_trajectory,
            "predictions": self.predictions,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# KPI-mined causal loop",
            "",
            f"**Source:** `{self.source}`",
            "",
            self.plan.to_markdown(),
            self.fit.to_markdown(),
        ]
        if self.edges:
            lines += ["## Causal edges", ""]
            for e in self.edges[:20]:
                lines.append(
                    f"- `{e.get('source')}` → `{e.get('target')}` "
                    f"(score={e.get('score', e.get('confidence', ''))})"
                )
            lines.append("")
        if self.kpi_trajectory:
            lines += ["## KPI trajectory", ""]
            pts = self.kpi_trajectory.get("points") or self.kpi_trajectory.get("trajectory") or []
            if isinstance(pts, list) and pts:
                for p in pts[:12]:
                    lines.append(f"- {p}")
            else:
                lines.append(f"```json\n{json.dumps(self.kpi_trajectory, indent=2, default=str)[:1200]}\n```")
            lines.append("")
        if self.predictions:
            lines += ["## Predictions", "", f"```json\n{json.dumps(self.predictions, indent=2, default=str)[:800]}\n```", ""]
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]
        return "\n".join(lines)


def _mine_kpis(df: pd.DataFrame, *, min_score: float = 0.15) -> dict[str, Any]:
    """Prefer DataMine when installed; else autocausal.mining."""
    try:
        from autocausal.datamine_adapter import available, mine_via_datamine

        if available():
            payload = mine_via_datamine(df, prefer_autocausal=True, min_score=min_score)
            if payload and payload.get("ok") is not False:
                # normalize kpis list
                kpis = payload.get("kpis") or payload.get("kpi_names") or []
                if isinstance(kpis, list) and kpis and isinstance(kpis[0], dict):
                    payload = dict(payload)
                    payload["kpi_names"] = [k.get("id") or k.get("name") for k in kpis]
                    payload["kpis"] = payload["kpi_names"]
                payload = dict(payload)
                payload.setdefault("backend", "datamine")
                return payload
    except Exception:
        pass

    from autocausal.mining import mine

    report = mine(df, min_score=min_score)
    d = report.to_dict()
    d["backend"] = "autocausal.mining"
    return d


class KPIMinedCausalLoop:
    """
    End-to-end KPI-mined loop with SLM/Rule model construction.

    Pipeline::

        mine KPIs → ModelConstructPlan → impute (median|sklearn|torch)
        → discover → optional predictor → optional physics rollout → report
    """

    def __init__(self, df: pd.DataFrame, *, source: str = "memory") -> None:
        self._df = df.copy()
        self.source = source
        self.last_result: Optional[KPILoopResult] = None
        self._ac: Any = None

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: Any) -> "KPIMinedCausalLoop":
        from autocausal.ingest import load_csv

        df = load_csv(path, **kwargs)
        return cls(df, source=f"csv:{path}")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, source: str = "dataframe") -> "KPIMinedCausalLoop":
        return cls(df, source=source)

    @classmethod
    def from_autocausal(cls, ac: Any) -> "KPIMinedCausalLoop":
        loop = cls(ac.df, source=getattr(ac, "source", "autocausal"))
        loop._ac = ac
        return loop

    def run(
        self,
        *,
        text: str = "",
        use_slm: bool = False,
        use_torch: Optional[bool] = None,
        guides: Optional[Sequence[str]] = None,
        horizon: int = 5,
        physics: bool = True,
        min_score: float = 0.15,
        epochs: int = 40,
        imputer_backend: Optional[str] = None,
        **discover_kwargs: Any,
    ) -> KPILoopResult:
        from autocausal.api import AutoCausal

        notes: list[str] = []
        ac = self._ac or AutoCausal(self._df, source=self.source)
        self._ac = ac

        # 1) Mine KPIs
        mining = _mine_kpis(ac.df, min_score=min_score)
        ac.mining = mining  # store raw dict; discover tolerates via hasattr checks
        # Also attach a lightweight object for guide context columns
        try:
            from autocausal.mining import mine as ac_mine

            # keep structured MiningReport for AutoCausal.guide context
            ac.mining = ac_mine(ac.df, min_score=min_score)
            mining_dict = ac.mining.to_dict()
            mining_dict["backend"] = mining.get("backend", "autocausal.mining")
            if mining.get("backend") == "datamine":
                notes.append("DataMine available; used for KPI enrichment notes.")
                # merge datamine kpi names if richer
                dm_kpis = mining.get("kpis") or mining.get("kpi_names") or []
                if dm_kpis:
                    existing = list(ac.mining.kpis or [])
                    for k in dm_kpis:
                        ks = k if isinstance(k, str) else str(k.get("id") or k.get("name") or k)
                        if ks not in existing and ks in ac.df.columns:
                            existing.append(ks)
                    ac.mining.kpis = existing
                    mining_dict = ac.mining.to_dict()
                    mining_dict["backend"] = "datamine+autocausal"
            mining = mining_dict
        except Exception as e:
            notes.append(f"mining wrap: {type(e).__name__}: {e}")

        kpis = list(mining.get("kpis") or [])
        context = {
            "text": text,
            "columns": mining.get("columns") or [{"name": c} for c in ac.df.columns],
            "associations": mining.get("associations") or [],
            "kpis": kpis,
            "mining": mining,
            "edges": [],
            "candidates": {},
        }

        # 2) SLM/Rule constructs plan
        plan = construct_model_plan(
            context,
            use_slm=use_slm,
            use_torch=use_torch,
            guides=guides,
            prefer_physics=physics,
        )
        if imputer_backend:
            plan = plan_from_backend_override(plan, backend=imputer_backend)
        notes.extend(plan.notes)
        notes.append(f"construct backend={plan.backend} imputer={plan.imputer} predictor={plan.predictor}")

        # 3) Impute per plan
        focus_cols = plan.feature_columns or plan.kpi_focus or list(ac.df.columns)
        imputed, impute_meta, fit = apply_imputer(
            ac.df, plan.imputer, columns=focus_cols, epochs=epochs
        )
        ac._df = imputed
        from autocausal.impute import ImputationReport, ColumnImputation

        # lightweight imputation stamp for discover
        ac.imputation = ImputationReport(
            method=str(impute_meta.get("method", plan.imputer)),
            columns=[
                ColumnImputation(
                    column=c,
                    strategy=str(impute_meta.get("method", plan.imputer)),
                    missing_before=0,
                    missing_after=0,
                )
                for c in (plan.kpi_focus or [])[:12]
            ],
        )
        fit.predictor = plan.predictor
        fit.outcome = plan.outcome
        fit.kpi_focus = list(plan.kpi_focus)

        # 4) Discover
        focus = [c for c in (plan.focus_columns if hasattr(plan, "focus_columns") else plan.kpi_focus) if c in ac.df.columns]
        # ModelConstructPlan uses kpi_focus / feature_columns
        focus = [c for c in plan.feature_columns if c in ac.df.columns]
        if len(focus) < 2:
            focus = None
        result = ac.discover(focus_columns=focus, **discover_kwargs)
        edges = list(result.edges or [])
        candidates = dict(result.candidates or {})

        # 5) Optional predictor on imputed KPI state
        pred_model, pred_meta = fit_predictor(
            ac.df,
            plan.predictor,
            outcome=plan.outcome,
            features=[c for c in plan.feature_columns if c != plan.outcome],
            epochs=epochs,
        )
        predictions = None
        if pred_model is not None:
            yhat = predict_outcome(pred_model, ac.df, pred_meta)
            predictions = {
                "method": pred_meta.get("method"),
                "outcome": plan.outcome,
                "metrics": {
                    k: pred_meta[k]
                    for k in ("train_mae", "heldout_mask_mae", "n_rows")
                    if k in pred_meta
                },
                "yhat_head": yhat[:10].tolist() if yhat is not None else None,
            }
            fit.predictor = str(pred_meta.get("method", plan.predictor))
            fit.metrics = {**fit.metrics, **(predictions.get("metrics") or {})}
            if pred_meta.get("method") == "torch_mlp" or (
                hasattr(pred_model, "stats") and "torch" in str(getattr(pred_model, "stats", ""))
            ):
                fit.torch_used = True
            if pred_meta.get("method") == "sklearn_rf":
                fit.sklearn_used = True
            # detect torch regressor
            if hasattr(pred_model, "stats") and getattr(pred_model.stats, "backend", "").startswith("torch"):
                fit.torch_used = True
                fit.metrics["predictor_heldout_mae"] = pred_model.stats.heldout_mask_mae
                fit.metrics["predictor_train_mae"] = pred_model.stats.train_mae

        # 6) Optional physics rollout on imputed KPI state
        physics_payload = None
        kpi_traj = None
        if physics and plan.use_physics and horizon > 0:
            try:
                from autocausal.physics import PhysicsCausalSuite

                suite = PhysicsCausalSuite.from_autocausal(ac)
                cols = [c for c in plan.kpi_focus if c in ac.df.columns]
                traj = suite.rollout(horizon=horizon, columns=cols or None, use_edges=True)
                kpi_traj = traj.to_dict() if hasattr(traj, "to_dict") else {"raw": str(traj)}
                physics_payload = {
                    "system": suite.system,
                    "horizon": horizon,
                    "columns": cols,
                }
                notes.append(f"physics rollout horizon={horizon} system={suite.system}")
            except Exception as e:
                notes.append(f"physics soft-skip: {type(e).__name__}: {e}")

        fit.notes = list(dict.fromkeys(list(fit.notes) + notes[:8]))

        out = KPILoopResult(
            plan=plan,
            fit=fit,
            edges=edges,
            candidates=candidates,
            mining=mining,
            physics=physics_payload,
            kpi_trajectory=kpi_traj,
            predictions=predictions,
            source=self.source,
            notes=notes,
        )
        self.last_result = out
        return out
