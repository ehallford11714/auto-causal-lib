"""AutoEDASuite — SLM-directed orchestration over EDAActions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pandas as pd

from autocausal.suites.autoeda.actions import EDA_REGISTRY, EDAActions
from autocausal.suites.autoeda.report import EDAReport, RoleProposal
from autocausal.suites.base import resolve_frame
from autocausal.suites.director import SLMAutoDirector, SLMDirectives, resolve_suite_slm

__all__ = ["AutoEDASuite", "auto_eda"]


class AutoEDASuite:
    """Library-first EDA suite (SLM picks action sequence when available).

    Example::

        from autocausal.suites.autoeda import AutoEDASuite, EDAActions
        EDAActions.suggest_roles(df)
        eda = AutoEDASuite(df, use_slm=True).run()
    """

    def __init__(
        self,
        source: Any = None,
        *,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        actions: Optional[Sequence[str]] = None,
        text: str = "",
        max_corr_cols: int = 12,
        include_plots: bool = False,
        table: Optional[str] = None,
        query: Optional[str] = None,
    ) -> None:
        self.source = source
        self.use_slm = resolve_suite_slm(use_slm)
        self.model_name = model_name
        self.actions_override = list(actions) if actions else None
        self.text = text
        self.max_corr_cols = max_corr_cols
        self.include_plots = include_plots
        self.table = table
        self.query = query
        self.frame: Optional[pd.DataFrame] = None
        self.report: Optional[EDAReport] = None
        self.directives: Optional[SLMDirectives] = None

    def run(self, source: Any = None, *, text: Optional[str] = None) -> "AutoEDASuite":
        src = self.source if source is None else source
        if src is None:
            raise ValueError("AutoEDASuite requires a DataFrame, path, or AutoCausal")
        df, label, _ = resolve_frame(src, table=self.table, query=self.query)
        self.frame = df
        txt = self.text if text is None else text

        director = SLMAutoDirector(use_slm=self.use_slm, model_name=self.model_name)
        directives = director.direct("eda", df, text=txt)
        self.directives = directives

        sequence = self.actions_override or list(directives.actions) or EDAActions.default_sequence()
        sequence = [a for a in sequence if a in EDA_REGISTRY]
        if not sequence:
            sequence = EDAActions.default_sequence()

        n_rows, n_cols = df.shape
        columns = [str(c) for c in df.columns]
        dtypes: dict[str, str] = {c: str(df[c].dtype) for c in columns}
        missingness: dict[str, float] = {}
        cardinality: dict[str, int] = {}
        numeric_summary: dict[str, dict[str, float]] = {}
        correlations: dict[str, dict[str, float]] = {}
        roles = RoleProposal()
        warnings: list[str] = []
        suggestions: list[str] = []
        leakage: list[str] = []
        notes = [
            "AutoEDA proposes roles and readiness heuristics — not causal proof.",
        ]
        qc: Optional[dict[str, Any]] = None
        mining: Optional[dict[str, Any]] = None
        action_results: list[dict[str, Any]] = []
        actions_run: list[str] = []
        tools_invoked: list[dict[str, Any]] = []

        focus = [c for c in directives.focus_columns if c in df.columns]
        for name in sequence:
            kwargs: dict[str, Any] = {}
            if name == "summarize_distributions" and focus:
                kwargs["columns"] = focus
            elif name == "correlation_matrix":
                kwargs["max_cols"] = self.max_corr_cols
                if focus:
                    kwargs["columns"] = focus
            elif name == "suggest_roles":
                kwargs["role_hypotheses"] = directives.role_hypotheses
            elif name == "leakage_hints":
                kwargs["outcome"] = roles.outcome
                kwargs["correlations"] = correlations or None

            try:
                result = EDA_REGISTRY.run(name, df, **kwargs)
            except Exception as e:
                warnings.append(f"Action `{name}` soft-fail: {type(e).__name__}: {e}")
                continue

            actions_run.append(name)
            action_results.append(result.to_dict())
            tools_invoked.append({"tool": f"autoeda.{name}", "ok": True})
            warnings.extend(result.warnings)
            notes.extend(result.notes)
            payload = result.payload or {}

            if name == "summarize_distributions":
                dtypes.update(payload.get("dtypes") or {})
                missingness.update(payload.get("missingness") or {})
                numeric_summary.update(payload.get("numeric_summary") or {})
            elif name == "correlation_matrix":
                correlations = payload.get("correlations") or correlations
            elif name == "cardinality_report":
                cardinality = payload.get("cardinality") or cardinality
                missingness.update(payload.get("missingness") or {})
            elif name == "suggest_roles" and payload.get("roles"):
                rd = payload["roles"]
                roles = RoleProposal(
                    outcome=rd.get("outcome"),
                    treatment=rd.get("treatment"),
                    instruments=list(rd.get("instruments") or []),
                    confounders=list(rd.get("confounders") or []),
                    time_col=rd.get("time_col"),
                    group_col=rd.get("group_col"),
                    scores=dict(rd.get("scores") or {}),
                )
            elif name == "qc_snapshot" and payload.get("qc"):
                qc = payload["qc"]
                for issue in qc.get("issues") or []:
                    if issue.get("severity") in ("warn", "block"):
                        warnings.append(f"QC [{issue.get('code')}]: {issue.get('message')}")
            elif name == "leakage_hints":
                leakage = list(payload.get("leakage_hints") or [])
            elif name == "mining_profile":
                mining = payload.get("mining_profile")

        if not missingness:
            missingness = {c: float(df[c].isna().mean()) for c in columns}
        if not cardinality:
            cardinality = {c: int(df[c].nunique(dropna=True)) for c in columns}

        score = 1.0
        if n_rows < 30:
            warnings.append(f"Few rows ({n_rows}); estimates will be noisy.")
            score -= 0.25
        if roles.outcome is None or roles.treatment is None:
            warnings.append("Could not confidently propose outcome and treatment.")
            score -= 0.3
        if not roles.instruments:
            warnings.append("No instrument candidates — IV may be unavailable.")
            score -= 0.2
            suggestions.append("Add an exogenous Z or use DiD if treated×post exists.")
        for c, miss in missingness.items():
            if miss > 0.2:
                warnings.append(f"High missingness in `{c}` ({miss:.0%}).")
                score -= 0.05
        score -= 0.05 * len(leakage)
        if roles.instruments:
            suggestions.append("Validate first-stage F before trusting IV.")
        if directives.focus_columns:
            suggestions.append(f"Director focus: {directives.focus_columns[:6]}")

        plots = None
        if self.include_plots:
            try:
                import matplotlib  # noqa: F401

                plots = {"available": True, "note": "Use notebooks for figures."}
            except Exception:
                plots = {"available": False, "note": "matplotlib not installed."}

        dir_dict = directives.to_dict()
        dir_dict["tools_invoked"] = tools_invoked

        report = EDAReport(
            n_rows=n_rows,
            n_cols=n_cols,
            columns=columns,
            dtypes=dtypes,
            missingness=missingness,
            cardinality=cardinality,
            numeric_summary=numeric_summary,
            correlations=correlations,
            roles=roles,
            readiness_score=round(float(max(0.0, min(1.0, score))), 4),
            warnings=warnings,
            suggestions=suggestions,
            leakage_hints=leakage,
            qc=qc,
            mining_profile=mining,
            plots=plots,
            action_results=action_results,
            actions_run=actions_run,
            slm_directives=dir_dict,
            notes=list(dict.fromkeys(notes + list(directives.notes))),
            source=label,
            backend=directives.backend,
        )
        self.report = report
        return self

    def to_autocausal(self) -> Any:
        from autocausal.api import AutoCausal

        if self.frame is None:
            self.run()
        assert self.frame is not None
        ac = AutoCausal.from_dataframe(
            self.frame, source=f"eda:{self.report.source if self.report else 'memory'}"
        )
        ac.eda_report = self.report
        return ac

    def to_dict(self) -> dict[str, Any]:
        if self.report is None:
            self.run()
        assert self.report is not None
        return self.report.to_dict()

    def to_markdown(self) -> str:
        if self.report is None:
            self.run()
        assert self.report is not None
        return self.report.to_markdown()

    def write(self, path: Union[str, Path], *, fmt: str = "auto") -> Path:
        if self.report is None:
            self.run()
        assert self.report is not None
        return self.report.write(path, fmt=fmt)


def auto_eda(source: Any, *, use_slm: Optional[bool] = None, **kwargs: Any) -> EDAReport:
    suite = AutoEDASuite(source, use_slm=use_slm, **kwargs).run()
    assert suite.report is not None
    return suite.report
