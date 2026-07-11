"""AutoEDASuite — SLM-directed exploratory analysis for causal readiness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from autocausal.suites.base import resolve_frame, write_report
from autocausal.suites.director import (
    EPISTEMIC_NOTE,
    SLMAutoDirector,
    SLMDirectives,
    resolve_suite_slm,
)

__all__ = ["RoleProposal", "EDAReport", "AutoEDASuite", "auto_eda"]


@dataclass
class RoleProposal:
    outcome: Optional[str] = None
    treatment: Optional[str] = None
    instruments: list[str] = field(default_factory=list)
    confounders: list[str] = field(default_factory=list)
    time_col: Optional[str] = None
    group_col: Optional[str] = None
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EDAReport:
    n_rows: int
    n_cols: int
    columns: list[str]
    dtypes: dict[str, str]
    missingness: dict[str, float]
    cardinality: dict[str, int]
    numeric_summary: dict[str, dict[str, float]]
    correlations: dict[str, dict[str, float]]
    roles: RoleProposal
    readiness_score: float
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    leakage_hints: list[str] = field(default_factory=list)
    qc: Optional[dict[str, Any]] = None
    mining_profile: Optional[dict[str, Any]] = None
    plots: Optional[dict[str, Any]] = None
    slm_directives: Optional[dict[str, Any]] = None
    notes: list[str] = field(default_factory=list)
    source: str = ""
    backend: str = "rule"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": list(self.columns),
            "dtypes": dict(self.dtypes),
            "missingness": dict(self.missingness),
            "cardinality": dict(self.cardinality),
            "numeric_summary": dict(self.numeric_summary),
            "correlations": dict(self.correlations),
            "roles": self.roles.to_dict(),
            "readiness_score": self.readiness_score,
            "warnings": list(self.warnings),
            "suggestions": list(self.suggestions),
            "leakage_hints": list(self.leakage_hints),
            "qc": self.qc,
            "mining_profile": self.mining_profile,
            "plots": self.plots,
            "slm_directives": self.slm_directives,
            "notes": list(self.notes),
            "source": self.source,
            "backend": self.backend,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# AutoEDA report (causal readiness)",
            "",
            f"- rows={self.n_rows}, cols={self.n_cols}",
            f"- readiness_score=**{self.readiness_score:.2f}**",
            f"- backend: `{self.backend}`",
            "",
            f"> {EPISTEMIC_NOTE}",
            "",
            "## Proposed roles (hypotheses — not ground truth)",
            f"- Y (outcome): `{self.roles.outcome}`",
            f"- X (treatment): `{self.roles.treatment}`",
            f"- Z (instruments): {self.roles.instruments}",
            f"- W (confounders): {self.roles.confounders}",
            "",
            "## Missingness (top)",
            "",
            "| column | missing% | cardinality |",
            "|---|---:|---:|",
        ]
        ranked = sorted(self.missingness.items(), key=lambda kv: -kv[1])[:15]
        for c, m in ranked:
            lines.append(f"| `{c}` | {m:.1%} | {self.cardinality.get(c, '')} |")
        lines.append("")
        if self.leakage_hints:
            lines += ["## Leakage hints", ""]
            for h in self.leakage_hints:
                lines.append(f"- {h}")
            lines.append("")
        if self.warnings:
            lines += ["## Warnings", ""]
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")
        if self.suggestions:
            lines += ["## Suggestions", ""]
            for s in self.suggestions:
                lines.append(f"- {s}")
            lines.append("")
        if self.notes:
            lines += ["## Notes", ""]
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        return "\n".join(lines)

    def write(self, path: Union[str, Path], *, fmt: str = "auto") -> Path:
        return write_report(self, path, fmt=fmt)


class AutoEDASuite:
    """Library-first EDA suite for causal readiness (SLM-directed).

    Example::

        from autocausal import AutoEDASuite
        eda = AutoEDASuite(df, use_slm=True).run()
        print(eda.report.to_markdown())
    """

    def __init__(
        self,
        source: Any = None,
        *,
        use_slm: Optional[bool] = None,
        model_name: Optional[str] = None,
        text: str = "",
        max_corr_cols: int = 12,
        include_plots: bool = False,
        include_mining: bool = True,
        table: Optional[str] = None,
        query: Optional[str] = None,
    ) -> None:
        self.source = source
        self.use_slm = resolve_suite_slm(use_slm)
        self.model_name = model_name
        self.text = text
        self.max_corr_cols = max_corr_cols
        self.include_plots = include_plots
        self.include_mining = include_mining
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

        report = _build_eda(
            df,
            directives=directives,
            text=txt,
            max_corr_cols=self.max_corr_cols,
            include_plots=self.include_plots,
            include_mining=self.include_mining,
            source_label=label,
        )
        report.slm_directives = directives.to_dict()
        report.backend = directives.backend
        report.notes = list(dict.fromkeys(list(report.notes) + list(directives.notes)))
        self.report = report
        return self

    def to_autocausal(self) -> Any:
        from autocausal.api import AutoCausal

        if self.frame is None:
            self.run()
        assert self.frame is not None
        ac = AutoCausal.from_dataframe(self.frame, source=f"eda:{self.report.source if self.report else 'memory'}")
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


def auto_eda(
    source: Any,
    *,
    use_slm: Optional[bool] = None,
    **kwargs: Any,
) -> EDAReport:
    suite = AutoEDASuite(source, use_slm=use_slm, **kwargs).run()
    assert suite.report is not None
    return suite.report


def _score_role(name: str, series: pd.Series, role: str) -> float:
    cl = name.lower()
    score = 0.0
    if role == "outcome":
        if any(k in cl for k in ("y", "outcome", "sales", "return", "target", "revenue", "churn")):
            score += 2.0
        if pd.api.types.is_numeric_dtype(series):
            score += 0.5
    elif role == "treatment":
        if any(k in cl for k in ("x", "treat", "campaign", "exposure", "spend", "demand")):
            score += 2.0
        if series.nunique(dropna=True) <= 3:
            score += 0.3
    elif role == "instrument":
        if any(k in cl for k in ("z", "instrument", "iv", "lottery", "assign", "random")):
            score += 2.5
    elif role == "confounder":
        if any(k in cl for k in ("age", "size", "macro", "control", "covar", "region", "gender")):
            score += 1.5
    return score


def _propose_roles(df: pd.DataFrame, directives: SLMDirectives) -> RoleProposal:
    scores_y: dict[str, float] = {}
    scores_x: dict[str, float] = {}
    scores_z: dict[str, float] = {}
    scores_w: dict[str, float] = {}
    for c in df.columns:
        s = df[c]
        scores_y[c] = _score_role(c, s, "outcome")
        scores_x[c] = _score_role(c, s, "treatment")
        scores_z[c] = _score_role(c, s, "instrument")
        scores_w[c] = _score_role(c, s, "confounder")

    # Boost from SLM role hypotheses
    for c in directives.role_hypotheses.get("outcome") or []:
        if c in scores_y:
            scores_y[c] += 2.0
    for c in directives.role_hypotheses.get("treatment") or []:
        if c in scores_x:
            scores_x[c] += 2.0
    for c in directives.role_hypotheses.get("instrument") or []:
        if c in scores_z:
            scores_z[c] += 2.0
    for c in directives.role_hypotheses.get("confounder") or []:
        if c in scores_w:
            scores_w[c] += 1.5

    def _best(scores: dict[str, float], exclude: set[str]) -> Optional[str]:
        cands = [(c, v) for c, v in scores.items() if c not in exclude and v > 0]
        if not cands:
            for c in df.columns:
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c]):
                    return str(c)
            return None
        cands.sort(key=lambda kv: kv[1], reverse=True)
        return str(cands[0][0])

    y = _best(scores_y, set())
    x = _best(scores_x, {y} if y else set())
    used = {c for c in (y, x) if c}
    z_sorted = sorted(
        ((c, v) for c, v in scores_z.items() if c not in used and v > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    instruments = [str(c) for c, _ in z_sorted[:3]]
    used |= set(instruments)
    confounders = [
        str(c)
        for c, v in sorted(scores_w.items(), key=lambda kv: -kv[1])
        if c not in used and v > 0
    ][:5]

    time_col = next(
        (
            str(c)
            for c in df.columns
            if any(k in str(c).lower() for k in ("date", "time", "year", "month"))
            or str(df[c].dtype).startswith("datetime")
        ),
        None,
    )
    group_col = next(
        (str(c) for c in df.columns if any(k in str(c).lower() for k in ("group", "unit", "region", "id"))),
        None,
    )

    return RoleProposal(
        outcome=y,
        treatment=x,
        instruments=instruments,
        confounders=confounders,
        time_col=time_col,
        group_col=group_col,
        scores={
            **{f"y:{k}": v for k, v in scores_y.items()},
            **{f"x:{k}": v for k, v in scores_x.items()},
            **{f"z:{k}": v for k, v in scores_z.items()},
        },
    )


def _build_eda(
    df: pd.DataFrame,
    *,
    directives: SLMDirectives,
    text: str,
    max_corr_cols: int,
    include_plots: bool,
    include_mining: bool,
    source_label: str,
) -> EDAReport:
    n_rows, n_cols = df.shape
    columns = [str(c) for c in df.columns]
    # Prefer director focus order for summaries
    focus = [c for c in directives.focus_columns if c in df.columns]
    ordered = list(dict.fromkeys(focus + columns))

    dtypes = {c: str(df[c].dtype) for c in ordered}
    missingness = {c: float(df[c].isna().mean()) for c in ordered}
    cardinality = {c: int(df[c].nunique(dropna=True)) for c in ordered}

    numeric_cols = [c for c in ordered if pd.api.types.is_numeric_dtype(df[c])]
    numeric_summary: dict[str, dict[str, float]] = {}
    for c in numeric_cols:
        s = df[c].dropna()
        if s.empty:
            continue
        numeric_summary[c] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
            "nunique": float(s.nunique()),
        }

    corr_cols = numeric_cols[:max_corr_cols]
    correlations: dict[str, dict[str, float]] = {}
    if len(corr_cols) >= 2:
        cm = df[corr_cols].corr(numeric_only=True)
        for a in cm.columns:
            correlations[str(a)] = {
                str(b): float(cm.loc[a, b]) for b in cm.columns if a != b and np.isfinite(cm.loc[a, b])
            }

    roles = _propose_roles(df, directives)

    warnings: list[str] = []
    suggestions: list[str] = []
    leakage: list[str] = []
    notes = [
        "AutoEDA proposes roles and readiness heuristics — not causal proof.",
        EPISTEMIC_NOTE,
    ]
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

    # Leakage: near-perfect corr with outcome
    if roles.outcome and roles.outcome in correlations:
        for other, r in correlations[roles.outcome].items():
            if abs(r) >= 0.99 and other != roles.outcome:
                leakage.append(
                    f"`{other}` |corr|={abs(r):.3f} with outcome `{roles.outcome}` — possible leakage."
                )
                score -= 0.1

    # Name-based leakage
    for c in columns:
        cl = c.lower()
        if any(k in cl for k in ("target_leak", "future_", "y_true", "ground_truth")):
            leakage.append(f"Column name `{c}` looks like label leakage.")

    qc_dict: Optional[dict[str, Any]] = None
    try:
        from autocausal.qc import validate_frame

        qc_dict = validate_frame(df).to_dict()
        for issue in qc_dict.get("issues") or []:
            if issue.get("severity") in ("warn", "block"):
                warnings.append(f"QC [{issue.get('code')}]: {issue.get('message')}")
    except Exception as e:
        notes.append(f"QC soft-fail: {type(e).__name__}")

    mining_profile: Optional[dict[str, Any]] = None
    if include_mining:
        try:
            from autocausal.mining import profile_dataframe

            mining_profile = profile_dataframe(df)
        except Exception:
            try:
                from autocausal.mining import mine

                mining_profile = {"columns": mine(df, min_score=0.99).columns}
            except Exception as e:
                notes.append(f"Mining profile soft-fail: {type(e).__name__}: {e}")

    plots: Optional[dict[str, Any]] = None
    if include_plots:
        try:
            import matplotlib  # noqa: F401

            plots = {
                "available": True,
                "note": "Plot generation skipped in library path; use notebooks for figures.",
            }
        except Exception:
            plots = {"available": False, "note": "matplotlib not installed — text tables only."}

    if roles.instruments:
        suggestions.append("Validate first-stage F before trusting IV.")
    if directives.focus_columns:
        suggestions.append(f"Director focus: {directives.focus_columns[:6]}")

    score = float(max(0.0, min(1.0, score)))
    return EDAReport(
        n_rows=n_rows,
        n_cols=n_cols,
        columns=columns,
        dtypes=dtypes,
        missingness=missingness,
        cardinality=cardinality,
        numeric_summary=numeric_summary,
        correlations=correlations,
        roles=roles,
        readiness_score=round(score, 4),
        warnings=warnings,
        suggestions=suggestions,
        leakage_hints=leakage,
        qc=qc_dict,
        mining_profile=mining_profile,
        plots=plots,
        notes=notes,
        source=source_label,
        backend=directives.backend,
    )
