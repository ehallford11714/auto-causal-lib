"""AutoEDASuite — SLM-directed orchestration over EDAActions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd

from autocausal.suites.autoeda.actions import EDA_REGISTRY, EDAActions
from autocausal.suites.autoeda.report import EDAReport, RoleProposal
from autocausal.suites.base import resolve_frame
from autocausal.suites.director import SLMAutoDirector, SLMDirectives, resolve_suite_slm
from autocausal.production import ProductionPolicy, resolve_policy
from autocausal.statistical_gates import run_statistical_gates

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
        mode: str = "exploratory",
        policy: Optional[ProductionPolicy | Mapping[str, Any]] = None,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        instrument: Optional[str | Sequence[str]] = None,
        confounders: Optional[Sequence[str]] = None,
        unit: Optional[str] = None,
        time: Optional[str] = None,
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
        self.mode = mode
        self.policy_input = policy
        self.explicit_treatment = treatment
        self.explicit_outcome = outcome
        self.explicit_instruments = (
            [instrument] if isinstance(instrument, str) else list(instrument or [])
        )
        self.explicit_confounders = list(confounders or [])
        self.explicit_unit = unit
        self.explicit_time = time
        self.frame: Optional[pd.DataFrame] = None
        self.report: Optional[EDAReport] = None
        self.directives: Optional[SLMDirectives] = None

    def run(self, source: Any = None, *, text: Optional[str] = None) -> "AutoEDASuite":
        src = self.source if source is None else source
        if src is None:
            raise ValueError("AutoEDASuite requires a DataFrame, path, or AutoCausal")
        df, label, ac_in = resolve_frame(src, table=self.table, query=self.query)
        self.frame = df
        txt = self.text if text is None else text
        inherited_policy = (
            getattr(ac_in, "policy", None) if self.policy_input is None else None
        )
        inherited_mode = getattr(ac_in, "mode", self.mode)
        policy = resolve_policy(
            inherited_mode,
            self.policy_input or inherited_policy,
        )

        director = SLMAutoDirector(
            use_slm=self.use_slm and policy.allow_slm,
            model_name=self.model_name,
        )
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
        association_scan: list[dict[str, Any]] = []

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

        # Explicit design metadata always supersedes role hypotheses.
        if self.explicit_outcome is not None:
            roles.outcome = self.explicit_outcome
        if self.explicit_treatment is not None:
            roles.treatment = self.explicit_treatment
        if self.explicit_instruments:
            roles.instruments = list(self.explicit_instruments)
        if self.explicit_confounders:
            roles.confounders = list(self.explicit_confounders)
        if self.explicit_time is not None:
            roles.time_col = self.explicit_time
        if self.explicit_unit is not None:
            roles.group_col = self.explicit_unit

        if not missingness:
            missingness = {c: float(df[c].isna().mean()) for c in columns}
        if not cardinality:
            cardinality = {c: int(df[c].nunique(dropna=True)) for c in columns}

        # Typed descriptive association scan; q-values are BH-FDR adjusted.
        proposed_association_columns = (
            focus[: self.max_corr_cols]
            if len(focus) >= 2
            else columns[: self.max_corr_cols]
        )
        association_columns = [
            column
            for column in proposed_association_columns
            if pd.api.types.is_numeric_dtype(df[column])
            or df[column].nunique(dropna=True) <= 100
        ][: self.max_corr_cols]
        if len(association_columns) >= 2:
            try:
                from autocausal.correlation import correlation_matrix

                typed = correlation_matrix(
                    df,
                    columns=association_columns,
                    method="auto",
                    alpha=policy.statistical_validity.fdr_alpha,  # type: ignore[union-attr]
                    random_state=policy.random_state,
                )
                association_scan = [
                    result.to_dict() for result in typed.results
                ]
                notes.append(
                    "Typed associations are descriptive and never causal identification evidence."
                )
            except Exception as exc:
                warnings.append(
                    f"Typed association scan soft-fail: {type(exc).__name__}: {exc}"
                )

        row_missing = df.isna().sum(axis=1)
        missingness_patterns = {
            "rows_complete": int((row_missing == 0).sum()),
            "rows_one_missing": int((row_missing == 1).sum()),
            "rows_multiple_missing": int((row_missing > 1).sum()),
            "columns_with_missing": int(
                sum(value > 0 for value in missingness.values())
            ),
        }
        categorical_imbalance: dict[str, Any] = {}
        for column in columns:
            series = df[column]
            is_categorical = (
                not pd.api.types.is_numeric_dtype(series)
                or series.nunique(dropna=True) <= 10
            )
            if not is_categorical or len(series.dropna()) == 0:
                continue
            shares = series.value_counts(normalize=True, dropna=True)
            categorical_imbalance[column] = {
                "levels": int(series.nunique(dropna=True)),
                "largest_level_fraction": float(shares.iloc[0]),
                "smallest_level_fraction": float(shares.iloc[-1]),
                "level_values_redacted": True,
            }
        subgroup_imbalance: dict[str, Any] = {}
        group_column = roles.group_col
        if group_column and group_column in df.columns:
            counts = df[group_column].value_counts(dropna=False)
            subgroup_imbalance = {
                "group_column": group_column,
                "n_groups": int(len(counts)),
                "min_group_size": int(counts.min()) if len(counts) else 0,
                "max_group_size": int(counts.max()) if len(counts) else 0,
                "size_ratio": (
                    float(counts.max() / max(counts.min(), 1))
                    if len(counts)
                    else None
                ),
                "group_values_redacted": True,
            }
        if (
            roles.time_col
            and roles.time_col in df.columns
            and roles.group_col
            and roles.group_col in df.columns
        ):
            duplicate_unit_time = int(
                df.duplicated([roles.group_col, roles.time_col]).sum()
            )
            if duplicate_unit_time:
                leakage.append(
                    f"{duplicate_unit_time} duplicate unit-time rows require panel design review."
                )

        causal_readiness: dict[str, Any] = {
            "treatment": roles.treatment,
            "outcome": roles.outcome,
            "instruments": list(roles.instruments),
            "confounders": list(roles.confounders),
            "unit": roles.group_col,
            "time": roles.time_col,
            "roles_explicit": {
                "treatment": self.explicit_treatment is not None,
                "outcome": self.explicit_outcome is not None,
                "instrument": bool(self.explicit_instruments),
                "confounders": bool(self.explicit_confounders),
            },
            "instrument_assumptions_verified": False,
            "association_used_as_identification": False,
        }
        if roles.treatment and roles.treatment in df.columns:
            treatment_counts = df[roles.treatment].value_counts(dropna=True)
            causal_readiness["treatment_levels"] = int(len(treatment_counts))
            causal_readiness["minimum_treatment_group"] = (
                int(treatment_counts.min()) if len(treatment_counts) else 0
            )
        if roles.outcome and roles.outcome in df.columns:
            causal_readiness["outcome_missing_fraction"] = float(
                df[roles.outcome].isna().mean()
            )

        assumption_readiness: dict[str, Any] = {}
        gate_report = None
        if (
            roles.treatment
            and roles.outcome
            and roles.treatment in df.columns
            and roles.outcome in df.columns
        ):
            diagnostics = run_statistical_gates(
                df,
                treatment=roles.treatment,
                outcome=roles.outcome,
                instrument=roles.instruments or None,
                controls=roles.confounders,
                policy=policy,
                random_state=policy.random_state,
            )
            assumption_readiness = diagnostics.metrics
            gate_report = diagnostics.gates.to_dict()

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

        descriptive_findings = [
            f"{sum(value > 0 for value in missingness.values())} columns contain missing values.",
            f"{len(categorical_imbalance)} categorical/low-cardinality columns were checked for imbalance.",
            f"{len(association_scan)} typed pairwise associations were scanned with FDR where p-values exist.",
        ]
        predictive_findings = [
            "High association or feature relevance may support prediction but does not identify an intervention effect."
        ]
        causal_findings = [
            (
                "Treatment/outcome design fields were explicit."
                if self.explicit_treatment and self.explicit_outcome
                else "At least one treatment/outcome role is heuristic; production inference requires explicit fields."
            ),
            "Instrument exclusion and independence are unverified unless documented outside this report.",
        ]

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
            association_scan=association_scan,
            missingness_patterns=missingness_patterns,
            categorical_imbalance=categorical_imbalance,
            subgroup_imbalance=subgroup_imbalance,
            causal_readiness=causal_readiness,
            assumption_readiness=assumption_readiness,
            descriptive_findings=descriptive_findings,
            predictive_findings=predictive_findings,
            causal_readiness_findings=causal_findings,
            gate_report=gate_report,
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
