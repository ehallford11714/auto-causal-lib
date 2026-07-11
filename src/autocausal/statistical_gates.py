"""Practical statistical assumption diagnostics for production-oriented gates.

These checks assess data/design assumptions.  They do not identify causal
effects and do not replace domain review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from autocausal.production import (
    EPISTEMIC,
    GateReport,
    GateResult,
    ProductionPolicy,
)

__all__ = [
    "StatisticalDiagnostics",
    "apply_fdr_to_edges",
    "benjamini_hochberg",
    "first_stage_diagnostics",
    "run_statistical_gates",
]


@dataclass
class StatisticalDiagnostics:
    """Serializable assumptions/evidence diagnostics with no raw rows."""

    schema: str = "AutoCausalStatisticalDiagnostics.v1"
    n: int = 0
    treatment: Optional[str] = None
    outcome: Optional[str] = None
    instrument: Optional[str] = None
    controls: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: GateReport = field(default_factory=GateReport)
    notes: list[str] = field(default_factory=lambda: [EPISTEMIC])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "n": self.n,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "instrument": self.instrument,
            "controls": list(self.controls),
            "metrics": dict(self.metrics),
            "gates": self.gates.to_dict(),
            "notes": list(self.notes),
        }


def _status(
    ok: bool,
    *,
    profile: str,
    warning_only: bool = False,
    escalate: bool = False,
) -> str:
    if ok:
        return "pass"
    if warning_only or profile == "exploratory":
        return "warn"
    if profile == "review" or escalate:
        return "escalate"
    return "fail"


def _gate(
    gate_id: str,
    ok: bool,
    detail: str,
    *,
    profile: str,
    stage: str = "statistics",
    metric: Any = None,
    threshold: Any = None,
    evidence: Optional[dict[str, Any]] = None,
    remediation: Optional[str] = None,
    warning_only: bool = False,
    escalate: bool = False,
) -> GateResult:
    status = _status(
        ok,
        profile=profile,
        warning_only=warning_only,
        escalate=escalate,
    )
    return GateResult(
        id=gate_id,
        ok=status in ("pass", "warn", "skip"),
        status=status,
        detail=detail,
        metric=metric,
        threshold=threshold,
        evidence=dict(evidence or {}),
        remediation=remediation,
        stage=stage,
    )


def benjamini_hochberg(
    pvalues: Sequence[Optional[float]],
    *,
    alpha: float = 0.05,
) -> tuple[list[Optional[float]], list[bool]]:
    """Benjamini-Hochberg adjusted q-values preserving missing positions."""
    valid = [
        (index, float(value))
        for index, value in enumerate(pvalues)
        if value is not None and np.isfinite(float(value))
    ]
    qvalues: list[Optional[float]] = [None] * len(pvalues)
    rejected = [False] * len(pvalues)
    if not valid:
        return qvalues, rejected
    valid.sort(key=lambda item: item[1])
    m = len(valid)
    adjusted = [0.0] * m
    running = 1.0
    for reverse_rank in range(m - 1, -1, -1):
        rank = reverse_rank + 1
        candidate = valid[reverse_rank][1] * m / rank
        running = min(running, candidate)
        adjusted[reverse_rank] = min(1.0, running)
    for position, ((original_index, _), qvalue) in enumerate(
        zip(valid, adjusted)
    ):
        qvalues[original_index] = float(qvalue)
        rejected[original_index] = bool(qvalue <= alpha)
    return qvalues, rejected


def apply_fdr_to_edges(
    edges: Sequence[dict[str, Any]],
    *,
    alpha: float = 0.05,
) -> tuple[list[dict[str, Any]], GateResult]:
    """Attach BH q-values to edge scans; edges without p-values are untested."""
    out = [dict(edge) for edge in edges]
    pvalues = [
        float(edge["pvalue"])
        if edge.get("pvalue") is not None
        and np.isfinite(float(edge.get("pvalue")))
        else None
        for edge in out
    ]
    qvalues, rejected = benjamini_hochberg(pvalues, alpha=alpha)
    tested = 0
    significant = 0
    for edge, qvalue, keep in zip(out, qvalues, rejected):
        edge["qvalue"] = round(qvalue, 6) if qvalue is not None else None
        edge["fdr_reject_null"] = bool(keep) if qvalue is not None else None
        if qvalue is not None:
            tested += 1
            significant += int(keep)
    gate = GateResult(
        id="multiple_testing_fdr",
        ok=True,
        status="pass" if tested else "skip",
        detail=(
            f"BH-FDR alpha={alpha}: {significant}/{tested} tested edges retained"
            if tested
            else "No meaningful edge p-values; BH-FDR skipped"
        ),
        metric=significant,
        threshold=alpha,
        evidence={"tested": tested, "significant": significant},
        remediation=(
            "Treat untested edges as exploratory; do not infer significance from score."
        ),
        stage="discovery",
    )
    return out, gate


def _numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    keep = [column for column in columns if column in df.columns]
    if not keep:
        return pd.DataFrame(index=df.index)
    return df[keep].apply(pd.to_numeric, errors="coerce")


def _ols(
    y: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    fitted = x @ beta
    return beta, residual, fitted


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _two_sided_normal_p(statistic: float) -> float:
    return float(2.0 * (1.0 - _normal_cdf(abs(float(statistic)))))


def first_stage_diagnostics(
    df: pd.DataFrame,
    *,
    treatment: str,
    instrument: str | Sequence[str],
    controls: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Partial first-stage F for observed instrument(s), with explicit columns."""
    instruments = (
        [instrument] if isinstance(instrument, str) else list(instrument)
    )
    controls_list = list(controls or [])
    columns = [treatment, *instruments, *controls_list]
    work = _numeric_frame(df, columns).dropna()
    n = len(work)
    k_z = len(instruments)
    if n <= len(controls_list) + k_z + 2 or k_z == 0:
        return {
            "ok": False,
            "first_stage_f": None,
            "n": n,
            "n_instruments": k_z,
            "reason": "insufficient complete rows",
        }
    y = work[treatment].to_numpy(dtype=float)
    w = (
        work[controls_list].to_numpy(dtype=float)
        if controls_list
        else np.empty((n, 0))
    )
    z = work[instruments].to_numpy(dtype=float)
    restricted = np.column_stack([np.ones(n), w])
    unrestricted = np.column_stack([np.ones(n), w, z])
    _, residual_r, _ = _ols(y, restricted)
    _, residual_u, fitted = _ols(y, unrestricted)
    ssr_r = float(residual_r @ residual_r)
    ssr_u = float(residual_u @ residual_u)
    denominator_df = n - unrestricted.shape[1]
    if denominator_df <= 0 or ssr_u <= 1e-15:
        f_stat = float("inf") if ssr_r > ssr_u else 0.0
    else:
        numerator = max(0.0, (ssr_r - ssr_u) / k_z)
        f_stat = float(numerator / (ssr_u / denominator_df))
    centered = y - y.mean()
    r2 = 1.0 - ssr_u / max(float(centered @ centered), 1e-15)
    return {
        "ok": True,
        "first_stage_f": f_stat,
        "first_stage_r2": float(r2),
        "n": n,
        "n_instruments": k_z,
        "instruments": [str(value) for value in instruments],
        "fitted_variance": float(np.var(fitted)),
    }


def _vif_and_condition(work: pd.DataFrame) -> tuple[dict[str, float], float]:
    if work.empty or work.shape[1] == 0:
        return {}, 0.0
    standardized = work.copy()
    for column in standardized.columns:
        values = standardized[column].to_numpy(dtype=float)
        sd = float(np.std(values))
        standardized[column] = (values - float(np.mean(values))) / (
            sd if sd > 1e-12 else 1.0
        )
    matrix = standardized.to_numpy(dtype=float)
    condition = float(np.linalg.cond(np.column_stack([np.ones(len(matrix)), matrix])))
    vifs: dict[str, float] = {}
    if matrix.shape[1] <= 1:
        return {str(work.columns[0]): 1.0}, condition
    for index, column in enumerate(work.columns):
        y = matrix[:, index]
        others = np.delete(matrix, index, axis=1)
        x = np.column_stack([np.ones(len(matrix)), others])
        _, residual, _ = _ols(y, x)
        total = float((y - y.mean()) @ (y - y.mean()))
        r2 = 1.0 - float(residual @ residual) / max(total, 1e-15)
        vifs[str(column)] = float(1.0 / max(1.0 - r2, 1e-12))
    return vifs, condition


def _breusch_pagan(
    residual: np.ndarray,
    design: np.ndarray,
) -> tuple[float, Optional[float], str]:
    """Breusch-Pagan LM statistic; scipy p-value when available."""
    squared = residual**2
    _, auxiliary_residual, _ = _ols(squared, design)
    total = float((squared - squared.mean()) @ (squared - squared.mean()))
    r2 = 1.0 - float(auxiliary_residual @ auxiliary_residual) / max(total, 1e-15)
    lm = max(0.0, len(squared) * r2)
    df = max(1, design.shape[1] - 1)
    try:
        from scipy.stats import chi2

        return lm, float(chi2.sf(lm, df)), "breusch_pagan"
    except Exception:
        # Monotonic approximation; clearly marked as fallback.
        return lm, None, "breusch_pagan_lm_no_scipy"


def _durbin_watson(residual: np.ndarray) -> float:
    denominator = float(residual @ residual)
    if denominator <= 1e-15 or len(residual) < 2:
        return 2.0
    differences = np.diff(residual)
    return float((differences @ differences) / denominator)


def _overlap(
    df: pd.DataFrame,
    *,
    treatment: str,
    controls: Sequence[str],
    epsilon: float,
    random_state: int,
) -> dict[str, Any]:
    columns = [treatment, *controls]
    work = df[[column for column in columns if column in df.columns]].copy()
    if treatment not in work.columns:
        return {"ok": False, "reason": "treatment missing"}
    treatment_values = pd.to_numeric(work[treatment], errors="coerce")
    mask = treatment_values.notna()
    work = work.loc[mask]
    treatment_values = treatment_values.loc[mask].astype(int)
    if set(treatment_values.unique()) - {0, 1} or treatment_values.nunique() != 2:
        return {"ok": False, "reason": "treatment is not binary"}
    if not controls:
        proportion = float(treatment_values.mean())
        overlap = 1.0 if epsilon < proportion < 1.0 - epsilon else 0.0
        return {
            "ok": True,
            "overlap_fraction": overlap,
            "propensity_min": proportion,
            "propensity_max": proportion,
            "n": len(work),
            "method": "marginal_treatment_rate",
        }
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        x = work[list(controls)]
        numeric = [
            column
            for column in controls
            if pd.api.types.is_numeric_dtype(x[column])
        ]
        categorical = [column for column in controls if column not in numeric]
        transformer = ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    categorical,
                ),
            ],
            remainder="drop",
        )
        model = Pipeline(
            [
                ("preprocess", transformer),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=int(random_state),
                    ),
                ),
            ]
        )
        model.fit(x, treatment_values)
        propensity = model.predict_proba(x)[:, 1]
        inside = (propensity >= epsilon) & (propensity <= 1.0 - epsilon)
        return {
            "ok": True,
            "overlap_fraction": float(np.mean(inside)),
            "propensity_min": float(np.min(propensity)),
            "propensity_max": float(np.max(propensity)),
            "propensity_p01": float(np.quantile(propensity, 0.01)),
            "propensity_p99": float(np.quantile(propensity, 0.99)),
            "n": len(work),
            "method": "logistic_propensity",
        }
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "method": "logistic_propensity_soft_fail",
        }


def run_statistical_gates(
    df: pd.DataFrame,
    *,
    treatment: Optional[str] = None,
    outcome: Optional[str] = None,
    instrument: Optional[str | Sequence[str]] = None,
    controls: Optional[Sequence[str]] = None,
    policy: Optional[ProductionPolicy] = None,
    profile: Optional[str] = None,
    random_state: Optional[int] = None,
) -> StatisticalDiagnostics:
    """Run deterministic design/statistical checks with explicit soft deps."""
    policy = policy or ProductionPolicy.review()
    statistical = policy.statistical_validity
    assert statistical is not None
    profile_name = str(profile or policy.profile)
    controls_list = [
        str(column)
        for column in (controls or [])
        if str(column) in df.columns
    ]
    rng_seed = policy.random_state if random_state is None else int(random_state)
    report = GateReport(profile=profile_name, policy_version=policy.policy_version)
    diagnostics = StatisticalDiagnostics(
        n=len(df),
        treatment=treatment,
        outcome=outcome,
        instrument=(
            instrument
            if isinstance(instrument, str) or instrument is None
            else ",".join(str(value) for value in instrument)
        ),
        controls=controls_list,
        gates=report,
    )

    n = len(df)
    sample_ok = n >= statistical.min_sample_size
    report.add(
        _gate(
            "sample_size",
            sample_ok,
            f"n={n}; minimum={statistical.min_sample_size}",
            profile=profile_name,
            metric=n,
            threshold=statistical.min_sample_size,
            remediation="Collect more independent units or reduce model complexity.",
        )
    )

    # Events per variable for binary treatment/outcome.
    binary_name = None
    for candidate in (outcome, treatment):
        if candidate and candidate in df.columns:
            unique = pd.to_numeric(
                df[candidate], errors="coerce"
            ).dropna().unique()
            if len(unique) == 2 and set(unique).issubset({0, 1}):
                binary_name = candidate
                break
    if binary_name:
        values = pd.to_numeric(df[binary_name], errors="coerce").dropna()
        events = int(min((values == 0).sum(), (values == 1).sum()))
        n_parameters = max(1, len(controls_list) + 1)
        epv = events / n_parameters
        diagnostics.metrics["events_per_variable"] = epv
        report.add(
            _gate(
                "events_per_variable",
                epv >= statistical.min_events_per_variable,
                (
                    f"minority events/parameter={epv:.2f} "
                    f"for `{binary_name}`"
                ),
                profile=profile_name,
                metric=round(epv, 4),
                threshold=statistical.min_events_per_variable,
                remediation="Collect more minority events or reduce covariates.",
            )
        )
    else:
        report.add(
            GateResult(
                id="events_per_variable",
                ok=True,
                status="skip",
                detail="No binary treatment/outcome; EPV not applicable.",
                stage="statistics",
            )
        )

    # Missingness mechanism heuristic: indicator correlations are not a formal MCAR test.
    numeric_columns = [
        str(column)
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    max_missing_indicator_corr = 0.0
    missing_pairs = 0
    for column in df.columns:
        missing = df[column].isna().astype(float)
        if missing.sum() == 0 or missing.nunique() < 2:
            continue
        for numeric_column in numeric_columns[:30]:
            if numeric_column == column:
                continue
            pair = pd.concat(
                [
                    missing.rename("missing"),
                    pd.to_numeric(
                        df[numeric_column], errors="coerce"
                    ).rename("value"),
                ],
                axis=1,
            ).dropna()
            if len(pair) < 10 or pair["value"].std() <= 1e-12:
                continue
            correlation = abs(float(pair["missing"].corr(pair["value"])))
            if np.isfinite(correlation):
                max_missing_indicator_corr = max(
                    max_missing_indicator_corr, correlation
                )
                missing_pairs += 1
    diagnostics.metrics["missingness_indicator_max_abs_corr"] = (
        max_missing_indicator_corr
    )
    report.add(
        _gate(
            "missingness_mechanism_heuristic",
            max_missing_indicator_corr < 0.30,
            (
                "Heuristic missing-indicator association "
                f"max|r|={max_missing_indicator_corr:.3f}; "
                "this is not a formal MCAR/MAR/MNAR test."
            ),
            profile=profile_name,
            metric=round(max_missing_indicator_corr, 6),
            threshold="<0.30",
            evidence={"pairs_scanned": missing_pairs, "formal_test": False},
            remediation="Model missingness, inspect collection process, and run sensitivity analysis.",
            warning_only=True,
        )
    )

    # Variance and normality (normality is warning-only for discovery).
    near_zero = []
    normality: dict[str, Any] = {}
    for column in numeric_columns[:50]:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        variance = float(values.var(ddof=1)) if len(values) > 1 else 0.0
        if not np.isfinite(variance) or variance <= policy.data_quality.near_zero_variance:  # type: ignore[union-attr]
            near_zero.append(column)
        if len(values) >= 8:
            sample = values.iloc[: min(len(values), 5000)].to_numpy(dtype=float)
            try:
                from scipy.stats import shapiro

                statistic, pvalue = shapiro(sample)
                normality[column] = {
                    "method": "shapiro",
                    "statistic": float(statistic),
                    "pvalue": float(pvalue),
                }
            except Exception:
                skew = float(values.skew())
                normality[column] = {
                    "method": "skew_fallback",
                    "skew": skew,
                    "pvalue": None,
                }
    diagnostics.metrics["near_zero_variance_columns"] = near_zero
    diagnostics.metrics["normality"] = normality
    relevant_near_zero = [
        column for column in near_zero if column in (treatment, outcome)
    ]
    report.add(
        _gate(
            "near_zero_variance",
            not relevant_near_zero,
            (
                "No treatment/outcome near-zero variance"
                if not relevant_near_zero
                else f"Near-zero variance: {relevant_near_zero}"
            ),
            profile=profile_name,
            metric=relevant_near_zero,
            threshold=policy.data_quality.near_zero_variance,  # type: ignore[union-attr]
            remediation="Choose varying treatment/outcome or collect informative observations.",
        )
    )
    non_normal = [
        column
        for column, values in normality.items()
        if values.get("pvalue") is not None and values["pvalue"] < statistical.alpha
    ]
    report.add(
        _gate(
            "normality_diagnostic",
            not non_normal,
            (
                "No strong Shapiro warnings"
                if not non_normal
                else f"Non-normality flagged: {non_normal[:12]}"
            ),
            profile=profile_name,
            metric=len(non_normal),
            threshold=f"p>={statistical.alpha}",
            remediation="Use robust/nonparametric inference where relevant.",
            warning_only=True,
        )
    )

    # Multicollinearity on treatment + controls (not outcome).
    design_columns = [
        column
        for column in [treatment, *controls_list]
        if column and column in numeric_columns
    ]
    design = _numeric_frame(df, design_columns).dropna()
    vifs, condition = _vif_and_condition(design)
    max_vif = max(vifs.values(), default=1.0)
    diagnostics.metrics["vif"] = vifs
    diagnostics.metrics["condition_number"] = condition
    collinearity_ok = (
        max_vif <= statistical.max_vif
        and condition <= statistical.max_condition_number
    )
    report.add(
        _gate(
            "multicollinearity",
            collinearity_ok,
            f"max VIF={max_vif:.3f}; condition number={condition:.3f}",
            profile=profile_name,
            metric={"max_vif": max_vif, "condition_number": condition},
            threshold={
                "max_vif": statistical.max_vif,
                "max_condition_number": statistical.max_condition_number,
            },
            remediation="Remove/recode redundant variables or use regularized/nuisance models.",
            escalate=True,
        )
    )

    # OLS residual diagnostics when treatment/outcome are numeric.
    if (
        outcome
        and treatment
        and outcome in df.columns
        and treatment in df.columns
    ):
        regression_columns = [outcome, treatment, *controls_list]
        work = _numeric_frame(df, regression_columns).dropna()
        if len(work) >= max(10, len(regression_columns) + 3):
            y = work[outcome].to_numpy(dtype=float)
            x = np.column_stack(
                [
                    np.ones(len(work)),
                    work[[treatment, *controls_list]].to_numpy(dtype=float),
                ]
            )
            _, residual, _ = _ols(y, x)
            lm, bp_pvalue, bp_method = _breusch_pagan(residual, x)
            dw = _durbin_watson(residual)
            diagnostics.metrics["heteroskedasticity"] = {
                "method": bp_method,
                "lm": lm,
                "pvalue": bp_pvalue,
            }
            diagnostics.metrics["durbin_watson"] = dw
            hetero_ok = (
                bp_pvalue is None
                or bp_pvalue >= statistical.heteroskedasticity_alpha
            )
            report.add(
                _gate(
                    "heteroskedasticity",
                    hetero_ok,
                    (
                        f"{bp_method}: LM={lm:.3f}, p={bp_pvalue}"
                        "; robust SE recommended regardless."
                    ),
                    profile=profile_name,
                    metric=bp_pvalue if bp_pvalue is not None else lm,
                    threshold=statistical.heteroskedasticity_alpha,
                    remediation="Use HC/cluster/HAC robust standard errors.",
                    warning_only=True,
                )
            )
            autocorrelation_ok = (
                statistical.durbin_watson_min
                <= dw
                <= statistical.durbin_watson_max
            )
            report.add(
                _gate(
                    "autocorrelation",
                    autocorrelation_ok,
                    f"Durbin-Watson={dw:.3f}; row order must represent time for interpretation.",
                    profile=profile_name,
                    metric=dw,
                    threshold=[
                        statistical.durbin_watson_min,
                        statistical.durbin_watson_max,
                    ],
                    remediation="Use time/panel design with HAC or cluster-robust SE.",
                    warning_only=True,
                )
            )
        else:
            report.add(
                GateResult(
                    id="residual_diagnostics",
                    ok=True,
                    status="skip",
                    detail="Insufficient complete rows for residual diagnostics.",
                    stage="statistics",
                )
            )

    # Positivity/overlap for binary treatment.
    if treatment and treatment in df.columns:
        overlap = _overlap(
            df,
            treatment=treatment,
            controls=controls_list,
            epsilon=statistical.propensity_epsilon,
            random_state=rng_seed,
        )
        diagnostics.metrics["overlap"] = overlap
        if overlap.get("ok"):
            fraction = float(overlap.get("overlap_fraction") or 0.0)
            report.add(
                _gate(
                    "positivity_overlap",
                    fraction >= statistical.min_overlap_fraction,
                    (
                        f"propensity overlap fraction={fraction:.3f} "
                        f"within [{statistical.propensity_epsilon:.2f}, "
                        f"{1-statistical.propensity_epsilon:.2f}]"
                    ),
                    profile=profile_name,
                    metric=fraction,
                    threshold=statistical.min_overlap_fraction,
                    evidence=overlap,
                    remediation="Restrict estimand/population, trim under policy, or collect overlap.",
                )
            )
        else:
            report.add(
                GateResult(
                    id="positivity_overlap",
                    ok=True,
                    status="skip",
                    detail=f"Overlap check skipped: {overlap.get('reason')}",
                    stage="statistics",
                    evidence=overlap,
                )
            )

    # Observed IV relevance only. Exclusion remains explicitly unverified.
    if treatment and instrument:
        instruments = (
            [instrument] if isinstance(instrument, str) else list(instrument)
        )
        synthetic = any(
            str(value).startswith("auto_instrument") for value in instruments
        )
        if synthetic:
            first_stage = {
                "ok": False,
                "reason": "synthetic instrument forbidden for evidence",
                "instruments": [str(value) for value in instruments],
            }
        else:
            first_stage = first_stage_diagnostics(
                df,
                treatment=treatment,
                instrument=instruments,
                controls=controls_list,
            )
        diagnostics.metrics["first_stage"] = first_stage
        f_value = first_stage.get("first_stage_f")
        iv_ok = bool(
            first_stage.get("ok")
            and f_value is not None
            and float(f_value) >= policy.causal_evidence.min_first_stage_f  # type: ignore[union-attr]
            and not synthetic
        )
        report.add(
            _gate(
                "weak_instrument",
                iv_ok,
                (
                    f"first-stage F={f_value}; observed={not synthetic}; "
                    "exclusion/independence remain unverified"
                ),
                profile=profile_name,
                stage="causal_evidence",
                metric=f_value,
                threshold=policy.causal_evidence.min_first_stage_f,  # type: ignore[union-attr]
                evidence={
                    **first_stage,
                    "instrument_origin": "synthetic" if synthetic else "observed",
                    "exclusion_verified": False,
                    "independence_verified": False,
                },
                remediation="Collect a stronger observed instrument and domain evidence for exclusion.",
            )
        )

    # Approximate MDE for binary treatment and numeric outcome.
    if treatment and outcome and treatment in df.columns and outcome in df.columns:
        pair = _numeric_frame(df, [treatment, outcome]).dropna()
        unique_treatment = set(pair[treatment].unique())
        if unique_treatment.issubset({0, 1}) and len(unique_treatment) == 2:
            treated = pair.loc[pair[treatment] == 1, outcome]
            control = pair.loc[pair[treatment] == 0, outcome]
            if len(treated) >= 2 and len(control) >= 2:
                pooled_sd = float(
                    np.sqrt(
                        (
                            (len(treated) - 1) * treated.var(ddof=1)
                            + (len(control) - 1) * control.var(ddof=1)
                        )
                        / max(len(pair) - 2, 1)
                    )
                )
                try:
                    from scipy.stats import norm

                    z_alpha = float(norm.ppf(1 - statistical.alpha / 2))
                    z_power = float(norm.ppf(statistical.target_power))
                except Exception:
                    z_alpha, z_power = 1.959964, 0.841621
                mde = (
                    (z_alpha + z_power)
                    * pooled_sd
                    * sqrt(1 / len(treated) + 1 / len(control))
                )
                diagnostics.metrics["minimum_detectable_effect"] = {
                    "mde_outcome_units": float(mde),
                    "pooled_sd": pooled_sd,
                    "standardized_mde": float(mde / pooled_sd)
                    if pooled_sd > 1e-12
                    else None,
                    "target_power": statistical.target_power,
                    "alpha": statistical.alpha,
                    "n_treated": len(treated),
                    "n_control": len(control),
                }
                report.add(
                    _gate(
                        "power_mde",
                        True,
                        f"Approximate MDE={mde:.4g} outcome units.",
                        profile=profile_name,
                        metric=mde,
                        threshold={
                            "power": statistical.target_power,
                            "alpha": statistical.alpha,
                        },
                        evidence=diagnostics.metrics[
                            "minimum_detectable_effect"
                        ],
                        remediation="Compare MDE with the smallest substantively meaningful effect.",
                        warning_only=True,
                    )
                )

    diagnostics.notes.append(
        "Normality/missingness/residual diagnostics are assumption checks, not causal proof."
    )
    return diagnostics
