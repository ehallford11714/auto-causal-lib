"""Unified causal inference orchestration, planning, and policy gates."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from autocausal.__version__ import __version__
from autocausal.inference import native
from autocausal.inference.types import (
    CausalInferenceResult,
    CausalSpec,
    MethodRecommendation,
)
from autocausal.production import (
    EvidenceGateError,
    EvidenceGrade,
    GateReport,
    GateResult,
    ProductionGateError,
    ProductionPolicy,
    RunRecorder,
    build_manifest,
    resolve_mode,
    resolve_policy,
)
from autocausal.statistical_gates import run_statistical_gates

METHOD_SUPPORT: dict[str, dict[str, Any]] = {
    "regression": {
        "status": "native",
        "capability": "effect estimation",
        "dependency": "numpy/pandas",
    },
    "propensity_score": {
        "status": "native",
        "capability": "binary-treatment diagnostics and adjusted estimate",
        "dependency": "scikit-learn",
    },
    "iptw": {
        "status": "native",
        "capability": "stabilized ATE weighting",
        "dependency": "scikit-learn",
    },
    "aipw": {
        "status": "native",
        "capability": "cross-fitted doubly robust ATE",
        "dependency": "scikit-learn",
    },
    "matching": {
        "status": "native",
        "capability": "nearest-neighbor propensity ATT",
        "dependency": "scikit-learn",
    },
    "iv_2sls": {
        "status": "native",
        "capability": "observed-instrument 2SLS",
        "dependency": "numpy/pandas",
    },
    "difference_in_differences": {
        "status": "native",
        "capability": "two-group/two-period-or-panel DiD",
        "dependency": "numpy/pandas",
    },
    "panel_fixed_effects": {
        "status": "native",
        "capability": "unit-within estimator with time effects",
        "dependency": "numpy/pandas",
    },
    "regression_discontinuity": {
        "status": "native",
        "capability": "sharp local-linear RDD",
        "dependency": "numpy/pandas",
    },
    "interrupted_time_series": {
        "status": "native",
        "capability": "segmented regression with HAC covariance",
        "dependency": "numpy/pandas",
    },
    "doubleml": {
        "status": "optional_adapter",
        "capability": "PLR/DML ATE",
        "dependency": "doubleml; optional causal-extra",
    },
    "econml_linear_dml": {
        "status": "optional_adapter",
        "capability": "DML/CATE",
        "dependency": "econml; optional causal-extra",
    },
    "econml_causal_forest": {
        "status": "optional_adapter",
        "capability": "causal forest CATE",
        "dependency": "econml; optional causal-extra",
    },
    "dowhy": {
        "status": "optional_adapter",
        "capability": "identification/refutation (not native point estimation here)",
        "dependency": "dowhy; optional",
    },
    "causal_learn": {
        "status": "optional_adapter",
        "capability": "discovery only",
        "dependency": "causal-learn; optional",
    },
    "lingam": {
        "status": "optional_adapter",
        "capability": "discovery only",
        "dependency": "lingam; optional",
    },
    "gcastle": {
        "status": "optional_adapter",
        "capability": "discovery only",
        "dependency": "gcastle; optional; review license/deployment",
    },
    "tigramite": {
        "status": "planned_deferred",
        "capability": "temporal discovery",
        "dependency": "not wired; review GPL licensing",
    },
    "granger": {
        "status": "planned_deferred",
        "capability": "predictive temporal precedence, not causal effect",
        "dependency": "not wired",
    },
    "mediation": {
        "status": "planned_deferred",
        "capability": "mediation effects",
        "dependency": "no maintained validated adapter selected",
    },
    "synthetic_control": {
        "status": "planned_deferred",
        "capability": "comparative case study",
        "dependency": "no validated native implementation",
    },
    "tmle": {
        "status": "planned_deferred",
        "capability": "targeted learning",
        "dependency": "no maintained validated adapter selected",
    },
    "front_door": {
        "status": "planned_deferred",
        "capability": "front-door identification",
        "dependency": "design validation not implemented",
    },
    "proximal_inference": {
        "status": "planned_deferred",
        "capability": "proxy-variable causal inference",
        "dependency": "specialized bridge assumptions not implemented",
    },
    "survival_causal_inference": {
        "status": "planned_deferred",
        "capability": "time-to-event causal estimands",
        "dependency": "censoring/risk-set support not implemented",
    },
    "causalml_uplift": {
        "status": "planned_deferred",
        "capability": "uplift/CATE",
        "dependency": "intentionally not added; stale-package risk",
    },
}

ALIASES = {
    "ols": "regression",
    "regression_adjustment": "regression",
    "ps": "propensity_score",
    "propensity": "propensity_score",
    "ipw": "iptw",
    "doubly_robust": "aipw",
    "psm": "matching",
    "2sls": "iv_2sls",
    "iv": "iv_2sls",
    "did": "difference_in_differences",
    "fixed_effects": "panel_fixed_effects",
    "within": "panel_fixed_effects",
    "rdd": "regression_discontinuity",
    "rd": "regression_discontinuity",
    "its": "interrupted_time_series",
    "linear_dml": "econml_linear_dml",
    "causal_forest": "econml_causal_forest",
}


def method_support_matrix() -> list[dict[str, Any]]:
    return [
        {"method": method, **metadata}
        for method, metadata in METHOD_SUPPORT.items()
    ]


class AutoInferencePlanner:
    """Recommend candidate designs; never silently execute one."""

    def __init__(
        self,
        spec: CausalSpec,
        *,
        policy: Optional[ProductionPolicy | dict[str, Any]] = None,
        mode: str = "exploratory",
    ) -> None:
        self.spec = spec
        self.mode = resolve_mode(mode)
        self.policy = resolve_policy(self.mode, policy)

    def recommend(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> list[MethodRecommendation]:
        candidates: list[tuple[str, str, list[str], list[str]]] = []
        if self.spec.instruments:
            candidates.append(
                (
                    "iv_2sls",
                    "Observed instrument metadata supplied; relevance and domain validity still require gates.",
                    ["instrument"],
                    [
                        "relevance",
                        "exclusion",
                        "independence",
                        "monotonicity for LATE",
                    ],
                )
            )
        if self.spec.running is not None or self.spec.cutoff is not None:
            candidates.append(
                (
                    "regression_discontinuity",
                    "Running variable/cutoff metadata suggests a local discontinuity design.",
                    ["running", "cutoff"],
                    ["continuity", "no sorting", "bandwidth stability"],
                )
            )
        if self.spec.unit and self.spec.time and self.spec.post:
            candidates.append(
                (
                    "difference_in_differences",
                    "Unit, time, and post metadata support evaluating a DiD design.",
                    ["unit", "time", "post"],
                    ["parallel trends", "no anticipation", "no spillovers"],
                )
            )
        if self.spec.unit and self.spec.time:
            candidates.append(
                (
                    "panel_fixed_effects",
                    "Repeated unit/time metadata supports a within-unit model.",
                    ["unit", "time"],
                    ["strict exogeneity", "no time-varying unmeasured confounding"],
                )
            )
        if self.spec.time and self.spec.post and not self.spec.unit:
            candidates.append(
                (
                    "interrupted_time_series",
                    "Ordered time and intervention metadata support segmented regression.",
                    ["time", "post"],
                    ["stable counterfactual trend", "no coincident intervention"],
                )
            )

        binary_treatment = False
        if df is not None and self.spec.treatment in df.columns:
            values = set(df[self.spec.treatment].dropna().unique())
            binary_treatment = values.issubset(
                {self.spec.control_value, self.spec.treatment_value}
            ) and len(values) == 2
        if binary_treatment:
            candidates.extend(
                [
                    (
                        "aipw",
                        "Binary treatment with measured confounders; cross-fitted doubly robust ATE is a candidate.",
                        ["confounders"],
                        ["exchangeability", "positivity", "nuisance-model adequacy"],
                    ),
                    (
                        "iptw",
                        "Binary treatment permits stabilized propensity weighting with explicit overlap gates.",
                        ["confounders"],
                        ["exchangeability", "propensity model", "positivity"],
                    ),
                    (
                        "matching",
                        "Binary treatment permits propensity matching as an ATT sensitivity analysis.",
                        ["confounders"],
                        ["exchangeability", "common support", "match quality"],
                    ),
                ]
            )
        candidates.append(
            (
                "regression",
                "Regression adjustment is a transparent baseline, not automatic identification.",
                ["confounders"],
                ["exchangeability", "functional form", "positivity"],
            )
        )

        recommendations: list[MethodRecommendation] = []
        for rank, (method, rationale, fields, assumptions) in enumerate(
            candidates, start=1
        ):
            missing = [
                field
                for field in fields
                if field == "confounders" and not self.spec.confounders
                or field == "instrument" and not self.spec.instruments
                or field not in ("confounders", "instrument")
                and getattr(self.spec, field, None) is None
            ]
            recommendations.append(
                MethodRecommendation(
                    method=method,
                    rank=rank,
                    rationale=rationale,
                    required_fields=fields,
                    missing_fields=missing,
                    status="candidate" if not missing else "defer",
                    assumptions=assumptions,
                )
            )
        return recommendations


def _gate_status(ok: bool, profile: str, *, warning_only: bool = False) -> str:
    if ok:
        return "pass"
    if warning_only or profile == "exploratory":
        return "warn"
    if profile == "review":
        return "escalate"
    return "fail"


def _gate(
    gate_id: str,
    ok: bool,
    detail: str,
    *,
    profile: str,
    metric: Any = None,
    threshold: Any = None,
    remediation: Optional[str] = None,
    evidence: Optional[dict[str, Any]] = None,
    warning_only: bool = False,
    stage: str = "causal_inference",
) -> GateResult:
    status = _gate_status(ok, profile, warning_only=warning_only)
    return GateResult(
        id=gate_id,
        ok=status in ("pass", "warn", "skip"),
        status=status,
        detail=detail,
        metric=metric,
        threshold=threshold,
        remediation=remediation,
        evidence=dict(evidence or {}),
        stage=stage,
    )


def _validate_design(
    df: pd.DataFrame,
    spec: CausalSpec,
    method: str,
    policy: ProductionPolicy,
) -> GateReport:
    profile = policy.profile
    report = GateReport(profile=profile, policy_version=policy.policy_version)
    roles_ok = bool(spec.treatment and spec.outcome)
    report.add(
        _gate(
            "explicit_treatment_outcome",
            roles_ok,
            "Treatment and outcome are explicit in CausalSpec.",
            profile=profile,
            remediation="Provide CausalSpec(treatment=..., outcome=...).",
        )
    )
    required = spec.required_columns(method)
    missing = [column for column in required if column not in df.columns]
    report.add(
        _gate(
            "required_design_columns",
            not missing,
            "All required design columns are present."
            if not missing
            else f"Missing required columns: {missing}",
            profile=profile,
            metric=missing,
            threshold=[],
            remediation="Supply/rename the explicit design fields before estimation.",
        )
    )
    leakage = [
        column
        for column in spec.confounders
        if column in {spec.treatment, spec.outcome, spec.post, spec.running}
    ]
    report.add(
        _gate(
            "design_leakage",
            not leakage,
            "No treatment/outcome/design field duplicated among confounders."
            if not leakage
            else f"Leaking/conflicting confounders: {leakage}",
            profile=profile,
            metric=leakage,
            threshold=[],
            remediation="Remove outcomes, treatments, and post-treatment variables from confounders.",
        )
    )
    unconfounded_methods = {
        "regression",
        "propensity_score",
        "iptw",
        "aipw",
        "matching",
    }
    confounder_ok = bool(spec.confounders) or method not in unconfounded_methods
    report.add(
        _gate(
            "confounder_readiness",
            confounder_ok,
            "Measured confounders documented."
            if confounder_ok
            else "No confounders supplied for an exchangeability-based method.",
            profile=profile,
            remediation="Collect and specify pre-treatment common causes; seek domain review.",
            warning_only=not bool(
                policy.causal_evidence
                and policy.causal_evidence.require_confounders
            ),
        )
    )
    sutva_acknowledged = bool(
        spec.assumptions.get("sutva")
        or spec.assumptions.get("no_interference")
    )
    report.add(
        _gate(
            "sutva_interference",
            sutva_acknowledged,
            (
                "SUTVA/no-interference explicitly acknowledged."
                if sutva_acknowledged
                else "SUTVA/interference is not testable here and remains a caveat."
            ),
            profile=profile,
            evidence={"domain_acknowledgement": sutva_acknowledged},
            remediation="Document treatment versions and plausible spillovers.",
            warning_only=True,
        )
    )
    if method == "iv_2sls":
        origin_ok = (
            bool(spec.instruments)
            and spec.instrument_provenance == "observed"
            and not any(value.startswith("auto_instrument") for value in spec.instruments)
        )
        report.add(
            _gate(
                "observed_instrument_provenance",
                origin_ok,
                f"Instrument provenance is `{spec.instrument_provenance}`.",
                profile=profile,
                evidence={
                    "instrument_count": len(spec.instruments),
                    "origin": spec.instrument_provenance,
                },
                remediation="Provide a real observed instrument and document its origin.",
            )
        )
    method_fields: dict[str, list[str]] = {
        "difference_in_differences": ["unit", "time", "post"],
        "panel_fixed_effects": ["unit", "time"],
        "regression_discontinuity": ["running", "cutoff"],
        "interrupted_time_series": ["time", "post"],
    }
    if method in method_fields:
        absent = [
            field
            for field in method_fields[method]
            if getattr(spec, field, None) is None
        ]
        report.add(
            _gate(
                f"{method}_design_metadata",
                not absent,
                "Required design metadata supplied."
                if not absent
                else f"Missing design metadata: {absent}",
                profile=profile,
                metric=absent,
                threshold=[],
                remediation=f"Provide explicit {', '.join(method_fields[method])}.",
            )
        )
    return report


def _post_estimation_gates(
    raw: dict[str, Any],
    *,
    method: str,
    policy: ProductionPolicy,
) -> GateReport:
    profile = policy.profile
    causal = policy.causal_evidence
    statistical = policy.statistical_validity
    assert causal is not None and statistical is not None
    diagnostics = dict(raw.get("diagnostics") or {})
    report = GateReport(profile=profile, policy_version=policy.policy_version)
    estimate = raw.get("estimate")
    stable_estimate = estimate is not None and np.isfinite(float(estimate))
    report.add(
        _gate(
            "finite_effect_estimate",
            stable_estimate,
            "Effect estimate is finite."
            if stable_estimate
            else "Effect estimate is missing or non-finite.",
            profile=profile,
            metric=estimate,
            remediation="Inspect design matrix, sample sufficiency, and estimator diagnostics.",
        )
    )
    complete_fraction = float(
        (raw.get("sample") or {}).get("complete_fraction", 1.0)
    )
    missing_ok = complete_fraction >= 1.0 - policy.data_quality.max_missing_fraction  # type: ignore[union-attr]
    report.add(
        _gate(
            "complete_case_fraction",
            missing_ok,
            f"Complete-case fraction={complete_fraction:.3f}.",
            profile=profile,
            metric=complete_fraction,
            threshold=1.0 - policy.data_quality.max_missing_fraction,  # type: ignore[union-attr]
            remediation="Use policy-audited train/design-aware imputation or collect missing data.",
        )
    )
    if method in ("propensity_score", "iptw", "aipw"):
        overlap = float(diagnostics.get("overlap_fraction", 0.0))
        report.add(
            _gate(
                "estimator_positivity",
                overlap >= statistical.min_overlap_fraction,
                f"Estimator overlap fraction={overlap:.3f}.",
                profile=profile,
                metric=overlap,
                threshold=statistical.min_overlap_fraction,
                remediation="Restrict population/estimand or collect overlap; do not extrapolate.",
            )
        )
    if method in ("iptw", "matching"):
        balance = float(diagnostics.get("max_abs_smd_after", float("inf")))
        report.add(
            _gate(
                "post_adjustment_balance",
                balance <= causal.max_abs_standardized_mean_difference,
                f"Maximum post-adjustment |SMD|={balance:.3f}.",
                profile=profile,
                metric=balance,
                threshold=causal.max_abs_standardized_mean_difference,
                remediation="Revise propensity model/matching or defer the estimate.",
            )
        )
    if method == "iptw":
        maximum_weight = float(
            diagnostics.get("weight_max_before_policy", float("inf"))
        )
        report.add(
            _gate(
                "extreme_propensity_weights",
                maximum_weight <= causal.max_propensity_weight,
                f"Maximum pre-policy stabilized weight={maximum_weight:.3f}.",
                profile=profile,
                metric=maximum_weight,
                threshold=causal.max_propensity_weight,
                remediation="Investigate positivity; report sensitivity to trimming/capping.",
            )
        )
    if method == "iv_2sls":
        f_value = diagnostics.get("first_stage_f")
        strong = f_value is not None and float(f_value) >= causal.min_first_stage_f
        report.add(
            _gate(
                "first_stage_relevance",
                strong,
                (
                    f"First-stage partial F={f_value}; exclusion and independence "
                    "remain unverified."
                ),
                profile=profile,
                metric=f_value,
                threshold=causal.min_first_stage_f,
                remediation="Collect a stronger observed instrument or use weak-IV-robust methods.",
            )
        )
        over_id = diagnostics.get("overidentification")
        if isinstance(over_id, dict) and over_id.get("p_value") is not None:
            over_ok = float(over_id["p_value"]) >= statistical.alpha
            report.add(
                _gate(
                    "overidentification_diagnostic",
                    over_ok,
                    (
                        f"Sargan p={over_id['p_value']:.4g}; non-rejection does "
                        "not prove instrument validity."
                    ),
                    profile=profile,
                    metric=over_id["p_value"],
                    threshold=statistical.alpha,
                    remediation="Review each exclusion restriction with domain experts.",
                    warning_only=True,
                )
            )
    if method == "difference_in_differences":
        pretrend = diagnostics.get("pretrend") or {}
        pre_periods = int(diagnostics.get("pre_periods") or 0)
        enough_periods = pre_periods >= causal.min_pre_periods
        report.add(
            _gate(
                "did_preperiods",
                enough_periods,
                f"Pre-period count={pre_periods}.",
                profile=profile,
                metric=pre_periods,
                threshold=causal.min_pre_periods,
                remediation="Collect more pre-periods or defer DiD.",
            )
        )
        pvalue = pretrend.get("p_value")
        no_detected_pretrend = pvalue is not None and float(pvalue) >= statistical.alpha
        report.add(
            _gate(
                "parallel_trends_diagnostic",
                no_detected_pretrend,
                (
                    f"Differential pretrend p={pvalue}; parallel trends is never proven."
                ),
                profile=profile,
                metric=pvalue,
                threshold=statistical.alpha,
                evidence={"parallel_trends_verified": False},
                remediation="Inspect event-study plots, alternative controls, and domain evidence.",
                warning_only=not causal.require_parallel_trends_review,
            )
        )
    if method == "regression_discontinuity":
        minimum_side = min(
            int(diagnostics.get("n_left") or 0),
            int(diagnostics.get("n_right") or 0),
        )
        report.add(
            _gate(
                "rdd_bandwidth_support",
                minimum_side >= causal.min_rdd_side,
                f"Smaller cutoff side contains n={minimum_side}.",
                profile=profile,
                metric=minimum_side,
                threshold=causal.min_rdd_side,
                remediation="Increase sample/support or reconsider bandwidth/design.",
            )
        )
        assignment = float(diagnostics.get("assignment_agreement") or 0.0)
        report.add(
            _gate(
                "rdd_assignment_continuity",
                assignment >= 0.95,
                f"Sharp-assignment agreement={assignment:.3f}.",
                profile=profile,
                metric=assignment,
                threshold=0.95,
                remediation="Use a fuzzy-RDD estimator (deferred) or correct assignment metadata.",
            )
        )
    if method == "interrupted_time_series":
        lag1 = diagnostics.get("residual_lag1_correlation")
        autocorrelation_ok = lag1 is None or abs(float(lag1)) <= 0.5
        report.add(
            _gate(
                "its_residual_autocorrelation",
                autocorrelation_ok,
                f"Residual lag-1 correlation={lag1}; HAC covariance used.",
                profile=profile,
                metric=lag1,
                threshold="|rho1|<=0.5",
                remediation="Model serial structure or revise segmented trend.",
                warning_only=True,
            )
        )
    return report


def _merge_reports(
    policy: ProductionPolicy,
    *reports: GateReport,
) -> GateReport:
    merged = GateReport(
        profile=policy.profile,
        policy_version=policy.policy_version,
    )
    for report in reports:
        merged.extend(report.results)
        merged.notes.extend(report.notes)
    return merged


def _adapter_result(
    df: pd.DataFrame,
    spec: CausalSpec,
    method: str,
    *,
    random_state: int,
) -> dict[str, Any]:
    if method == "doubleml":
        from autocausal.backends import doubleml_backend

        output = doubleml_backend.estimate(
            df,
            y=spec.outcome,
            d=spec.treatment,
            x=spec.confounders,
            random_state=random_state,
        )
    elif method in ("econml_linear_dml", "econml_causal_forest"):
        from autocausal.backends import econml_backend

        output = econml_backend.estimate(
            df,
            y=spec.outcome,
            d=spec.treatment,
            x=spec.confounders,
            method=method,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"{method!r} is not an effect-estimation adapter. "
            "See method_support_matrix() for its capability."
        )
    estimate_payload = output.get("estimate") or {}
    estimate = estimate_payload.get("ate")
    standard_error = estimate_payload.get("se")
    if standard_error is None:
        standard_error = estimate_payload.get("ate_se_heterogeneity")
    pvalue = estimate_payload.get("pvalue")
    if estimate is not None and standard_error is not None:
        low = float(estimate) - 1.959964 * float(standard_error)
        high = float(estimate) + 1.959964 * float(standard_error)
    else:
        low = high = None
    return {
        "ok": bool(output.get("ok")),
        "soft_skip": bool(output.get("soft_skip")),
        "error": output.get("error"),
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": low,
        "ci_high": high,
        "p_value": pvalue,
        "n": int(estimate_payload.get("n") or 0),
        "sample": {
            "n_input": len(df),
            "n_complete": int(estimate_payload.get("n") or 0),
        },
        "diagnostics": {
            "backend": output.get("backend"),
            "adapter_data": output.get("data") or {},
        },
        "assumptions": list(output.get("notes") or []),
        "warnings": [output.get("error")] if output.get("error") else [],
    }


class AutoInference:
    """Fit an explicit causal design under a composable run policy."""

    def __init__(
        self,
        spec: CausalSpec,
        *,
        policy: Optional[ProductionPolicy | dict[str, Any]] = None,
        mode: str = "exploratory",
        random_state: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> None:
        if not isinstance(spec, CausalSpec):
            raise TypeError("AutoInference requires a CausalSpec.")
        self.spec = spec
        self.mode = resolve_mode(mode)
        self.policy = resolve_policy(
            self.mode,
            policy,
            random_state=random_state,
        )
        self.random_state = self.policy.random_state
        self.run_id = run_id
        self.last_result: Optional[CausalInferenceResult] = None

    def planner(self) -> AutoInferencePlanner:
        return AutoInferencePlanner(
            self.spec,
            policy=self.policy,
            mode=self.mode,
        )

    def fit(
        self,
        df: pd.DataFrame,
        *,
        method: str = "aipw",
        fail_closed: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalInferenceResult:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("AutoInference.fit expects a pandas DataFrame.")
        normalized = ALIASES.get(method.lower().strip(), method.lower().strip())
        if normalized == "auto":
            recommendations = self.planner().recommend(df)
            if self.mode == "production":
                raise ProductionGateError(
                    "Production inference requires an explicit method; use "
                    "AutoInferencePlanner.recommend() for reviewed candidates.",
                    code="automatic_method_selection_forbidden",
                    recommendations=[
                        f"Review candidate `{value.method}`: {value.rationale}"
                        for value in recommendations[:5]
                    ],
                )
            viable = [value for value in recommendations if value.status == "candidate"]
            normalized = viable[0].method if viable else "regression"
        if normalized not in METHOD_SUPPORT:
            raise ValueError(
                f"Unknown inference method {method!r}. "
                f"Supported catalog: {sorted(METHOD_SUPPORT)}"
            )
        support = METHOD_SUPPORT[normalized]
        if support["status"] == "planned_deferred":
            raise NotImplementedError(
                f"{normalized} is planned/deferred, not implemented. "
                f"Reason: {support['dependency']}"
            )
        if normalized in ("dowhy", "causal_learn", "lingam", "gcastle"):
            raise ValueError(
                f"{normalized} provides {support['capability']}, not a unified "
                "native effect estimate. Use discovery/refute APIs."
            )

        manifest = build_manifest(
            df,
            mode=self.mode,
            policy=self.policy,
            run_id=self.run_id,
            config={
                "inference": {
                    "method_requested": method,
                    "method_selected": normalized,
                    "spec": self.spec.to_dict(),
                    "fallback": (
                        None if normalized == method.lower().strip() else normalized
                    ),
                }
            },
        )
        recorder = RunRecorder(manifest)
        design_report = _validate_design(
            df, self.spec, normalized, self.policy
        )
        manifest.gates.extend(design_report.results)
        hard_fail = self.mode == "production" if fail_closed is None else fail_closed
        if hard_fail and design_report.failed:
            manifest.finish("blocked")
            raise ProductionGateError(
                "Causal design metadata failed production gates.",
                code="causal_design_gate_failed",
                gates=design_report.failed,
                recommendations=[
                    gate.remediation or "Review the causal design."
                    for gate in design_report.failed
                ],
                manifest=manifest,
            )

        with recorder.span(
            "statistical_gates",
            method=normalized,
            n_rows=len(df),
            n_columns=len(df.columns),
        ):
            statistical = run_statistical_gates(
                df,
                treatment=self.spec.treatment,
                outcome=self.spec.outcome,
                instrument=self.spec.instruments or None,
                controls=self.spec.confounders,
                policy=self.policy,
                random_state=self.random_state,
            )
        manifest.gates.extend(statistical.gates.results)
        if hard_fail and statistical.gates.failed:
            manifest.finish("blocked")
            raise EvidenceGateError(
                "Statistical/design assumptions failed production gates.",
                code="statistical_validity_gate_failed",
                gates=statistical.gates.failed,
                recommendations=[
                    gate.remediation or "Review statistical assumptions."
                    for gate in statistical.gates.failed
                ],
                manifest=manifest,
            )

        recorder.check_deadline(self.policy.max_seconds)
        with recorder.span(
            "causal_estimation",
            method=normalized,
            support=support["status"],
        ):
            raw = self._fit_native_or_adapter(
                df,
                normalized,
                **kwargs,
            )
        recorder.check_deadline(self.policy.max_seconds)

        if raw.get("soft_skip"):
            gate = _gate(
                f"optional_adapter:{normalized}",
                False,
                f"Optional adapter `{normalized}` was unavailable or deferred.",
                profile=self.policy.profile,
                evidence={"dependency": support["dependency"]},
                remediation=f"Install/validate {support['dependency']}.",
            )
            adapter_report = GateReport(
                profile=self.policy.profile,
                policy_version=self.policy.policy_version,
                results=[gate],
            )
            merged = _merge_reports(
                self.policy, design_report, statistical.gates, adapter_report
            )
            manifest.gates.append(gate)
            manifest.finish("soft_skip")
            result = self._build_result(
                normalized,
                raw,
                merged,
                manifest,
                evidence_grade=EvidenceGrade.INSUFFICIENT.value,
            )
            self.last_result = result
            if hard_fail:
                raise ProductionGateError(
                    f"Required inference adapter `{normalized}` did not run.",
                    code="inference_adapter_unavailable",
                    gates=[gate],
                    partial_result=result,
                    manifest=manifest,
                )
            return result
        if not raw.get("ok", True):
            manifest.finish("error")
            raise ProductionGateError(
                f"Inference method `{normalized}` failed: {raw.get('error')}",
                code="inference_estimator_failed",
                manifest=manifest,
            )

        post_report = _post_estimation_gates(
            raw,
            method=normalized,
            policy=self.policy,
        )
        # RDD bandwidth stability is estimated after the primary fit.
        if normalized == "regression_discontinuity":
            self._add_rdd_stability_gate(df, raw, post_report)
        manifest.gates.extend(post_report.results)
        merged = _merge_reports(
            self.policy,
            design_report,
            statistical.gates,
            post_report,
        )
        grade = (
            EvidenceGrade.SUPPORTED.value
            if not merged.failed
            else EvidenceGrade.INSUFFICIENT.value
        )
        if self.mode == "exploratory" and grade == EvidenceGrade.SUPPORTED.value:
            grade = EvidenceGrade.EXPLORATORY.value
        manifest.finish("ok" if not merged.failed else "blocked")
        result = self._build_result(
            normalized,
            raw,
            merged,
            manifest,
            evidence_grade=grade,
        )
        self.last_result = result
        if hard_fail and post_report.failed:
            raise EvidenceGateError(
                f"`{normalized}` estimate failed post-estimation evidence gates.",
                code="causal_evidence_gate_failed",
                gates=post_report.failed,
                recommendations=[
                    gate.remediation or "Escalate for design review."
                    for gate in post_report.failed
                ],
                partial_result=result,
                manifest=manifest,
            )
        return result

    def _fit_native_or_adapter(
        self,
        df: pd.DataFrame,
        method: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        statistical = self.policy.statistical_validity
        causal = self.policy.causal_evidence
        assert statistical is not None and causal is not None
        alpha = statistical.alpha
        if method == "regression":
            return {
                "ok": True,
                **native.regression_adjustment(df, self.spec, alpha=alpha),
            }
        if method == "propensity_score":
            return {
                "ok": True,
                **native.propensity_score_adjustment(
                    df,
                    self.spec,
                    epsilon=statistical.propensity_epsilon,
                    random_state=self.random_state,
                    alpha=alpha,
                ),
            }
        if method == "iptw":
            return {
                "ok": True,
                **native.iptw(
                    df,
                    self.spec,
                    epsilon=statistical.propensity_epsilon,
                    trim_quantile=float(
                        kwargs.get(
                            "trim_quantile",
                            causal.propensity_trim_quantile,
                        )
                    ),
                    max_weight=float(
                        kwargs.get("max_weight", causal.max_propensity_weight)
                    ),
                    random_state=self.random_state,
                    alpha=alpha,
                ),
            }
        if method == "aipw":
            return {
                "ok": True,
                **native.aipw(
                    df,
                    self.spec,
                    epsilon=statistical.propensity_epsilon,
                    folds=int(kwargs.get("folds", causal.crossfit_folds)),
                    random_state=self.random_state,
                    alpha=alpha,
                ),
            }
        if method == "matching":
            return {
                "ok": True,
                **native.propensity_matching(
                    df,
                    self.spec,
                    epsilon=statistical.propensity_epsilon,
                    caliper=kwargs.get("caliper"),
                    random_state=self.random_state,
                    alpha=alpha,
                ),
            }
        if method == "iv_2sls":
            return {
                "ok": True,
                **native.iv_2sls(df, self.spec, alpha=alpha),
            }
        if method == "difference_in_differences":
            return {
                "ok": True,
                **native.difference_in_differences(
                    df,
                    self.spec,
                    alpha=alpha,
                    min_pre_periods=causal.min_pre_periods,
                ),
            }
        if method == "panel_fixed_effects":
            return {
                "ok": True,
                **native.panel_fixed_effects(df, self.spec, alpha=alpha),
            }
        if method == "regression_discontinuity":
            return {
                "ok": True,
                **native.regression_discontinuity(
                    df,
                    self.spec,
                    alpha=alpha,
                    min_side=causal.min_rdd_side,
                ),
            }
        if method == "interrupted_time_series":
            return {
                "ok": True,
                **native.interrupted_time_series(
                    df,
                    self.spec,
                    alpha=alpha,
                    hac_lags=int(kwargs.get("hac_lags", causal.hac_max_lags)),
                ),
            }
        return _adapter_result(
            df,
            self.spec,
            method,
            random_state=self.random_state,
        )

    def _add_rdd_stability_gate(
        self,
        df: pd.DataFrame,
        raw: dict[str, Any],
        report: GateReport,
    ) -> None:
        diagnostics = raw.get("diagnostics") or {}
        bandwidth = diagnostics.get("bandwidth")
        estimate = raw.get("estimate")
        if bandwidth is None or estimate is None:
            return
        alternatives: dict[str, Optional[float]] = {}
        for factor in (0.75, 1.25):
            alternative_spec = replace(
                self.spec, bandwidth=float(bandwidth) * factor
            )
            try:
                fitted = native.regression_discontinuity(
                    df,
                    alternative_spec,
                    alpha=self.policy.statistical_validity.alpha,  # type: ignore[union-attr]
                    min_side=self.policy.causal_evidence.min_rdd_side,  # type: ignore[union-attr]
                )
                alternatives[str(factor)] = fitted.get("estimate")
            except Exception:
                alternatives[str(factor)] = None
        valid = [
            float(value)
            for value in alternatives.values()
            if value is not None and np.isfinite(float(value))
        ]
        denominator = max(abs(float(estimate)), 1e-8)
        disagreement = (
            max(abs(value - float(estimate)) for value in valid) / denominator
            if valid
            else float("inf")
        )
        threshold = self.policy.statistical_validity.max_engine_disagreement  # type: ignore[union-attr]
        report.add(
            _gate(
                "rdd_bandwidth_stability",
                disagreement <= threshold,
                f"Relative estimate change across 0.75x/1.25x bandwidth={disagreement:.3f}.",
                profile=self.policy.profile,
                metric=disagreement,
                threshold=threshold,
                evidence={"alternative_estimates": alternatives},
                remediation="Report bandwidth sensitivity or defer unstable RDD.",
            )
        )
        diagnostics["bandwidth_sensitivity"] = {
            "base": estimate,
            "alternatives": alternatives,
            "relative_max_change": disagreement,
        }

    def _build_result(
        self,
        method: str,
        raw: dict[str, Any],
        gates: GateReport,
        manifest: Any,
        *,
        evidence_grade: str,
    ) -> CausalInferenceResult:
        return CausalInferenceResult(
            method=method,
            estimand=str(raw.get("estimand") or self.spec.estimand),
            estimate=(
                float(raw["estimate"])
                if raw.get("estimate") is not None
                else None
            ),
            standard_error=(
                float(raw["standard_error"])
                if raw.get("standard_error") is not None
                else None
            ),
            ci_low=(
                float(raw["ci_low"]) if raw.get("ci_low") is not None else None
            ),
            ci_high=(
                float(raw["ci_high"])
                if raw.get("ci_high") is not None
                else None
            ),
            p_value=(
                float(raw["p_value"])
                if raw.get("p_value") is not None
                else None
            ),
            n=int(raw.get("n") or 0),
            treatment=self.spec.treatment,
            outcome=self.spec.outcome,
            controls=list(self.spec.confounders),
            sample_used=dict(raw.get("sample") or {}),
            assumptions=list(raw.get("assumptions") or []),
            diagnostics=dict(raw.get("diagnostics") or {}),
            provenance={
                "run_id": manifest.run_id,
                "package_version": __version__,
                "method_support": METHOD_SUPPORT[method]["status"],
                "method_requested": manifest.config["inference"][
                    "method_requested"
                ],
                "method_selected": method,
                "instrument_origin": self.spec.instrument_provenance
                if self.spec.instruments
                else None,
                "random_state": self.random_state,
                "policy_profile": self.policy.profile,
                "policy_version": self.policy.policy_version,
                "policy_overrides": [
                    value.to_dict() for value in self.policy.overrides
                ],
                "correlation_used_as_identification": False,
            },
            evidence_grade=evidence_grade,
            gates=gates,
            warnings=[
                value
                for value in list(raw.get("warnings") or [])
                if value is not None
            ],
            manifest=manifest,
            ok=bool(raw.get("ok", True)),
            soft_skip=bool(raw.get("soft_skip")),
            error=raw.get("error"),
        )
