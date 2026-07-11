"""Native causal estimators used by :mod:`autocausal.inference`.

The estimators are intentionally compact and transparent. They provide
robust uncertainty where practical, but their validity still depends on the
design assumptions surfaced by the gate layer.
"""

from __future__ import annotations

from math import erf, sqrt
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from autocausal.inference.types import CausalSpec
from autocausal.statistical_gates import first_stage_diagnostics


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _normal_p(value: float) -> float:
    return float(2.0 * (1.0 - _normal_cdf(abs(float(value)))))


def _effect_summary(
    estimate: float,
    standard_error: Optional[float],
    *,
    alpha: float = 0.05,
) -> dict[str, Optional[float]]:
    if standard_error is None or not np.isfinite(standard_error):
        return {
            "estimate": float(estimate),
            "standard_error": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
        }
    try:
        from scipy.stats import norm

        critical = float(norm.ppf(1.0 - alpha / 2.0))
    except Exception:
        critical = 1.959963984540054
    statistic = estimate / standard_error if standard_error > 0 else float("inf")
    return {
        "estimate": float(estimate),
        "standard_error": float(standard_error),
        "ci_low": float(estimate - critical * standard_error),
        "ci_high": float(estimate + critical * standard_error),
        "p_value": _normal_p(statistic),
    }


def _encode_controls(
    frame: pd.DataFrame,
    controls: Sequence[str],
    *,
    include_time: Optional[str] = None,
) -> tuple[np.ndarray, list[str]]:
    selected = frame[list(controls)].copy() if controls else pd.DataFrame(index=frame.index)
    if include_time and include_time in frame.columns:
        time_values = frame[include_time]
        if time_values.nunique() <= min(50, max(2, len(frame) // 4)):
            selected[f"{include_time}__fixed_effect"] = time_values.astype(str)
    if selected.empty:
        return np.empty((len(frame), 0), dtype=float), []
    encoded = pd.get_dummies(
        selected,
        drop_first=True,
        dummy_na=False,
        dtype=float,
    )
    for column in encoded.columns:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce")
    if encoded.isna().any().any():
        raise ValueError(
            "Controls contain missing/non-numeric values after encoding; "
            "clean or impute under policy before inference."
        )
    return encoded.to_numpy(dtype=float), [str(value) for value in encoded.columns]


def _binary_treatment(
    series: pd.Series,
    *,
    treated_value: Any,
    control_value: Any,
) -> np.ndarray:
    valid = series.isin([treated_value, control_value])
    if not bool(valid.all()):
        unexpected = int((~valid).sum())
        raise ValueError(
            f"Binary-treatment method found {unexpected} values outside "
            f"control={control_value!r}, treated={treated_value!r}."
        )
    values = (series == treated_value).astype(float).to_numpy()
    if np.unique(values).size != 2:
        raise ValueError("Both treatment and control groups are required.")
    return values


def _complete_frame(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    unique = list(dict.fromkeys(str(value) for value in columns if value))
    missing_columns = [column for column in unique if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns not found: {missing_columns}")
    before = len(df)
    frame = df[unique].dropna().copy()
    return frame, {
        "n_input": before,
        "n_complete": len(frame),
        "n_dropped_missing": before - len(frame),
        "complete_fraction": len(frame) / before if before else 0.0,
    }


def _fit_linear(
    y: np.ndarray,
    x: np.ndarray,
    *,
    covariance: str = "HC1",
    clusters: Optional[np.ndarray] = None,
    hac_lags: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n, k = x.shape
    if n <= k:
        raise ValueError(f"Linear model requires n>parameters; n={n}, k={k}.")
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    covariance_name = covariance
    if clusters is not None:
        unique_clusters = pd.unique(clusters)
        if len(unique_clusters) < 2:
            raise ValueError("Cluster-robust covariance requires at least two clusters.")
        meat = np.zeros((k, k), dtype=float)
        for cluster in unique_clusters:
            mask = clusters == cluster
            score = x[mask].T @ residual[mask]
            meat += np.outer(score, score)
        correction = (
            len(unique_clusters)
            / max(len(unique_clusters) - 1, 1)
            * (n - 1)
            / max(n - k, 1)
        )
        variance = correction * xtx_inv @ meat @ xtx_inv
        covariance_name = "cluster"
    elif hac_lags > 0:
        score = x * residual[:, None]
        meat = score.T @ score
        maximum_lag = min(int(hac_lags), n - 1)
        for lag in range(1, maximum_lag + 1):
            weight = 1.0 - lag / (maximum_lag + 1.0)
            cross = score[lag:].T @ score[:-lag]
            meat += weight * (cross + cross.T)
        variance = xtx_inv @ meat @ xtx_inv
        covariance_name = f"HAC({maximum_lag})"
    elif covariance.upper() in ("HC0", "HC1"):
        score = x * residual[:, None]
        meat = score.T @ score
        correction = n / max(n - k, 1) if covariance.upper() == "HC1" else 1.0
        variance = correction * xtx_inv @ meat @ xtx_inv
        covariance_name = covariance.upper()
    else:
        sigma2 = float(residual @ residual) / max(n - k, 1)
        variance = sigma2 * xtx_inv
        covariance_name = "classical"
    standard_error = np.sqrt(np.clip(np.diag(variance), 0.0, None))
    return beta, standard_error, residual, {
        "covariance": covariance_name,
        "n_parameters": k,
        "residual_degrees_of_freedom": n - k,
        "condition_number": float(np.linalg.cond(x)),
    }


def _standardized_mean_differences(
    controls: np.ndarray,
    treatment: np.ndarray,
    names: Sequence[str],
    *,
    weights: Optional[np.ndarray] = None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    if controls.shape[1] == 0:
        return values
    treated_mask = treatment == 1
    control_mask = treatment == 0
    for index, name in enumerate(names):
        column = controls[:, index]
        if weights is None:
            mean_t = float(np.mean(column[treated_mask]))
            mean_c = float(np.mean(column[control_mask]))
            variance_t = float(np.var(column[treated_mask], ddof=1))
            variance_c = float(np.var(column[control_mask], ddof=1))
        else:
            wt = weights[treated_mask]
            wc = weights[control_mask]
            wt = wt / max(float(wt.sum()), 1e-15)
            wc = wc / max(float(wc.sum()), 1e-15)
            treated_values = column[treated_mask]
            control_values = column[control_mask]
            mean_t = float(np.sum(wt * treated_values))
            mean_c = float(np.sum(wc * control_values))
            variance_t = float(np.sum(wt * (treated_values - mean_t) ** 2))
            variance_c = float(np.sum(wc * (control_values - mean_c) ** 2))
        pooled = sqrt(max((variance_t + variance_c) / 2.0, 0.0))
        values[str(name)] = (
            float((mean_t - mean_c) / pooled) if pooled > 1e-12 else 0.0
        )
    return values


def _propensity(
    controls: np.ndarray,
    treatment: np.ndarray,
    *,
    random_state: int,
) -> np.ndarray:
    if controls.shape[1] == 0:
        return np.repeat(float(np.mean(treatment)), len(treatment))
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            max_iter=2000,
            random_state=int(random_state),
        )
        model.fit(controls, treatment.astype(int))
        return model.predict_proba(controls)[:, 1]
    except Exception as exc:
        raise ImportError(
            "Propensity methods require scikit-learn LogisticRegression."
        ) from exc


def regression_adjustment(
    df: pd.DataFrame,
    spec: CausalSpec,
    *,
    alpha: float,
) -> dict[str, Any]:
    columns = [spec.outcome, spec.treatment, *spec.confounders]
    frame, sample = _complete_frame(df, columns)
    y = pd.to_numeric(frame[spec.outcome], errors="raise").to_numpy(dtype=float)
    d = pd.to_numeric(frame[spec.treatment], errors="raise").to_numpy(dtype=float)
    controls, names = _encode_controls(frame, spec.confounders)
    design = np.column_stack([np.ones(len(frame)), d, controls])
    beta, standard_errors, residual, model = _fit_linear(
        y, design, covariance="HC1"
    )
    effect = _effect_summary(float(beta[1]), float(standard_errors[1]), alpha=alpha)
    model.update(
        {
            "r_squared": float(
                1.0
                - residual @ residual
                / max(float((y - y.mean()) @ (y - y.mean())), 1e-15)
            ),
            "encoded_controls": names,
        }
    )
    return {
        **effect,
        "n": len(frame),
        "sample": sample,
        "diagnostics": model,
        "assumptions": [
            "conditional exchangeability given listed confounders",
            "correct conditional-mean specification",
            "positivity and consistency/SUTVA",
            "HC1 robust standard errors address heteroskedasticity, not confounding",
        ],
        "warnings": [],
    }


def propensity_score_adjustment(
    df: pd.DataFrame,
    spec: CausalSpec,
    *,
    epsilon: float,
    random_state: int,
    alpha: float,
) -> dict[str, Any]:
    columns = [spec.outcome, spec.treatment, *spec.confounders]
    frame, sample = _complete_frame(df, columns)
    y = pd.to_numeric(frame[spec.outcome], errors="raise").to_numpy(dtype=float)
    d = _binary_treatment(
        frame[spec.treatment],
        treated_value=spec.treatment_value,
        control_value=spec.control_value,
    )
    controls, names = _encode_controls(frame, spec.confounders)
    propensity = np.clip(
        _propensity(controls, d, random_state=random_state),
        epsilon,
        1.0 - epsilon,
    )
    design = np.column_stack([np.ones(len(frame)), d, propensity])
    beta, standard_errors, _, model = _fit_linear(y, design, covariance="HC1")
    effect = _effect_summary(float(beta[1]), float(standard_errors[1]), alpha=alpha)
    balance = _standardized_mean_differences(controls, d, names)
    model.update(
        {
            "propensity_min": float(np.min(propensity)),
            "propensity_max": float(np.max(propensity)),
            "overlap_fraction": float(
                np.mean((propensity > epsilon) & (propensity < 1 - epsilon))
            ),
            "max_abs_smd_unweighted": max(
                (abs(value) for value in balance.values()), default=0.0
            ),
            "balance_unweighted": balance,
            "encoded_controls": names,
        }
    )
    return {
        **effect,
        "n": len(frame),
        "sample": sample,
        "diagnostics": model,
        "assumptions": [
            "conditional exchangeability given listed confounders",
            "correct propensity-score model",
            "positivity and consistency/SUTVA",
            "propensity covariate adjustment relies on outcome-model linearity in score",
        ],
        "warnings": [
            "Propensity score alone does not prove balance or exchangeability."
        ],
    }


def iptw(
    df: pd.DataFrame,
    spec: CausalSpec,
    *,
    epsilon: float,
    trim_quantile: float,
    max_weight: float,
    random_state: int,
    alpha: float,
) -> dict[str, Any]:
    columns = [spec.outcome, spec.treatment, *spec.confounders]
    frame, sample = _complete_frame(df, columns)
    y = pd.to_numeric(frame[spec.outcome], errors="raise").to_numpy(dtype=float)
    d = _binary_treatment(
        frame[spec.treatment],
        treated_value=spec.treatment_value,
        control_value=spec.control_value,
    )
    controls, names = _encode_controls(frame, spec.confounders)
    raw_propensity = _propensity(controls, d, random_state=random_state)
    propensity = np.clip(raw_propensity, epsilon, 1.0 - epsilon)
    prevalence = float(np.mean(d))
    weights = np.where(
        d == 1,
        prevalence / propensity,
        (1.0 - prevalence) / (1.0 - propensity),
    )
    original_weights = weights.copy()
    if trim_quantile > 0:
        lower, upper = np.quantile(
            weights, [trim_quantile, 1.0 - trim_quantile]
        )
        weights = np.clip(weights, lower, upper)
    weights = np.minimum(weights, max_weight)
    sqrt_weights = np.sqrt(weights)
    design = np.column_stack([np.ones(len(frame)), d])
    beta, standard_errors, _, model = _fit_linear(
        y * sqrt_weights,
        design * sqrt_weights[:, None],
        covariance="HC1",
    )
    effect = _effect_summary(float(beta[1]), float(standard_errors[1]), alpha=alpha)
    before = _standardized_mean_differences(controls, d, names)
    after = _standardized_mean_differences(
        controls, d, names, weights=weights
    )
    effective_sample = float(weights.sum() ** 2 / max(weights @ weights, 1e-15))
    model.update(
        {
            "propensity_min_raw": float(np.min(raw_propensity)),
            "propensity_max_raw": float(np.max(raw_propensity)),
            "propensity_epsilon": epsilon,
            "overlap_fraction": float(
                np.mean(
                    (raw_propensity >= epsilon)
                    & (raw_propensity <= 1.0 - epsilon)
                )
            ),
            "stabilized": True,
            "trim_quantile": trim_quantile,
            "max_weight_policy": max_weight,
            "weight_max_before_policy": float(np.max(original_weights)),
            "weight_max_after_policy": float(np.max(weights)),
            "weight_fraction_modified": float(
                np.mean(np.abs(weights - original_weights) > 1e-12)
            ),
            "effective_sample_size": effective_sample,
            "balance_before": before,
            "balance_after": after,
            "max_abs_smd_before": max(
                (abs(value) for value in before.values()), default=0.0
            ),
            "max_abs_smd_after": max(
                (abs(value) for value in after.values()), default=0.0
            ),
            "encoded_controls": names,
        }
    )
    return {
        **effect,
        "n": len(frame),
        "sample": sample,
        "diagnostics": model,
        "assumptions": [
            "conditional exchangeability given listed confounders",
            "correct propensity-score model",
            "positivity and consistency/SUTVA",
            "ATE after explicit weight clipping policy",
        ],
        "warnings": [
            "Weight clipping changes the finite-sample target; diagnostics record the policy."
        ],
    }


def aipw(
    df: pd.DataFrame,
    spec: CausalSpec,
    *,
    epsilon: float,
    folds: int,
    random_state: int,
    alpha: float,
) -> dict[str, Any]:
    columns = [spec.outcome, spec.treatment, *spec.confounders]
    frame, sample = _complete_frame(df, columns)
    y = pd.to_numeric(frame[spec.outcome], errors="raise").to_numpy(dtype=float)
    d = _binary_treatment(
        frame[spec.treatment],
        treated_value=spec.treatment_value,
        control_value=spec.control_value,
    )
    controls, names = _encode_controls(frame, spec.confounders)
    n = len(frame)
    minimum_group = int(min(np.sum(d == 0), np.sum(d == 1)))
    actual_folds = min(max(2, int(folds)), minimum_group)
    if actual_folds < 2:
        raise ValueError("AIPW cross-fitting requires at least two units per group.")
    propensity = np.empty(n, dtype=float)
    mu0 = np.empty(n, dtype=float)
    mu1 = np.empty(n, dtype=float)
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(
            n_splits=actual_folds,
            shuffle=True,
            random_state=int(random_state),
        )
        for train, test in splitter.split(np.zeros(n), d.astype(int)):
            if controls.shape[1] == 0:
                propensity[test] = float(np.mean(d[train]))
                mu0[test] = float(np.mean(y[train][d[train] == 0]))
                mu1[test] = float(np.mean(y[train][d[train] == 1]))
                continue
            propensity_model = LogisticRegression(
                max_iter=2000,
                random_state=int(random_state),
            )
            propensity_model.fit(controls[train], d[train].astype(int))
            propensity[test] = propensity_model.predict_proba(controls[test])[:, 1]
            for treatment_value, target in ((0, mu0), (1, mu1)):
                group_train = train[d[train] == treatment_value]
                outcome_model = Ridge(alpha=1.0)
                outcome_model.fit(controls[group_train], y[group_train])
                target[test] = outcome_model.predict(controls[test])
    except Exception as exc:
        raise ImportError(
            "AIPW cross-fitting requires scikit-learn."
        ) from exc
    raw_propensity = propensity.copy()
    propensity = np.clip(propensity, epsilon, 1.0 - epsilon)
    pseudo_outcome = (
        mu1
        - mu0
        + d * (y - mu1) / propensity
        - (1.0 - d) * (y - mu0) / (1.0 - propensity)
    )
    estimate = float(np.mean(pseudo_outcome))
    influence = pseudo_outcome - estimate
    standard_error = float(np.std(influence, ddof=1) / sqrt(n))
    effect = _effect_summary(estimate, standard_error, alpha=alpha)
    diagnostics = {
        "crossfit_folds": actual_folds,
        "fold_local_nuisance_fits": True,
        "propensity_min_raw": float(np.min(raw_propensity)),
        "propensity_max_raw": float(np.max(raw_propensity)),
        "propensity_epsilon": epsilon,
        "overlap_fraction": float(
            np.mean(
                (raw_propensity >= epsilon)
                & (raw_propensity <= 1.0 - epsilon)
            )
        ),
        "influence_sd": float(np.std(influence, ddof=1)),
        "encoded_controls": names,
        "nuisance_models": (
            "logistic propensity + ridge outcome"
            if controls.shape[1]
            else "fold-local group means"
        ),
    }
    return {
        **effect,
        "n": n,
        "sample": sample,
        "diagnostics": diagnostics,
        "assumptions": [
            "conditional exchangeability given listed confounders",
            "positivity and consistency/SUTVA",
            "at least one of propensity or outcome nuisance models is correctly specified",
            "cross-fitted nuisance predictions",
        ],
        "warnings": [
            "Doubly robust does not mean robust to unmeasured confounding."
        ],
    }


def propensity_matching(
    df: pd.DataFrame,
    spec: CausalSpec,
    *,
    epsilon: float,
    caliper: Optional[float],
    random_state: int,
    alpha: float,
) -> dict[str, Any]:
    columns = [spec.outcome, spec.treatment, *spec.confounders]
    frame, sample = _complete_frame(df, columns)
    y = pd.to_numeric(frame[spec.outcome], errors="raise").to_numpy(dtype=float)
    d = _binary_treatment(
        frame[spec.treatment],
        treated_value=spec.treatment_value,
        control_value=spec.control_value,
    )
    controls, names = _encode_controls(frame, spec.confounders)
    propensity = np.clip(
        _propensity(controls, d, random_state=random_state),
        epsilon,
        1.0 - epsilon,
    )
    logit = np.log(propensity / (1.0 - propensity))
    treated_indices = np.flatnonzero(d == 1)
    control_indices = np.flatnonzero(d == 0)
    if caliper is None:
        caliper = 0.2 * float(np.std(logit, ddof=1))
    pairs: list[tuple[int, int, float]] = []
    for treated_index in treated_indices:
        distances = np.abs(logit[control_indices] - logit[treated_index])
        nearest_position = int(np.argmin(distances))
        distance = float(distances[nearest_position])
        if distance <= caliper:
            pairs.append(
                (
                    int(treated_index),
                    int(control_indices[nearest_position]),
                    distance,
                )
            )
    if len(pairs) < 2:
        raise ValueError("Fewer than two treated units matched within the caliper.")
    treated_match = np.array([pair[0] for pair in pairs], dtype=int)
    control_match = np.array([pair[1] for pair in pairs], dtype=int)
    differences = y[treated_match] - y[control_match]
    estimate = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / sqrt(len(differences)))
    effect = _effect_summary(estimate, standard_error, alpha=alpha)
    before = _standardized_mean_differences(controls, d, names)
    matched_controls = np.vstack(
        [controls[treated_match], controls[control_match]]
    )
    matched_treatment = np.concatenate(
        [np.ones(len(pairs)), np.zeros(len(pairs))]
    )
    after = _standardized_mean_differences(
        matched_controls, matched_treatment, names
    )
    diagnostics = {
        "estimand_detail": "ATT among matched treated units",
        "matching": "nearest neighbor on propensity logit, with replacement",
        "caliper_logit_sd": caliper,
        "n_treated": len(treated_indices),
        "n_matched_treated": len(pairs),
        "matched_fraction": len(pairs) / len(treated_indices),
        "unique_matched_controls": len(np.unique(control_match)),
        "mean_match_distance": float(np.mean([pair[2] for pair in pairs])),
        "balance_before": before,
        "balance_after": after,
        "max_abs_smd_before": max(
            (abs(value) for value in before.values()), default=0.0
        ),
        "max_abs_smd_after": max(
            (abs(value) for value in after.values()), default=0.0
        ),
        "encoded_controls": names,
    }
    return {
        **effect,
        "n": 2 * len(pairs),
        "sample": {
            **sample,
            "n_matched_pairs": len(pairs),
            "n_unique_controls": len(np.unique(control_match)),
        },
        "diagnostics": diagnostics,
        "assumptions": [
            "conditional exchangeability given listed confounders",
            "positivity and consistency/SUTVA",
            "propensity model and caliper produce comparable matched units",
        ],
        "warnings": [
            "Nearest-neighbor matching uses replacement; naive paired SE does not "
            "fully account for propensity estimation or reused controls.",
            "The native matching estimand is ATT, not population ATE.",
        ],
        "estimand": "ATT",
    }


def iv_2sls(
    df: pd.DataFrame,
    spec: CausalSpec,
    *,
    alpha: float,
) -> dict[str, Any]:
    instruments = spec.instruments
    if not instruments:
        raise ValueError("2SLS requires at least one observed instrument.")
    columns = [
        spec.outcome,
        spec.treatment,
        *instruments,
        *spec.confounders,
    ]
    frame, sample = _complete_frame(df, columns)
    numeric = frame.apply(pd.to_numeric, errors="raise")
    y = numeric[spec.outcome].to_numpy(dtype=float)
    d = numeric[spec.treatment].to_numpy(dtype=float)
    controls, control_names = _encode_controls(numeric, spec.confounders)
    z_excluded = numeric[instruments].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(frame)), d, controls])
    z = np.column_stack([np.ones(len(frame)), z_excluded, controls])
    ztz_inv = np.linalg.pinv(z.T @ z)
    projection_x = z @ ztz_inv @ z.T @ x
    a = x.T @ projection_x
    a_inv = np.linalg.pinv(a)
    beta = a_inv @ x.T @ z @ ztz_inv @ z.T @ y
    residual = y - x @ beta
    projected_design = z @ ztz_inv @ z.T @ x
    score = projected_design * residual[:, None]
    meat = score.T @ score
    correction = len(frame) / max(len(frame) - x.shape[1], 1)
    variance = correction * a_inv @ meat @ a_inv
    standard_errors = np.sqrt(np.clip(np.diag(variance), 0.0, None))
    effect = _effect_summary(float(beta[1]), float(standard_errors[1]), alpha=alpha)
    first_stage = first_stage_diagnostics(
        numeric,
        treatment=spec.treatment,
        instrument=instruments,
        controls=spec.confounders,
    )
    diagnostics: dict[str, Any] = {
        "covariance": "heteroskedasticity-robust sandwich",
        "first_stage": first_stage,
        "first_stage_f": first_stage.get("first_stage_f"),
        "n_instruments": len(instruments),
        "instruments": instruments,
        "instrument_provenance": spec.instrument_provenance,
        "encoded_controls": control_names,
        "exclusion_restriction": "unverified",
        "instrument_independence": "unverified",
        "monotonicity_for_late": "unverified",
    }
    if len(instruments) > 1:
        auxiliary = np.column_stack([np.ones(len(frame)), z_excluded, controls])
        auxiliary_beta, *_ = np.linalg.lstsq(auxiliary, residual, rcond=None)
        auxiliary_residual = residual - auxiliary @ auxiliary_beta
        total = float((residual - residual.mean()) @ (residual - residual.mean()))
        r2 = 1.0 - float(auxiliary_residual @ auxiliary_residual) / max(total, 1e-15)
        statistic = max(0.0, len(frame) * r2)
        degrees = len(instruments) - 1
        try:
            from scipy.stats import chi2

            pvalue = float(chi2.sf(statistic, degrees))
        except Exception:
            pvalue = None
        diagnostics["overidentification"] = {
            "method": "Sargan_nR2",
            "statistic": statistic,
            "degrees_of_freedom": degrees,
            "p_value": pvalue,
            "caveat": "A non-rejection does not prove instrument validity.",
        }
    return {
        **effect,
        "n": len(frame),
        "sample": sample,
        "diagnostics": diagnostics,
        "assumptions": [
            "observed instrument relevance",
            "instrument exclusion restriction (unverified by data alone)",
            "instrument independence/exogeneity (unverified by data alone)",
            "SUTVA; monotonicity if interpreted as LATE",
        ],
        "warnings": [
            "First-stage strength does not verify exclusion or independence."
        ],
        "estimand": "LATE/structural coefficient under IV assumptions",
    }


def difference_in_differences(
    df: pd.DataFrame,
    spec: CausalSpec,
    *,
    alpha: float,
    min_pre_periods: int,
) -> dict[str, Any]:
    if not spec.post or not spec.unit or not spec.time:
        raise ValueError("DiD requires explicit unit=, time=, and post= fields.")
    columns = [
        spec.outcome,
        spec.treatment,
        spec.post,
        spec.unit,
        spec.time,
        *spec.confounders,
    ]
    frame, sample = _complete_frame(df, columns)
    y = pd.to_numeric(frame[spec.outcome], errors="raise").to_numpy(dtype=float)
    treated = _binary_treatment(
        frame[spec.treatment],
        treated_value=spec.treatment_value,
        control_value=spec.control_value,
    )
    post = _binary_treatment(
        frame[spec.post],
        treated_value=1,
        control_value=0,
    )
    controls, names = _encode_controls(
        frame, spec.confounders, include_time=spec.time
    )
    interaction = treated * post
    design = np.column_stack([np.ones(len(frame)), treated, post, interaction, controls])
    beta, standard_errors, _, model = _fit_linear(
        y,
        design,
        clusters=frame[spec.unit].to_numpy(),
    )
    effect = _effect_summary(float(beta[3]), float(standard_errors[3]), alpha=alpha)
    pre = frame.loc[post == 0]
    pretrend: dict[str, Any] = {
        "status": "not_estimable",
        "parallel_trends_verified": False,
    }
    pre_periods = int(pre[spec.time].nunique())
    if pre_periods >= 2:
        pre_y = pd.to_numeric(pre[spec.outcome], errors="raise").to_numpy(dtype=float)
        pre_treated = _binary_treatment(
            pre[spec.treatment],
            treated_value=spec.treatment_value,
            control_value=spec.control_value,
        )
        time_numeric = pd.to_numeric(pre[spec.time], errors="coerce")
        if time_numeric.notna().all():
            centered_time = (
                time_numeric.to_numpy(dtype=float)
                - float(time_numeric.to_numpy(dtype=float).mean())
            )
            pre_design = np.column_stack(
                [
                    np.ones(len(pre)),
                    pre_treated,
                    centered_time,
                    pre_treated * centered_time,
                ]
            )
            try:
                pre_beta, pre_se, _, _ = _fit_linear(
                    pre_y,
                    pre_design,
                    clusters=pre[spec.unit].to_numpy(),
                )
                statistic = float(pre_beta[3] / pre_se[3]) if pre_se[3] > 0 else float("inf")
                pretrend = {
                    "status": "diagnostic_only",
                    "pre_periods": pre_periods,
                    "interaction_slope": float(pre_beta[3]),
                    "standard_error": float(pre_se[3]),
                    "p_value": _normal_p(statistic),
                    "parallel_trends_verified": False,
                    "caveat": "Failure to reject a differential pretrend does not prove parallel trends.",
                }
            except Exception as exc:
                pretrend["reason"] = f"{type(exc).__name__}: {exc}"
    model.update(
        {
            "cluster": spec.unit,
            "pretrend": pretrend,
            "pre_periods": pre_periods,
            "minimum_pre_periods_policy": min_pre_periods,
            "encoded_controls": names,
        }
    )
    warnings = []
    if pre_periods < min_pre_periods:
        warnings.append(
            f"Only {pre_periods} pre-period(s); policy requests {min_pre_periods}."
        )
    warnings.append(
        "Parallel trends cannot be proven; the pretrend regression is diagnostic only."
    )
    return {
        **effect,
        "n": len(frame),
        "sample": sample,
        "diagnostics": model,
        "assumptions": [
            "parallel untreated potential-outcome trends",
            "no differential time-varying confounding",
            "no anticipation and stable treatment composition",
            "SUTVA/no cross-unit spillovers",
        ],
        "warnings": warnings,
        "estimand": "ATT (DiD interaction)",
    }


def panel_fixed_effects(
    df: pd.DataFrame,
    spec: CausalSpec,
    *,
    alpha: float,
) -> dict[str, Any]:
    if not spec.unit or not spec.time:
        raise ValueError("Panel fixed effects requires explicit unit= and time=.")
    columns = [
        spec.outcome,
        spec.treatment,
        spec.unit,
        spec.time,
        *spec.confounders,
    ]
    frame, sample = _complete_frame(df, columns)
    y = pd.to_numeric(frame[spec.outcome], errors="raise")
    d = pd.to_numeric(frame[spec.treatment], errors="raise")
    controls, names = _encode_controls(
        frame, spec.confounders, include_time=spec.time
    )
    raw_design = np.column_stack([d.to_numpy(dtype=float), controls])
    design_frame = pd.DataFrame(
        raw_design,
        index=frame.index,
        columns=["__treatment__", *names],
    )
    unit = frame[spec.unit]
    within_y = y - y.groupby(unit).transform("mean")
    within_x = design_frame - design_frame.groupby(unit).transform("mean")
    varying = within_x.std(axis=0) > 1e-12
    if not bool(varying.iloc[0]):
        raise ValueError("Treatment has no within-unit variation.")
    within_x = within_x.loc[:, varying]
    treatment_position = list(within_x.columns).index("__treatment__")
    beta, standard_errors, _, model = _fit_linear(
        within_y.to_numpy(dtype=float),
        within_x.to_numpy(dtype=float),
        clusters=unit.to_numpy(),
    )
    effect = _effect_summary(
        float(beta[treatment_position]),
        float(standard_errors[treatment_position]),
        alpha=alpha,
    )
    model.update(
        {
            "unit_fixed_effects": spec.unit,
            "time_fixed_effects": spec.time,
            "n_units": int(unit.nunique()),
            "n_periods": int(frame[spec.time].nunique()),
            "within_varying_columns": list(within_x.columns),
        }
    )
    return {
        **effect,
        "n": len(frame),
        "sample": sample,
        "diagnostics": model,
        "assumptions": [
            "strict exogeneity conditional on unit/time fixed effects",
            "no time-varying unmeasured confounding",
            "treatment varies within units",
            "SUTVA/no cross-unit spillovers",
        ],
        "warnings": [
            "Fixed effects remove time-invariant unit confounding only."
        ],
    }


def regression_discontinuity(
    df: pd.DataFrame,
    spec: CausalSpec,
    *,
    alpha: float,
    min_side: int,
) -> dict[str, Any]:
    if spec.running is None or spec.cutoff is None:
        raise ValueError("RDD requires explicit running= and cutoff=.")
    columns = [spec.outcome, spec.treatment, spec.running, *spec.confounders]
    frame, sample = _complete_frame(df, columns)
    running = pd.to_numeric(frame[spec.running], errors="raise").to_numpy(dtype=float)
    centered = running - float(spec.cutoff)
    if spec.bandwidth is None:
        bandwidth = float(1.84 * np.std(centered, ddof=1) * len(frame) ** (-1 / 5))
        bandwidth_source = "rule_of_thumb"
    else:
        bandwidth = float(spec.bandwidth)
        bandwidth_source = "explicit"
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("RDD bandwidth must be positive.")
    local_mask = np.abs(centered) <= bandwidth
    local = frame.loc[local_mask].copy()
    local_centered = centered[local_mask]
    above = (local_centered >= 0).astype(float)
    if len(local) < 6 or np.unique(above).size < 2:
        raise ValueError("RDD bandwidth does not contain observations on both sides.")
    y = pd.to_numeric(local[spec.outcome], errors="raise").to_numpy(dtype=float)
    controls, names = _encode_controls(local, spec.confounders)
    design = np.column_stack(
        [
            np.ones(len(local)),
            above,
            local_centered,
            above * local_centered,
            controls,
        ]
    )
    # Triangular kernel gives more weight near the cutoff.
    kernel = np.maximum(0.0, 1.0 - np.abs(local_centered) / bandwidth)
    weighted_design = design * np.sqrt(kernel)[:, None]
    weighted_y = y * np.sqrt(kernel)
    beta, standard_errors, _, model = _fit_linear(
        weighted_y, weighted_design, covariance="HC1"
    )
    effect = _effect_summary(float(beta[1]), float(standard_errors[1]), alpha=alpha)
    n_left = int(np.sum(above == 0))
    n_right = int(np.sum(above == 1))
    treatment_numeric = pd.to_numeric(
        local[spec.treatment], errors="coerce"
    ).to_numpy(dtype=float)
    assignment_agreement = float(
        np.mean(
            (treatment_numeric == spec.treatment_value) == (above == 1)
        )
    )
    # A simple count-density diagnostic, not a formal McCrary test.
    density_ratio = max(n_left, n_right) / max(min(n_left, n_right), 1)
    model.update(
        {
            "running": spec.running,
            "cutoff": spec.cutoff,
            "bandwidth": bandwidth,
            "bandwidth_source": bandwidth_source,
            "kernel": "triangular",
            "local_linear": True,
            "n_left": n_left,
            "n_right": n_right,
            "minimum_side_policy": min_side,
            "assignment_agreement": assignment_agreement,
            "cutoff_count_ratio": density_ratio,
            "density_test": "count ratio heuristic; not McCrary",
            "encoded_controls": names,
        }
    )
    warnings = []
    if bandwidth_source != "explicit":
        warnings.append(
            "Bandwidth is a rule-of-thumb; report sensitivity to alternative bandwidths."
        )
    if min(n_left, n_right) < min_side:
        warnings.append(
            f"Only {min(n_left, n_right)} units on the smaller cutoff side."
        )
    return {
        **effect,
        "n": len(local),
        "sample": {
            **sample,
            "n_within_bandwidth": len(local),
            "n_left": n_left,
            "n_right": n_right,
        },
        "diagnostics": model,
        "assumptions": [
            "potential outcomes are continuous at the cutoff",
            "no precise manipulation/sorting around the cutoff",
            "local functional form and bandwidth are adequate",
            "SUTVA near the cutoff",
        ],
        "warnings": warnings,
        "estimand": "local discontinuity at cutoff",
    }


def interrupted_time_series(
    df: pd.DataFrame,
    spec: CausalSpec,
    *,
    alpha: float,
    hac_lags: int,
) -> dict[str, Any]:
    if not spec.time:
        raise ValueError("Interrupted time series requires explicit time=.")
    intervention = spec.post or spec.treatment
    columns = [spec.outcome, spec.time, intervention, *spec.confounders]
    frame, sample = _complete_frame(df, columns)
    frame = frame.sort_values(spec.time)
    time = pd.to_numeric(frame[spec.time], errors="raise").to_numpy(dtype=float)
    post = _binary_treatment(
        frame[intervention],
        treated_value=1 if spec.post else spec.treatment_value,
        control_value=0 if spec.post else spec.control_value,
    )
    if np.sum(post == 1) == 0 or np.sum(post == 0) == 0:
        raise ValueError("ITS requires observations before and after interruption.")
    first_post = float(np.min(time[post == 1]))
    centered_time = time - first_post
    time_after = np.where(post == 1, centered_time, 0.0)
    controls, names = _encode_controls(frame, spec.confounders)
    design = np.column_stack(
        [np.ones(len(frame)), centered_time, post, time_after, controls]
    )
    y = pd.to_numeric(frame[spec.outcome], errors="raise").to_numpy(dtype=float)
    beta, standard_errors, residual, model = _fit_linear(
        y,
        design,
        hac_lags=hac_lags,
    )
    effect = _effect_summary(float(beta[2]), float(standard_errors[2]), alpha=alpha)
    slope_summary = _effect_summary(
        float(beta[3]), float(standard_errors[3]), alpha=alpha
    )
    denominator = float(residual @ residual)
    lag1 = (
        float(np.corrcoef(residual[1:], residual[:-1])[0, 1])
        if len(residual) > 2 and denominator > 1e-15
        else None
    )
    model.update(
        {
            "interruption_column": intervention,
            "interruption_time": first_post,
            "level_change": effect,
            "slope_change": slope_summary,
            "hac_lags": hac_lags,
            "residual_lag1_correlation": lag1,
            "encoded_controls": names,
        }
    )
    return {
        **effect,
        "n": len(frame),
        "sample": {
            **sample,
            "n_pre": int(np.sum(post == 0)),
            "n_post": int(np.sum(post == 1)),
        },
        "diagnostics": model,
        "assumptions": [
            "no concurrent intervention or time-varying confounder at interruption",
            "pre-interruption trend would have continued",
            "segmented linear trend is adequately specified",
            "HAC covariance addresses residual autocorrelation up to configured lag",
        ],
        "warnings": [
            "ITS is vulnerable to coincident events and trend misspecification."
        ],
        "estimand": "immediate level change at interruption",
    }
