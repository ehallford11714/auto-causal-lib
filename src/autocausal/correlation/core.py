"""Typed, data-type-aware association analysis.

Association measures describe dependence.  They do not identify causal
effects, validate instruments, or establish a causal direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import erf, sqrt
from typing import Any, Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from autocausal.statistical_gates import benjamini_hochberg

ASSOCIATION_NOTICE = (
    "Association is not causation. Coefficients and p-values do not establish "
    "identification, causal direction, or freedom from confounding."
)

MissingPolicy = Literal["pairwise", "listwise", "raise"]

__all__ = [
    "ASSOCIATION_NOTICE",
    "CorrelationMatrixResult",
    "CorrelationResult",
    "CorrelationSuite",
    "correlation",
    "correlation_matrix",
]


@dataclass
class CorrelationResult:
    """A serializable association estimate with explicit limitations."""

    x: str
    y: str
    measure: str
    coefficient: Optional[float]
    n: int
    effect_size: Optional[float] = None
    effect_size_name: str = "coefficient"
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    confidence_level: float = 0.95
    p_value: Optional[float] = None
    q_value: Optional[float] = None
    fdr_reject_null: Optional[bool] = None
    missing_data: str = "pairwise_complete"
    controls: list[str] = field(default_factory=list)
    weights: Optional[str] = None
    cluster: Optional[str] = None
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    random_state: int = 0
    schema: str = "AutoCausalCorrelationResult.v1"
    epistemic_notice: str = ASSOCIATION_NOTICE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def report(self) -> str:
        estimate = (
            "not estimable"
            if self.coefficient is None
            else f"{self.coefficient:.6g}"
        )
        interval = (
            "not computed"
            if self.ci_low is None or self.ci_high is None
            else f"[{self.ci_low:.6g}, {self.ci_high:.6g}]"
        )
        pvalue = (
            "not meaningful/computed"
            if self.p_value is None
            else f"{self.p_value:.6g}"
        )
        qvalue = (
            ""
            if self.q_value is None
            else f"\n- BH-FDR q-value: {self.q_value:.6g}"
        )
        warning_lines = "\n".join(f"- {value}" for value in self.warnings)
        return "\n".join(
            [
                "# Association result",
                "",
                f"> **{self.epistemic_notice}**",
                "",
                f"- Variables: `{self.x}` and `{self.y}`",
                f"- Measure: `{self.measure}`",
                f"- Coefficient/effect size: {estimate}",
                f"- {self.confidence_level:.0%} bootstrap CI: {interval}",
                f"- p-value: {pvalue}{qvalue}",
                f"- Complete sample: {self.n}",
                f"- Missing-data policy: `{self.missing_data}`",
                f"- Controls: {self.controls or 'none'}",
                "",
                "## Warnings",
                warning_lines or "- None beyond the epistemic notice.",
            ]
        )


@dataclass
class CorrelationMatrixResult:
    """Pairwise scan results with BH-FDR correction."""

    columns: list[str]
    results: list[CorrelationResult]
    alpha: float = 0.05
    missing_data: str = "pairwise"
    random_state: int = 0
    warnings: list[str] = field(default_factory=list)
    schema: str = "AutoCausalCorrelationMatrixResult.v1"
    epistemic_notice: str = ASSOCIATION_NOTICE

    def coefficients(self) -> pd.DataFrame:
        matrix = pd.DataFrame(
            np.eye(len(self.columns)),
            index=self.columns,
            columns=self.columns,
            dtype=float,
        )
        for result in self.results:
            value = (
                np.nan if result.coefficient is None else result.coefficient
            )
            matrix.loc[result.x, result.y] = value
            matrix.loc[result.y, result.x] = value
        return matrix

    def q_values(self) -> pd.DataFrame:
        matrix = pd.DataFrame(
            np.nan,
            index=self.columns,
            columns=self.columns,
            dtype=float,
        )
        np.fill_diagonal(matrix.values, 0.0)
        for result in self.results:
            value = np.nan if result.q_value is None else result.q_value
            matrix.loc[result.x, result.y] = value
            matrix.loc[result.y, result.x] = value
        return matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "columns": list(self.columns),
            "alpha": self.alpha,
            "missing_data": self.missing_data,
            "random_state": self.random_state,
            "warnings": list(self.warnings),
            "epistemic_notice": self.epistemic_notice,
            "results": [result.to_dict() for result in self.results],
        }

    def report(self) -> str:
        tested = sum(result.p_value is not None for result in self.results)
        retained = sum(bool(result.fdr_reject_null) for result in self.results)
        rows = [
            "| x | y | measure | coefficient | p | q | n |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for result in self.results:
            coefficient = (
                "NA"
                if result.coefficient is None
                else f"{result.coefficient:.4g}"
            )
            pvalue = (
                "NA" if result.p_value is None else f"{result.p_value:.4g}"
            )
            qvalue = (
                "NA" if result.q_value is None else f"{result.q_value:.4g}"
            )
            rows.append(
                f"| {result.x} | {result.y} | {result.measure} | "
                f"{coefficient} | {pvalue} | {qvalue} | {result.n} |"
            )
        return "\n".join(
            [
                "# Association matrix",
                "",
                f"> **{self.epistemic_notice}**",
                "",
                f"BH-FDR alpha={self.alpha}: {retained}/{tested} tested pairs retained.",
                "",
                *rows,
            ]
        )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _normal_p(value: float) -> float:
    return float(2.0 * (1.0 - _normal_cdf(abs(float(value)))))


def _role(series: pd.Series) -> str:
    nonmissing = series.dropna()
    unique = int(nonmissing.nunique())
    if unique == 2:
        return "binary"
    if (
        pd.api.types.is_bool_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "continuous"
    return "categorical"


def _encode_binary(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    categories = list(pd.unique(series))
    if len(categories) != 2:
        raise ValueError("Binary association requires exactly two levels.")
    ordered = sorted(categories, key=lambda value: str(value))
    mapping = {ordered[0]: 0.0, ordered[1]: 1.0}
    return series.map(mapping).to_numpy(dtype=float), [
        str(ordered[0]),
        str(ordered[1]),
    ]


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _pearson_arrays(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    if weights is None:
        if len(x) < 2 or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("Correlation weights must be finite and non-negative.")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Correlation weights must have positive total.")
    normalized = weights / total
    mean_x = float(np.sum(normalized * x))
    mean_y = float(np.sum(normalized * y))
    dx = x - mean_x
    dy = y - mean_y
    covariance = float(np.sum(normalized * dx * dy))
    variance_x = float(np.sum(normalized * dx * dx))
    variance_y = float(np.sum(normalized * dy * dy))
    denominator = sqrt(max(variance_x * variance_y, 0.0))
    return covariance / denominator if denominator > 1e-15 else float("nan")


def _correlation_p(coefficient: float, n: int, df_adjustment: int = 0) -> float:
    df = n - 2 - df_adjustment
    if df <= 0 or not np.isfinite(coefficient):
        return float("nan")
    clipped = max(-0.999999999, min(0.999999999, coefficient))
    statistic = clipped * sqrt(df / max(1.0 - clipped**2, 1e-15))
    try:
        from scipy.stats import t

        return float(2.0 * t.sf(abs(statistic), df))
    except Exception:
        return _normal_p(statistic)


def _residualize(
    values: np.ndarray,
    controls: pd.DataFrame,
) -> np.ndarray:
    encoded = pd.get_dummies(
        controls,
        dummy_na=False,
        drop_first=True,
        dtype=float,
    )
    matrix = encoded.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(values)), matrix])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Székely-Rizzo sample distance correlation for one-dimensional inputs."""
    x_distance = np.abs(x[:, None] - x[None, :])
    y_distance = np.abs(y[:, None] - y[None, :])
    a = (
        x_distance
        - x_distance.mean(axis=0)[None, :]
        - x_distance.mean(axis=1)[:, None]
        + x_distance.mean()
    )
    b = (
        y_distance
        - y_distance.mean(axis=0)[None, :]
        - y_distance.mean(axis=1)[:, None]
        + y_distance.mean()
    )
    distance_covariance_sq = float(np.mean(a * b))
    distance_variance_x = float(np.mean(a * a))
    distance_variance_y = float(np.mean(b * b))
    denominator = sqrt(max(distance_variance_x * distance_variance_y, 0.0))
    if denominator <= 1e-15:
        return float("nan")
    return float(sqrt(max(distance_covariance_sq, 0.0) / denominator))


def _cramers_v(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, Optional[float], dict[str, Any]]:
    table = pd.crosstab(x, y)
    observed = table.to_numpy(dtype=float)
    n = float(observed.sum())
    if n <= 1 or min(observed.shape) < 2:
        return float("nan"), None, {"table_shape": list(observed.shape)}
    row_sum = observed.sum(axis=1, keepdims=True)
    column_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ column_sum / n
    valid = expected > 0
    chi_square = float(np.sum(((observed - expected) ** 2)[valid] / expected[valid]))
    phi2 = chi_square / n
    rows, columns = observed.shape
    phi2_corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    columns_corrected = columns - ((columns - 1) ** 2) / (n - 1)
    denominator = min(columns_corrected - 1, rows_corrected - 1)
    coefficient = (
        sqrt(phi2_corrected / denominator) if denominator > 0 else 0.0
    )
    degrees = (rows - 1) * (columns - 1)
    try:
        from scipy.stats import chi2

        pvalue = float(chi2.sf(chi_square, degrees))
    except Exception:
        pvalue = None
    return coefficient, pvalue, {
        "chi_square": chi_square,
        "degrees_of_freedom": degrees,
        "table_shape": [rows, columns],
        "bias_corrected": True,
    }


def _eta(
    categorical: pd.Series,
    continuous: np.ndarray,
) -> tuple[float, Optional[float], dict[str, Any]]:
    codes, categories = pd.factorize(categorical, sort=True)
    overall = float(np.mean(continuous))
    total = float(np.sum((continuous - overall) ** 2))
    between = 0.0
    groups: list[np.ndarray] = []
    for code in range(len(categories)):
        group = continuous[codes == code]
        if len(group):
            groups.append(group)
            between += len(group) * (float(np.mean(group)) - overall) ** 2
    eta_squared = between / total if total > 1e-15 else float("nan")
    eta = sqrt(max(eta_squared, 0.0)) if np.isfinite(eta_squared) else float("nan")
    pvalue: Optional[float]
    f_statistic: Optional[float]
    if len(groups) >= 2 and len(continuous) > len(groups):
        within = max(total - between, 0.0)
        df_between = len(groups) - 1
        df_within = len(continuous) - len(groups)
        f_statistic = (
            (between / df_between) / (within / df_within)
            if within > 1e-15
            else float("inf")
        )
        try:
            from scipy.stats import f

            pvalue = float(f.sf(f_statistic, df_between, df_within))
        except Exception:
            pvalue = None
    else:
        f_statistic = None
        pvalue = None
    return eta, pvalue, {
        "eta_squared": eta_squared,
        "group_count": len(groups),
        "anova_f": f_statistic,
        "group_test": "one_way_anova",
    }


def _mutual_information(
    x: pd.Series,
    y: pd.Series,
    *,
    normalized: bool,
    random_state: int,
) -> tuple[float, dict[str, Any]]:
    x_role = _role(x)
    y_role = _role(y)
    if normalized:
        from sklearn.metrics import normalized_mutual_info_score

        def discretize(series: pd.Series, role: str) -> np.ndarray:
            if role == "continuous":
                bins = min(10, max(2, int(sqrt(len(series)))))
                return pd.qcut(
                    series,
                    q=bins,
                    labels=False,
                    duplicates="drop",
                ).to_numpy()
            return pd.factorize(series, sort=True)[0]

        value = normalized_mutual_info_score(
            discretize(x, x_role),
            discretize(y, y_role),
            average_method="arithmetic",
        )
        return float(value), {
            "normalization": "arithmetic_entropy",
            "continuous_discretization": "quantile_bins",
        }
    from sklearn.feature_selection import (
        mutual_info_classif,
        mutual_info_regression,
    )

    if x_role == "continuous":
        feature = pd.to_numeric(x).to_numpy(dtype=float).reshape(-1, 1)
        discrete_feature = False
    else:
        feature = pd.factorize(x, sort=True)[0].reshape(-1, 1)
        discrete_feature = True
    if y_role in ("categorical", "binary"):
        target = pd.factorize(y, sort=True)[0]
        value = mutual_info_classif(
            feature,
            target,
            discrete_features=[discrete_feature],
            random_state=random_state,
        )[0]
        target_kind = "classification"
    else:
        target = pd.to_numeric(y).to_numpy(dtype=float)
        value = mutual_info_regression(
            feature,
            target,
            discrete_features=[discrete_feature],
            random_state=random_state,
        )[0]
        target_kind = "regression"
    return float(value), {
        "units": "nats",
        "target_kind": target_kind,
        "symmetric": False,
    }


def _select_auto_method(x: pd.Series, y: pd.Series) -> str:
    x_role = _role(x)
    y_role = _role(y)
    if x_role == "continuous" and y_role == "continuous":
        # Prefer rank association under severe skew or visible extreme tails.
        skew = max(abs(float(x.skew())), abs(float(y.skew())))
        outlier = False
        for series in (x, y):
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            if iqr > 0:
                outlier |= bool(
                    ((series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)).any()
                )
        return "spearman" if skew > 2.0 or outlier else "pearson"
    if {x_role, y_role} == {"binary", "continuous"}:
        return "point_biserial"
    if x_role in ("binary", "categorical") and y_role in (
        "binary",
        "categorical",
    ):
        return "phi" if x_role == y_role == "binary" else "cramers_v"
    if "continuous" in (x_role, y_role):
        return "correlation_ratio"
    return "cramers_v"


def _prepare(
    x: pd.Series,
    y: pd.Series,
    *,
    controls: Optional[pd.DataFrame],
    weights: Optional[pd.Series],
    cluster: Optional[pd.Series],
    missing: MissingPolicy,
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.DataFrame({"__x__": x, "__y__": y})
    control_columns: list[str] = []
    if controls is not None:
        for index, column in enumerate(controls.columns):
            name = f"__control_{index}__"
            frame[name] = controls[column]
            control_columns.append(name)
    if weights is not None:
        frame["__weights__"] = weights
    if cluster is not None:
        frame["__cluster__"] = cluster
    if missing == "raise" and frame.isna().any().any():
        raise ValueError("Missing values present with missing='raise'.")
    frame = frame.dropna()
    return frame, control_columns


def _estimate_core(
    frame: pd.DataFrame,
    *,
    method: str,
    control_columns: Sequence[str],
    random_state: int,
    permutation_n: int,
) -> tuple[float, Optional[float], str, dict[str, Any], list[str], list[str]]:
    x_series = frame["__x__"]
    y_series = frame["__y__"]
    x_role = _role(x_series)
    y_role = _role(y_series)
    weights = (
        frame["__weights__"].to_numpy(dtype=float)
        if "__weights__" in frame
        else None
    )
    assumptions: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {"x_role": x_role, "y_role": y_role}
    pvalue: Optional[float] = None
    measure = method

    if method == "pearson":
        x = pd.to_numeric(x_series).to_numpy(dtype=float)
        y = pd.to_numeric(y_series).to_numpy(dtype=float)
        coefficient = _pearson_arrays(x, y, weights)
        if weights is None:
            pvalue = _correlation_p(coefficient, len(frame))
        else:
            warnings.append(
                "Analytic p-value is omitted for weighted Pearson correlation."
            )
        assumptions.extend(
            [
                "linear association",
                "independent observations for analytic p-value",
                "finite second moments",
            ]
        )
    elif method == "spearman":
        if weights is not None:
            raise ValueError("Weighted Spearman is not implemented.")
        x = _rank(pd.to_numeric(x_series).to_numpy(dtype=float))
        y = _rank(pd.to_numeric(y_series).to_numpy(dtype=float))
        coefficient = _pearson_arrays(x, y)
        pvalue = _correlation_p(coefficient, len(frame))
        assumptions.extend(
            ["monotonic association", "independent observations for p-value"]
        )
    elif method == "kendall":
        if weights is not None:
            raise ValueError("Weighted Kendall is not implemented.")
        try:
            from scipy.stats import kendalltau

            coefficient, pvalue = kendalltau(x_series, y_series, nan_policy="omit")
            coefficient = float(coefficient)
            pvalue = float(pvalue)
        except Exception as exc:
            raise ImportError("Kendall correlation requires scipy.") from exc
        assumptions.append("independent observation pairs for p-value")
    elif method == "partial":
        if not control_columns:
            raise ValueError("Partial correlation requires explicit controls.")
        if weights is not None:
            raise ValueError("Weighted partial correlation is not implemented.")
        x = pd.to_numeric(x_series).to_numpy(dtype=float)
        y = pd.to_numeric(y_series).to_numpy(dtype=float)
        controls = frame[list(control_columns)]
        residual_x = _residualize(x, controls)
        residual_y = _residualize(y, controls)
        coefficient = _pearson_arrays(residual_x, residual_y)
        encoded_k = pd.get_dummies(
            controls, drop_first=True, dtype=float
        ).shape[1]
        pvalue = _correlation_p(coefficient, len(frame), encoded_k)
        metadata["control_degrees"] = encoded_k
        assumptions.extend(
            [
                "linear additive adjustment for controls",
                "no unmeasured confounding is NOT implied",
                "independent observations for p-value",
            ]
        )
    elif method in ("winsorized", "winsorized_pearson"):
        x = pd.to_numeric(x_series).to_numpy(dtype=float)
        y = pd.to_numeric(y_series).to_numpy(dtype=float)
        x_low, x_high = np.quantile(x, [0.05, 0.95])
        y_low, y_high = np.quantile(y, [0.05, 0.95])
        coefficient = _pearson_arrays(
            np.clip(x, x_low, x_high),
            np.clip(y, y_low, y_high),
            weights,
        )
        if weights is None:
            pvalue = _correlation_p(coefficient, len(frame))
        measure = "winsorized_pearson"
        metadata["winsor_limits"] = [0.05, 0.95]
        warnings.append(
            "Analytic p-value treats winsorization limits as fixed; prefer bootstrap CI."
        )
    elif method in ("point_biserial", "point-biserial"):
        if weights is not None:
            raise ValueError("Weighted point-biserial is not implemented.")
        if x_role == "binary" and y_role == "continuous":
            binary, levels = _encode_binary(x_series)
            continuous = pd.to_numeric(y_series).to_numpy(dtype=float)
        elif y_role == "binary" and x_role == "continuous":
            binary, levels = _encode_binary(y_series)
            continuous = pd.to_numeric(x_series).to_numpy(dtype=float)
        else:
            raise ValueError(
                "Point-biserial requires one binary and one continuous variable."
            )
        coefficient = _pearson_arrays(binary, continuous)
        pvalue = _correlation_p(coefficient, len(frame))
        metadata["binary_levels"] = levels
        measure = "point_biserial"
        assumptions.extend(
            ["binary grouping variable", "independent observations for p-value"]
        )
    elif method == "phi":
        if x_role != "binary" or y_role != "binary":
            raise ValueError("Phi requires two binary variables.")
        x, x_levels = _encode_binary(x_series)
        y, y_levels = _encode_binary(y_series)
        coefficient = _pearson_arrays(x, y)
        pvalue = _correlation_p(coefficient, len(frame))
        metadata.update(
            {"x_levels": x_levels, "y_levels": y_levels, "table_shape": [2, 2]}
        )
    elif method in ("cramers_v", "cramer_v", "cramér_v"):
        if weights is not None:
            raise ValueError("Weighted Cramér's V is not implemented.")
        coefficient, pvalue, extra = _cramers_v(x_series, y_series)
        metadata.update(extra)
        measure = "cramers_v_bias_corrected"
        assumptions.append("independent observations for chi-square p-value")
    elif method in ("correlation_ratio", "eta", "eta_squared"):
        if weights is not None:
            raise ValueError("Weighted correlation ratio is not implemented.")
        if x_role == "continuous" and y_role != "continuous":
            categorical = y_series
            continuous = pd.to_numeric(x_series).to_numpy(dtype=float)
        elif y_role == "continuous" and x_role != "continuous":
            categorical = x_series
            continuous = pd.to_numeric(y_series).to_numpy(dtype=float)
        else:
            raise ValueError(
                "Correlation ratio requires one categorical and one continuous variable."
            )
        coefficient, pvalue, extra = _eta(categorical, continuous)
        metadata.update(extra)
        measure = "correlation_ratio_eta"
        assumptions.extend(
            [
                "independent groups for ANOVA p-value",
                "homoskedastic normal residuals for classical ANOVA p-value",
            ]
        )
    elif method in ("mutual_information", "mi", "nmi"):
        if weights is not None:
            raise ValueError("Weighted mutual information is not implemented.")
        normalized = method == "nmi"
        coefficient, extra = _mutual_information(
            x_series,
            y_series,
            normalized=normalized,
            random_state=random_state,
        )
        metadata.update(extra)
        pvalue = None
        measure = "normalized_mutual_information" if normalized else "mutual_information"
        warnings.append(
            "Mutual-information magnitude is estimator/discretization dependent; no analytic p-value."
        )
    elif method in ("distance", "distance_correlation", "dcor"):
        if weights is not None:
            raise ValueError("Weighted distance correlation is not implemented.")
        x = pd.to_numeric(x_series).to_numpy(dtype=float)
        y = pd.to_numeric(y_series).to_numpy(dtype=float)
        if len(frame) > 5000:
            raise ValueError(
                "Distance correlation is O(n^2); sample or set an operational limit."
            )
        coefficient = _distance_correlation(x, y)
        if permutation_n > 0:
            rng = np.random.default_rng(random_state)
            null = np.empty(permutation_n, dtype=float)
            for index in range(permutation_n):
                null[index] = _distance_correlation(x, rng.permutation(y))
            pvalue = float(
                (1 + np.sum(null >= coefficient)) / (permutation_n + 1)
            )
            metadata["permutations"] = permutation_n
        else:
            warnings.append(
                "Distance-correlation p-value omitted; set permutation_n>0."
            )
        measure = "distance_correlation"
        assumptions.append("exchangeable observations for permutation p-value")
    else:
        raise ValueError(f"Unknown association method: {method!r}")

    if "__cluster__" in frame:
        warnings.append(
            "Clustered/repeated observations: analytic p-values assume i.i.d.; "
            "use the cluster bootstrap CI for uncertainty."
        )
    return (
        float(coefficient) if np.isfinite(coefficient) else float("nan"),
        pvalue if pvalue is None or np.isfinite(pvalue) else None,
        measure,
        metadata,
        assumptions,
        warnings,
    )


def _bootstrap_interval(
    frame: pd.DataFrame,
    *,
    method: str,
    control_columns: Sequence[str],
    bootstrap_n: int,
    confidence_level: float,
    random_state: int,
) -> tuple[Optional[float], Optional[float], int]:
    if bootstrap_n <= 0 or len(frame) < 3:
        return None, None, 0
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    clusters = (
        pd.unique(frame["__cluster__"])
        if "__cluster__" in frame
        else None
    )
    for _ in range(int(bootstrap_n)):
        if clusters is not None:
            sampled_clusters = rng.choice(
                clusters, size=len(clusters), replace=True
            )
            pieces = []
            for bootstrap_cluster, cluster_value in enumerate(sampled_clusters):
                piece = frame.loc[frame["__cluster__"] == cluster_value].copy()
                # Keep duplicated sampled clusters distinct without exposing IDs.
                piece["__cluster__"] = bootstrap_cluster
                pieces.append(piece)
            sample = pd.concat(pieces, ignore_index=True)
        else:
            indices = rng.integers(0, len(frame), size=len(frame))
            sample = frame.iloc[indices].reset_index(drop=True)
        try:
            value, _, _, _, _, _ = _estimate_core(
                sample,
                method=method,
                control_columns=control_columns,
                random_state=random_state,
                permutation_n=0,
            )
            if np.isfinite(value):
                values.append(value)
        except Exception:
            continue
    if len(values) < max(20, int(bootstrap_n * 0.5)):
        return None, None, len(values)
    alpha = 1.0 - confidence_level
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high), len(values)


def correlation(
    x: str | pd.Series | Sequence[Any],
    y: str | pd.Series | Sequence[Any],
    *,
    data: Optional[pd.DataFrame] = None,
    method: str = "auto",
    controls: Optional[Sequence[str] | pd.DataFrame] = None,
    weights: Optional[str | pd.Series | Sequence[float]] = None,
    cluster: Optional[str | pd.Series | Sequence[Any]] = None,
    missing: MissingPolicy = "pairwise",
    bootstrap_n: int = 0,
    confidence_level: float = 0.95,
    permutation_n: int = 0,
    random_state: int = 0,
) -> CorrelationResult:
    """Estimate one association with explicit measure selection and metadata."""
    if isinstance(x, str):
        if data is None or x not in data.columns:
            raise ValueError(f"Column {x!r} requires a data frame containing it.")
        x_series = data[x]
        x_name = x
    else:
        x_series = pd.Series(x)
        x_name = str(getattr(x, "name", None) or "x")
    if isinstance(y, str):
        if data is None or y not in data.columns:
            raise ValueError(f"Column {y!r} requires a data frame containing it.")
        y_series = data[y]
        y_name = y
    else:
        y_series = pd.Series(y)
        y_name = str(getattr(y, "name", None) or "y")

    if isinstance(controls, pd.DataFrame):
        control_frame = controls
        control_names = [str(column) for column in controls.columns]
    elif controls:
        if data is None:
            raise ValueError("Named controls require data=.")
        control_names = [str(column) for column in controls]
        missing_controls = [
            column for column in control_names if column not in data.columns
        ]
        if missing_controls:
            raise ValueError(f"Control columns not found: {missing_controls}")
        control_frame = data[control_names]
    else:
        control_names = []
        control_frame = None

    def resolve_optional(
        value: Optional[str | pd.Series | Sequence[Any]],
        label: str,
    ) -> tuple[Optional[pd.Series], Optional[str]]:
        if value is None:
            return None, None
        if isinstance(value, str):
            if data is None or value not in data.columns:
                raise ValueError(f"{label} column {value!r} not found.")
            return data[value], value
        series = pd.Series(value, index=x_series.index)
        return series, str(getattr(value, "name", None) or label)

    weight_series, weight_name = resolve_optional(weights, "weights")
    cluster_series, cluster_name = resolve_optional(cluster, "cluster")
    frame, control_columns = _prepare(
        x_series,
        y_series,
        controls=control_frame,
        weights=weight_series,
        cluster=cluster_series,
        missing=missing,
    )
    if len(frame) < 3:
        raise ValueError("At least three complete observations are required.")
    selected = (
        _select_auto_method(frame["__x__"], frame["__y__"])
        if method == "auto"
        else method.lower().replace(" ", "_")
    )
    if control_names and method == "auto":
        if (
            _role(frame["__x__"]) == "continuous"
            and _role(frame["__y__"]) == "continuous"
        ):
            selected = "partial"
        else:
            raise ValueError(
                "Controls are currently supported natively only by partial "
                "correlation for two continuous variables."
            )
    (
        coefficient,
        pvalue,
        measure,
        metadata,
        assumptions,
        warnings,
    ) = _estimate_core(
        frame,
        method=selected,
        control_columns=control_columns,
        random_state=int(random_state),
        permutation_n=int(permutation_n),
    )
    low, high, successful_bootstraps = _bootstrap_interval(
        frame,
        method=selected,
        control_columns=control_columns,
        bootstrap_n=int(bootstrap_n),
        confidence_level=float(confidence_level),
        random_state=int(random_state),
    )
    if bootstrap_n and successful_bootstraps < max(20, bootstrap_n // 2):
        warnings.append(
            f"Bootstrap interval unavailable: only {successful_bootstraps}/"
            f"{bootstrap_n} resamples were estimable."
        )
    if method == "auto":
        metadata["auto_selected_method"] = selected
    metadata["bootstrap_successful"] = successful_bootstraps
    effect_size = coefficient
    effect_size_name = "coefficient"
    if measure == "correlation_ratio_eta":
        effect_size_name = "eta"
        effect_size = coefficient
    return CorrelationResult(
        x=x_name,
        y=y_name,
        measure=measure,
        coefficient=coefficient if np.isfinite(coefficient) else None,
        effect_size=effect_size if np.isfinite(effect_size) else None,
        effect_size_name=effect_size_name,
        n=len(frame),
        ci_low=low,
        ci_high=high,
        confidence_level=confidence_level,
        p_value=pvalue,
        missing_data=f"{missing}_complete",
        controls=control_names,
        weights=weight_name,
        cluster=cluster_name,
        assumptions=assumptions,
        warnings=warnings,
        metadata=metadata,
        random_state=int(random_state),
    )


def correlation_matrix(
    data: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    method: str = "auto",
    missing: MissingPolicy = "pairwise",
    alpha: float = 0.05,
    bootstrap_n: int = 0,
    confidence_level: float = 0.95,
    permutation_n: int = 0,
    random_state: int = 0,
) -> CorrelationMatrixResult:
    """Scan pairwise associations and attach BH-FDR q-values."""
    selected_columns = (
        [str(column) for column in columns]
        if columns is not None
        else [str(column) for column in data.columns]
    )
    missing_columns = [
        column for column in selected_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")
    work = data[selected_columns]
    if missing == "listwise":
        work = work.dropna()
    results = []
    for pair_index, (left, right) in enumerate(combinations(selected_columns, 2)):
        result = correlation(
            left,
            right,
            data=work,
            method=method,
            missing="raise" if missing == "raise" else "pairwise",
            bootstrap_n=bootstrap_n,
            confidence_level=confidence_level,
            permutation_n=permutation_n,
            random_state=int(random_state) + pair_index,
        )
        results.append(result)
    qvalues, rejected = benjamini_hochberg(
        [result.p_value for result in results],
        alpha=alpha,
    )
    for result, qvalue, reject in zip(results, qvalues, rejected):
        result.q_value = qvalue
        result.fdr_reject_null = reject if qvalue is not None else None
    warnings = []
    untested = sum(result.p_value is None for result in results)
    if untested:
        warnings.append(
            f"{untested} pair(s) lacked meaningful p-values and were excluded from BH-FDR."
        )
    return CorrelationMatrixResult(
        columns=selected_columns,
        results=results,
        alpha=alpha,
        missing_data=missing,
        random_state=int(random_state),
        warnings=warnings,
    )


class CorrelationSuite:
    """Bound-data convenience API for association analysis."""

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        missing: MissingPolicy = "pairwise",
        bootstrap_n: int = 0,
        confidence_level: float = 0.95,
        random_state: int = 0,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("CorrelationSuite expects a pandas DataFrame.")
        self.data = data
        self.missing = missing
        self.bootstrap_n = int(bootstrap_n)
        self.confidence_level = float(confidence_level)
        self.random_state = int(random_state)

    def correlation(
        self,
        x: str,
        y: str,
        *,
        method: str = "auto",
        controls: Optional[Sequence[str]] = None,
        weights: Optional[str] = None,
        cluster: Optional[str] = None,
        bootstrap_n: Optional[int] = None,
        permutation_n: int = 0,
    ) -> CorrelationResult:
        return correlation(
            x,
            y,
            data=self.data,
            method=method,
            controls=controls,
            weights=weights,
            cluster=cluster,
            missing=self.missing,
            bootstrap_n=(
                self.bootstrap_n if bootstrap_n is None else int(bootstrap_n)
            ),
            confidence_level=self.confidence_level,
            permutation_n=permutation_n,
            random_state=self.random_state,
        )

    def matrix(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        method: str = "auto",
        alpha: float = 0.05,
        bootstrap_n: Optional[int] = None,
        permutation_n: int = 0,
    ) -> CorrelationMatrixResult:
        return correlation_matrix(
            self.data,
            columns=columns,
            method=method,
            missing=self.missing,
            alpha=alpha,
            bootstrap_n=(
                self.bootstrap_n if bootstrap_n is None else int(bootstrap_n)
            ),
            confidence_level=self.confidence_level,
            permutation_n=permutation_n,
            random_state=self.random_state,
        )
