"""SciPy and statsmodels adapters."""

from __future__ import annotations

from typing import Any

from autocausal.integrations.adapters.base import (
    LazyAdapter,
    as_2d_controls,
    bounded_int,
    residualize,
)


class ScipyAdapter(LazyAdapter):
    id = "scipy.stats"
    integration_id = "scipy"
    module_name = "scipy"
    package_name = "scipy"
    capabilities = ("stats.test", "stats.partial_correlation")

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability == "stats.partial_correlation":
            return self.partial_correlation(**kwargs)
        if capability == "stats.test":
            return self.statistical_test(**kwargs)
        raise KeyError(capability)

    @staticmethod
    def partial_correlation(
        *,
        x: Any,
        y: Any,
        controls: Any = None,
        method: str = "pearson",
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np
        from scipy import stats

        x_values = np.asarray(x, dtype=float).reshape(-1)
        y_values = np.asarray(y, dtype=float).reshape(-1)
        if len(x_values) != len(y_values):
            raise ValueError("x and y must have equal length")
        matrix = as_2d_controls(controls, len(x_values))
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        if matrix.shape[1]:
            finite &= np.isfinite(matrix).all(axis=1)
        rx, _ = residualize(x_values[finite], matrix[finite])
        ry, _ = residualize(y_values[finite], matrix[finite])
        selected = str(method).lower()
        if selected == "spearman":
            statistic, pvalue = stats.spearmanr(rx, ry)
        elif selected == "pearson":
            statistic, pvalue = stats.pearsonr(rx, ry)
        else:
            raise ValueError("method must be pearson or spearman")
        return {
            "method": f"partial_{selected}",
            "correlation": float(statistic),
            "pvalue": float(pvalue),
            "n": int(len(rx)),
            "n_controls": int(matrix.shape[1]),
            "caveat": "Association only; no causal identification.",
        }

    @staticmethod
    def statistical_test(
        *,
        method: str,
        x: Any,
        y: Any = None,
        alternative: str = "two-sided",
        equal_var: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import numpy as np
        from scipy import stats

        name = str(method).lower()
        x_values = np.asarray(x)
        if x_values.size > 1_000_000:
            raise ValueError("statistical test input exceeds 1,000,000 values")
        y_values = np.asarray(y) if y is not None else None
        if y_values is not None and y_values.size > 1_000_000:
            raise ValueError("statistical test input exceeds 1,000,000 values")
        if name in ("ttest_ind", "welch"):
            if y_values is None:
                raise ValueError("ttest_ind requires y")
            result = stats.ttest_ind(
                x_values.astype(float),
                y_values.astype(float),
                equal_var=bool(equal_var),
                alternative=alternative,
                nan_policy="omit",
            )
        elif name == "ttest_rel":
            if y_values is None:
                raise ValueError("ttest_rel requires y")
            result = stats.ttest_rel(
                x_values.astype(float),
                y_values.astype(float),
                alternative=alternative,
                nan_policy="omit",
            )
        elif name in ("mannwhitneyu", "mann_whitney"):
            if y_values is None:
                raise ValueError("mannwhitneyu requires y")
            result = stats.mannwhitneyu(
                x_values.astype(float),
                y_values.astype(float),
                alternative=alternative,
            )
        elif name in ("pearsonr", "spearmanr"):
            if y_values is None:
                raise ValueError(f"{name} requires y")
            result = getattr(stats, name)(
                x_values.astype(float),
                y_values.astype(float),
            )
        elif name in ("chi2", "chi2_contingency"):
            result = stats.chi2_contingency(
                x_values,
                correction=bool(kwargs.get("correction", True)),
            )
            return {
                "method": "chi2_contingency",
                "statistic": float(result.statistic),
                "pvalue": float(result.pvalue),
                "dof": int(result.dof),
            }
        elif name == "shapiro":
            sample = x_values.astype(float)
            sample = sample[np.isfinite(sample)][:5_000]
            result = stats.shapiro(sample)
        else:
            raise ValueError(
                "method must be ttest_ind, ttest_rel, mannwhitneyu, pearsonr, "
                "spearmanr, chi2_contingency, or shapiro"
            )
        return {
            "method": name,
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
        }


class StatsmodelsAdapter(LazyAdapter):
    id = "statsmodels.regression"
    integration_id = "statsmodels"
    module_name = "statsmodels"
    package_name = "statsmodels"
    capabilities = ("stats.robust_covariance", "stats.partial_correlation")

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability == "stats.partial_correlation":
            return ScipyAdapter.partial_correlation(**kwargs)
        if capability == "stats.robust_covariance":
            return self.robust_covariance(**kwargs)
        raise KeyError(capability)

    @staticmethod
    def robust_covariance(
        *,
        y: Any,
        x: Any,
        cov_type: str = "HC3",
        groups: Any = None,
        maxlags: int = 1,
        return_model: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        import numpy as np
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.stattools import durbin_watson

        y_values = np.asarray(y, dtype=float).reshape(-1)
        x_values = np.asarray(x, dtype=float)
        if x_values.ndim == 1:
            x_values = x_values.reshape(-1, 1)
        if len(y_values) != len(x_values) or len(y_values) > 1_000_000:
            raise ValueError("x/y shapes are incompatible or exceed row cap")
        design = sm.add_constant(x_values, has_constant="add")
        finite = np.isfinite(y_values) & np.isfinite(design).all(axis=1)
        model = sm.OLS(y_values[finite], design[finite]).fit()
        selected = str(cov_type).upper()
        allowed = {"HC0", "HC1", "HC2", "HC3", "HAC", "CLUSTER", "NONROBUST"}
        if selected not in allowed:
            raise ValueError(f"cov_type must be one of {sorted(allowed)}")
        if selected == "NONROBUST":
            robust = model
        elif selected == "HAC":
            robust = model.get_robustcov_results(
                cov_type="HAC",
                maxlags=bounded_int(
                    maxlags,
                    default=1,
                    minimum=1,
                    maximum=100,
                    name="maxlags",
                ),
            )
        elif selected == "CLUSTER":
            if groups is None:
                raise ValueError("cluster covariance requires groups")
            robust = model.get_robustcov_results(
                cov_type="cluster",
                groups=np.asarray(groups)[finite],
            )
        else:
            robust = model.get_robustcov_results(cov_type=selected)
        lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(
            model.resid,
            model.model.exog,
        )
        output: dict[str, Any] = {
            "method": "statsmodels_ols",
            "cov_type": selected,
            "params": [float(item) for item in robust.params],
            "standard_errors": [float(item) for item in robust.bse],
            "pvalues": [float(item) for item in robust.pvalues],
            "n": int(model.nobs),
            "r_squared": float(model.rsquared),
            "condition_number": float(model.condition_number),
            "durbin_watson": float(durbin_watson(model.resid)),
            "breusch_pagan": {
                "lm": float(lm),
                "lm_pvalue": float(lm_pvalue),
                "fvalue": float(fvalue),
                "f_pvalue": float(f_pvalue),
            },
            "caveat": (
                "Regression diagnostics do not establish causal identification."
            ),
        }
        if return_model:
            output["model"] = robust
        return output


__all__ = ["ScipyAdapter", "StatsmodelsAdapter"]
