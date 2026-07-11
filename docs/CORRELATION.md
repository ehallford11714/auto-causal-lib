# Correlation and association methods

`autocausal.correlation` provides typed descriptive association estimates.

> **Association is not causation.** A coefficient, p-value, confidence interval,
> mutual-information score, or FDR result does not identify an intervention
> effect, establish direction, validate an instrument, or remove confounding.

## Public API

```python
from autocausal.correlation import (
    CorrelationSuite,
    correlation,
    correlation_matrix,
)

one = correlation(
    "exposure",
    "outcome",
    data=df,
    method="partial",
    controls=["age", "baseline_score"],
    bootstrap_n=500,
    random_state=42,
)

scan = correlation_matrix(
    df,
    columns=["exposure", "outcome", "segment", "converted"],
    method="auto",
    alpha=0.05,
    random_state=42,
)
print(scan.report())
```

`CorrelationSuite(df, ...)` binds a frame and exposes `.correlation(...)` and
`.matrix(...)`. `AutoCausal.correlate(...)` is the fluent wrapper.

Every `CorrelationResult` records the measure, coefficient/effect size,
bootstrap interval, meaningful analytic/permutation p-value when available,
complete-case `n`, missing-data policy, controls, weights, cluster column,
assumptions, warnings, and deterministic seed. Matrix scans attach
Benjamini-Hochberg q-values only to pairs with meaningful p-values.

## Shipped native portfolio

| Data roles | Method | Status | Inference and limitations |
|---|---|---|---|
| continuous–continuous | Pearson | native | Linear association. Analytic p-value assumes independent pairs; bootstrap CI available. |
| continuous–continuous | weighted Pearson | native | Non-negative observation weights. Analytic p-value is deliberately omitted; bootstrap CI available. |
| continuous–continuous | Spearman | native | Rank/monotonic association. No weighted version. |
| continuous–continuous | Kendall tau-b | native via SciPy | Rank concordance with SciPy p-value. Raises a clear dependency error without SciPy. |
| continuous–continuous + controls | partial Pearson | native | Residualizes both variables on explicitly supplied controls, including one-hot categorical controls. This does not imply no unmeasured confounding. |
| robust continuous | 5% winsorized Pearson | native | Explicit fixed 5th/95th-percentile clipping. Prefer bootstrap CI because the analytic p-value treats clipping limits as fixed. |
| binary–continuous | point-biserial | native | Binary levels and orientation are recorded. Algebraically Pearson on binary coding. |
| binary–binary | phi | native | Signed 2×2 association. |
| categorical–categorical | bias-corrected Cramér's V | native | Chi-square p-value when SciPy is available; table shape is recorded but cell values are not. |
| categorical–continuous | correlation ratio eta / eta² | native | Eta is the main coefficient; eta² and one-way ANOVA metadata are included. Classical ANOVA p-value has normality/equal-variance assumptions. |
| nonlinear/mixed | mutual information | native via scikit-learn | k-nearest-neighbor/discrete estimator in nats. Direction used by the estimator is recorded; no analytic p-value. |
| nonlinear/mixed | normalized mutual information | native via scikit-learn | Continuous inputs are quantile-discretized; normalization/discretization metadata is explicit. |
| continuous nonlinear | distance correlation | native | Correct centered-distance implementation. Optional deterministic permutation p-value; O(n²), hard-limited to 5,000 rows per call. |

### `method="auto"`

Role inference is local to the two columns:

- two continuous variables: Pearson, or Spearman under severe skew/extreme
  tails;
- binary + continuous: point-biserial;
- two binary variables: phi;
- categorical + categorical: bias-corrected Cramér's V;
- categorical + continuous: correlation ratio;
- two continuous variables plus controls: partial correlation.

Auto selection is recorded in `metadata["auto_selected_method"]`. Choose an
explicit method when the estimand is prescribed.

### Missing values, weights, and repeated observations

- Single estimates use complete rows across the two variables, controls,
  weights, and cluster field.
- Matrix scans default to pairwise complete rows. `missing="listwise"` first
  restricts the whole scan; `missing="raise"` refuses missing data.
- `cluster=` switches bootstrap resampling from rows to whole clusters.
  Analytic p-values remain marked as i.i.d. and should not be reported as
  cluster robust.
- Weighted Pearson is implemented. Weighted rank, Cramér's V, eta, mutual
  information, and distance correlation are not silently approximated.
- Panel-aware uncertainty beyond cluster bootstrap is deferred.

## Support matrix: adapter/deferred methods

| Method family | Status | Reason / path forward |
|---|---|---|
| biweight midcorrelation | planned/deferred | No native implementation is shipped until edge cases and finite-sample behavior are validated against a maintained reference. Use explicit winsorized Pearson or an external reviewed implementation. |
| Theil's U / uncertainty coefficient | planned/deferred | Directional entropy conventions and bias correction need a reviewed contract. |
| polychoric / polyserial / tetrachoric correlation | planned/deferred | Requires latent-normal threshold estimation and robust convergence diagnostics. |
| repeated-measures correlation | planned/deferred | Requires explicit subject/time design and suitable degrees-of-freedom handling. |
| survey-weighted correlation with design-based SE | planned/deferred | Native weighted Pearson does not model strata/PSUs/replicate weights. |
| cluster-robust analytic covariance | planned/deferred | Cluster bootstrap is shipped; sandwich p-values are not. |
| maximal information coefficient (MIC) | optional external only | No dependency is installed or claimed by AutoCausal. |
| HSIC / kernel dependence | planned/deferred | Kernel/bandwidth/permutation policy is not yet standardized. |
| Hoeffding's D / distance covariance tests | planned/deferred | Distance correlation is shipped, not the broader test portfolio. |
| Goodman–Kruskal gamma/lambda, Somers' D | planned/deferred | Ordinal/tie semantics need explicit typed roles. |
| canonical correlation / RV coefficient | planned/deferred | Multivariate association and regularization are outside the current pairwise result contract. |

There are no placeholder return values for deferred methods. Requesting an
unknown method raises `ValueError`.

## FDR and interpretation

`correlation_matrix` applies Benjamini-Hochberg correction over non-missing
p-values in that scan. It records `q_value` and `fdr_reject_null`; measures with
no meaningful p-value are excluded rather than treated as p=1.

FDR controls an expected false-discovery proportion under its assumptions. It
does not:

- make effect sizes important;
- correct selection performed before the scan;
- establish direction or causality;
- validate conditioning choices in partial correlation;
- repair dependence from repeated, clustered, or temporal sampling.

Use association output in AutoEDA/AutoMine as descriptive or predictive
evidence only. Production causal gates explicitly record
`correlation_used_as_identification=False`.
