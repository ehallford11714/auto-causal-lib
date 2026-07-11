# Causal inference methods and support boundaries

AutoCausal 0.13 separates five activities:

1. **description/association** measures observed dependence;
2. **discovery** proposes graph structure or temporal precedence;
3. **identification** links a target estimand to observed data under explicit
   design and domain assumptions;
4. **estimation** computes an estimate and uncertainty for that estimand;
5. **refutation/sensitivity** probes how conclusions change.

Prediction is a sixth, separate activity. High predictive accuracy,
correlation, feature importance, or discovery stability is never causal
identification evidence by itself.

This release provides production-oriented safeguards, not production
certification or an automated identification oracle.

## Unified API

```python
from autocausal.inference import CausalSpec, AutoInference
from autocausal.production import ProductionPolicy

spec = CausalSpec(
    treatment="treatment",
    outcome="outcome",
    confounders=["age", "baseline"],
    estimand="ATE",
    assumptions={"sutva": "reviewed by study owner"},
)

result = AutoInference(
    spec,
    mode="production",
    policy=ProductionPolicy.strict(),
    random_state=42,
).fit(df, method="aipw")

print(result.report())
```

Every `CausalInferenceResult` exposes:

- method and estimand;
- estimate, robust/influence-function SE, 95% CI, and p-value;
- complete sample summary (never raw rows);
- documented assumptions and warnings;
- method-specific diagnostics;
- run/method/instrument provenance;
- gate-derived evidence grade;
- structured `GateReport` and privacy-safe `RunManifest`;
- `to_fabric_metadata()` preserving estimand, diagnostics, provenance, and
  gates.

`AutoCausal.infer(spec=..., method=...)` is the fluent wrapper.
`AutoCausal.estimate(...)` remains compatible and records a
`unified_method_mapping` in its manifest.

## Planner behavior

```python
from autocausal.inference import AutoInferencePlanner

for candidate in AutoInferencePlanner(spec).recommend(df):
    print(candidate.method, candidate.status, candidate.missing_fields)
```

The planner recommends candidates from explicit design metadata. It does not
run them. `fit(..., method="auto")` is allowed only as an explicitly
exploratory convenience; production refuses automatic method selection.

## Native effect-estimation portfolio

| Method | Status | Estimand / required fields | Main assumptions | Valid use | Invalid or unsupported use |
|---|---|---|---|---|---|
| regression adjustment | native | Treatment coefficient / `treatment`, `outcome`, pre-treatment `confounders` | conditional exchangeability, positivity, consistency/SUTVA, correct conditional mean | transparent baseline with HC1 robust SE | post-treatment controls, arbitrary causal interpretation of a misspecified regression |
| propensity-score adjustment | native | ATE-style treatment coefficient conditional on estimated score / binary treatment | exchangeability, correct propensity and outcome-in-score specification, positivity | diagnostics and sensitivity baseline | treating propensity fit as proof of balance/exchangeability |
| stabilized IPTW | native | ATE after explicit clipping policy / binary treatment | exchangeability, correct propensity, positivity, consistency | weighting with overlap, effective-sample-size, weight, and SMD diagnostics | extrapolation under failed positivity; hiding trimming |
| cross-fitted AIPW | native | ATE / binary treatment and confounders | exchangeability, positivity, consistency; propensity or outcome nuisance model adequate | fold-local logistic propensity + ridge outcome nuisance fits; influence-function SE | “doubly robust” as protection from unmeasured confounding |
| nearest-neighbor propensity matching | native | ATT among matched treated / binary treatment | exchangeability, common support, adequate propensity/caliper | sensitivity analysis with before/after SMD | population ATE claim; ignoring reused controls or match uncertainty |
| observed-instrument 2SLS | native | structural coefficient / LATE under assumptions; explicit observed `instrument` | relevance, exclusion, independence, SUTVA; monotonicity for LATE | linear IV with robust sandwich SE, partial first-stage F, optional Sargan nR² over-ID diagnostic | synthetic Z in production; treating F or over-ID non-rejection as proof of validity |
| difference-in-differences | native | treated×post ATT / `unit`, `time`, `treatment`, `post`, `outcome` | parallel untreated trends, no anticipation, no time-varying differential confounding/spillovers | repeated units with cluster-robust SE and differential-pretrend diagnostic | no pre-period support; claiming pretrend non-rejection proves parallel trends |
| panel fixed effects / within | native | within-unit treatment coefficient / `unit`, `time` | strict exogeneity, within-unit treatment variation, no time-varying omitted causes | unit demeaning, time fixed effects when tractable, unit-clustered SE | time-invariant treatment; claim that FE removes time-varying confounding |
| sharp local-linear RDD | native | local cutoff discontinuity / `running`, explicit `cutoff`, treatment, outcome | continuity, no precise sorting, valid local form/bandwidth, SUTVA | triangular-kernel local linear fit, side counts, assignment and ±25% bandwidth stability | fuzzy RDD, manipulated running variable, unsupported bandwidth extrapolation |
| interrupted time series | native | immediate level change; slope change in diagnostics / ordered `time`, `post` or treatment indicator | stable counterfactual trend, no coincident event, adequate segmented trend | segmented regression with Newey-West/HAC covariance and residual warning | causal claim when another intervention/event coincides or pretrend is inadequate |

Native code depends on NumPy/pandas. Propensity, matching, and AIPW methods use
scikit-learn. SciPy improves distribution p-values but the core normal
fallback remains explicit. Installed dependency licenses remain their upstream
licenses; review the exact locked versions for deployment.

### Robust uncertainty shipped

- OLS: HC1 heteroskedasticity-robust covariance.
- IPTW: HC1 covariance on the weighted treatment model; weight-estimation
  uncertainty is a limitation.
- AIPW: cross-fitted influence-function standard error.
- matching: paired-difference SE, explicitly labeled incomplete for propensity
  fitting and reused controls.
- 2SLS: heteroskedasticity-robust sandwich covariance.
- DiD/panel FE: unit-clustered covariance.
- ITS: Newey-West/HAC covariance at the policy lag.
- RDD: HC1 covariance after triangular local weighting.

## Optional adapters

| Adapter | Status | Capability | Dependency / license note |
|---|---|---|---|
| DoubleML | optional adapter | PLR/DML ATE | `auto-causal-lib[causal-extra]`; upstream package/license, verify locked version |
| EconML LinearDML | optional adapter | DML/CATE | causal-extra; upstream MIT at time of writing, verify locked version |
| EconML CausalForestDML | optional adapter | forest CATE | causal-extra; optional heavy dependency |
| DoWhy | optional adapter | identification/refutation through existing refute surface | causal-extra; no fake native point-estimator claim |
| causal-learn PC/GES/FCI | optional adapter | discovery only | causal-extra; does not estimate an intervention effect |
| LiNGAM / DirectLiNGAM | optional adapter | discovery only under non-Gaussian linear assumptions | causal-extra |
| gCastle NOTEARS adapter | optional adapter | discovery only | causal-extra; verify package/runtime license before deployment |

An unavailable adapter soft-skips with a structured insufficient result in
exploratory mode. Production fails closed when the requested adapter does not
run. No optional adapter silently falls back to native regression.

Tigramite and Granger are cataloged as temporal-discovery concepts, not effect
estimators. They are not wired in 0.13. Tigramite's exact package/version
license must be reviewed before any adapter is added.

## Planned/deferred: no literal support claim

| Technique | Status | Why deferred |
|---|---|---|
| mediation / natural direct-indirect effects | planned/deferred | cross-world assumptions, exposure-mediator interaction, and sensitivity contract are not implemented |
| synthetic control / augmented synthetic control | planned/deferred | donor-pool, pre-fit, placebo, and inference policy need a validated implementation |
| TMLE | planned/deferred | no maintained adapter and targeting/variance contract selected |
| front-door adjustment | planned/deferred | mediator identification checks are not implemented |
| proximal causal inference | planned/deferred | proxy/bridge completeness assumptions require specialized diagnostics |
| survival causal inference | planned/deferred | censoring, competing-risk, and time-varying treatment support is absent |
| marginal structural models | planned/deferred | longitudinal treatment/censoring weights and history design are absent |
| g-methods / g-computation beyond simple regression | planned/deferred | longitudinal intervention regimes are absent |
| regression kink / fuzzy RDD | planned/deferred | native method is sharp RDD only |
| staggered-adoption heterogeneous DiD | planned/deferred | native DiD is not Callaway–Sant'Anna or Sun–Abraham |
| Bayesian causal models / BART | planned/deferred | posterior diagnostics and maintained adapter not selected |
| causal survival forests | planned/deferred | time-to-event estimand and censoring support absent |
| interference/network causal effects | planned/deferred | native gates only surface a SUTVA/interference caveat |
| CausalML uplift | intentionally deferred | stale-package risk; no package was added |

Requesting a deferred method raises `NotImplementedError` or a capability error;
it never returns a placeholder estimate.

## Production gates by design

All methods gate explicit treatment/outcome, required columns, sample size,
missing complete-case fraction, leakage, near-zero variance,
multicollinearity, and a visible SUTVA/interference caveat. Additional gates:

- propensity methods: overlap/positivity;
- IPTW/matching: post-adjustment SMD and extreme-weight policy;
- IV: observed provenance and first-stage F; exclusion and independence remain
  `unverified`;
- DiD: minimum pre-periods and differential-pretrend diagnostic;
- RDD: observations on both sides, sharp assignment, bandwidth sensitivity;
- ITS: residual autocorrelation warning with HAC covariance;
- all methods: finite estimate and deterministic manifest.

Production raises `ProductionGateError` or `EvidenceGateError` with gate
records, remediation, partial result where available, and manifest. Exploratory
mode continues with visible warnings.

Evidence grades are gate-derived:

- `supported`: required design/statistical/post-estimation gates passed;
- `exploratory`: exploratory run without blocking failures;
- `insufficient`: a requested adapter skipped or an evidence gate failed;
- `refuted`: set by refutation outcomes elsewhere.

`supported` deliberately does not mean “identified.”

## Unified production pipeline

```python
from autocausal.production import ProductionPolicy, run_production_pipeline

# Review-only check: recommends candidates but does not select one.
check = run_production_pipeline(
    df,
    treatment="treatment",
    outcome="outcome",
    confounders=["age", "baseline"],
    policy=ProductionPolicy.strict(),
    random_state=42,
)
assert check.status == "review_required"

# Run only after a reviewer selects and justifies the design.
run = run_production_pipeline(
    df,
    treatment="treatment",
    outcome="outcome",
    confounders=["age", "baseline"],
    method="aipw",
    policy=ProductionPolicy.strict(),
    random_state=42,
)
print(run.gates.report())
print(run.inference_result.report())
```

The pipeline aligns policy-aware cleanse, AutoEDA gate inputs, optional AutoML,
the inference planner, the selected estimator, gates, and stage spans. Raw
frames are omitted from serialization.
