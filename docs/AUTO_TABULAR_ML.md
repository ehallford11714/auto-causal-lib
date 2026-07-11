# AutoTabularML

`AutoTabularML` is a bounded predictive-model selection system. It extends the
predictive surface alongside `autocausal.ml.AutoML` and the existing KPI/torch
loop without replacing either API. The core `AutoML` path is the compact
production-gate evaluator; `AutoTabularML` adds a fitted selected pipeline,
gradient-boosting candidate, richer metric ledger, held-out permutation
importance, subgroup summaries, and guarded persistence.

```python
from autocausal.automl import AutoTabularML

report = AutoTabularML(
    mode="production",
    policy=ac.policy,  # optional; accepts the shared policy without hard imports
    random_state=17,
    cv=5,
    max_candidates=4,
).run(
    df,
    target="survived",
    group_column="family_id",
    subgroup_columns=["sex", "passenger_class"],
)

print(report.report())
pipeline = report.selected_pipeline
```

Task and splits

Regression, binary classification, and multiclass classification are inferred
from an explicit target. Production mode requires `target=`. Split strategy is
selected in this order when `split_strategy="auto"`:

1. forward-chaining time splits when `time_column` is supplied;
2. nonoverlapping group splits when `group_column` is supplied;
3. stratified random splits for classification;
4. deterministic random folds for regression.

Every split is audited for row overlap. Group and time columns are excluded
from predictive features by default.

Leakage-safe preprocessing

Numeric imputation/scaling, categorical imputation/one-hot encoding, datetime
components, and the estimator live in one sklearn `Pipeline` with a
`ColumnTransformer`. The pipeline is cloned and fitted separately inside each
training fold. Raw text and ID-like columns are excluded; use AutoNLP's
fold-safe transformer for reviewed text features.

Candidate set and ledger

The bounded defaults are:

- dummy baseline;
- ridge or logistic regression;
- random forest;
- histogram gradient boosting.

Candidates are ranked by appropriate cross-validation performance with fixed
complexity and expected-latency penalties. Reports include fold values,
variance, 95% confidence summaries, fit/predict latency, errors, ranks, and the
selection reason. Binary probability models report Brier score and ROC AUC;
multiclass models report weighted one-vs-rest AUC when defined.

Permutation importance is computed on a held-out CV fold and remains
predictive—not causal. Correlated features may split or mask importance.

Policy hooks

`gate_inputs` accepts plain mappings or policy-like objects without importing
the production-policy module. Hooks cover sample size, leakage hints,
calibration, CV instability, class imbalance, subgroup gaps, and resource
limits. Existing gate results can be appended with `gate_results=`.
`enforce_gates=True` raises `AutoMLGateError` with the completed report.

Persistence

```python
artifact, manifest = report.save_model("artifacts/model.joblib")

from autocausal.automl import load_trusted_model
model = load_trusted_model(artifact, trusted=True)
```

The sidecar manifest records package versions and a SHA-256 hash. Joblib
artifacts can execute Python code during deserialization; loading is refused
unless the caller explicitly sets `trusted=True`. Never load an artifact from
an untrusted source.

Epistemic boundary and integration

Prediction quality and feature importance do not identify causal effects. The
stable entry point is `from autocausal.automl import AutoTabularML`. A fluent
`AutoCausal.tabular_ml()` method was deferred while the shared facade and
production policies are changing concurrently; it can later delegate to
`AutoTabularML(self).run(...)`.
