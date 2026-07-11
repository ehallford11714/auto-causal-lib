# Production vs exploratory

AutoCausalLib 0.13 is **alpha research software**. Production mode means
fail-closed execution, reproducibility, privacy-safe manifests, and honest
evidence labels. It does **not** turn heuristic discovery into causal
identification.

> **EPISTEMIC:** Associations are not effects. Synthetic instruments are demo
> plumbing (`identification=none`). An observed Z is still
> `identification=unverified` until a human-reviewed design establishes the IV
> assumptions.

## Copy-paste production pipeline

```python
from autocausal import AutoCausal, ProductionPolicy

policy = ProductionPolicy(
    qc="block",
    stability=True,
    bootstrap_n=20,
    ensemble=True,
    required_evidence="supported",
    required_engines=(),       # e.g. ("doubleml", "dowhy") when required
    min_stability=0.60,
    min_first_stage_f=10.0,
    max_rows=100_000,
    max_columns=200,
    max_rounds=3,
    max_seconds=300,
    allow_slm=False,
    allow_raw_data_external=False,
    random_state=42,
)

ac = AutoCausal.from_dataframe(
    df,
    source="warehouse:approved_study",
    mode="production",
    policy=policy,
)

# Production defaults apply QC block + ensemble + bootstrap stability.
# auto_instrument=True, allow_iv_fallback=True, and qc="off" are refused.
result = ac.discover(
    candidates={
        "treatment": ["treatment"],
        "outcome": ["outcome"],
        "instrument": ["observed_assignment_z"],
    }
)

estimate = ac.estimate(
    backend="builtin_2sls",
    y="outcome",
    d="treatment",
    z="observed_assignment_z",
)
refutation = ac.refute(
    edge=next(e for e in result.edges if e["type"] == "iv_2sls"),
    method="placebo",
    y="outcome",
    d="treatment",
)

print(result.report())
print(result.manifest.to_json())
replay_config = result.replay_config()  # no raw rows
```

For the aligned cleanse → EDA → statistics → optional AutoML → causal
inference path:

```python
from autocausal.production import ProductionPolicy, run_production_pipeline

# No method: perform checks and return reviewed recommendations without fitting.
check = run_production_pipeline(
    df,
    treatment="treatment",
    outcome="outcome",
    instrument="observed_assignment_z",
    confounders=["age", "baseline"],
    policy=ProductionPolicy.strict(),
    random_state=42,
)
assert check.status == "review_required"

# A human/domain reviewer selects and justifies the estimator.
run = run_production_pipeline(
    df,
    treatment="treatment",
    outcome="outcome",
    instrument="observed_assignment_z",
    confounders=["age", "baseline"],
    method="iv_2sls",
    policy=ProductionPolicy.strict(),
    random_state=42,
)
print(run.gates.report())
```

The same entry points are available as
`ac.production_check(treatment=..., outcome=...)` and
`ac.run_production(..., method=...)`.

When a gate fails, catch the typed exception. The exception includes safe gate
details, escalation recommendations, a manifest, and a partial result when one
exists:

```python
from autocausal import ProductionGateError

try:
    result = ac.discover()
except ProductionGateError as exc:
    print(exc.code)
    print(exc.to_dict())  # no raw frame
    partial = exc.partial_result
```

## Copy-paste exploratory pipeline

```python
from autocausal import AutoCausal, ProductionPolicy

ac = AutoCausal.from_dataframe(
    df,
    mode="exploratory",
    policy=ProductionPolicy.exploratory(random_state=42),
)

# Safe default still does not invent a Z.
result = ac.discover(use_iv=True)

# Explicit demo-only opt-in. Every resulting edge is tagged:
# auto_instrument=True, synthetic=True, identification="none",
# evidence_grade="insufficient", confidence <= 0.25.
demo = ac.discover(auto_instrument=True, qc="warn")
```

Exploratory mode may use deterministic soft fallback. Every fallback must be
recorded in notes/manifest warnings. It may not call synthetic output
identified.

## Modes and policy

`ProductionPolicy` (alias `RunPolicy`) is JSON serializable:

```python
blob = policy.to_json()
restored = ProductionPolicy.from_json(blob)
assert restored.to_dict() == policy.to_dict()
```

The policy profiles are aligned with `mode=`:

- `exploratory`: continue through nonfatal gates with visible warnings;
- `review`: run strong checks, suppress synthetic-IV shortcuts, and emit
  `escalate` decisions for unresolved human/domain review;
- `production`: fail closed on unresolved required gates.

Passing a policy whose profile conflicts with the requested mode is refused
instead of silently weakening the run.

Important fields:

- `qc`: `block` by default in production; production refuses `qc="off"`.
- `stability`, `bootstrap_n`, `ensemble`, `min_methods`: evidence support gates.
- `use_iv`, `require_observed_instrument`, `min_first_stage_f`: IV gates.
- `required_evidence`: `exploratory`, `supported`, `refuted`, or `insufficient`.
  There is intentionally no `identified` grade.
- `required_engines`: explicit engines that must be available. Missing required
  engines raise `ProductionGateError`; no silent soft backend.
- `fallback_behavior`: `fail` in production, `warn` in exploratory.
- `max_rows`, `max_columns`, `max_rounds`, `max_seconds`: resource limits.
- `allow_slm`: false in production. If explicitly enabled, heuristic/raw-text
  parsing still fails the structured parse gate.
- `allow_raw_data_external`: false by default for MCP/SLM payload boundaries.
- `redact_sample_values`: production reports redact imputation/fill values.
- `fail_on_pii`: optionally convert PII warnings into a blocking gate.
- `random_state`: propagated to discovery bootstrap, demo IV, built-in refutes,
  sensitivity, and supported estimators.

The aggregate policy composes:

- `DataQualityPolicy`: minimum rows/missingness, audited range constraints,
  and permissions for row/column drops, coercion, winsorization, and imputation;
- `StatisticalValidityPolicy`: EPV, VIF/condition number, overlap, FDR,
  stability/disagreement, power, and residual diagnostic thresholds;
- `CausalEvidencePolicy`: observed-IV/F threshold, balance/weight limits,
  cross-fit folds, DiD pre-periods, RDD side support, and HAC lag;
- `AutoMLRiskPolicy`: group/time split requirements, CV variance,
  calibration, class balance, and raw-prediction control;
- `PrivacyPolicy`: PII warnings/failure, sample redaction, and raw external
  payload permission;
- `OperationalPolicy`: engines, fallback behavior, SLM and resource limits.

Use `ProductionPolicy.with_overrides(..., reason=..., actor=...)` for auditable
exceptions. The override ledger is preserved in manifests and inference
provenance.

## Evidence and provenance

Each edge contains:

```text
evidence_grade: exploratory | supported | refuted | insufficient
identification: unverified | none
provenance:
  source_columns
  datasets
  discovery_methods
  bootstrap_stability
  estimator
  refuters
  instrument_origin: observed | synthetic | none
  run_id
  package_version
```

Production rejects synthetic IV, weak observed IV, and edges below the configured
evidence grade. Rejected edges move to `result.rejected_edges` for audit; they are
removed from `result.edges` and the accepted graph. `result.evidence_gates`
explains every failure.

`supported` means the configured stability/agreement or observed-IV strength
gate passed. It does **not** mean identified.

## Reproducibility and observability

`RunManifest` includes:

- run ID, package version, UTC timestamps, mode, policy, and discover config;
- deterministic `random_state`;
- data schema/shape and SHA-256 fingerprint (no raw/sample values);
- installed engine versions;
- privacy warnings;
- stage events (`policy`, `qc`, `impute`, `discover`, `evidence_gates`,
  `estimate`, `refute`) and durations;
- failed gates and warnings.

Replay against the attached or explicitly supplied frame:

```python
replayed = result.reproduce(df)
```

Fingerprint mismatch fails unless deliberately overridden with
`verify_fingerprint=False`.

## Security and privacy baseline

- PII-like column names and high-cardinality columns are reported without sample
  values.
- Production reports redact imputation fill values.
- Manifests contain only schema, shape, hashes, versions, config, and events.
- `ac.external_payload()` is summary-only.
- `ac.external_payload(include_frame=True)` raises `UnsafePayloadError` unless
  the policy explicitly allows raw external payloads.
- Production SLM use is denied by default, and raw DataFrames are not placed in
  the guide/create/interpret contexts.

This is a baseline, not a compliance certification. Perform your own access,
retention, encryption, and regulatory review.

## Doctor

```bash
python -m autocausal doctor --production
python -m autocausal doctor --production --json
```

The production checklist verifies safe IV defaults, policy serialization, QC,
resource controls, required/optional engine availability, and version.

Install optional causal engines when your policy requires them:

```bash
pip install "auto-causal-lib[causal-extra]"
```

## Migration from 0.12

Behavior change:

```python
# 0.12 and earlier could synthesize auto_instrument_z by default.
# 0.13: safe default; no synthetic Z.
result = ac.discover()  # auto_instrument=False

# Explicit demo-only opt-in:
demo = ac.discover(auto_instrument=True, mode="exploratory")
```

Other production changes:

- `mode="production"` now applies a typed `ProductionPolicy`.
- QC off, synthetic/weak IV, disabled required stability/ensemble, missing
  required engines, unsafe SLM, resource overruns, and estimator/refuter soft
  skips fail with structured exceptions.
- `estimate` and `refute` require explicit `y` and `d`; production IV estimate
  also requires observed `z`.
- Production may remove insufficient edges or raise
  `EvidenceGateError` when none pass.

## Maturity

- Core imputation/discovery: stable-alpha.
- Heuristic PC-lite/ensemble: alpha.
- Numpy 2SLS with observed Z: alpha.
- `auto_instrument`: demo-only.
- DoubleML/EconML/DoWhy adapters: soft-optional beta.
- GRAIL offline implementation: stub/scaffold.
- Physics, insight, and agentic loops: alpha/demo.

Reports and README must retain these labels.

## Shipped in 0.13

- Typed policy, mode integration, JSON round-trip, and public exports.
- Safe `auto_instrument=False` default and production synthetic-IV refusal.
- Structured evidence/provenance, rejected-edge audit, and honest reports.
- Unified random state, private data fingerprint, run manifest, replay config,
  deterministic `result.reproduce()`.
- Structured gate/resource/privacy/unsafe-payload exceptions.
- Fail-closed QC, required engines, weak IV, insufficient edges, backend soft
  skips, SLM parse, and resource limits.
- Lightweight redacted stage spans.
- PII/high-cardinality warnings and raw external payload gate.
- Doctor production checks and contract/integration tests.
- Typed association package with data-role auto selection, deterministic
  bootstrap/cluster bootstrap, and BH-FDR matrix scans.
- Unified causal spec/result/planner with ten native estimators and
  method-specific gates. See [CAUSAL_INFERENCE.md](CAUSAL_INFERENCE.md).
- Policy-aware reversible/dry-run cleanse, before/after fingerprints, redacted
  action ledger, range/schema checks, and train-only split transformation hook.
- AutoEDA machine-readable gate inputs, typed descriptive associations,
  imbalance/missingness/design-readiness sections.
- Modest AutoML with fold-local preprocessing, deterministic group/time CV,
  baseline/candidate ledger, calibration, leakage, imbalance, and stability
  gates. Predictive output is never causal evidence.

## Deferred (do not claim shipped)

- **Mandatory external CausalIV design-object handoff** before every production
  IV edge. 0.13 enforces observed Z + first-stage threshold, but cannot validate
  exclusion/independence; edges remain `identification=unverified`.
- **Durable/pickle sessions.** Manifests/results/config serialize; raw frames are
  intentionally not persisted.
- Formal weak-IV robust intervals, Anderson-Rubin tests, exclusion tests, and
  domain-specific identification proofs.
- Hard process-level cancellation for long native-library calls. 0.13 checks
  elapsed time between stages and returns structured partial diagnostics.
- Full distributed tracing/metrics exporters. 0.13 provides in-process redacted
  events only.
- Compliance-grade PII classification/DLP. 0.13 is name/cardinality-based.
- Biweight/polychoric/repeated-measures/survey-design correlations; see the
  honest support matrix in [CORRELATION.md](CORRELATION.md).
- Mediation, synthetic control, TMLE, front-door, proximal, longitudinal
  g-methods, and survival causal inference. No placeholder estimates are
  shipped; reasons are cataloged in [CAUSAL_INFERENCE.md](CAUSAL_INFERENCE.md).
- Fuzzy RDD, staggered-adoption heterogeneous DiD, formal McCrary testing, and
  weak-IV-robust confidence sets.
