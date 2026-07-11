# AutoCausal integrations

AutoCausal uses a lazy integration registry, a curated adapter set, and an
opt-in plugin API. It does **not** import every scientific package, install
dependencies at runtime, load third-party entry points automatically, enable
telemetry, or treat package availability as causal validity.

## Public API

```python
from autocausal.integrations import (
    CapabilityRouter,
    CapabilitySpec,
    IntegrationRegistry,
    IntegrationSpec,
    build_install_plan,
    get_capability,
    integration_status,
    invoke_capability,
    list_integrations,
)
```

Normal catalog/status calls use `importlib.util.find_spec` and
`importlib.metadata`; optional packages are imported only on invocation or an
explicit `deep=True` health probe.

```python
status = integration_status("statsmodels")
decision = CapabilityRouter(...).route(
    "causal.estimate.ate",
    policy={"production": True},
    context={"design": "dml", "design_validated": False},
)
plan = build_install_plan(profile="all-safe", hardware="cpu")
```

`RoutingDecision` records all candidates, selected integration/adapter,
versions, fallbacks, policy reasons, escalation, and caveats. A selected causal
estimator still requires a defensible design, overlap/exchangeability checks,
evidence gates, and human review.

## Catalog matrix

Status meanings:

- **native adapter**: maintained in-process fallback or existing AutoCausal engine.
- **external adapter**: real, lazy, bounded adapter covered by contract tests.
- **awareness-only**: install/status documentation only; no capability is claimed.
- **deprecated/blocked**: never routed unless replaced by a reviewed plugin/catalog change.

| Package | Category | Status | Implemented capabilities | Profile | License / runtime | Deterministic fallback or deferral reason |
|---|---|---|---|---|---|---|
| auto-causal-lib | native | native adapter | partial correlation; bounded sklearn estimators; hashed TF-IDF; cosine search; PC-lite; cross-fitted AIPW; chart/DAG specs | all-safe | MIT / CPU | terminal local fallback; causal outputs remain assumption-dependent |
| numpy | statistics | awareness-only | — | stats | BSD-3 / CPU | foundation used by native adapters |
| scipy | statistics | external adapter | `stats.test`, `stats.partial_correlation` | stats | BSD-3 / CPU | native residual partial correlation |
| statsmodels | statistics | external adapter | `stats.robust_covariance`, `stats.partial_correlation` | stats | BSD-3 / CPU | SciPy/native |
| pingouin | statistics | awareness-only | — | research opt-in | GPL-3 / CPU | license-gated |
| linearmodels | econometrics | awareness-only | — | stats, causal-extra | NCSA / CPU | IV/panel API requires a design-specific adapter |
| arch | econometrics | awareness-only | — | stats | NCSA / CPU | volatility models are not generic causal estimators |
| pymc | statistics | awareness-only | — | research | Apache-2 / CPU, optional GPU | sampling/resource policy not yet standardized |
| arviz | statistics | awareness-only | — | research | Apache-2 / CPU | retained for Bayesian diagnostics awareness |
| lifelines | statistics | awareness-only | — | stats | MIT / CPU | survival estimands need a dedicated contract |
| scikit-posthocs | statistics | awareness-only | — | stats | MIT / CPU | no generic production adapter claimed |
| patsy | statistics | awareness-only | — | stats | BSD-2 / CPU | formula evaluation requires caller-controlled namespaces |
| scikit-learn | ML | external adapter | preprocessing; classifier/regressor factory; bounded CV; metrics; TF-IDF embeddings | automl, all-safe | BSD-3 / CPU | native bounded estimators / hashed TF-IDF |
| xgboost | ML | external adapter | CPU-bounded classifier/regressor/estimator factory | automl | Apache-2 / native CPU | scikit-learn |
| lightgbm | ML | external adapter | CPU-bounded classifier/regressor/estimator factory | automl | MIT / native CPU | scikit-learn; wheel/compiler availability varies |
| catboost | ML | external adapter | CPU-bounded classifier/regressor/estimator factory | automl | Apache-2 / native CPU | scikit-learn |
| optuna | AutoML | external adapter | in-memory, trial/time-bounded tuning | automl, all-safe | MIT / CPU | manual bounded search |
| imbalanced-learn | ML | external adapter | fold-local resampling pipeline | automl, production | MIT / CPU | sklearn `class_weight`; never resample before splitting |
| shap | explainability | external adapter | sample/background-capped predictive explanation | automl | MIT / native CPU | permutation/model-native importance; SHAP is not causal attribution |
| joblib | ML | awareness-only | — | automl | BSD-3 / CPU | untrusted deserialization intentionally not exposed |
| torch | ML | awareness-only | — | research | BSD-3 / native CPU/GPU | existing SLM/model subsystem owns runtime policy |
| TensorFlow | ML | awareness-only | — | research opt-in | Apache-2 / native CPU/GPU | platform/binary complexity; no clean generic adapter |
| MLflow | operations | external adapter | explicitly enabled, scalar-only manifest logging | production | Apache-2 / local or network | disabled by default; remote URI needs network approval |
| NLTK | NLP | awareness-only | — | nlp-full | Apache-2 / CPU | existing AutoCausal NLP subsystem owns corpus resources |
| spaCy | NLP | external adapter | bounded local entity extraction | nlp-full, all-safe | MIT / native CPU | rule/NLTK features; models are never auto-downloaded |
| transformers | NLP | awareness-only | — | nlp-full | Apache-2 / native CPU/GPU | separately gated SLM subsystem |
| sentence-transformers | NLP | external adapter | local-only embeddings by default | nlp-full | Apache-2 / native CPU | sklearn TF-IDF, then native hashed TF-IDF |
| gensim | NLP | awareness-only | — | research opt-in | LGPL-2.1+ / CPU | copyleft policy-gated |
| stanza | NLP | awareness-only | — | nlp-full | Apache-2 / CPU | model download/runtime policy not standardized |
| textblob | NLP | awareness-only | — | nlp-full | MIT / CPU | convenience API only |
| keybert | NLP | awareness-only | — | nlp-full | MIT / CPU | depends on embedding model policy |
| FAISS | retrieval | external adapter | bounded in-memory local vector search | nlp-full | MIT / native CPU | NumPy cosine search; wheels vary by platform |
| Chroma | retrieval | external adapter | ephemeral local vector search only | nlp-full | Apache-2 / CPU, optional network | native cosine search; server/data-egress policy blocks default routing |
| causal-learn | causal | external adapter | PC plus existing PC/GES/FCI bridge | causal-extra | MIT / CPU | PC-lite with maturity note |
| DoWhy | causal | external adapter | existing refutation bridge | causal-extra | MIT / CPU | native placebo refuter |
| EconML | causal | external adapter | existing LinearDML/CausalForest bridge | causal-extra | MIT / native CPU | native AIPW/OLS association path; design review required |
| DoubleML | causal | external adapter | existing partially linear estimate bridge | causal-extra | BSD-3 / CPU | native AIPW/OLS association path; design review required |
| LiNGAM | causal | external adapter | existing DirectLiNGAM bridge | causal-extra | MIT / CPU | PC-lite; non-Gaussian/linearity assumptions must be reviewed |
| gCastle | causal | external adapter | existing NOTEARS bridge | causal-extra | Apache-2 / native CPU | PC-lite; optional deep backends remain out of scope |
| tigramite | causal | awareness-only | — | research opt-in | GPL-3 / CPU | explicit license approval required |
| pgmpy | causal/PGM | awareness-only | — | causal-extra | MIT / CPU | probabilistic graph support is not generic identification |
| CausalML | uplift | awareness-only | — | research | Apache-2 / native CPU | large compatibility surface; no stale uplift dependency is bundled |
| PySensemakr | sensitivity | awareness-only | — | causal-extra | MIT / CPU | API compatibility must be pinned before adapter promotion |
| linearmodels IV/panel | causal/econometrics | awareness-only | — | stats, causal-extra | NCSA / CPU | explicit estimand/design contract deferred |
| py-tetrad | causal | awareness-only | — | research opt-in | unknown/mixed / Java | Java and artifact/license management are caller-owned |
| CausalNex | causal | deprecated/blocked | — | none | Apache-2 / CPU | stale release line and incompatible modern Python/networkx constraints |
| CDT | causal | deprecated/blocked | — | none | GPL-3 / R, native, optional CUDA | stale dependency surface and external runtime requirements |
| Pandera | data quality | external adapter | caller-supplied schema validation | production, all-safe | MIT / CPU | AutoCausal QC |
| Great Expectations | data quality | awareness-only | — | research | Apache-2 / optional network | context/checkpoint execution can write or egress data |
| ydata-profiling | data quality | awareness-only | — | research | MIT / CPU | expensive HTML generation is not auto-routed |
| matplotlib | visualization | external adapter | headless AutoChart rendering | viz, production | PSF-based / CPU | AutoChart data/spec fallback |
| seaborn | visualization | awareness-only | — | viz | BSD-3 / CPU | AutoChart uses backend-neutral specs |
| Plotly | visualization | external adapter | AutoChart and local DAG rendering | viz, production | MIT / CPU | matplotlib, then spec/data fallback |
| Kaleido | visualization | external adapter | bounded static Plotly export | viz, production | MIT / native CPU | HTML/JSON chart artifact |
| networkx | visualization | external adapter | deterministic graph structure/layout | viz, production | BSD-3 / CPU | DAG spec-only fallback |
| graphviz | visualization | awareness-only | — | viz | Python MIT; native runtime separate | executable/runtime availability is platform-specific |
| Polars | data path | external adapter | local pandas/Arrow/Polars conversion | production, all-safe | MIT / native CPU | pandas |
| PyArrow | data path | external adapter | local Arrow/pandas conversion | production, all-safe | Apache-2 / native CPU | pandas |

Catalog entries with no implemented capability cannot be selected by
`CapabilityRouter`, even if installed.

## Routing policy

Routing considers:

- explicit integration selection;
- package presence, version metadata, and optional deep health probe;
- callable adapter registration;
- license allowlist;
- CPU/GPU/Java/R/native runtime policy;
- network and data-egress policy;
- row/memory budget;
- deterministic and production-ready requirements;
- capability-specific caveats and deterministic fallbacks.

Default examples:

- `stats.partial_correlation`: statsmodels → SciPy → native residual method.
- `ml.tabular_classifier`: sklearn → installed CPU boosted engines → native bounded estimator.
- `nlp.embeddings`: local sentence-transformers → sklearn TF-IDF → native hashed TF-IDF.
- `causal.discovery.pc`: causal-learn PC → PC-lite.
- `causal.estimate.ate`: DoubleML/EconML/native cross-fitted AIPW, reordered by explicit design context.
- `viz.dag`: networkx → Plotly → spec-only.

Copyleft, unknown-license, remote/data-egress, Java, R, and GPU integrations
are denied by default. Optional telemetry is disabled by default.

## Plugin safety

Third parties may advertise an entry point in the
`autocausal.integrations` group. Discovery returns only
`PluginDescriptor` metadata and never calls `EntryPoint.load()`. Loading
requires a `PluginLoadPolicy` with `allow_entry_point_loading=True` and an
explicit trusted entry-point name or distribution. The loaded object must be
an `IntegrationPlugin`; factories are not invoked automatically.

Applications can avoid entry points entirely and register an
`IntegrationSpec` plus an `IntegrationAdapter` explicitly.

## Install profiles

Available plans are `stats`, `automl`, `nlp-full`, `causal-extra`, `viz`,
`research`, `production`, and `all-safe`.

`all-safe` is the permissive-license, maintained, CPU-oriented set. It excludes
GPL/LGPL packages, CUDA-specific wheels, Java/R integrations, remote tracking
clients, TensorFlow, and blocked packages. Platform-sensitive native wheels are
reported as warnings. `build_install_plan` returns packages, constraints,
exclusions, warnings, and a suggested command, but never executes it.

```bash
python -m autocausal integrations list
python -m autocausal integrations status scipy --deep
python -m autocausal integrations doctor --json
python -m autocausal integrations plan --profile all-safe
```

## MCP and production manifests

The AgentHook/MCP registry exposes read-only integration operations:

- `autocausal_list_integrations`
- `autocausal_integration_status`
- `autocausal_route_capability`

The routing tool returns a decision and never invokes an adapter. When
`invoke_capability(..., manifest=run_manifest)` is used, AutoCausal records the
routing decision and detected package versions under
`manifest.config["integrations"]`; no raw rows or sample values are recorded.
