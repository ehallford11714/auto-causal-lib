# AutoViz

`AutoVizSuite` plans diagnostics from a frame plus available causal, panel,
model, and gate metadata. It is a planner, not a plotting wrapper.

```python
from autocausal.autoviz import AutoVizSuite

report = AutoVizSuite(
    df,
    mode="production",
    panel={"entity": "account_id", "time": "date"},
    candidates={
        "treatment": ["offer"],
        "outcome": ["retention"],
        "confounder": ["tenure"],
    },
    edges=discovery_edges,
    model_metrics=automl_report,
    gate_results=gate_report,
).run()

print(report.report())
report.write("artifacts/viz-plan.json")
```

The deterministic planner inspects roles, missingness, cardinality, proposed
causal roles, IV metadata, edge stability, panel/time columns, predictive
metrics, feature importance, and gate results. Recommendations are ranked and
include rationale, required columns, data requirements, chart hints, and
caveats.

Optional structured enrichment

An SLM/Qwen adapter is optional. It receives schema and aggregate metadata, not
sample values, and must return a list of typed recommendations. Unknown chart
types, missing columns, duplicate IDs, and malformed responses are rejected.

```python
def adapter(safe_request):
    return {"recommendations": generated_json}

report = AutoVizSuite(df).run(use_slm=True, slm_enricher=adapter)
```

The rule plan remains usable if no adapter or model is installed. In production
mode, reports contain no raw frame values.

Epistemic boundary

Recommended views include distributions, missingness, associations,
treatment–outcome descriptions, balance/overlap, IV first stage, edge
stability, graph structure, trends, subgroup summaries, residuals,
calibration/ROC/PR, importance, and gate dashboards. None of these views
establishes causality. Proposed treatment/outcome roles and graph arrows remain
hypotheses until supported by an identification design.

Integration

The stable entry point is `from autocausal.autoviz import AutoVizSuite`.
`AutoCausal.autoviz()` was intentionally not added while the core facade is
being changed concurrently. A future facade method can delegate to
`AutoVizSuite.from_autocausal(self).run(...)` without changing this API.
