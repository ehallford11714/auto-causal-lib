# AutoChart

AutoChart separates a typed chart contract from rendering:

```python
from autocausal.autochart import AutoChart, ChartSpec

spec = ChartSpec(
    id="first-stage",
    type="iv_first_stage",
    title="Observed instrument first stage",
    x="assignment",
    y="treatment",
    provenance={"design": "reviewed encouragement design"},
)

chart = AutoChart(spec, backend="auto", production=True).render(df)
chart.save("artifacts/first-stage.html")  # Plotly, when available
chart.write("artifacts/first-stage.json") # every backend
```

`ChartSpec` validates referenced columns, filters, chart requirements,
cardinality, and row limits. It carries aggregation, annotations, provenance,
and accessibility metadata. Alt text, labels, and a colorblind-safe palette
have safe defaults.

Backends

Backend resolution is Plotly, then Matplotlib, then a dependency-free
data/spec result. Matplotlib forces the noninteractive `Agg` backend and never
opens a GUI. JSON is always supported. Plotly supports HTML and, with a
compatible image engine such as Kaleido, PNG/SVG. Matplotlib supports PNG/SVG.

Sampling and privacy

Sampling is deterministic (`random_state` is part of the spec). Production
rendering defaults to aggregate or binned data:

- categorical labels are aliased;
- scatter/first-stage rows become binned means;
- overlap views become grouped histograms;
- missingness, correlations, balance, curves, importance, edges, and gates use
  aggregate metadata.

Set `allow_raw_values=True` on `AutoChart` only after an explicit privacy
review. That override is recorded in provenance.

Causal-specific inputs

DAG/network, edge-stability, and gate/evidence views consume structured context:

```python
chart = AutoChart(
    ChartSpec("network", "dag", "Discovery hypotheses"),
    backend="data",
).render(df, context={"edges": result.edges})
```

Graph arrows, first-stage strength, overlap, balance, and displayed
associations are diagnostics. A renderer never labels them as identified causal
effects.

Integration

The stable entry points are `AutoChart`, `ChartSpec`, and `AutoChartReport` from
`autocausal.autochart`. A later core-facade adapter can translate an AutoViz
recommendation with `ChartSpec.from_recommendation(...)`.
