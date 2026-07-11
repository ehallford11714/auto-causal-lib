# SLM-directed report engine

`autocausal.reporting` turns completed AutoCausal artifacts into a normalized,
provenance-linked report bundle and renders that bundle without exposing the
source frame to the report director. PDF generation uses ReportLab directly and
does not require Chromium, a browser service, or an HTML-to-PDF executable.

## Quick start

```python
from autocausal.reporting import ReportEngine, ReportPolicy

engine = ReportEngine(
    use_slm=True,
    policy=ReportPolicy.production(),
)
artifact = engine.generate(
    source=result,
    output="autocausal-report.pdf",
)

print(artifact.path)
print(artifact.sha256)
print(artifact.warnings)
```

An AutoCausal session, a result, one report, or a list of report artifacts may
be supplied. The convenience methods use the same engine:

```python
artifact = ac.generate_report(
    "autocausal-report.pdf",
    use_slm=False,
    siblings=("markdown", "html", "json"),
)

artifact = discovery_result.generate_report("discovery.pdf")
artifact = auto_result.generate_report("pipeline.pdf")
```

Set `use_slm=False` for a completely deterministic report plan. If SLM
direction is requested but unavailable, disallowed, malformed, or unsafe, the
engine records the reason and falls back to that deterministic plan.

## Processing model

The engine has four explicit stages:

1. **Normalize** supplied artifacts into `ReportSource` objects containing only
   approved facts, tables, chart specifications, citations, caveats, and
   provenance identifiers.
2. **Plan** section selection and ordering with `SLMReportDirector`, constrained
   by a deterministic baseline and strict identifier allowlists.
3. **Validate** claims, citations, chart inputs, caveats, synthetic-IV labels,
   raw-data absence, and production privacy.
4. **Render** the validated `ReportBundle` to PDF, Markdown, HTML, or JSON and
   compute artifact metadata and SHA-256.

The typed public objects are:

- `ReportSource`: normalized evidence from one source artifact.
- `ReportPlan`: audience, purpose, section order, included/excluded artifacts,
  chart requests, appendix policy, citation policy, and redaction policy.
- `ReportSection`: render-ready facts, tables, charts, narrative claims,
  caveats, and provenance links.
- `ReportBundle`: sources, validated plan, sections, citations, audit notes,
  director record, and validation result.
- `ReportArtifact`: path, format, size, SHA-256, timestamp, run identifiers,
  package version, provenance summary, sibling outputs, and warnings.
- `ReportPolicy`: safety, SLM, content, citation, privacy, theme, page, table,
  chart, and output controls.

All typed models provide `to_dict()`, `to_json()`, `to_markdown()`, `report()`,
and JSON/Markdown `write()` methods where appropriate. `ReportArtifact.write()`
writes artifact metadata and never silently overwrites the rendered report.

## Source adapters

Adapters are duck typed and lazy. Optional or concurrently developed modules do
not need to be importable for the core package to load. The default registry
recognizes:

- `AutoCausal`, `DiscoveryResult`, and `AutoResult`
- mining and association reports (`MiningReport`, `MineReport`)
- quality, cleanse, imputation, and EDA reports
- causal estimate/inference, refutation, sensitivity, and validation reports
- production gate reports and run manifests
- insight, grounding, Agentic/SLM-loop, and GRAIL reports
- AutoML fit/model-selection reports
- AutoNLP hints, sentiment, and behavioral reports
- AutoViz plans and AutoChart/RenderedChart artifacts
- deep-research `ResearchReport`/`DeepResearchReport` and `SourceRecord`
- public-causal reports

Nested result artifacts are normalized separately. This preserves distinct
source and fact identifiers for discovery, estimates, refutations, sensitivity,
and other downstream results.

Raw `DataFrame` and `Series` objects are deliberately rejected. Run analysis
first and pass the resulting aggregate report artifacts.

## Default section catalog

The comprehensive template has deterministic ordering:

1. Cover
2. Executive summary
3. Scope, data provenance, schema/fingerprint, and run policy
4. Data quality, AutoCleanse ledger, and QC gates
5. AutoEDA and causal-readiness findings
6. Correlations and associations, explicitly labeled non-causal
7. Discovery graph, method agreement, stability, and FDR
8. Causal specification, estimates, intervals, assumptions, and diagnostics
9. IV evidence, separating observed and synthetic instruments
10. Refutations, sensitivity, and failed/escalated gates
11. AutoML results, explicitly separated from causal estimates
12. NLP/behavioral findings and privacy caveats
13. Insight, Agentic, SLM actions, and experiment recommendations
14. Deep-research evidence, citations, and contradictions
15. Visualizations
16. Limitations, unresolved questions, escalation, and recommendations
17. Technical appendix

Sections without supporting normalized evidence are omitted. The omission is
recorded in report audit notes; no placeholder evidence is fabricated. Core
scope, limitation, and appendix sections remain available for audit context.

## Constrained SLM direction

The SLM sees a bounded inventory of source IDs, fact IDs, fact labels,
categories, caveats, evidence eligibility, selected safe attributes, citation
IDs, existing charts, and redacted normalized fact values. Raw frames,
documents, snippets, sample values, and evidence-span text are not sent. Set
`ReportPolicy.send_fact_values_to_slm=False` to withhold even normalized fact
values.

The SLM may:

- prioritize sections already available for the run;
- summarize one or more existing facts;
- propose transitions grounded in fact IDs;
- select existing charts or recommend an allowed chart type over existing
  normalized facts;
- recommend follow-up work linked to existing facts.

Validation rejects a proposal that:

- creates a section, source, fact, table, chart, edge, or citation ID;
- introduces a number not present in the referenced facts;
- cites a record not attached to the referenced facts;
- suppresses required failed-gate, contradiction, limitation, or synthetic-IV
  disclosure;
- excludes a protected source carrying those disclosures;
- requests a chart without existing fact/table/provenance support;
- exceeds policy limits.

Rejected SLM output is never partially merged. The deterministic plan is used
and the rejection is retained in `director_actions`, warnings, and status.
Rendered SLM narrative is labeled with its backend/model identity.

The existing Qwen/Hugging Face backend is loaded lazily through
`autocausal.slm.get_backend`. No model download occurs when `use_slm=False`.

## Production safety and privacy

`ReportPolicy.production()` enables:

- raw-data prohibition;
- PII and secret redaction;
- citation integrity and verified-source requirements;
- synthetic-IV exclusion from production evidence;
- fail-closed validation.

Sensitive keys and common email, SSN, and token patterns are redacted during
normalization. NLP adapters retain aggregate profile values and structured
hypotheses, but omit documents, snippets, sample values, and evidence-span
text. AutoChart filter values and annotation text are redacted under production
policy.

Associations, correlations, discovery edges, linguistic hypotheses, predictive
metrics, feature importance, and behavioral edges retain explicit epistemic
caveats. They are not converted into identified causal effects.

Synthetic instruments may appear only as labeled, audit-only material in
production. They cannot be marked evidence eligible. Observed-instrument
evidence is rendered separately.

## Citation and provenance rules

The citation list is built only from supplied or fetched `SourceRecord`
identifiers. A literature claim may cite only IDs present in the same normalized
research source, and every rendered narrative citation must also be attached to
its referenced facts.

Production validation fails when:

- a claim refers to a missing source-record ID;
- a fact or SLM claim uses a detached citation;
- a required citation is unverified;
- a key claim has no normalized fact ID;
- a chart lacks a fact, table, image, or provenance mapping.

Abstracts, snippets, and exact evidence-span text are not retained in the report
bundle. Citation metadata includes safe bibliographic fields and retrieval
provenance.

## PDF and other renderers

ReportLab is a base dependency:

```bash
python -m pip install reportlab
```

The PDF renderer supports:

- letter and A4 page sizes;
- `professional`, `high_contrast`, and `monochrome` themes;
- headers, footers, page numbers, metadata, outlines/bookmarks, and a contents
  page;
- bounded tables with deterministic row ordering;
- PNG/JPEG images;
- SVG through the optional `svglib` package;
- in-memory Matplotlib export;
- Plotly static export when `kaleido` is installed;
- accessible chart captions and alt-text labels;
- chart-spec/table fallback with an artifact warning when image export is not
  available.

Example policy customization:

```python
policy = ReportPolicy.production(
    theme="high_contrast",
    page_size="A4",
    max_pages=120,
    max_rows_per_table=40,
    max_charts=16,
    sibling_formats=("markdown", "json"),
)
```

PDF page limits are checked after rendering. In production, exceeding the limit
raises `ReportRenderError` and the incomplete output is removed. Every completed
artifact includes its byte size and SHA-256 digest.

## Tool integrations

The reporting package registers these skilling tools lazily:

- `report.plan`
- `report.generate`
- `report.validate`

The MCP/AgentHook registry exposes:

- `autocausal_report_plan`
- `autocausal_generate_report`
- `autocausal_report_status`

Agentic report generation requires an explicit policy approval mapping through
`generate_approved_agentic_report`; a missing or rejected approval fails closed.

## Limitations

- The report engine summarizes completed artifacts; it does not perform causal
  identification or repair an invalid analysis.
- The deterministic renderer does not infer chart data from raw frames.
- SVG embedding requires `svglib`; Plotly static images require `kaleido`.
  Missing image tooling produces a specification/table fallback.
- PDF bookmarks and the contents page are practical navigation aids, not a
  fully typeset, dynamically numbered textbook TOC.
- Heuristic PII scanning cannot recognize every domain-specific identifier.
  Callers must avoid placing raw or sensitive values in custom report objects.
- SLM output is advisory composition only. All accepted claims remain bounded
  by normalized facts and validation.
