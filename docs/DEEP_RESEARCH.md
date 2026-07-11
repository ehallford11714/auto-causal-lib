# Deep research handoff

`autocausal.research` turns an AutoCausal finding into a bounded research
agenda, retrieves real source records, verifies exact evidence spans, and
returns recommendations to the causal workflow. It is library-first and
offline-first. It does **not** treat model memory as a source and does **not**
upgrade an AutoCausal edge's identification grade.

This package is intentionally separate from the production, inference,
AutoViz, ML, and NLP packages so it can be integrated into a coordinated
release without publishing a partial independent release.

## Quick start

```python
from autocausal.research import (
    DeepResearchSuite,
    LocalDocumentProvider,
    ResearchPolicy,
    SourceRecord,
)

# Importing autocausal.research installs conflict-light adapters on
# DiscoveryResult, AutoResult, InsightReport, AgenticLoopReport, GateReport,
# and AutoCausal when those classes do not already provide them.
handoff = result.to_research_handoff(
    domain="public health",
    context={"population": "adults", "setting": "community"},
)

policy = ResearchPolicy(
    allowed_providers=("local",),
    minimum_independent_sources=2,
)
suite = DeepResearchSuite(
    policy=policy,
    providers=[LocalDocumentProvider(my_source_records)],
    use_slm=False,
)
report = suite.run(handoff, intensity="standard")
print(report.report())
```

The direct convenience form is:

```python
report = result.deep_research(
    intensity="deep",
    policy=policy,
    providers=[LocalDocumentProvider(my_source_records)],
)
```

`AutoCausal.deep_research()` requires an existing discovery result. It never
silently runs discovery or changes an edge.

## Evidence layers

The report keeps four layers separate:

1. **AutoCausal empirical findings** retain their original stability, gate,
   and evidence-grade metadata.
2. **Retrieved literature evidence** consists only of `SourceRecord` objects
   and verified `EvidenceSpan` text.
3. **SLM synthesis** is optional, structurally validated, and rejected if it
   refers to an unknown source.
4. **Contradictions and unresolved assumptions** remain explicit rather than
   being averaged into a confident narrative.

A `supported_literature_context` label means that the configured number of
independent sources contain supporting evidence. It does not mean that the
AutoCausal estimate is identified, unbiased, transportable, or production
eligible.

## Typed contracts

The public contracts are JSON-serializable dataclasses:

- `ResearchHandoff`: findings, edges, grades, failed gates, uncertainty
  signals, candidate roles, privacy-safe labels, approved context, and
  recommended experiments. It rejects raw-frame/raw-text provenance.
- `ResearchQuestion`: priority, linked findings, rationale, inclusion and
  exclusion criteria, query variants, and PECO/PICO-style fields.
- `SourceRecord`: provider, stable identifier, title, authors, date, abstract
  or snippet, retrieval timestamp, DOI/arXiv/URL, license, availability,
  references, and bounded metadata.
- `EvidenceSpan`: exact source text, relation, extraction method, confidence,
  and source field.
- `ResearchClaim`: normalized claim, linked edge/finding, exact spans,
  contradiction status, literature label, and independent-source count.
- `ResearchReport`: agenda, sources, claims, contradictions, provenance,
  costs, limits, intensity, cross-matches, gaps, saturation, and handback.
- `ResearchPolicy`, `ResearchBudget`, `BudgetUsage`, and `SearchIntensity`.
- `CrossMatch`, `MatchReason`, `ComparabilityScore`, and
  `ClaimEvidenceGraph`.

Use `to_dict()`, `to_json()`, `from_dict()`, and `from_json()` at persistence
boundaries. A deserialized report can be resumed with
`suite.resume(report, handoff=original_handoff)`. A live report can call
`report.deepen(...)` directly because it retains its suite/cache reference.

## Finding-to-agenda rules

Rules run before an SLM and assign the highest priority to:

- low bootstrap stability or discovery-engine disagreement;
- weak, synthetic, or unverified IV assumptions;
- failed production or evidence gates;
- refuted or sensitivity-unstable estimates;
- overlap/positivity problems;
- uncertain confounders, collider adjustment, or selection bias;
- score-, SLM-, GRAIL-, or NLP-only direction;
- surprising subgroup or context-dependent effects; and
- conflicting empirical signals.

Rules produce bounded search queries and criteria. A local Qwen/Hugging Face
backend may enrich the agenda after the rule plan. Invalid JSON, unknown
finding IDs, or malformed questions are recorded and ignored.

## Search intensity

Intensity selects a default `ResearchBudget`; it never bypasses privacy,
provider, network, citation, or approval policy.

### `quick`

- Up to two questions, one query each, one provider, six unique sources, and
  one round.
- Intended for metadata/abstract orientation and a fast unresolved-gap check.
- No automatic related-work expansion.

### `standard`

- Up to five questions, three query variants, two providers, 24 unique
  sources, and two rounds.
- Adds query expansion, deduplication, deterministic cross-match, and a basic
  contradiction check.

### `deep`

- Up to eight questions, five variants, three providers, 60 unique sources,
  and four rounds.
- Adds iterative gap analysis, related/reference expansion when retrieved
  metadata exposes it, population/context comparability, and independent
  source requirements.

### `exhaustive`

- Up to 15 questions, eight variants, six providers, 150 unique sources, and
  eight rounds.
- Keeps a systematic include/exclude screening log.
- Requires explicit human approval in production or high-impact contexts by
  default. An exploratory general-domain policy may permit it within caps.

These are defaults, not guarantees of coverage. Provider caps and source
availability can result in fewer records.

### Budget overrides

```python
report = suite.run(
    handoff,
    intensity="deep",
    budget_overrides={
        "max_questions": 4,
        "queries_per_question": 3,
        "sources_per_provider": 10,
        "max_sources": 25,
        "max_providers": 3,
        "max_rounds": 3,
        "wall_time_seconds": 180,
        "max_tokens": 12_000,
        "max_bytes": 10_000_000,
        "publication_year_min": 2015,
        "publication_year_max": 2026,
        "languages": ["en"],
    },
)
```

Overrides are capped by `ResearchPolicy.maximum_budget`. `BudgetUsage` records
questions, query calls, provider count, fetched and retained sources, rounds,
estimated SLM tokens, response bytes, elapsed time, cache hits, failures, and
the policy limit that stopped the run.

## Adaptive routing

`IntensityRouter` emits an auditable `IntensityRecommendation` containing the
selected level, recommended level, reasons, metrics, approval requirement, and
route decision.

Deterministic routing considers production/high-impact risk, failed gates,
stability and engine disagreement, refutation/IV risks, independent-source
coverage, contradictions, population/context mismatches, saturation, and
budget use. An SLM request for more context is logged but cannot by itself
increase intensity.

The recommended level does not automatically replace the selected level.
Exhaustive work in production/high-impact contexts and policy-defined
high-cost external production work stop with `ResearchApprovalRequired`
unless approval is explicitly supplied:

```python
report = suite.run(
    handoff,
    intensity="exhaustive",
    approval_granted=True,
)
```

## Resuming and deepening

```python
first = suite.run(handoff, intensity="quick")
deeper = first.deepen(intensity="deep")
exhaustive = deeper.deepen(
    intensity="exhaustive",
    approval_granted=True,
)
```

Deepening:

- carries forward retrieved sources, claims, screening history, budget usage,
  query log, round history, and saturation data;
- reuses the in-memory or disk metadata cache;
- limits the new agenda to unresolved findings and evidence gaps;
- skips an exact provider/query/filter signature already completed;
- uses bounded query, contradiction, population, and related-work expansions;
  and
- never reruns completed discovery or mutates AutoCausal edges.

For a detached JSON report:

```python
restored = ResearchReport.from_json(saved_json)
deeper = suite.resume(restored, handoff=handoff, intensity="deep")
```

## Iterative workflow

The base workflow records timestamped agent spans and provider tool traces:

```text
prepare_handoff
  -> prioritize_uncertainty
  -> plan_queries
  -> retrieve
  -> deduplicate_screen
  -> extract_evidence
  -> cross_check_contradiction
  -> synthesize
  -> recommend_experiments
  -> handback
```

Subsequent rounds expose the deeper routing graph:

```text
gap_analysis
  -> intensity_route
  -> query_expand
  -> retrieve
  -> cross_match
  -> contradiction_probe
  -> source_independence_check
  -> evidence_saturation
  -> route(deepen | stop | human)
```

The run stops when coverage is met without unresolved contradiction, evidence
saturates, a budget is exhausted, the configured round limit is reached, no
provider is permitted, comparability remains insufficient, or human approval
is required.

## Retrieval providers

Included adapters implement a common `ResearchProvider.search()` interface:

- `LocalDocumentProvider`: `SourceRecord` objects, JSON/JSONL, bounded local
  text/Markdown documents, or a soft `query`/`search` vector-store adapter.
- `ArxivProvider`: official Atom API metadata.
- `CrossrefProvider`: official REST works API.
- `OpenAlexProvider`: official works API and reconstructed abstract index.
- `SemanticScholarProvider`: low-volume, no-key Graph API adapter with
  backoff; it is not enabled by the default provider allowlist.
- `GenericWebSearchProvider`: an explicit callback only. It is disabled unless
  generic web, the provider, network access, and external consent are all
  enabled.

Network providers use HTTPS host allowlists, no redirects, bounded response
sizes, timeouts, a responsible User-Agent, and retry/backoff for transient
failures. Failures are recorded per query and never replaced with invented
metadata.

Disk caching is optional:

```python
policy = ResearchPolicy(
    allowed_providers=("arxiv", "crossref", "openalex"),
    allow_network=True,
    external_network_consent=True,
    cache_dir=".autocausal/research-cache",
)
```

The cache contains query metadata and `SourceRecord` payloads, not credentials
or raw AutoCausal observations.

## Privacy and production policy

Deep-research handoffs omit raw frames, row samples, and raw text columns.
PII-like variable names are pseudonymized. User-approved context is bounded
and redacts common emails, phone numbers, long tokens, and secret-like keys.

Network retrieval is off by default. Both conditions are required:

```python
ResearchPolicy(
    allowed_providers=("arxiv", "crossref", "openalex"),
    allow_network=True,
    external_network_consent=True,
)
```

To inherit production mode plus operational round/time caps:

```python
research_policy = ResearchPolicy.from_production_policy(
    autocausal_policy,
    allowed_providers=("local", "arxiv"),
)
```

The serialized production-policy snapshot is audit context; research-specific
provider consent and citation settings remain explicit.

Production mode applies the same requirement and adds human approval for
high-cost external work. Provider allowlists and maximum budgets still apply
at every intensity. Credentials, private URLs, localhost, redirects, and
arbitrary scraping are not supported.

## Cross-match semantics

`CrossMatchEngine` deterministically generates candidates using:

- normalized tokens and approved aliases;
- exact DOI, arXiv, or stable identifiers;
- corpus TF-IDF/cosine retrieval relevance;
- treatment/outcome/instrument/confounder role overlap;
- direction and local negation scanning;
- population, setting/context, and time-period overlap;
- design terms; and
- retrieved reference links.

`ComparabilityScore` exposes each component plus an overall weighted value and
warnings. Population, context, time, abstract-only evidence, and contradictory
direction can lower comparability. The value is a triage aid, not a causal
effect confidence.

Optional SLM semantic adjudication runs only after deterministic candidate
generation. It may label relevance or support/contradiction, but it cannot add
a source and its score remains a separate `MatchReason`.

The claim/evidence graph has typed finding, source, evidence-span, and claim
nodes. Reference edges may point to unretrieved identifiers and are explicitly
marked as such; they are not citations until retrieved.

## Citation and synthesis safeguards

- Sources deduplicate by normalized DOI, arXiv ID, then title fingerprint.
- Companion papers sharing an explicit study, cohort, trial, or dataset ID
  share one source-independence group.
- Metadata-only sources can support discovery of related work but cannot
  contribute an `EvidenceSpan`.
- Abstract/snippet evidence is labeled separately from full-text evidence.
- Rule and SLM extraction must return text that is an exact substring of the
  retrieved abstract/snippet.
- A claim's source ID must resolve to a `SourceRecord`.
- Supporting and contradicting spans remain separate.
- Publication dates and context metadata are included in contradiction
  records.
- SLM synthesis is discarded if it emits an unknown source ID or DOI.
- `ResearchReport.validate_citations(strict=True)` is called before handoff
  and before Markdown rendering.

These checks prevent unsupported citations from surviving any intensity.
They cannot determine whether a paper's methods or reported result are
correct; that remains a human-review task.

## Handback and agentic use

The report proposes only explicit actions:

- `collect_data`
- `run_refutation`
- `revise_roles`
- `search_deeper`
- `human_review`

`report.to_agentic_handback()` returns questions, query variants, experiments,
unresolved gaps, warnings, and source IDs. It states that discovered edges are
not mutated and identification grades are unchanged.

`research_escalation_node(state, suite=...)` is an FSM/LangGraph-compatible
adapter. It copies mapping state where possible, invokes policy routing, adds a
research report/handback, and preserves the original `edges` list.

## MCP and AgentHook

The default registry exposes:

- `autocausal_research_plan`
- `autocausal_deep_research`
- `autocausal_research_status`
- `autocausal_research_report`

The run tool accepts a session containing a discovery result or a serialized
`ResearchHandoff`. Offline `sources` are accepted as `SourceRecord` payloads.
Network and approval flags are explicit and default to false.

## Limitations

- API availability, indexing delay, abstracts, licenses, and metadata quality
  vary by provider.
- Title and identifier deduplication cannot discover every preprint/published
  or translated duplicate.
- Explicit study IDs improve source-independence grouping; absent IDs are not
  inferred from author similarity alone.
- Deterministic direction and contradiction scans are conservative lexical
  signals, not semantic or methodological proof.
- Abstract evidence can omit subgroup definitions, estimands, caveats, and
  null results present in full text.
- Systematic-review standards require protocol registration, calibrated
  dual screening, full-text assessment, risk-of-bias tools, and human domain
  expertise beyond this workflow.
- Search intensity increases cost and potential coverage; it does not
  guarantee saturation or correctness.

## Intensity and cross-match

Levels:

| Intensity | Behavior |
|-----------|----------|
| `quick` | One bounded round, one/few providers, metadata/abstract evidence |
| `standard` | Query expansion, multi-provider, basic contradiction check |
| `deep` | Related-work DOI/arXiv expansion from fetched references, prior/episode/public corpus cross-match, stronger independence requirements |
| `exhaustive` | Highest configured budget; production/high-impact domains require explicit approval |

Cross-match exposes component scores (`lexical_alias`, `identifier`, `role_compatibility`,
`direction_agreement`, `population_overlap`, `context_overlap`, `time_overlap`,
`design_relevance`) plus optional prior-corpus reasons. Overall comparability is
retrieval relevance, **not** causal confirmation.

Pass prior corpora into the suite for episode/public reuse:

```python
suite = DeepResearchSuite(
    policy=policy,
    providers=[LocalDocumentProvider(my_source_records)],
    prior_sources=prior_episode_records,
    episode_sources=current_session_records,
    public_sources=curated_corpus,
)
report = suite.run(handoff, intensity="deep")
deeper = report.deepen(intensity="exhaustive", approval_granted=True)
print(deeper.selected_intensity, deeper.recommended_intensity)
print(deeper.cross_matches[0].comparability.to_dict())
```

Local providers boost exact DOI/arXiv queries produced by related-work expansion
so identifier deepen rounds can resolve without inventing citations.

CLI:

```bash
python -m autocausal research plan --csv data.csv --intensity deep
python -m autocausal research run --csv data.csv --intensity standard --sources-json sources.json
python -m autocausal research deepen --report-json report.json --handoff-json handoff.json --intensity deep
```

## Offline verification

```bash
python -m pytest tests/test_deep_research.py -q
```

The tests cover budget ordering/caps, privacy-safe handoff, deterministic
agenda and cross-match, context mismatch, deduplication/source independence,
contradictions, exact citations, resume/query reuse, related-work DOI/arXiv
expansion, prior-corpus cross-match, saturation/policy stops, production
exhaustive approval, mocked unsafe Qwen output, result adapters, and MCP
plan/run/status/report.
