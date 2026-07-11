# Module reference (`src/autocausal/`)

Inventory of packages and public symbols as of **0.13.x**. Soft-optional heavy deps never hard-require install.

> Exploratory outputs ≠ causal identification.

Canonical production path: `load → cleanse → EDA/association → plan → infer → refute/sensitivity`, with optional prediction kept separate.

---

## Top-level

| Module | Purpose | Key symbols |
|--------|---------|-------------|
| `__init__.py` | Lazy re-exports | `AutoCausal`, `DiscoveryResult`, `AutoResult`, suites, insight, engines, … |
| `__version__.py` | Version string | `__version__` |
| `__main__.py` | `python -m autocausal` | → `cli.main` |
| `api.py` | Primary façade | `AutoCausal` (`mode`, `policy`, `random_state`) |
| `production.py` | Policy, evidence, manifests, gates, orchestration | `ProductionPolicy`, `RunManifest`, `EvidenceGrade`, `ProductionRun`, `run_production_pipeline` |
| `statistical_gates.py` | Assumption/evidence diagnostics | sample/EPV, VIF, residual, overlap, weak-IV, power, BH-FDR gates |
| `cli.py` | CLI | `main` (`doctor --production`) |
| `engines.py` | Unified engine surface | `list_engines`, `engine_status`, `estimate`, `refute`, `discover_with`, `connectivity_map` |
| `suite_tools.py` | Tool registry + refute | `list_tools`, `invoke_tool`, `validate_pipeline`, `refute`, `RefuteResult` |
| `connective.py` | In-process agent broker | re-exports `AgentHook`, `call_tool`, `list_tools` |
| `doctor.py` | Health + production checklist | `doctor_report`, `format_doctor_markdown` |

### `AutoCausal` (api.py) — main methods

Constructors: `from_csv`, `from_parquet`, `from_sqlalchemy`, `from_dataframe`, `connect`, `ping`, `auto`.

Pipeline: `mine`, `impute`, `discover`, `discover_ensemble`, `cleanse`, `eda`, `automine`, `run`, `validate_qc`, `enrich_from_text`.

Causal: `infer`, `estimate`, `refute`, `production_check`, `run_production`, `engines_status`, `to_causaliv_request`, `sensitivity`, `set_panel`, `panel_features`.

Association: `correlate` (typed pair or BH-FDR matrix scan).

Guides: `guide`, `direct`, `create`, `interpret`, `ground`.

Loops: `physics_loop`, `ml_loop`, `insight_loop`, `agentic_loop`, `mine_behavioral_traces`, `attach_behavioral`, `mine_public`, `join_public`, `join_frames`.

Export: `report`, `to_fabric_bundle`, `validate_tools`.

---

## Core tabular / causal

| Module | Purpose | Key symbols |
|--------|---------|-------------|
| `ingest.py` | CSV / Parquet / SQL load | `load_csv`, `load_parquet`, `load_sqlalchemy`, `dialect_from_url` |
| `db.py` | Connect / ping / sample | `connect`, `ping`, `PingResult`, `list_tables`, `ping_public` |
| `impute.py` | Missing-value fill | `impute_dataframe`, `ImputationReport`, `ColumnImputation` |
| `roles.py` | Column typing | `ColumnRole`, `infer_column_roles`, `numeric_matrix` |
| `mining.py` | Profiles / associations / KPIs | `mine`, `MiningReport`, `profile_dataframe` |
| `discovery.py` | PC-lite + ensemble + soft backends | `discover_relationships`, `discover_ensemble`, `propose_candidates`, `consensus_edges` |
| `iv.py` | 2SLS / CausalIV soft | `try_iv_edges` |
| `results.py` | Result dataclasses + replay | `DiscoveryResult`, `AutoResult`, `replay_config`, `reproduce` |
| `report.py` | Markdown render | `render_markdown_report`, `render_auto_markdown` |
| `qc.py` | ID leakage / hygiene gate | `validate_frame`, `QCReport`, `QCIssue` |
| `join.py` | Multi-frame align | `align`, `suggest_keys`, `AlignReport` |
| `panel.py` | Longitudinal helpers | `PanelSpec`, `panel_lag`, `panel_diff`, `panel_within` |
| `sensitivity.py` | Sensitivity metrics | `compute_sensitivity`, `SensitivityReport` |
| `grounding.py` | Edge grounding | `ground_edges`, `GroundingReport` |
| `contracts/` | Fabric envelopes | `fabric_bundle`, `mining_to_mine_report`, `edges_to_causal_edge_envelopes` |
| `correlation/` | Typed descriptive association | `CorrelationSuite`, `correlation`, `correlation_matrix`, typed results |
| `inference/` | Explicit-design effect estimation | `CausalSpec`, `AutoInference`, `AutoInferencePlanner`, `CausalInferenceResult`, native estimators |
| `ml/automl.py` | Leakage-safe prediction | `AutoML`, `AutoMLReport`, `run_automl` |

---

## Soft causal backends (`backends/`)

| Module | Purpose | Key symbols |
|--------|---------|-------------|
| `backends/__init__.py` | Catalog + status | `backend_status`, `DISCOVERY_BACKENDS`, `ESTIMATE_BACKENDS`, `REFUTE_BACKENDS` |
| `_common.py` | Shared helpers | `soft_import`, `resolve_roles`, `adjacency_to_edges` |
| `causal_learn.py` | PC / GES / FCI | `discover`, `discover_pc`, `discover_ges`, `discover_fci` |
| `lingam_backend.py` | DirectLiNGAM | `discover` |
| `gcastle_backend.py` | NOTEARS | `discover` |
| `doubleml_backend.py` | PLR ATE | `estimate` |
| `econml_backend.py` | LinearDML / CausalForestDML | `estimate` |
| `dowhy_refute.py` | Real `refute_estimate` | `refute` |

Install: `pip install "auto-causal-lib[causal-extra]"`. Details: [CAUSAL_BACKENDS.md](CAUSAL_BACKENDS.md).

---

## Guides / SLM

| Module | Purpose | Key symbols |
|--------|---------|-------------|
| `slm.py` | Rule + HF creation/inference | `RuleBackend`, `HuggingFaceSLM`, `create_from_context`, `infer_from_results`, `guide_pipeline`, `slm_status` |
| `guides/` | Direction backends | `direct`, `list_guides`, `DirectionPlan`, `RuleGuide`, `LLMIntentGuide`, `RetracementGuide`, `KineteqPivotEmbeddingGuide`, `KineteqGrailGuide` |

---

## Suites / skilling

| Module | Purpose | Key symbols |
|--------|---------|-------------|
| `suites/` | AutoCleanse / AutoEDA / AutoMine | `AutoCleanseSuite`, `AutoEDASuite`, `AutoMineSuite`, `SLMAutoDirector`, `CleanseActions`, `EDAActions`, `MineActions` |
| `skilling/` | SLM tool surface | `ToolSurface`, `suite_tool_surface`, `SkillRegistry`, `SLMToolBroker`, `skill_catalog`, `SkillDrill`, `SkillTrace` |

---

## Insight / agentic / GRAIL

| Module | Purpose | Key symbols |
|--------|---------|-------------|
| `insight/` | Research insight loop | `InsightSuite`, `InsightReport`, `ExperimentRecommender`, `run_insight_loop`, `demo_insight` |
| `agentic/` | Cyclic FSM loop | `AgenticCausalLoop`, `run_agentic_loop`, `AgenticLoopReport`, `LoopState`, `Compactor`, `AgentMemory`, `GraphRuntime` |
| `grail/` | Kineteq GRAIL soft | `GrailEngine`, `run_grail`, `GrailReport`, `grail_backend_status` |

---

## MCP / connective

| Module | Purpose | Key symbols |
|--------|---------|-------------|
| `mcp/` | Stdio MCP server | `build_default_registry`, `AgentHook`, `SessionStore`, `run_stdio`, `main` |
| `connective.py` | Same tools in-process | `AgentHook`, `call_tool`, `list_tools` |

Needs `[mcp]` only for the SDK stdio server. See [MCP.md](MCP.md).

---

## NLP / behavioral / physics / ML

| Module | Purpose | Key symbols |
|--------|---------|-------------|
| `nlp/` | Text → role hints | `TextCausalHints`, `NlpFeatureBuilder`, `extract_causal_hints_from_text`, `nltk_status` |
| `behavioral/` | Habit/nudge traces | `BehavioralTraceStore`, `mine_behavioral_traces`, `list_demos` |
| `physics/` | Dynamics loop | `PhysicsCausalSuite`, `PhysicsEngine` |
| `ml/` | KPI ML loop | `KPIMinedCausalLoop`, `ModelConstructPlan`, `FitReport` |
| `apps/` | Streamlit demo paths | `physics_demo_path` |

---

## Data / public / bridges

| Module | Purpose | Key symbols |
|--------|---------|-------------|
| `datasets.py` | Bundled examples | `load_dataset`, `list_datasets` |
| `public_suite.py` | Public/demo sources | `list_public`, `load_public`, `join_public_frames` |
| `public_causal.py` | Multi-source mine | `mine_public`, `PublicCausalMiner` |
| `isolates_bridge.py` | IntentIsolates soft | `run_isolates_causal` |
| `datamine_adapter.py` | DataMine soft | `mine_via_datamine` |
| `data/**` | Package data CSVs | (package-data) |

---

## How surfaces reach engines

| Surface | Entry |
|---------|-------|
| Library | `engines.*`, `AutoCausal.discover/estimate/refute` |
| CLI | `engines`, `estimate`, `refute` — [CLI.md](CLI.md) |
| MCP / AgentHook | `autocausal_list_engines`, `_estimate`, `_refute`, `_discover` |
| Skilling | suite actions + `suite_tools` adapters |
| Insight | session `AutoCausal` |

See [INDEX.md](INDEX.md) for the full doc map.
