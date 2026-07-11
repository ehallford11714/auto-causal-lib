# AutoCausalLib roadmap

## Shipped in 0.11.4 — report() aliases + gap-closure

- [x] `report(*, as_markdown=True)` on PublicCausalReport, BehavioralReport, GroundingReport
- [x] GuideResult / DirectionPlan, ModelConstructPlan, FitReport, KPILoopResult
- [x] Trajectory / PhysicalGroundingReport / PhysicsLoopResult
- [x] SLM GuideResult / CreationResult / InferenceResult, ValidationReport, SLMDirectives
- [x] Ergonomics tests for callable `report()` on remaining result types
- [x] `mi` / `mi_binned` real binned NMI (`mi_stub` alias; edges labeled `mi_binned`)
- [x] `doctor` CLI + `doctor_report` / `format_doctor_markdown`
- [x] `apply_grail` / `boost_edges` → discover merge (`grail_boost`); `direct()` merges plan boosts
- [x] `session_snapshot()` + LIBRARY_API persistence notes (in-memory; MCP/EpisodeStore)

## Shipped in 0.11.3 — DiscoveryResult post-discover handle

- [x] `DiscoveryResult.to_fabric_bundle()` / `to_mine_report()` / `to_causaliv_request()` / `engines_status()`
- [x] `DiscoveryResult.estimate()` / `refute()` / `sensitivity()` (session weakref + attached frame)
- [x] `AutoResult` fabric / estimate / refute / sensitivity / causaliv parity
- [x] `report()` aliases on MiningReport, suite reports, Insight/Agentic/Grail reports
- [x] `tests/test_api_ergonomics.py` iris chain regression

## Shipped in 0.11.1 — docs + packaging

- [x] Comprehensive docs: INDEX / MODULES / CLI / MCP / LIBRARY_API / CAUSAL_BACKENDS
- [x] README callout: `pip install auto-causal-lib` → `import autocausal`
- [x] MANIFEST.in includes docs in sdist; wheel verified in clean venv

## Shipped in 0.11.0 — causal backends + engine connectivity

- [x] Soft `causal-learn` PC / GES / FCI discovery adapters
- [x] Real DoWhy `CausalModel.refute_estimate` (placebo / random common cause / data subset)
- [x] DoubleML PLR ATE + EconML LinearDML / CausalForestDML estimate
- [x] LiNGAM DirectLiNGAM + gCastle NOTEARS discovery (soft)
- [x] Unified `autocausal.engines` + CLI `engines` / `estimate` / `refute`
- [x] MCP tools: `autocausal_list_engines`, `autocausal_estimate`, `autocausal_refute`
- [x] Insight / MCP / skilling / CLI first-class in wheel; `[causal-extra]` expanded
- [x] Docs: [CAUSAL_BACKENDS.md](CAUSAL_BACKENDS.md)

## Shipped in 0.8.0 (P1 → P3)

### Priority 1 — library credibility
- [x] Fabric contracts (`MineReport` / `CausalEdge` / `FabricBundle` / `SearchDAG`) via `autocausal.contracts`
- [x] `MiningReport.to_mine_report()`, `DiscoveryResult.to_causal_edges()` / `to_fabric_bundle()`, `AutoResult.to_fabric_bundle()`
- [x] Discovery stability (`discover(stability=True, bootstrap_n=…)`) with honest confidence
- [x] QC gate (`autocausal.qc.validate_frame` → `QCReport`; hooked on `discover`)
- [x] NLP wired into guide/direct (`enrich_from_text`)

### Priority 2 — serious causal mining
- [x] Multi-method discovery (`score_pc_lite` + `corr_skeleton` + `mi_binned` → consensus)
- [x] Panel / longitudinal (`PanelSpec`, lag/diff/within, soft DiD notes)
- [x] CausalIV handoff (`AutoCausal.to_causaliv_request()`)
- [x] Sensitivity plumbed onto `AutoCausal` / `AutoResult` / `auto()`
- [x] Generic multi-frame `autocausal.join.align`
- [x] `docs/LIBRARY_API.md`

### Priority 3 — research / scale
- [x] Public/example loaders: local cache + soft network (`use_cache`)
- [x] Refute hooks (`suite_tools.refute` / `ac.refute`) — DoWhy real path + builtin placebo
- [x] Chunked/sampled SQL (`chunksize`, `sample_n`)
- [x] CausalSearch DAG export (`to_search_dag`, soft-optional)
- [x] `py.typed` marker
- [x] Imputation mechanism diagnostics on `ImputationReport`

## Deferred / out of scope
- CausalNex, CDT, stale scikit-uplift (skipped by design)
- Formal Little's MCAR test / MNAR models
- Hard dependency on CausalIV / CausalSearch / CausalBridge (remain soft)
- Production AutoML OS / guaranteed identification

## Next candidates
- Stronger ensemble scoring + calibration plots
- Native CausalBridge schema validation CI
- Optional parquet cache for public open mirrors
