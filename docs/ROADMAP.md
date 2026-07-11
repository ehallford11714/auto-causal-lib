# AutoCausalLib roadmap

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
- [x] `MiningReport.to_mine_report()`, `DiscoveryResult.to_causal_edges()`, `AutoResult.to_fabric_bundle()`
- [x] Discovery stability (`discover(stability=True, bootstrap_n=…)`) with honest confidence
- [x] QC gate (`autocausal.qc.validate_frame` → `QCReport`; hooked on `discover`)
- [x] NLP wired into guide/direct (`enrich_from_text`)

### Priority 2 — serious causal mining
- [x] Multi-method discovery (`score_pc_lite` + `corr_skeleton` + `mi_stub` → consensus)
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
