"""autocausal — auto-impute and exploratory causal discovery for tabular data."""

from __future__ import annotations

from autocausal.api import AutoCausal
from autocausal.production import (
    EvidenceGateError,
    EvidenceGrade,
    ProductionGateError,
    ProductionPolicy,
    ProductionRun,
    ResourceLimitError,
    RunManifest,
    RunPolicy,
    UnsafePayloadError,
    run_production_pipeline,
)
from autocausal.results import AutoResult, DiscoveryResult
from autocausal.__version__ import __version__
from autocausal.correlation import (
    CorrelationSuite,
    correlation,
    correlation_matrix,
)
from autocausal.inference import (
    AutoInference,
    AutoInferencePlanner,
    CausalInferenceResult,
    CausalSpec,
)

__all__ = [
    "AutoCausal",
    "DiscoveryResult",
    "AutoResult",
    "ProductionPolicy",
    "ProductionRun",
    "run_production_pipeline",
    "RunPolicy",
    "RunManifest",
    "EvidenceGrade",
    "ProductionGateError",
    "EvidenceGateError",
    "ResourceLimitError",
    "UnsafePayloadError",
    "CorrelationSuite",
    "correlation",
    "correlation_matrix",
    "AutoInference",
    "AutoInferencePlanner",
    "CausalInferenceResult",
    "CausalSpec",
    "__version__",
    "create_from_context",
    "infer_from_results",
    "list_tools",
    "validate_pipeline",
    "slm_status",
    "ensure_local_qwen",
    "probe_hardware",
    "recommend_qwen_model",
    "list_guides",
    "direct",
    "KPIMinedCausalLoop",
    "ModelConstructPlan",
    "PublicCausalMiner",
    "PublicCausalReport",
    "mine_public",
    "TextCausalHints",
    "NlpFeatureBuilder",
    "extract_causal_hints_from_text",
    "BehavioralTraceStore",
    "mine_behavioral_traces",
    "InsightSuite",
    "InsightReport",
    "ExperimentRecommender",
    "run_insight_loop",
    "load_dataset",
    "list_datasets",
    "validate_frame",
    "QCReport",
    "align",
    "PanelSpec",
    "refute",
    "estimate",
    "list_engines",
    "engine_status",
    "discover_with",
    "connectivity_map",
    "AutoCleanseSuite",
    "AutoEDASuite",
    "AutoMineSuite",
    "CleanseActions",
    "EDAActions",
    "MineActions",
    "CleanseReport",
    "EDAReport",
    "MineReport",
    "SLMAutoDirector",
    "ToolSurface",
    "SkillRegistry",
    "SLMToolBroker",
    "suite_tool_surface",
    "AgentHook",
    "GrailEngine",
    "GrailReport",
    "run_grail",
    "AgenticCausalLoop",
    "AgenticLoopReport",
    "run_agentic_loop",
    "SLMLangGraphChain",
    "SLMChainReport",
    "run_slm_langgraph_loop",
    "doctor_report",
    "DeepResearchSuite",
    "ResearchReport",
    "ResearchPolicy",
    "ResearchHandoff",
    "ResearchBudget",
    "SearchIntensity",
    "CrossMatchEngine",
    "IntensityRouter",
    "to_research_handoff",
    "expand_related_work_queries",
    "match_prior_sources",
    "ReportEngine",
    "ReportPolicy",
    "generate_report",
    "AutoTabularML",
    "AutoMLReport",
    "AutoVizSuite",
    "AutoVizReport",
    "list_integrations",
    "integration_status",
    "invoke_capability",
    "CapabilityRouter",
]


def __getattr__(name: str):
    if name in ("create_from_context", "infer_from_results", "slm_status"):
        from autocausal import slm as _slm

        return getattr(_slm, name)
    if name in ("list_tools", "validate_pipeline", "refute"):
        from autocausal import suite_tools as _st

        return getattr(_st, name)
    if name in ("estimate", "list_engines", "engine_status", "discover_with", "connectivity_map"):
        from autocausal import engines as _eng

        return getattr(_eng, name)
    if name in ("list_guides", "direct"):
        from autocausal import guides as _guides

        return getattr(_guides, name)
    if name in ("KPIMinedCausalLoop", "ModelConstructPlan", "FitReport", "construct_model_plan"):
        from autocausal import ml as _ml

        return getattr(_ml, name)
    if name in ("PublicCausalMiner", "PublicCausalReport", "mine_public"):
        from autocausal import public_causal as _pc

        return getattr(_pc, name)
    if name in ("TextCausalHints", "NlpFeatureBuilder", "extract_causal_hints_from_text"):
        from autocausal import nlp as _nlp

        return getattr(_nlp, name)
    if name in ("BehavioralTraceStore", "mine_behavioral_traces"):
        from autocausal import behavioral as _beh

        return getattr(_beh, name)
    if name in (
        "InsightSuite",
        "InsightReport",
        "ExperimentRecommender",
        "run_insight_loop",
        "run_slm_research_loop",
        "demo_insight",
    ):
        from autocausal import insight as _insight

        return getattr(_insight, name)
    if name in ("load_dataset", "list_datasets", "get_dataset", "DATASET_IDS"):
        from autocausal import datasets as _ds

        return getattr(_ds, name)
    if name in ("validate_frame", "QCReport", "QCIssue"):
        from autocausal import qc as _qc

        return getattr(_qc, name)
    if name in ("align", "AlignReport", "suggest_keys"):
        from autocausal import join as _join

        return getattr(_join, name)
    if name in ("PanelSpec", "panel_lag", "panel_diff", "panel_within"):
        from autocausal import panel as _panel

        return getattr(_panel, name)
    if name in (
        "AutoCleanseSuite",
        "AutoEDASuite",
        "AutoMineSuite",
        "CleanseActions",
        "EDAActions",
        "MineActions",
        "CleanseReport",
        "EDAReport",
        "MineReport",
        "SLMAutoDirector",
        "SLMDirectives",
        "auto_cleanse",
        "auto_eda",
        "auto_mine",
    ):
        from autocausal import suites as _suites

        return getattr(_suites, name)
    if name in (
        "ToolSurface",
        "SkillRegistry",
        "SLMToolBroker",
        "suite_tool_surface",
        "skill_catalog",
        "SkillDrill",
        "SkillTrace",
        "ToolDef",
        "ToolResult",
    ):
        from autocausal import skilling as _sk

        return getattr(_sk, name)
    if name in ("AgentHook", "call_tool"):
        from autocausal.connective import AgentHook, call_tool

        return {"AgentHook": AgentHook, "call_tool": call_tool}[name]
    if name in ("GrailEngine", "GrailReport", "run_grail", "grail_backend_status"):
        from autocausal import grail as _grail

        return getattr(_grail, name)
    if name in (
        "AgenticCausalLoop",
        "AgenticLoopReport",
        "run_agentic_loop",
        "LoopState",
        "Compactor",
        "GraphRuntime",
        "VectorStoreMemory",
        "AgentMemory",
        "SLMLangGraphChain",
        "SLMChainReport",
        "run_slm_langgraph_loop",
        "langgraph_available",
    ):
        from autocausal import agentic as _agentic

        return getattr(_agentic, name)
    if name in ("ensure_local_qwen", "probe_hardware", "recommend_qwen_model"):
        from autocausal import slm as _slm

        return getattr(_slm, name)
    if name == "doctor_report":
        from autocausal.doctor import doctor_report

        return doctor_report
    if name in (
        "DeepResearchSuite",
        "ResearchReport",
        "ResearchPolicy",
        "ResearchHandoff",
        "ResearchBudget",
        "to_research_handoff",
        "SearchIntensity",
        "CrossMatchEngine",
        "IntensityRouter",
        "LocalDocumentProvider",
        "expand_related_work_queries",
        "extract_reference_identifiers",
        "match_prior_sources",
    ):
        from autocausal import research as _research

        return getattr(_research, name)
    if name in (
        "ReportEngine",
        "ReportPolicy",
        "ReportArtifact",
        "generate_report",
    ):
        from autocausal import reporting as _reporting

        return getattr(_reporting, name)
    if name in ("AutoTabularML", "AutoMLReport", "AutoMLGateError"):
        from autocausal import automl as _automl

        return getattr(_automl, name)
    if name in ("AutoVizSuite", "AutoVizReport", "VizPlan"):
        from autocausal import autoviz as _autoviz

        return getattr(_autoviz, name)
    if name in (
        "list_integrations",
        "integration_status",
        "invoke_capability",
        "CapabilityRouter",
        "build_install_plan",
    ):
        from autocausal import integrations as _integrations

        return getattr(_integrations, name)
    raise AttributeError(f"module 'autocausal' has no attribute {name!r}")


# Install conflict-light research methods without importing network/SLM models.
# Existing host-class methods always win.
try:
    from autocausal.research.adapters import install_research_adapters

    install_research_adapters()
except Exception:
    pass
