"""autocausal — auto-impute and exploratory causal discovery for tabular data."""

from __future__ import annotations

from autocausal.api import AutoCausal
from autocausal.results import AutoResult, DiscoveryResult
from autocausal.__version__ import __version__

__all__ = [
    "AutoCausal",
    "DiscoveryResult",
    "AutoResult",
    "__version__",
    "create_from_context",
    "infer_from_results",
    "list_tools",
    "validate_pipeline",
    "slm_status",
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
]


def __getattr__(name: str):
    if name in ("create_from_context", "infer_from_results", "slm_status"):
        from autocausal import slm as _slm

        return getattr(_slm, name)
    if name in ("list_tools", "validate_pipeline", "refute"):
        from autocausal import suite_tools as _st

        return getattr(_st, name)
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
    raise AttributeError(f"module 'autocausal' has no attribute {name!r}")
