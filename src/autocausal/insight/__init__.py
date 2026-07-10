"""AutoCausal Insight suite — reports, SLM/rule guide, experiment research loop.

Library-first (apps / notebooks)::

    from autocausal.insight import (
        InsightSuite,
        InsightReport,
        ExperimentRecommender,
        run_insight_loop,
        run_slm_research_loop,
    )

    suite = InsightSuite(use_slm=True)  # soft-fails to rules without HF
    report = suite.run_loop("data.csv", max_rounds=3, join_sources=["demographics_demo"])
    print(report.to_markdown())
    report.write("insight_report.md")

Epistemic honesty: exploratory discovery ≠ identification. SLM text is
generative assistance only.
"""

from __future__ import annotations

from autocausal.insight.experiments import (
    ExperimentPlan,
    ExperimentRecommendation,
    ExperimentRecommender,
)
from autocausal.insight.report import CAVEATS, InsightReport, RoleHypotheses
from autocausal.insight.narrator import (
    build_insight_report,
    optional_slm_narrative,
    resolve_use_slm,
    rule_narrate,
    synthesize_insight,
    synthesize_summary,
)
from autocausal.insight.suite import (
    InsightSuite,
    demo_insight,
    edge_delta,
    edge_key,
    mine_further,
    run_insight_loop,
    run_slm_research_loop,
)

__all__ = [
    "CAVEATS",
    "ExperimentPlan",
    "ExperimentRecommendation",
    "ExperimentRecommender",
    "InsightReport",
    "InsightSuite",
    "RoleHypotheses",
    "build_insight_report",
    "demo_insight",
    "edge_delta",
    "edge_key",
    "mine_further",
    "optional_slm_narrative",
    "resolve_use_slm",
    "rule_narrate",
    "run_insight_loop",
    "run_slm_research_loop",
    "synthesize_insight",
    "synthesize_summary",
]
