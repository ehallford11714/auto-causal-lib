"""SLM-guided, citation-grounded deep research for AutoCausal findings.

The package is offline-first.  Network providers require both an allowlisted
provider and explicit network consent in :class:`ResearchPolicy`.
"""

from autocausal.research.adapters import (
    install_research_adapters,
    research_escalation_node,
    research_handback_context,
)
from autocausal.research.evidence import (
    CrossMatchEngine,
    EvidenceExtractor,
    citation_integrity_errors,
    deduplicate_sources,
    expand_related_work_queries,
    extract_reference_identifiers,
    match_prior_sources,
    source_independence_groups,
    title_fingerprint,
)
from autocausal.research.handoff import (
    handoff_from_gate_report,
    redact_context,
    safe_variable_label,
    to_research_handoff,
)
from autocausal.research.models import (
    BudgetUsage,
    ClaimEvidenceGraph,
    ComparabilityScore,
    CrossMatch,
    EvidenceSpan,
    IntensityRecommendation,
    MatchReason,
    ResearchBudget,
    ResearchClaim,
    ResearchHandoff,
    ResearchPolicy,
    ResearchQuestion,
    ResearchReport,
    SearchIntensity,
    SourceRecord,
)
from autocausal.research.planning import AgendaPlanner, IntensityRouter
from autocausal.research.providers import (
    ArxivProvider,
    CrossrefProvider,
    GenericWebSearchProvider,
    LocalDocumentProvider,
    OpenAlexProvider,
    ProviderError,
    ProviderQuery,
    ResearchCache,
    SemanticScholarProvider,
)
from autocausal.research.slm import (
    HuggingFaceResearchSLM,
    StructuredOutputError,
)
from autocausal.research.suite import (
    DEEPEN_NODES,
    WORKFLOW_NODES,
    DeepResearchSuite,
    PrivacyGateError,
    ResearchApprovalRequired,
    ResearchLimitStop,
    ResearchPolicyError,
)


__all__ = [
    "AgendaPlanner",
    "ArxivProvider",
    "BudgetUsage",
    "ClaimEvidenceGraph",
    "ComparabilityScore",
    "CrossMatch",
    "CrossMatchEngine",
    "CrossrefProvider",
    "DEEPEN_NODES",
    "DeepResearchSuite",
    "EvidenceExtractor",
    "EvidenceSpan",
    "GenericWebSearchProvider",
    "HuggingFaceResearchSLM",
    "IntensityRecommendation",
    "IntensityRouter",
    "LocalDocumentProvider",
    "MatchReason",
    "OpenAlexProvider",
    "PrivacyGateError",
    "ProviderError",
    "ProviderQuery",
    "ResearchApprovalRequired",
    "ResearchBudget",
    "ResearchCache",
    "ResearchClaim",
    "ResearchHandoff",
    "ResearchLimitStop",
    "ResearchPolicy",
    "ResearchPolicyError",
    "ResearchQuestion",
    "ResearchReport",
    "SearchIntensity",
    "SemanticScholarProvider",
    "SourceRecord",
    "StructuredOutputError",
    "WORKFLOW_NODES",
    "citation_integrity_errors",
    "deduplicate_sources",
    "expand_related_work_queries",
    "extract_reference_identifiers",
    "handoff_from_gate_report",
    "install_research_adapters",
    "match_prior_sources",
    "redact_context",
    "research_escalation_node",
    "research_handback_context",
    "safe_variable_label",
    "source_independence_groups",
    "title_fingerprint",
    "to_research_handoff",
]


# Conflict-light ergonomic methods. Existing methods always win.
ADAPTER_INSTALLATION = install_research_adapters()
