"""Privacy-aware AutoNLP built on ``autocausal.nlp`` fallbacks."""

from autocausal.autonlp.features import (
    FoldSafeTextVectorizer,
    HANDCRAFTED_NAMES,
    NLPFeaturePlan,
    SearchResult,
    TextRetriever,
    VectorStoreAdapter,
    aggregate_text_features,
    deterministic_text_features,
)
from autocausal.autonlp.profile import (
    NLPProfile,
    TextColumnProfile,
    detect_text_columns,
    profile_text_frame,
    redact_sensitive_text,
    sensitive_risk_counts,
)
from autocausal.autonlp.report import AutoNLPReport, NLP_CAVEAT
from autocausal.autonlp.roles import (
    CausalClaim,
    RoleHypothesis,
    extract_causal_claims,
    extract_role_hypotheses,
    role_hypotheses_to_guide_context,
)
from autocausal.autonlp.suite import AutoNLPSuite, StructuredNLPEnricher

__all__ = [
    "AutoNLPReport",
    "AutoNLPSuite",
    "CausalClaim",
    "FoldSafeTextVectorizer",
    "HANDCRAFTED_NAMES",
    "NLPFeaturePlan",
    "NLPProfile",
    "NLP_CAVEAT",
    "RoleHypothesis",
    "SearchResult",
    "StructuredNLPEnricher",
    "TextColumnProfile",
    "TextRetriever",
    "VectorStoreAdapter",
    "aggregate_text_features",
    "detect_text_columns",
    "deterministic_text_features",
    "extract_causal_claims",
    "extract_role_hypotheses",
    "profile_text_frame",
    "redact_sensitive_text",
    "role_hypotheses_to_guide_context",
    "sensitive_risk_counts",
]
