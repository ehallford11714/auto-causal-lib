"""Advanced NLTK NLP tooling for exploratory causal role hints.

Soft-optional NLTK: all entry points work with regex/lexicon fallbacks when
``nltk`` or corpora are missing.

Epistemic honesty: NLP hints are linguistic cues, not identified causation.

Library-first usage (apps / notebooks)::

    from autocausal.nlp import (
        TextCausalHints,
        NlpFeatureBuilder,
        extract_causal_hints_from_text,
        ensure_nltk_data,
    )

    hints = extract_causal_hints_from_text(
        "Randomized treatment leads to higher revenue, controlling for age."
    )
    print(hints.roles.to_dict())
    print(hints.to_guide_context())

    builder = NlpFeatureBuilder()
    features = builder.transform(["because spend increases sales"])
"""

from __future__ import annotations

from autocausal.nlp.backend import (
    DEFAULT_RESOURCES,
    NltkStatus,
    ensure_nltk_data,
    nltk_available,
    nltk_status,
    soft_import_nltk,
)
from autocausal.nlp.builder import NlpFeatureBuilder, extract_causal_hints_from_text
from autocausal.nlp.features import (
    DEFAULT_BOW_VOCAB,
    dataframe_text_features,
    feature_column_names,
    text_to_feature_row,
    texts_to_features,
)
from autocausal.nlp.hints import CAVEAT, RoleCandidates, TextCausalHints, extract_text_hints
from autocausal.nlp.keywords import KeywordExtraction, extract_keywords, extract_modality_markers
from autocausal.nlp.sentiment import SentimentResult, polarity
from autocausal.nlp.tokenize import TokenAnalysis, analyze, lemmatize, pos_tag, sent_tokenize, tokenize

__all__ = [
    "CAVEAT",
    "DEFAULT_BOW_VOCAB",
    "DEFAULT_RESOURCES",
    "KeywordExtraction",
    "NlpFeatureBuilder",
    "NltkStatus",
    "RoleCandidates",
    "SentimentResult",
    "TextCausalHints",
    "TokenAnalysis",
    "analyze",
    "dataframe_text_features",
    "ensure_nltk_data",
    "extract_causal_hints_from_text",
    "extract_keywords",
    "extract_modality_markers",
    "extract_text_hints",
    "feature_column_names",
    "lemmatize",
    "nltk_available",
    "nltk_status",
    "polarity",
    "pos_tag",
    "sent_tokenize",
    "soft_import_nltk",
    "text_to_feature_row",
    "texts_to_features",
    "tokenize",
]
