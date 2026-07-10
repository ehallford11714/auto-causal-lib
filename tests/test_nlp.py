"""Library API tests for autocausal.nlp (offline; NLTK optional)."""

from __future__ import annotations

import pandas as pd
import pytest

from autocausal.nlp import (
    NlpFeatureBuilder,
    TextCausalHints,
    analyze,
    extract_causal_hints_from_text,
    nltk_status,
    polarity,
    tokenize,
)


SAMPLE = (
    "A randomized treatment nudge leads to higher compliance and revenue, "
    "associated with age as a confounder because baseline differs."
)


def test_extract_causal_hints_from_text_roles():
    hints = extract_causal_hints_from_text(SAMPLE)
    assert isinstance(hints, TextCausalHints)
    roles = hints.roles.to_dict()
    assert "treatment" in roles
    assert "outcome" in roles
    # Should pick up some modality markers
    assert "leads_to" in hints.modality_markers or "associated_with" in hints.modality_markers
    assert "because" in hints.modality_markers or "associated_with" in hints.modality_markers
    assert "not identify" in hints.caveat.lower() or "linguistic" in hints.caveat.lower()
    ctx = hints.to_guide_context()
    assert "candidates" in ctx
    assert "focus_columns" in ctx


def test_text_causal_hints_extract_classmethod():
    hints = TextCausalHints.extract("lottery assignment causes sales")
    assert hints.backend in ("nltk", "regex", "nltk+fallback")
    assert isinstance(hints.keywords, list)


def test_tokenize_and_polarity_offline():
    toks = tokenize("Hello world because treatment")
    assert "Hello" in toks or "hello" in [t.lower() for t in toks]
    analysis = analyze("Policy affects outcome.")
    assert analysis.tokens
    assert analysis.sentences
    sent = polarity("great success and improvement")
    assert sent.compound >= 0
    assert sent.backend in ("vader", "lexicon_stub")


def test_nlp_feature_builder_frame():
    builder = NlpFeatureBuilder(prefix="nlp_")
    row = builder.row("treatment leads to sales")
    assert "nlp_polarity" in row
    assert row["nlp_mod_leads_to"] == 1 or row["nlp_role_treatment"] == 1
    df = pd.DataFrame({"notes": ["nudge improves habit", "no effect"]})
    out = builder.transform_frame(df, "notes")
    assert len(out) == 2
    assert any(c.startswith("nlp_") for c in out.columns)
    assert set(builder.column_names()).issubset(set(out.columns)) or "nlp_polarity" in out.columns


def test_nltk_status_no_network():
    status = nltk_status()
    d = status.to_dict()
    assert "installed" in d
    assert "resources" in d


def test_top_level_lazy_exports():
    import autocausal as ac

    assert ac.TextCausalHints is TextCausalHints
    assert ac.NlpFeatureBuilder is NlpFeatureBuilder
    assert ac.extract_causal_hints_from_text is extract_causal_hints_from_text


def test_autocausal_from_text_hints():
    from autocausal import AutoCausal

    ac = AutoCausal.from_text_hints(SAMPLE)
    assert ac.nlp_hints is not None
    assert ac.nlp_hints.roles.treatment or ac.nlp_hints.modality_markers
