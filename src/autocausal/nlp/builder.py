"""NlpFeatureBuilder — stable programmatic builder for text → feature columns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import pandas as pd

from autocausal.nlp.features import (
    DEFAULT_BOW_VOCAB,
    dataframe_text_features,
    feature_column_names,
    text_to_feature_row,
    texts_to_features,
)
from autocausal.nlp.hints import TextCausalHints


@dataclass
class NlpFeatureBuilder:
    """Build bag-of-keywords / polarity / modality feature columns from text.

    Embeddable in apps and notebooks::

        from autocausal.nlp import NlpFeatureBuilder

        builder = NlpFeatureBuilder(prefix="nlp_")
        row = builder.row("Nudge treatment leads to higher compliance")
        df = builder.transform_frame(df, text_col="notes")
    """

    vocab: Sequence[str] = field(default_factory=lambda: list(DEFAULT_BOW_VOCAB))
    prefix: str = "nlp_"

    def row(self, text: str) -> dict[str, Any]:
        return text_to_feature_row(text, vocab=self.vocab, prefix=self.prefix)

    def transform(self, texts: Sequence[str]) -> pd.DataFrame:
        return texts_to_features(texts, vocab=self.vocab, prefix=self.prefix)

    def transform_frame(
        self,
        df: pd.DataFrame,
        text_col: str,
        *,
        drop_text: bool = False,
    ) -> pd.DataFrame:
        return dataframe_text_features(
            df,
            text_col,
            vocab=self.vocab,
            prefix=self.prefix,
            drop_text=drop_text,
        )

    def column_names(self) -> list[str]:
        return feature_column_names(vocab=self.vocab, prefix=self.prefix)

    def hints(self, text: str) -> TextCausalHints:
        """Convenience: extract TextCausalHints for the same text."""
        return TextCausalHints.extract(text)


def extract_causal_hints_from_text(text: str, **kwargs: Any) -> TextCausalHints:
    """Stable alias: free text → :class:`TextCausalHints` for guide/discover."""
    return TextCausalHints.extract(text, **kwargs)


__all__ = [
    "NlpFeatureBuilder",
    "extract_causal_hints_from_text",
    "DEFAULT_BOW_VOCAB",
]
