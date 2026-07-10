"""Text → feature columns for causal mining (bag-of-keywords, polarity, modality)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd

from autocausal.nlp.keywords import (
    MODALITY_PATTERNS,
    extract_keywords,
    extract_modality_markers,
)
from autocausal.nlp.sentiment import polarity


# Default keyword vocabulary used as bag-of-keywords columns
DEFAULT_BOW_VOCAB = (
    "treatment",
    "intervention",
    "nudge",
    "stimulus",
    "outcome",
    "effect",
    "impact",
    "revenue",
    "sales",
    "confounder",
    "covariate",
    "baseline",
    "instrument",
    "random",
    "lottery",
    "habit",
    "compliance",
    "reward",
    "exposure",
    "policy",
)


def text_to_feature_row(
    text: str,
    *,
    vocab: Optional[Sequence[str]] = None,
    prefix: str = "nlp_",
) -> dict[str, Any]:
    """Convert one text into a flat feature dict for tabular mining."""
    text = text or ""
    vocab = tuple(vocab) if vocab is not None else DEFAULT_BOW_VOCAB
    lower = text.lower()
    kw = extract_keywords(text)
    sent = polarity(text)
    modality = extract_modality_markers(text)

    row: dict[str, Any] = {
        f"{prefix}polarity": sent.compound,
        f"{prefix}pos": sent.positive,
        f"{prefix}neg": sent.negative,
        f"{prefix}n_tokens": len(kw.keywords),
        f"{prefix}n_role_hits": len(kw.role_hits),
        f"{prefix}n_modality": len(modality),
        f"{prefix}backend": kw.backend,
    }
    for term in vocab:
        row[f"{prefix}kw_{term}"] = 1 if term in lower or term in kw.keywords else 0
    for _pat, name in MODALITY_PATTERNS:
        row[f"{prefix}mod_{name}"] = 1 if name in modality else 0

    roles = {h.role for h in kw.role_hits}
    for role in ("treatment", "outcome", "confounder", "instrument"):
        row[f"{prefix}role_{role}"] = 1 if role in roles else 0
    return row


def texts_to_features(
    texts: Sequence[str],
    *,
    vocab: Optional[Sequence[str]] = None,
    prefix: str = "nlp_",
) -> pd.DataFrame:
    """Vectorize a sequence of texts into a feature DataFrame."""
    rows = [text_to_feature_row(t, vocab=vocab, prefix=prefix) for t in texts]
    return pd.DataFrame(rows)


def dataframe_text_features(
    df: pd.DataFrame,
    text_col: str,
    *,
    vocab: Optional[Sequence[str]] = None,
    prefix: str = "nlp_",
    drop_text: bool = False,
) -> pd.DataFrame:
    """Append NLP feature columns derived from ``text_col`` onto a copy of ``df``."""
    if text_col not in df.columns:
        raise KeyError(f"text column {text_col!r} not in dataframe")
    feats = texts_to_features(
        ["" if pd.isna(x) else str(x) for x in df[text_col].tolist()],
        vocab=vocab,
        prefix=prefix,
    )
    out = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    if drop_text:
        out = out.drop(columns=[text_col])
    return out


def feature_column_names(
    *,
    vocab: Optional[Sequence[str]] = None,
    prefix: str = "nlp_",
) -> list[str]:
    """List feature column names produced by :func:`text_to_feature_row`."""
    vocab = tuple(vocab) if vocab is not None else DEFAULT_BOW_VOCAB
    names = [
        f"{prefix}polarity",
        f"{prefix}pos",
        f"{prefix}neg",
        f"{prefix}n_tokens",
        f"{prefix}n_role_hits",
        f"{prefix}n_modality",
        f"{prefix}backend",
    ]
    names.extend(f"{prefix}kw_{t}" for t in vocab)
    names.extend(f"{prefix}mod_{name}" for _pat, name in MODALITY_PATTERNS)
    names.extend(
        f"{prefix}role_{r}" for r in ("treatment", "outcome", "confounder", "instrument")
    )
    return names


__all__ = [
    "DEFAULT_BOW_VOCAB",
    "text_to_feature_row",
    "texts_to_features",
    "dataframe_text_features",
    "feature_column_names",
]
