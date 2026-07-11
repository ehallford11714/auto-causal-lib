"""Deterministic NLP features, fold-safe vectorization, aggregation, and search."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from autocausal.nlp.keywords import extract_modality_markers
from autocausal.nlp.sentiment import polarity


class VectorStoreAdapter(Protocol):
    """Minimal opt-in external vector-store interface."""

    def fit(self, documents: Sequence[str]) -> Any: ...

    def search(self, query: str, top_k: int) -> Sequence[Any]: ...


@dataclass
class NLPFeaturePlan:
    text_columns: list[str]
    vectorizer: str = "tfidf"
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 5_000
    include_readability: bool = True
    include_sentiment: bool = True
    include_modality: bool = True
    embeddings: Optional[str] = None
    fit_scope: str = "training_fold_only"
    notes: list[str] = field(default_factory=list)
    schema: str = "AutoCausalNLPFeaturePlan.v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ngram_range"] = list(self.ngram_range)
        return payload


def _coerce_texts(values: Any) -> list[str]:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            return (
                values.fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .tolist()
            )
        values = values.iloc[:, 0]
    if isinstance(values, pd.Series):
        return values.fillna("").astype(str).tolist()
    array = np.asarray(values, dtype=object)
    if array.ndim == 2:
        if array.shape[1] == 1:
            array = array[:, 0]
        else:
            return [
                " ".join("" if item is None else str(item) for item in row)
                for row in array
            ]
    return ["" if item is None else str(item) for item in array.tolist()]


def _syllable_count(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(groups))


HANDCRAFTED_NAMES = (
    "char_count",
    "word_count",
    "sentence_count",
    "mean_word_length",
    "lexical_diversity",
    "flesch_reading_ease",
    "sentiment_compound",
    "sentiment_positive",
    "sentiment_negative",
    "modality_count",
    "negation_count",
    "uncertainty_count",
    "causal_connector_count",
)


def deterministic_text_features(text: str) -> dict[str, float]:
    value = str(text or "")
    words = re.findall(r"\b[\w'-]+\b", value.lower())
    sentences = [item for item in re.split(r"[.!?]+", value) if item.strip()]
    word_count = len(words)
    sentence_count = max(len(sentences), 1 if value.strip() else 0)
    syllables = sum(_syllable_count(word) for word in words)
    reading_ease = (
        206.835
        - 1.015 * (word_count / max(sentence_count, 1))
        - 84.6 * (syllables / max(word_count, 1))
        if word_count
        else 0.0
    )
    sentiment = polarity(value)
    modality = extract_modality_markers(value)
    lower = value.lower()
    return {
        "char_count": float(len(value)),
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "mean_word_length": (
            float(np.mean([len(word) for word in words])) if words else 0.0
        ),
        "lexical_diversity": len(set(words)) / max(word_count, 1),
        "flesch_reading_ease": float(reading_ease),
        "sentiment_compound": float(sentiment.compound),
        "sentiment_positive": float(sentiment.positive),
        "sentiment_negative": float(sentiment.negative),
        "modality_count": float(len(modality)),
        "negation_count": float(
            len(re.findall(r"\b(?:no|not|never|without|cannot|can't)\b", lower))
        ),
        "uncertainty_count": float(
            len(
                re.findall(
                    r"\b(?:may|might|could|possibly|perhaps|suggest|uncertain)\w*\b",
                    lower,
                )
            )
        ),
        "causal_connector_count": float(
            len(
                re.findall(
                    r"\b(?:cause\w*|leads?\s+to|results?\s+in|affects?|"
                    r"increases?|decreases?|associated\s+with)\b",
                    lower,
                )
            )
        ),
    }


class FoldSafeTextVectorizer:
    """Sklearn-compatible TF-IDF + deterministic text-feature transformer.

    ``fit`` learns vocabulary only from the values supplied by the enclosing
    training fold.  It intentionally exposes ``fit_document_count_`` and a
    vocabulary hash for leakage audits without storing text in reports.
    """

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 5_000,
        min_df: int | float = 1,
        lowercase: bool = True,
        include_handcrafted: bool = True,
        embedding_adapter: Any = None,
        allow_external_text: bool = False,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.lowercase = lowercase
        self.include_handcrafted = include_handcrafted
        self.embedding_adapter = embedding_adapter
        self.allow_external_text = allow_external_text

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "min_df": self.min_df,
            "lowercase": self.lowercase,
            "include_handcrafted": self.include_handcrafted,
            "embedding_adapter": self.embedding_adapter,
            "allow_external_text": self.allow_external_text,
        }

    def set_params(self, **params: Any) -> "FoldSafeTextVectorizer":
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        return self

    def fit(self, values: Any, y: Any = None) -> "FoldSafeTextVectorizer":
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = _coerce_texts(values)
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=tuple(self.ngram_range),
            max_features=int(self.max_features),
            min_df=self.min_df,
            lowercase=bool(self.lowercase),
            strip_accents="unicode",
            sublinear_tf=True,
        )
        try:
            self.vectorizer_.fit(texts)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).lower():
                raise
            self.vectorizer_ = TfidfVectorizer(
                vocabulary={"__empty__": 0},
                lowercase=bool(self.lowercase),
            )
            self.vectorizer_.fit(["__empty__"])
        self.fit_document_count_ = len(texts)
        vocabulary = sorted(self.vectorizer_.vocabulary_.items())
        self.vocabulary_hash_ = hashlib.sha256(
            repr(vocabulary).encode("utf-8")
        ).hexdigest()
        if self.embedding_adapter is not None:
            if not self.allow_external_text:
                raise ValueError(
                    "embedding adapter requires allow_external_text=True; raw "
                    "text is never sent externally by default"
                )
            if hasattr(self.embedding_adapter, "fit"):
                self.embedding_adapter.fit(texts)
        return self

    def transform(self, values: Any) -> Any:
        from scipy import sparse

        if not hasattr(self, "vectorizer_"):
            raise ValueError("FoldSafeTextVectorizer must be fitted before transform")
        texts = _coerce_texts(values)
        matrix = self.vectorizer_.transform(texts)
        pieces = [matrix]
        if self.include_handcrafted:
            handcrafted = np.asarray(
                [
                    list(deterministic_text_features(text).values())
                    for text in texts
                ],
                dtype=float,
            )
            pieces.append(sparse.csr_matrix(handcrafted))
        if self.embedding_adapter is not None:
            if not self.allow_external_text:
                raise ValueError("external embedding transform lacks explicit consent")
            embeddings = np.asarray(self.embedding_adapter.transform(texts), dtype=float)
            pieces.append(sparse.csr_matrix(embeddings))
        return sparse.hstack(pieces, format="csr")

    def fit_transform(self, values: Any, y: Any = None) -> Any:
        return self.fit(values, y).transform(values)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if not hasattr(self, "vectorizer_"):
            raise ValueError("vectorizer is not fitted")
        names = [f"tfidf:{name}" for name in self.vectorizer_.get_feature_names_out()]
        if self.include_handcrafted:
            names.extend(f"text:{name}" for name in HANDCRAFTED_NAMES)
        if self.embedding_adapter is not None:
            dimensions = int(getattr(self.embedding_adapter, "n_features_out_", 0))
            names.extend(f"embedding:{index}" for index in range(dimensions))
        return np.asarray(names, dtype=object)

    def audit(self) -> dict[str, Any]:
        return {
            "fit_document_count": int(getattr(self, "fit_document_count_", 0)),
            "vocabulary_size": int(
                len(getattr(getattr(self, "vectorizer_", None), "vocabulary_", {}))
            ),
            "vocabulary_hash": getattr(self, "vocabulary_hash_", None),
            "fit_scope": "supplied_training_partition",
            "raw_text_included": False,
        }


def aggregate_text_features(
    frame: pd.DataFrame,
    *,
    text_column: str,
    group_columns: Sequence[str],
    time_column: Optional[str] = None,
    frequency: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate deterministic document features to trace/panel rows."""

    required = [text_column, *group_columns]
    if time_column:
        required.append(time_column)
    unknown = [column for column in required if column not in frame]
    if unknown:
        raise KeyError(f"unknown aggregation columns: {unknown}")
    feature_rows = pd.DataFrame(
        [
            deterministic_text_features("" if pd.isna(value) else str(value))
            for value in frame[text_column]
        ],
        index=frame.index,
    )
    work = pd.concat([frame[required].copy(), feature_rows], axis=1)
    groupers = list(group_columns)
    if time_column:
        parsed = pd.to_datetime(work[time_column], errors="coerce")
        if parsed.isna().any():
            raise ValueError("time aggregation requires parseable timestamps")
        if frequency:
            work["_time_bucket"] = parsed.dt.to_period(frequency).dt.to_timestamp()
            groupers.append("_time_bucket")
        else:
            groupers.append(time_column)
    aggregated = (
        work.groupby(groupers, dropna=False, observed=True)[list(HANDCRAFTED_NAMES)]
        .agg(["mean", "sum"])
    )
    aggregated.columns = [
        f"{name}_{operation}" for name, operation in aggregated.columns
    ]
    counts = work.groupby(groupers, dropna=False, observed=True).size().rename("document_count")
    return aggregated.join(counts).reset_index()


@dataclass
class SearchResult:
    index: int
    score: float
    document: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextRetriever:
    """Local cosine retrieval with an explicit-consent vector-store adapter."""

    def __init__(
        self,
        *,
        vector_store: Optional[VectorStoreAdapter] = None,
        allow_external_text: bool = False,
    ) -> None:
        self.vector_store = vector_store
        self.allow_external_text = allow_external_text

    def fit(self, documents: Sequence[str]) -> "TextRetriever":
        texts = [str(document or "") for document in documents]
        if not texts:
            raise ValueError("TextRetriever.fit requires at least one document")
        self._documents = texts
        if self.vector_store is not None:
            if not self.allow_external_text:
                raise ValueError(
                    "vector-store adapter requires allow_external_text=True"
                )
            self.vector_store.fit(texts)
            self.backend_ = "adapter"
            return self
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer_ = TfidfVectorizer(
            ngram_range=(1, 2), max_features=20_000, strip_accents="unicode"
        )
        try:
            self.matrix_ = self.vectorizer_.fit_transform(texts)
        except ValueError:
            self.vectorizer_ = TfidfVectorizer(vocabulary={"__empty__": 0})
            self.matrix_ = self.vectorizer_.fit_transform(["__empty__"] * len(texts))
        self.backend_ = "sklearn_cosine"
        return self

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_documents: bool = False,
    ) -> list[SearchResult]:
        if not hasattr(self, "backend_"):
            raise ValueError("TextRetriever must be fitted before search")
        if self.backend_ == "adapter":
            raw = self.vector_store.search(str(query), int(top_k))  # type: ignore[union-attr]
            output: list[SearchResult] = []
            for rank, value in enumerate(raw):
                if isinstance(value, dict):
                    output.append(
                        SearchResult(
                            index=int(value.get("index", rank)),
                            score=float(value.get("score", 0.0)),
                            document=(
                                str(value.get("document"))
                                if include_documents and value.get("document") is not None
                                else None
                            ),
                        )
                    )
            return output
        from sklearn.metrics.pairwise import cosine_similarity

        query_matrix = self.vectorizer_.transform([str(query)])
        scores = cosine_similarity(query_matrix, self.matrix_).ravel()
        indices = np.argsort(-scores, kind="stable")[: max(1, int(top_k))]
        return [
            SearchResult(
                index=int(index),
                score=round(float(scores[index]), 10),
                document=self._documents[index] if include_documents else None,
            )
            for index in indices
        ]


__all__ = [
    "FoldSafeTextVectorizer",
    "HANDCRAFTED_NAMES",
    "NLPFeaturePlan",
    "SearchResult",
    "TextRetriever",
    "VectorStoreAdapter",
    "aggregate_text_features",
    "deterministic_text_features",
]
