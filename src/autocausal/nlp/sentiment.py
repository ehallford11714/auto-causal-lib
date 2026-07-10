"""Light sentiment / polarity features (VADER when available, else lexicon stub)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autocausal.nlp.backend import resource_available, soft_import_nltk
from autocausal.nlp.tokenize import tokenize


# Minimal polarity lexicon for offline fallback
_POS = frozenset(
    {
        "good",
        "great",
        "excellent",
        "positive",
        "improve",
        "improved",
        "improvement",
        "success",
        "successful",
        "gain",
        "increase",
        "increased",
        "benefit",
        "beneficial",
        "happy",
        "love",
        "best",
        "effective",
        "helpful",
        "reward",
        "rewarding",
    }
)
_NEG = frozenset(
    {
        "bad",
        "poor",
        "negative",
        "worse",
        "worst",
        "fail",
        "failed",
        "failure",
        "loss",
        "decrease",
        "decreased",
        "harm",
        "harmful",
        "sad",
        "hate",
        "risk",
        "risky",
        "problem",
        "issue",
        "punish",
        "punishment",
        "aversive",
    }
)


@dataclass
class SentimentResult:
    compound: float
    positive: float
    negative: float
    neutral: float
    backend: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compound": self.compound,
            "positive": self.positive,
            "negative": self.negative,
            "neutral": self.neutral,
            "backend": self.backend,
            "notes": list(self.notes),
        }


def polarity(text: str) -> SentimentResult:
    """Return polarity scores; prefer NLTK VADER, else lexicon stub."""
    text = text or ""
    nltk = soft_import_nltk()
    if nltk is not None and resource_available("vader_lexicon", nltk):
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore

            sia = SentimentIntensityAnalyzer()
            scores = sia.polarity_scores(text)
            return SentimentResult(
                compound=float(scores.get("compound", 0.0)),
                positive=float(scores.get("pos", 0.0)),
                negative=float(scores.get("neg", 0.0)),
                neutral=float(scores.get("neu", 0.0)),
                backend="vader",
            )
        except Exception as e:
            # fall through to lexicon
            note = f"VADER soft-fail: {type(e).__name__}: {e}"
        else:
            note = None
    else:
        note = None

    tokens = [t.lower() for t in tokenize(text) if t.isalpha()]
    if not tokens:
        return SentimentResult(
            compound=0.0,
            positive=0.0,
            negative=0.0,
            neutral=1.0,
            backend="lexicon_stub",
            notes=[n for n in [note, "empty text"] if n],
        )
    pos_n = sum(1 for t in tokens if t in _POS)
    neg_n = sum(1 for t in tokens if t in _NEG)
    total = len(tokens)
    pos = pos_n / total
    neg = neg_n / total
    neu = max(0.0, 1.0 - pos - neg)
    # Map to roughly VADER-like compound in [-1, 1]
    compound = max(-1.0, min(1.0, (pos_n - neg_n) / max(1.0, (pos_n + neg_n) ** 0.5)))
    notes = ["lexicon stub polarity (install nltk + vader_lexicon for VADER)"]
    if note:
        notes.insert(0, note)
    if nltk is None:
        notes.append("nltk not installed")
    elif not resource_available("vader_lexicon", nltk):
        notes.append("vader_lexicon missing")
    return SentimentResult(
        compound=round(compound, 4),
        positive=round(pos, 4),
        negative=round(neg, 4),
        neutral=round(neu, 4),
        backend="lexicon_stub",
        notes=notes,
    )
