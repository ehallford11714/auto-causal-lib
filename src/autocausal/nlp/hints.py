"""TextCausalHints — extract causal role candidates from free text.

Epistemic note: these are *linguistic* hints for guide/discover pipelines,
not identified causal relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from autocausal.nlp.keywords import KeywordExtraction, extract_keywords
from autocausal.nlp.sentiment import SentimentResult, polarity
from autocausal.nlp.tokenize import TokenAnalysis, analyze


CAVEAT = (
    "NLP causal hints are exploratory linguistic cues only — "
    "they do not identify causation and must not be treated as causal estimates."
)


@dataclass
class RoleCandidates:
    """Role → candidate phrases/tokens for guide/discover focus."""

    treatment: list[str] = field(default_factory=list)
    outcome: list[str] = field(default_factory=list)
    confounder: list[str] = field(default_factory=list)
    instrument: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "treatment": list(self.treatment),
            "outcome": list(self.outcome),
            "confounder": list(self.confounder),
            "instrument": list(self.instrument),
        }

    def focus_columns(self, *, limit: int = 12) -> list[str]:
        """Flatten role candidates into a focus list (stable order)."""
        ordered: list[str] = []
        seen: set[str] = set()
        for bucket in (self.treatment, self.outcome, self.instrument, self.confounder):
            for item in bucket:
                key = item.lower().replace(" ", "_")
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
                if len(ordered) >= limit:
                    return ordered
        return ordered


@dataclass
class TextCausalHints:
    """Structured NLP hints usable by guide / discover pipelines."""

    text: str
    roles: RoleCandidates
    keywords: list[str] = field(default_factory=list)
    noun_phrases: list[str] = field(default_factory=list)
    modality_markers: list[str] = field(default_factory=list)
    sentiment: Optional[dict[str, Any]] = None
    analysis: Optional[dict[str, Any]] = None
    backend: str = "regex"
    notes: list[str] = field(default_factory=list)
    caveat: str = CAVEAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "roles": self.roles.to_dict(),
            "keywords": list(self.keywords),
            "noun_phrases": list(self.noun_phrases),
            "modality_markers": list(self.modality_markers),
            "sentiment": self.sentiment,
            "analysis": self.analysis,
            "backend": self.backend,
            "focus_columns": self.roles.focus_columns(),
            "notes": list(self.notes),
            "caveat": self.caveat,
        }

    def to_guide_context(self) -> dict[str, Any]:
        """Shape compatible with AutoCausal._guide_context / direct()."""
        roles = self.roles.to_dict()
        return {
            "text": self.text,
            "nlp_hints": self.to_dict(),
            "candidates": {
                "treatment": roles["treatment"],
                "outcome": roles["outcome"],
                "confounders": roles["confounder"],
                "instruments": roles["instrument"],
            },
            "focus_columns": self.roles.focus_columns(),
            "modality_markers": list(self.modality_markers),
            "notes": list(self.notes) + [self.caveat],
        }

    @classmethod
    def extract(cls, text: str, *, include_analysis: bool = True) -> "TextCausalHints":
        """Extract roles/candidates from free text for guide/discover."""
        text = text or ""
        kw: KeywordExtraction = extract_keywords(text)
        sent: SentimentResult = polarity(text)
        tok: Optional[TokenAnalysis] = analyze(text) if include_analysis else None

        roles = RoleCandidates()
        for hit in kw.role_hits:
            bucket = getattr(roles, hit.role, None)
            if bucket is None:
                continue
            token = hit.token
            if token not in bucket:
                bucket.append(token)

        # Also pull short noun phrases that look role-like into treatment/outcome
        for np in kw.noun_phrases:
            if len(np.split()) > 4:
                continue
            low = np.lower()
            if any(c in low for c in ("treat", "interven", "nudg", "policy", "stimul")):
                if np not in roles.treatment:
                    roles.treatment.append(np)
            if any(c in low for c in ("outcome", "revenue", "sales", "effect", "impact", "habit")):
                if np not in roles.outcome:
                    roles.outcome.append(np)

        notes = list(kw.notes)
        notes.append(CAVEAT)
        return cls(
            text=text,
            roles=roles,
            keywords=list(kw.keywords),
            noun_phrases=list(kw.noun_phrases),
            modality_markers=list(kw.modality_markers),
            sentiment=sent.to_dict(),
            analysis=tok.to_dict() if tok is not None else None,
            backend=kw.backend,
            notes=notes,
        )


def extract_text_hints(text: str, **kwargs: Any) -> TextCausalHints:
    """Module-level alias for :meth:`TextCausalHints.extract`."""
    return TextCausalHints.extract(text, **kwargs)


__all__ = [
    "CAVEAT",
    "RoleCandidates",
    "TextCausalHints",
    "extract_text_hints",
]
