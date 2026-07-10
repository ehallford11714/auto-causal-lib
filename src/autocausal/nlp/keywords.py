"""Keyword / noun-phrase extraction for causal role hints."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from autocausal.nlp.tokenize import analyze, pos_tag, tokenize


# Lexicons for causal-role language (exploratory hints only)
TREATMENT_CUES = frozenset(
    {
        "treatment",
        "treat",
        "intervention",
        "policy",
        "program",
        "nudge",
        "stimulus",
        "exposure",
        "assign",
        "assignment",
        "dose",
        "therapy",
        "campaign",
        "incentive",
        "subsidy",
        "training",
    }
)
OUTCOME_CUES = frozenset(
    {
        "outcome",
        "result",
        "effect",
        "impact",
        "revenue",
        "sales",
        "profit",
        "conversion",
        "retention",
        "churn",
        "mortality",
        "survival",
        "score",
        "performance",
        "y",
        "response",
        "behavior",
        "behaviour",
        "compliance",
        "habit",
    }
)
CONFOUNDER_CUES = frozenset(
    {
        "confound",
        "confounder",
        "covariate",
        "control",
        "baseline",
        "age",
        "gender",
        "sex",
        "income",
        "education",
        "region",
        "season",
        "prior",
        "history",
        "context",
    }
)
INSTRUMENT_CUES = frozenset(
    {
        "instrument",
        "lottery",
        "random",
        "randomized",
        "randomised",
        "iv",
        "exogenous",
        "rainfall",
        "judge",
        "distance",
        "eligibility",
    }
)

MODALITY_PATTERNS: list[tuple[str, str]] = [
    (r"\bbecause\b", "because"),
    (r"\bcauses?\b", "causes"),
    (r"\bcaused\b", "caused"),
    (r"\bleads?\s+to\b", "leads_to"),
    (r"\bresulting\s+in\b", "resulting_in"),
    (r"\bdue\s+to\b", "due_to"),
    (r"\bassociated\s+with\b", "associated_with"),
    (r"\bcorrelated\s+with\b", "correlated_with"),
    (r"\bincreases?\b", "increases"),
    (r"\bdecreases?\b", "decreases"),
    (r"\baffects?\b", "affects"),
    (r"\binfluences?\b", "influences"),
    (r"\bif\b.+\bthen\b", "if_then"),
]


@dataclass
class KeywordHit:
    token: str
    role: str
    score: float = 1.0
    source: str = "lexicon"

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "role": self.role,
            "score": self.score,
            "source": self.source,
        }


@dataclass
class KeywordExtraction:
    keywords: list[str]
    noun_phrases: list[str]
    role_hits: list[KeywordHit]
    modality_markers: list[str]
    backend: str = "regex"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": list(self.keywords),
            "noun_phrases": list(self.noun_phrases),
            "role_hits": [h.to_dict() for h in self.role_hits],
            "modality_markers": list(self.modality_markers),
            "backend": self.backend,
            "notes": list(self.notes),
        }


def extract_modality_markers(text: str) -> list[str]:
    """Detect causal/associative modality phrases in text."""
    lower = (text or "").lower()
    found: list[str] = []
    for pat, name in MODALITY_PATTERNS:
        if re.search(pat, lower):
            found.append(name)
    return found


def extract_noun_phrases(text: str, *, max_phrases: int = 24, max_tokens: int = 4) -> list[str]:
    """Simple NP chunks from POS tags (DT? JJ* NN+), capped length."""
    analysis = analyze(text)
    tagged = analysis.pos or pos_tag(tokenize(text))
    phrases: list[str] = []
    buf: list[str] = []

    def _flush() -> None:
        nonlocal buf
        if not buf:
            return
        # Keep short content NPs only (avoid whole-sentence heuristic chunks)
        if 1 <= len(buf) <= max_tokens and any(c.isalpha() for c in "".join(buf)):
            phrases.append(" ".join(buf))
        buf = []

    for tok, tag in tagged:
        if tag.startswith("DT") and not buf:
            buf = [tok]
        elif tag.startswith("JJ") and len(buf) < max_tokens:
            buf.append(tok)
        elif tag.startswith("NN") and len(buf) < max_tokens:
            buf.append(tok)
        else:
            _flush()
    _flush()
    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        key = p.lower()
        if key not in seen and len(key) > 1:
            seen.add(key)
            out.append(p)
        if len(out) >= max_phrases:
            break
    return out


def _role_for_token(tok: str) -> Optional[str]:
    t = tok.lower().strip()
    # Prefer full cue words; allow light stems without truncating display tokens upstream
    if t in TREATMENT_CUES or any(t.startswith(c) for c in ("treat", "interven", "nudg")):
        return "treatment"
    if t in OUTCOME_CUES or any(t.startswith(c) for c in ("outcom", "revenu", "perform")):
        return "outcome"
    if t in CONFOUNDER_CUES or t.startswith("confound"):
        return "confounder"
    if t in INSTRUMENT_CUES or t.startswith("random") or t in {"randomiz", "randomise"}:
        return "instrument"
    return None


def extract_keywords(text: str, *, top_k: int = 40) -> KeywordExtraction:
    """Extract keywords, NPs, and causal-role lexicon hits."""
    analysis = analyze(text)
    tokens = [t.lower() for t in analysis.tokens if any(c.isalpha() for c in t)]
    # Prefer lemmas when present
    lemmas = [L.lower() for L in analysis.lemmas] if analysis.lemmas else tokens

    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "with",
        "as",
        "by",
        "at",
        "from",
        "that",
        "this",
        "it",
        "we",
        "you",
        "they",
        "i",
    }
    keywords: list[str] = []
    seen: set[str] = set()
    for t in lemmas + tokens:
        if t in stop or len(t) < 2:
            continue
        if t not in seen:
            seen.add(t)
            keywords.append(t)
        if len(keywords) >= top_k:
            break

    role_hits: list[KeywordHit] = []
    hit_keys: set[tuple[str, str]] = set()
    for t in keywords:
        role = _role_for_token(t)
        if role and (t, role) not in hit_keys:
            hit_keys.add((t, role))
            role_hits.append(KeywordHit(token=t, role=role, score=1.0, source="lexicon"))

    nps = extract_noun_phrases(text)
    for np in nps:
        for part in np.lower().split():
            role = _role_for_token(part)
            if role and (np.lower(), role) not in hit_keys:
                hit_keys.add((np.lower(), role))
                role_hits.append(
                    KeywordHit(token=np, role=role, score=0.8, source="noun_phrase")
                )

    modality = extract_modality_markers(text)
    notes = list(analysis.notes)
    notes.append(
        "Keyword/role hits are linguistic cues only — not identified causation."
    )
    return KeywordExtraction(
        keywords=keywords,
        noun_phrases=nps,
        role_hits=role_hits,
        modality_markers=modality,
        backend=analysis.backend,
        notes=notes,
    )
