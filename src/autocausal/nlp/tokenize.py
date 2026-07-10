"""Tokenize / sentence split / POS / lemmatize with NLTK or regex fallbacks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from autocausal.nlp.backend import resource_available, soft_import_nltk


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class TokenAnalysis:
    """Structured NLP analysis of a text span."""

    text: str
    tokens: list[str]
    sentences: list[str]
    pos: list[tuple[str, str]] = field(default_factory=list)
    lemmas: list[str] = field(default_factory=list)
    backend: str = "regex"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": list(self.tokens),
            "sentences": list(self.sentences),
            "pos": [{"token": t, "tag": p} for t, p in self.pos],
            "lemmas": list(self.lemmas),
            "backend": self.backend,
            "notes": list(self.notes),
        }


def tokenize(text: str) -> list[str]:
    """Word tokenize; NLTK when available, else regex."""
    text = text or ""
    nltk = soft_import_nltk()
    if nltk is not None and (
        resource_available("punkt", nltk) or resource_available("punkt_tab", nltk)
    ):
        try:
            from nltk.tokenize import word_tokenize

            return word_tokenize(text)
        except Exception:
            pass
    return _WORD_RE.findall(text)


def sent_tokenize(text: str) -> list[str]:
    """Sentence split; NLTK when available, else naive regex."""
    text = (text or "").strip()
    if not text:
        return []
    nltk = soft_import_nltk()
    if nltk is not None and (
        resource_available("punkt", nltk) or resource_available("punkt_tab", nltk)
    ):
        try:
            from nltk.tokenize import sent_tokenize as nltk_sent

            return [s.strip() for s in nltk_sent(text) if s.strip()]
        except Exception:
            pass
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    """POS tag tokens; NLTK when available, else heuristic NN/VB/JJ stubs."""
    if not tokens:
        return []
    nltk = soft_import_nltk()
    if nltk is not None and (
        resource_available("averaged_perceptron_tagger", nltk)
        or resource_available("averaged_perceptron_tagger_eng", nltk)
    ):
        try:
            return list(nltk.pos_tag(tokens))
        except Exception:
            pass
    # Heuristic fallback
    function = {
        "to": "TO",
        "of": "IN",
        "in": "IN",
        "on": "IN",
        "for": "IN",
        "with": "IN",
        "from": "IN",
        "by": "IN",
        "as": "IN",
        "at": "IN",
        "the": "DT",
        "a": "DT",
        "an": "DT",
        "and": "CC",
        "or": "CC",
        "but": "CC",
        "because": "IN",
        "if": "IN",
        "then": "RB",
        "is": "VBZ",
        "are": "VBP",
        "was": "VBD",
        "were": "VBD",
        "be": "VB",
        "leads": "VBZ",
        "lead": "VB",
        "causes": "VBZ",
        "cause": "VB",
        "caused": "VBD",
        "affects": "VBZ",
        "affect": "VB",
        "increases": "VBZ",
        "decreases": "VBZ",
    }
    out: list[tuple[str, str]] = []
    for t in tokens:
        lower = t.lower()
        if lower in function:
            out.append((t, function[lower]))
        elif lower.endswith(("ing", "ed", "ize", "ise")):
            out.append((t, "VB"))
        elif lower.endswith(("ly",)):
            out.append((t, "RB"))
        elif lower.endswith(("ous", "ful", "ive", "al", "ic")):
            out.append((t, "JJ"))
        elif t[:1].isupper() and len(t) > 1 and t[1:].islower():
            out.append((t, "NNP"))
        else:
            out.append((t, "NN"))
    return out


def lemmatize(tokens: list[str], pos: Optional[list[tuple[str, str]]] = None) -> list[str]:
    """Lemmatize with WordNet when available; else light suffix stripping."""
    if not tokens:
        return []
    nltk = soft_import_nltk()
    if nltk is not None and resource_available("wordnet", nltk):
        try:
            from nltk.stem import WordNetLemmatizer

            lemmatizer = WordNetLemmatizer()
            tagged = pos if pos is not None else pos_tag(tokens)
            lemmas: list[str] = []
            for tok, tag in tagged:
                wn = _penn_to_wn(tag)
                if wn:
                    lemmas.append(lemmatizer.lemmatize(tok.lower(), wn))
                else:
                    lemmas.append(lemmatizer.lemmatize(tok.lower()))
            return lemmas
        except Exception:
            pass
    return [_simple_lemma(t) for t in tokens]


def analyze(text: str) -> TokenAnalysis:
    """Full tokenize + sentence + POS + lemma pass with backend notes."""
    text = text or ""
    notes: list[str] = []
    nltk = soft_import_nltk()
    backend = "nltk" if nltk is not None else "regex"
    if nltk is None:
        notes.append("nltk not installed; used regex/heuristic fallbacks")

    sentences = sent_tokenize(text)
    tokens = tokenize(text)
    # Prefer lowercased alpha tokens for POS/lemma of content words
    content = [t for t in tokens if any(c.isalpha() for c in t)]
    tagged = pos_tag(content)
    lemmas = lemmatize(content, tagged)

    if nltk is not None:
        if not (
            resource_available("punkt", nltk) or resource_available("punkt_tab", nltk)
        ):
            notes.append("punkt missing — sentence/tokenize fallback")
            backend = "nltk+fallback"
        if not (
            resource_available("averaged_perceptron_tagger", nltk)
            or resource_available("averaged_perceptron_tagger_eng", nltk)
        ):
            notes.append("POS tagger missing — heuristic tags")
            backend = "nltk+fallback"
        if not resource_available("wordnet", nltk):
            notes.append("wordnet missing — suffix lemma stub")
            backend = "nltk+fallback"

    return TokenAnalysis(
        text=text,
        tokens=tokens,
        sentences=sentences,
        pos=tagged,
        lemmas=lemmas,
        backend=backend,
        notes=notes,
    )


def _penn_to_wn(tag: str) -> Optional[str]:
    if not tag:
        return None
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return None


def _simple_lemma(token: str) -> str:
    t = token.lower()
    for suf in ("ingly", "edly", "ing", "ed", "ies", "es", "s"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            if suf == "ies":
                return t[:-3] + "y"
            return t[: -len(suf)]
    return t
