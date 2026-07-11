"""Causal-language claims, role hypotheses, and evidence offsets."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from autocausal.autonlp.profile import redact_sensitive_text


_CONNECTOR = re.compile(
    r"\b(?P<connector>"
    r"causes?|caused|leads?\s+to|results?\s+in|affects?|"
    r"increases?|decreases?|improves?|reduces?|drives?|"
    r"is\s+associated\s+with|are\s+associated\s+with|associated\s+with"
    r")\b",
    re.I,
)
_NEGATION = re.compile(
    r"\b(?:no|not|never|neither|without|doesn['’]?t|didn['’]?t|cannot|can['’]?t)\b",
    re.I,
)
_UNCERTAINTY = re.compile(
    r"\b(?:may|might|could|possibly|perhaps|suggests?|appears?|"
    r"likely|unlikely|uncertain|associated)\b",
    re.I,
)
_ROLE_TERMS: dict[str, re.Pattern[str]] = {
    "treatment": re.compile(
        r"\b(?:treatment|intervention|exposure|dose|policy|nudge|stimulus|program)\b",
        re.I,
    ),
    "outcome": re.compile(
        r"\b(?:outcome|response|revenue|sales|mortality|recovery|compliance|"
        r"conversion|retention|performance|score)\b",
        re.I,
    ),
    "instrument": re.compile(
        r"\b(?:instrument|lottery|random(?:ized)?\s+assignment|encouragement|"
        r"eligibility|distance)\b",
        re.I,
    ),
    "confounder": re.compile(
        r"\b(?:confounder|covariate|baseline|age|severity|socioeconomic|"
        r"prior\s+\w+)\b",
        re.I,
    ),
}


@dataclass
class RoleHypothesis:
    role: str
    text: str
    start: int
    end: int
    confidence: float
    basis: str = "lexical"
    hypothesis: bool = True

    def to_dict(self, *, production: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if production:
            payload["text"] = redact_sensitive_text(self.text)[0]
        return payload


@dataclass
class CausalClaim:
    id: str
    treatment: str
    outcome: str
    start: int
    end: int
    evidence_span: str
    connector: str
    modality: list[str] = field(default_factory=list)
    negated: bool = False
    uncertain: bool = False
    instruments: list[str] = field(default_factory=list)
    confounders: list[str] = field(default_factory=list)
    confidence: float = 0.5
    source_column: Optional[str] = None
    document_index: Optional[int] = None
    hypothesis: bool = True
    caveat: str = (
        "Linguistic causal wording is a hypothesis, not evidence of causal identification."
    )

    def to_dict(self, *, production: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if production:
            for key in ("treatment", "outcome", "evidence_span"):
                payload[key] = redact_sensitive_text(str(payload[key]))[0]
            payload["instruments"] = [
                redact_sensitive_text(value)[0] for value in self.instruments
            ]
            payload["confounders"] = [
                redact_sensitive_text(value)[0] for value in self.confounders
            ]
        return payload


def _clean_argument(text: str, *, left: bool) -> str:
    value = re.sub(r"\s+", " ", text).strip(" ,;:-")
    words = value.split()
    if len(words) > 12:
        words = words[-12:] if left else words[:12]
    return " ".join(words)


def extract_role_hypotheses(text: str) -> list[RoleHypothesis]:
    output: list[RoleHypothesis] = []
    for role, pattern in _ROLE_TERMS.items():
        for match in pattern.finditer(text or ""):
            output.append(
                RoleHypothesis(
                    role=role,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.68 if role in ("treatment", "outcome") else 0.58,
                )
            )
    return sorted(output, key=lambda value: (value.start, value.role))


def extract_causal_claims(
    text: str,
    *,
    source_column: Optional[str] = None,
    document_index: Optional[int] = None,
) -> list[CausalClaim]:
    """Extract connector-based claims with absolute evidence offsets."""

    source = str(text or "")
    claims: list[CausalClaim] = []
    sentence_pattern = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
    for sentence_match in sentence_pattern.finditer(source):
        sentence = sentence_match.group(0)
        for connector_match in _CONNECTOR.finditer(sentence):
            left = _clean_argument(sentence[: connector_match.start()], left=True)
            right = _clean_argument(sentence[connector_match.end() :], left=False)
            if not left or not right:
                continue
            absolute_start = sentence_match.start()
            absolute_end = sentence_match.end()
            role_hypotheses = extract_role_hypotheses(sentence)
            instruments = [
                hypothesis.text
                for hypothesis in role_hypotheses
                if hypothesis.role == "instrument"
            ]
            confounders = [
                hypothesis.text
                for hypothesis in role_hypotheses
                if hypothesis.role == "confounder"
            ]
            connector = re.sub(
                r"\s+", "_", connector_match.group("connector").lower()
            )
            negated = bool(_NEGATION.search(sentence))
            uncertain = bool(_UNCERTAINTY.search(sentence))
            modalities: list[str] = []
            if negated:
                modalities.append("negation")
            if uncertain:
                modalities.append("uncertainty")
            if "associated" in connector:
                modalities.append("association")
                uncertain = True
            confidence = 0.72
            if uncertain:
                confidence -= 0.18
            if negated:
                confidence -= 0.25
            claims.append(
                CausalClaim(
                    id=f"claim:{document_index if document_index is not None else 0}:{absolute_start}:{connector_match.start()}",
                    treatment=left,
                    outcome=right,
                    start=absolute_start,
                    end=absolute_end,
                    evidence_span=sentence.strip(),
                    connector=connector,
                    modality=modalities,
                    negated=negated,
                    uncertain=uncertain,
                    instruments=list(dict.fromkeys(instruments)),
                    confounders=list(dict.fromkeys(confounders)),
                    confidence=round(max(0.05, confidence), 3),
                    source_column=source_column,
                    document_index=document_index,
                )
            )
    return claims


def role_hypotheses_to_guide_context(
    hypotheses: Sequence[RoleHypothesis],
) -> dict[str, Any]:
    candidates: dict[str, list[str]] = {
        "treatment": [],
        "outcome": [],
        "instrument": [],
        "confounder": [],
    }
    for hypothesis in hypotheses:
        bucket = candidates.setdefault(hypothesis.role, [])
        normalized = hypothesis.text.lower().replace(" ", "_")
        if normalized not in bucket:
            bucket.append(normalized)
    return {
        "candidates": candidates,
        "focus_columns": list(
            dict.fromkeys(
                [
                    *candidates["treatment"],
                    *candidates["outcome"],
                    *candidates["instrument"],
                    *candidates["confounder"],
                ]
            )
        )[:20],
        "hypothesis_only": True,
        "notes": [
            "NLP-derived roles are lexical hypotheses; review against the frame "
            "schema and study design before discovery or estimation."
        ],
    }


__all__ = [
    "CausalClaim",
    "RoleHypothesis",
    "extract_causal_claims",
    "extract_role_hypotheses",
    "role_hypotheses_to_guide_context",
]
