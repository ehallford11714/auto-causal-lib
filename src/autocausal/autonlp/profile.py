"""Text-column detection, privacy scanning, and aggregate corpus profiling."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}
_SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    "api_key_assignment": re.compile(
        r"(?i)\b(?:api[_ -]?key|secret|token|password)\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{8,}"
    ),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def redact_sensitive_text(text: str) -> tuple[str, dict[str, int]]:
    """Redact common PII/secrets while retaining non-sensitive context."""

    output = str(text or "")
    counts: dict[str, int] = {}
    for name, pattern in {**_PII_PATTERNS, **_SECRET_PATTERNS}.items():
        output, count = pattern.subn(f"[REDACTED_{name.upper()}]", output)
        if count:
            counts[name] = int(count)
    return output, counts


def sensitive_risk_counts(texts: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        _, found = redact_sensitive_text(text)
        for name, count in found.items():
            counts[name] = counts.get(name, 0) + count
    return counts


def _language_hint(texts: Sequence[str]) -> dict[str, Any]:
    combined = " ".join(texts[:200]).lower()
    words = re.findall(r"[a-záéíóúñàâçèéêëîïôûùüÿœ]+", combined)
    if not words:
        return {"language": "unknown", "confidence": 0.0, "method": "heuristic"}
    vocab = {
        "en": {"the", "and", "is", "of", "to", "because", "with", "for"},
        "es": {"el", "la", "de", "que", "y", "con", "para", "porque"},
        "fr": {"le", "la", "de", "et", "que", "avec", "pour", "parce"},
        "de": {"der", "die", "das", "und", "mit", "für", "weil", "ist"},
    }
    scores = {
        language: sum(1 for word in words if word in terms)
        for language, terms in vocab.items()
    }
    language = max(scores, key=scores.get)
    best = scores[language]
    return {
        "language": language if best else "unknown",
        "confidence": round(best / max(sum(scores.values()), 1), 4),
        "method": "stopword_heuristic",
    }


@dataclass
class TextColumnProfile:
    column: str
    n_rows: int
    n_non_missing: int
    missing_fraction: float
    unique_count: int
    duplication_fraction: float
    mean_length: float
    median_length: float
    p95_length: float
    language_hint: dict[str, Any] = field(default_factory=dict)
    pii_risk: dict[str, int] = field(default_factory=dict)
    secret_risk: dict[str, int] = field(default_factory=dict)
    label_leakage_risk: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NLPProfile:
    text_columns: list[TextColumnProfile] = field(default_factory=list)
    candidate_columns: list[str] = field(default_factory=list)
    n_rows: int = 0
    target: Optional[str] = None
    privacy_risk: str = "low"
    notes: list[str] = field(default_factory=list)
    schema: str = "AutoCausalNLPProfile.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "n_rows": self.n_rows,
            "target": self.target,
            "candidate_columns": list(self.candidate_columns),
            "text_columns": [profile.to_dict() for profile in self.text_columns],
            "privacy_risk": self.privacy_risk,
            "notes": list(self.notes),
            "contains_sample_values": False,
        }


def detect_text_columns(
    frame: pd.DataFrame,
    *,
    minimum_mean_length: float = 12.0,
    minimum_unique_fraction: float = 0.05,
) -> list[str]:
    """Detect likely free-text columns without returning sample values."""

    candidates: list[str] = []
    n_rows = max(len(frame), 1)
    for raw_column in frame.columns:
        column = str(raw_column)
        series = frame[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue
        non_missing = series.dropna().astype(str)
        if non_missing.empty:
            continue
        mean_length = float(non_missing.str.len().mean())
        unique_fraction = float(non_missing.nunique()) / n_rows
        whitespace_fraction = float(non_missing.str.contains(r"\s", regex=True).mean())
        if (
            mean_length >= minimum_mean_length
            and (
                unique_fraction >= minimum_unique_fraction
                or whitespace_fraction >= 0.50
            )
        ):
            candidates.append(column)
    return candidates


def profile_text_frame(
    frame: pd.DataFrame,
    *,
    text_columns: Optional[Sequence[str]] = None,
    target: Optional[str] = None,
) -> NLPProfile:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("profile_text_frame frame must be a pandas DataFrame")
    columns = (
        [str(column) for column in text_columns]
        if text_columns is not None
        else detect_text_columns(frame)
    )
    unknown = [column for column in columns if column not in frame]
    if unknown:
        raise KeyError(f"unknown text columns: {unknown}")
    profiles: list[TextColumnProfile] = []
    all_risk_types: set[str] = set()
    label_values: list[str] = []
    if target is not None:
        if target not in frame:
            raise KeyError(f"target column {target!r} is not in the frame")
        values = frame[target].dropna().astype(str).unique().tolist()
        if len(values) <= 50:
            label_values = [value.lower() for value in values if len(value) >= 2]

    for column in columns:
        series = frame[column]
        values = series.dropna().astype(str)
        lengths = values.str.len().to_numpy(dtype=float)
        risks = sensitive_risk_counts(values.tolist())
        pii = {name: count for name, count in risks.items() if name in _PII_PATTERNS}
        secrets = {
            name: count for name, count in risks.items() if name in _SECRET_PATTERNS
        }
        all_risk_types.update(risks)
        duplicate_fraction = (
            1.0 - float(values.nunique()) / len(values) if len(values) else 0.0
        )
        target_name_hits = 0
        label_value_hits = 0
        if target:
            target_pattern = re.compile(rf"\b{re.escape(target)}\b", re.I)
            target_name_hits = int(values.str.contains(target_pattern, na=False).sum())
            if label_values:
                lower_values = values.str.lower()
                label_value_hits = int(
                    sum(
                        lower_values.str.contains(
                            rf"\b{re.escape(label)}\b", regex=True, na=False
                        ).sum()
                        for label in label_values
                    )
                )
        leakage = {
            "target_name_hits": target_name_hits,
            "label_value_hits": label_value_hits,
            "risk": (
                "high"
                if target_name_hits + label_value_hits > max(2, len(values) * 0.05)
                else "low"
            ),
        }
        notes: list[str] = []
        if risks:
            notes.append(
                "Potential PII/secrets detected locally; no matching values are "
                "included in this profile."
            )
        if leakage["risk"] == "high":
            notes.append(
                "Text may reveal the prediction label; review temporal ordering "
                "and remove post-outcome text before training."
            )
        profiles.append(
            TextColumnProfile(
                column=column,
                n_rows=len(frame),
                n_non_missing=len(values),
                missing_fraction=round(float(series.isna().mean()), 8),
                unique_count=int(values.nunique()),
                duplication_fraction=round(duplicate_fraction, 8),
                mean_length=round(float(np.mean(lengths)), 4) if len(lengths) else 0.0,
                median_length=(
                    round(float(np.median(lengths)), 4) if len(lengths) else 0.0
                ),
                p95_length=(
                    round(float(np.quantile(lengths, 0.95)), 4)
                    if len(lengths)
                    else 0.0
                ),
                language_hint=_language_hint(values.tolist()),
                pii_risk=pii,
                secret_risk=secrets,
                label_leakage_risk=leakage,
                notes=notes,
            )
        )
    privacy = "high" if all_risk_types else "low"
    return NLPProfile(
        text_columns=profiles,
        candidate_columns=columns,
        n_rows=len(frame),
        target=target,
        privacy_risk=privacy,
        notes=[
            "Profiles contain aggregate counts only; sample text is omitted.",
            "Language hints are lightweight heuristics, not language identification guarantees.",
        ],
    )


__all__ = [
    "NLPProfile",
    "TextColumnProfile",
    "detect_text_columns",
    "profile_text_frame",
    "redact_sensitive_text",
    "sensitive_risk_counts",
]
