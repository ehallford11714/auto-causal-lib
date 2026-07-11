"""Broader AutoNLP suite built on the existing offline-safe NLP helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

from autocausal.autonlp.features import NLPFeaturePlan
from autocausal.autonlp.profile import (
    profile_text_frame,
    redact_sensitive_text,
)
from autocausal.autonlp.report import AutoNLPReport, NLP_CAVEAT
from autocausal.autonlp.roles import (
    CausalClaim,
    RoleHypothesis,
    extract_causal_claims,
    extract_role_hypotheses,
)
from autocausal.nlp.sentiment import polarity
from autocausal.nlp.tokenize import analyze


StructuredNLPEnricher = Callable[[dict[str, Any]], Any]


def _external_summary(response: Any) -> dict[str, Any]:
    """Retain structured adapter metadata, not returned raw documents."""

    if not isinstance(response, Mapping):
        return {
            "accepted": False,
            "reason": "adapter response was not a mapping",
        }
    allowed: dict[str, Any] = {
        "accepted": True,
        "response_keys": sorted(str(key) for key in response.keys()),
    }
    recommendations = response.get("recommendations")
    if isinstance(recommendations, Sequence) and not isinstance(
        recommendations, (str, bytes)
    ):
        safe: list[dict[str, Any]] = []
        for item in recommendations[:50]:
            if not isinstance(item, Mapping):
                continue
            safe.append(
                {
                    key: item.get(key)
                    for key in ("role", "column", "feature", "priority", "rationale")
                    if isinstance(item.get(key), (str, int, float, bool))
                }
            )
        allowed["recommendations"] = safe
    return allowed


class AutoNLPSuite:
    """Profile text, extract linguistic hypotheses, and plan fold-safe features."""

    def __init__(
        self,
        source: Any,
        *,
        mode: Optional[str] = None,
        text_columns: Optional[Sequence[str]] = None,
        target: Optional[str] = None,
        max_documents: int = 10_000,
        max_claims: int = 2_000,
        table: Optional[str] = None,
        query: Optional[str] = None,
    ) -> None:
        if isinstance(source, pd.DataFrame):
            frame, source_label, ac = source.copy(), "dataframe", None
        elif hasattr(source, "df") and isinstance(source.df, pd.DataFrame):
            frame, source_label, ac = (
                source.df.copy(),
                str(getattr(source, "source", "autocausal")),
                source,
            )
        else:
            from autocausal.suites.base import resolve_frame

            frame, source_label, ac = resolve_frame(
                source, table=table, query=query
            )
        self.frame = frame
        self.source = source_label
        self.ac = ac
        self.mode = str(mode or getattr(ac, "mode", "exploratory")).lower()
        if self.mode not in ("exploratory", "production"):
            raise ValueError("mode must be 'exploratory' or 'production'")
        self.text_columns = (
            [str(column) for column in text_columns]
            if text_columns is not None
            else None
        )
        self.target = target
        self.max_documents = max(1, int(max_documents))
        self.max_claims = max(1, int(max_claims))

    @classmethod
    def from_autocausal(cls, ac: Any, **kwargs: Any) -> "AutoNLPSuite":
        return cls(ac, **kwargs)

    def run(
        self,
        *,
        external_enricher: Optional[StructuredNLPEnricher] = None,
        allow_external_text: bool = False,
        feature_max_features: int = 5_000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> AutoNLPReport:
        production = self.mode == "production"
        profile = profile_text_frame(
            self.frame,
            text_columns=self.text_columns,
            target=self.target,
        )
        claims: list[CausalClaim] = []
        hypotheses: list[RoleHypothesis] = []
        analysis_summary: dict[str, Any] = {}
        external_documents: list[dict[str, Any]] = []
        notes = [NLP_CAVEAT]

        for column_profile in profile.text_columns:
            column = column_profile.column
            series = self.frame[column]
            token_count = 0
            sentence_count = 0
            lemma_count = 0
            sentiments: list[float] = []
            pos_counts: Counter[str] = Counter()
            backends: Counter[str] = Counter()
            analyzed_documents = 0
            for document_index, value in series.items():
                if pd.isna(value):
                    continue
                if analyzed_documents >= self.max_documents:
                    break
                text = str(value)
                token_analysis = analyze(text)
                sentiment = polarity(text)
                token_count += len(token_analysis.tokens)
                sentence_count += len(token_analysis.sentences)
                lemma_count += len(token_analysis.lemmas)
                sentiments.append(float(sentiment.compound))
                pos_counts.update(tag for _, tag in token_analysis.pos)
                backends[token_analysis.backend] += 1
                claims.extend(
                    extract_causal_claims(
                        text,
                        source_column=column,
                        document_index=int(document_index)
                        if isinstance(document_index, (int, np.integer))
                        else analyzed_documents,
                    )
                )
                hypotheses.extend(extract_role_hypotheses(text))
                if external_enricher is not None and allow_external_text:
                    outgoing = redact_sensitive_text(text)[0] if production else text
                    external_documents.append(
                        {
                            "column": column,
                            "document_index": (
                                int(document_index)
                                if isinstance(document_index, (int, np.integer))
                                else analyzed_documents
                            ),
                            "text": outgoing,
                        }
                    )
                analyzed_documents += 1
            if analyzed_documents >= self.max_documents and series.notna().sum() > analyzed_documents:
                notes.append(
                    f"Column {column!r} analysis was capped at "
                    f"{self.max_documents} documents."
                )
            analysis_summary[column] = {
                "documents_analyzed": analyzed_documents,
                "token_count": token_count,
                "sentence_count": sentence_count,
                "lemma_count": lemma_count,
                "mean_sentiment": (
                    round(float(np.mean(sentiments)), 8) if sentiments else 0.0
                ),
                "pos_tag_counts": dict(pos_counts.most_common(20)),
                "analysis_backends": dict(backends),
                "contains_tokens_or_text": False,
            }

        # Stable de-duplication keeps the report bounded.
        unique_hypotheses: list[RoleHypothesis] = []
        seen_hypotheses: set[tuple[str, str]] = set()
        for hypothesis in hypotheses:
            key = (hypothesis.role, hypothesis.text.lower())
            if key not in seen_hypotheses:
                seen_hypotheses.add(key)
                unique_hypotheses.append(hypothesis)
        if len(claims) > self.max_claims:
            notes.append(
                f"Causal-language claims were capped at {self.max_claims}."
            )
            claims = claims[: self.max_claims]
        feature_plans = [
            NLPFeaturePlan(
                text_columns=[profile_item.column],
                ngram_range=ngram_range,
                max_features=int(feature_max_features),
                embeddings=None,
                notes=[
                    "Use FoldSafeTextVectorizer inside the estimator pipeline so "
                    "vocabulary is learned from training folds only.",
                    "Optional embedding adapters require explicit raw-text consent.",
                ],
            )
            for profile_item in profile.text_columns
        ]

        enrichment: Optional[dict[str, Any]] = None
        if external_enricher is not None:
            if not allow_external_text:
                raise ValueError(
                    "external NLP enrichment requires allow_external_text=True; "
                    "raw text is never sent to SLM/MCP by default"
                )
            policy = getattr(self.ac, "policy", None)
            if policy is not None and not bool(
                getattr(policy, "allow_raw_data_external", False)
            ):
                raise ValueError(
                    "active AutoCausal policy forbids external raw-text payloads"
                )
            request = {
                "schema": "AutoCausalNLPEnrichmentRequest.v1",
                "mode": self.mode,
                "documents": external_documents,
                "profile": profile.to_dict(),
                "consent": {
                    "allow_external_text": True,
                    "production_redaction_applied": production,
                },
            }
            response = external_enricher(request)
            enrichment = _external_summary(response)
            notes.append(
                "External text enrichment ran with explicit consent; only a "
                "structured response summary is retained."
            )
        elif production:
            notes.append(
                "Production mode processed text locally and redacts detected "
                "PII/secrets from serialized evidence spans."
            )
        else:
            notes.append(
                "No raw text was sent to an SLM, MCP server, embedding service, "
                "or vector store."
            )

        report = AutoNLPReport(
            profile=profile,
            claims=claims,
            role_hypotheses=unique_hypotheses,
            feature_plans=feature_plans,
            analysis_summary=analysis_summary,
            privacy={
                "mode": self.mode,
                "risk": profile.privacy_risk,
                "raw_text_sent_external": bool(
                    external_enricher is not None and allow_external_text
                ),
                "production_redaction": production,
                "sample_values_in_profile": False,
            },
            external_enrichment=enrichment,
            mode=self.mode,
            notes=notes,
        )
        return report

    def report(self, **kwargs: Any) -> AutoNLPReport:
        """Alias for :meth:`run`."""
        return self.run(**kwargs)


__all__ = ["AutoNLPSuite", "StructuredNLPEnricher"]
