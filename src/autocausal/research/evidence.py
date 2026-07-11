"""Deduplication, cross-matching, evidence extraction, and citation safeguards."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from hashlib import sha256
from typing import Any, Mapping, Optional, Protocol, Sequence

from autocausal.research.models import (
    ClaimEvidenceGraph,
    ComparabilityScore,
    CrossMatch,
    EvidenceSpan,
    MatchReason,
    ResearchClaim,
    ResearchHandoff,
    ResearchPolicy,
    ResearchQuestion,
    SourceRecord,
)


_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}
_NEGATION = {
    "no",
    "not",
    "neither",
    "null",
    "failed",
    "fails",
    "inconsistent",
    "unrelated",
    "absence",
    "without",
}
_DESIGN_TERMS = {
    "randomized",
    "randomised",
    "trial",
    "instrumental",
    "quasi",
    "experimental",
    "longitudinal",
    "cohort",
    "difference",
    "regression",
    "discontinuity",
    "negative",
    "control",
    "systematic",
    "review",
    "meta",
    "analysis",
}


def normalize_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.split(r"[^a-z0-9]+", str(value).lower())
        if len(token) > 1 and token not in _STOP
    ]


def title_fingerprint(title: str) -> str:
    normalized = " ".join(normalize_tokens(title))
    return sha256(normalized.encode("utf-8")).hexdigest()[:20]


def source_dedup_key(source: SourceRecord) -> str:
    if source.doi:
        return "doi:" + source.doi.lower()
    if source.arxiv_id:
        return "arxiv:" + re.sub(r"v\d+$", "", source.arxiv_id.lower())
    return "title:" + title_fingerprint(source.title)


def deduplicate_sources(
    sources: Sequence[SourceRecord],
) -> tuple[list[SourceRecord], list[dict[str, Any]], dict[str, str]]:
    """Deduplicate DOI/arXiv/title and retain the richest record."""

    by_key: dict[str, SourceRecord] = {}
    order: list[str] = []
    aliases: dict[str, str] = {}
    log: list[dict[str, Any]] = []

    def richness(source: SourceRecord) -> tuple[int, int, int]:
        return (
            2
            if source.availability == "full_text"
            else 1
            if source.evidence_text
            else 0,
            len(source.evidence_text),
            len(source.metadata),
        )

    for source in sources:
        key = source_dedup_key(source)
        current = by_key.get(key)
        if current is None:
            by_key[key] = source
            order.append(key)
            aliases[source.source_id] = source.source_id
            continue
        winner, duplicate = (
            (source, current)
            if richness(source) > richness(current)
            else (current, source)
        )
        by_key[key] = winner
        aliases[current.source_id] = winner.source_id
        aliases[source.source_id] = winner.source_id
        merged_references = list(
            dict.fromkeys([*winner.references, *duplicate.references])
        )
        winner.references = merged_references
        if not winner.doi:
            winner.doi = duplicate.doi
        if not winner.arxiv_id:
            winner.arxiv_id = duplicate.arxiv_id
        if not winner.abstract:
            winner.abstract = duplicate.abstract
        log.append(
            {
                "action": "deduplicate",
                "dedup_key": key,
                "retained": winner.source_id,
                "removed": duplicate.source_id,
                "reason": key.split(":", 1)[0],
            }
        )
    return [by_key[key] for key in order], log, aliases


def source_independence_groups(
    sources: Sequence[SourceRecord],
) -> dict[str, list[str]]:
    """Group companion papers sharing explicit study/cohort/trial identifiers."""

    groups: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        metadata = source.metadata or {}
        explicit = (
            metadata.get("trial_registration")
            or metadata.get("study_id")
            or metadata.get("cohort_id")
            or metadata.get("dataset_id")
        )
        if explicit:
            key = "study:" + str(explicit).lower()
        else:
            # Distinct deduplicated papers count independently unless metadata
            # explicitly links the same underlying study.
            key = source_dedup_key(source)
        groups[key].append(source.source_id)
    return dict(sorted(groups.items()))


def _tfidf_vectors(documents: Sequence[str]) -> list[dict[str, float]]:
    tokens = [normalize_tokens(document) for document in documents]
    df: Counter[str] = Counter()
    for row in tokens:
        df.update(set(row))
    n_docs = max(1, len(tokens))
    vectors: list[dict[str, float]] = []
    for row in tokens:
        counts = Counter(row)
        norm = max(1, sum(counts.values()))
        vector = {
            token: (count / norm) * (math.log((1 + n_docs) / (1 + df[token])) + 1.0)
            for token, count in counts.items()
        }
        vectors.append(vector)
    return vectors


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    ln = math.sqrt(sum(value * value for value in left.values()))
    rn = math.sqrt(sum(value * value for value in right.values()))
    return 0.0 if not ln or not rn else min(1.0, max(0.0, dot / (ln * rn)))


def _component_overlap(expected: str, observed: str) -> tuple[float, bool]:
    left = set(normalize_tokens(expected))
    right = set(normalize_tokens(observed))
    if not left:
        return 0.5, False
    if not right:
        return 0.25, True
    score = len(left & right) / max(1, len(left))
    return score, score == 0.0


def _time_overlap(expected: str, observed: str) -> tuple[float, bool]:
    expected_years = [
        int(value)
        for value in re.findall(r"\b(?:19|20)\d{2}\b", str(expected))
    ]
    observed_years = [
        int(value)
        for value in re.findall(r"\b(?:19|20)\d{2}\b", str(observed))
    ]
    if not expected_years:
        return 0.5, False
    if not observed_years:
        return 0.25, True
    observed_year = observed_years[0]
    low, high = min(expected_years), max(expected_years)
    if low <= observed_year <= high:
        return 1.0, False
    distance = min(abs(observed_year - low), abs(observed_year - high))
    score = max(0.0, 1.0 - distance / 10.0)
    return score, score < 0.5


def _direction_relation(
    source: str,
    target: str,
    text: str,
    *,
    expected_sign: Optional[float] = None,
) -> tuple[float, str]:
    low = " ".join(normalize_tokens(text))
    source_tokens = normalize_tokens(source)
    target_tokens = normalize_tokens(target)
    if not source_tokens or not target_tokens:
        return 0.3, "contextualizes"
    source_pos = min(
        (low.find(token) for token in source_tokens if token in low), default=-1
    )
    target_pos = min(
        (low.find(token) for token in target_tokens if token in low), default=-1
    )
    if source_pos < 0 or target_pos < 0:
        return 0.2, "contextualizes"
    window_start = max(0, min(source_pos, target_pos) - 80)
    window_end = min(len(low), max(source_pos, target_pos) + 160)
    window = low[window_start:window_end]
    negated = any(token in window.split() for token in _NEGATION)
    if negated:
        return 0.65, "contradicts"
    negative_effect = bool(
        set(window.split())
        & {
            "decrease",
            "decreased",
            "decreases",
            "decline",
            "declined",
            "lower",
            "lowered",
            "reduce",
            "reduced",
            "reduces",
        }
    )
    positive_effect = bool(
        set(window.split())
        & {
            "increase",
            "increased",
            "increases",
            "higher",
            "raise",
            "raised",
            "raises",
        }
    )
    if expected_sign not in (None, 0.0) and negative_effect != positive_effect:
        observed_sign = -1.0 if negative_effect else 1.0
        if (float(expected_sign) > 0) == (observed_sign > 0):
            return 0.90, "supports"
        return 0.25, "contradicts"
    if source_pos <= target_pos:
        return 0.75, "supports"
    return 0.35, "contextualizes"


class MatchAdjudicator(Protocol):
    def adjudicate_match(self, payload: Mapping[str, Any]) -> Any: ...


class CrossMatchEngine:
    """Connect empirical findings to literature with inspectable components."""

    def __init__(self, *, slm_backend: Optional[MatchAdjudicator] = None) -> None:
        self.slm_backend = slm_backend

    def match(
        self,
        handoff: ResearchHandoff,
        sources: Sequence[SourceRecord],
        *,
        use_slm: bool = False,
        minimum_relevance: float = 0.08,
    ) -> list[CrossMatch]:
        finding_rows: list[tuple[str, dict[str, Any], str]] = []
        for index, edge in enumerate(handoff.edges):
            finding_id = str(edge.get("finding_id") or f"edge:{index}")
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            aliases = [
                *(handoff.aliases.get(source) or []),
                *(handoff.aliases.get(target) or []),
            ]
            text = " ".join(
                [
                    source,
                    target,
                    str(edge.get("type") or ""),
                    str(edge.get("method") or ""),
                    *aliases,
                ]
            )
            finding_rows.append((finding_id, dict(edge), text))
        if not finding_rows:
            for finding in handoff.findings:
                finding_rows.append(
                    (
                        str(finding.get("id") or "finding"),
                        dict(finding),
                        str(finding.get("summary") or ""),
                    )
                )

        finding_docs = [item[2] for item in finding_rows]
        source_docs = [
            f"{source.title} {source.abstract or ''} {source.snippet or ''}"
            for source in sources
        ]
        vectors = _tfidf_vectors([*finding_docs, *source_docs])
        finding_vectors = vectors[: len(finding_docs)]
        source_vectors = vectors[len(finding_docs) :]
        matches: list[CrossMatch] = []

        expected_population = str(
            handoff.context.get("population") or handoff.context.get("cohort") or ""
        )
        expected_context = str(
            handoff.context.get("setting")
            or handoff.context.get("context")
            or (
                handoff.domain
                if handoff.domain.lower() not in ("", "general", "unknown")
                else ""
            )
        )
        expected_period = str(
            handoff.context.get("time_period") or handoff.context.get("period") or ""
        )
        roles = handoff.candidate_roles

        for f_index, (finding_id, finding, _) in enumerate(finding_rows):
            edge_source = str(finding.get("source") or "")
            edge_target = str(finding.get("target") or "")
            role_tokens = set(
                normalize_tokens(
                    " ".join(
                        [
                            *roles.get("treatment", []),
                            *roles.get("outcome", []),
                            *roles.get("instrument", []),
                            *roles.get("confounder", []),
                        ]
                    )
                )
            )
            finding_ids = {
                str(finding.get("doi") or "").lower(),
                str(finding.get("arxiv_id") or "").lower(),
            } - {""}
            for s_index, source in enumerate(sources):
                source_text = source_docs[s_index]
                lexical = _cosine(finding_vectors[f_index], source_vectors[s_index])
                source_tokens = set(normalize_tokens(source_text))
                alias_tokens = set(
                    normalize_tokens(
                        " ".join(
                            [
                                *(handoff.aliases.get(edge_source) or []),
                                *(handoff.aliases.get(edge_target) or []),
                            ]
                        )
                    )
                )
                alias_score = (
                    len(alias_tokens & source_tokens) / max(1, len(alias_tokens))
                    if alias_tokens
                    else lexical
                )
                lexical_alias = max(lexical, alias_score)
                source_ids = {
                    str(source.doi or "").lower(),
                    str(source.arxiv_id or "").lower(),
                    str(source.stable_id or "").lower(),
                } - {""}
                identifier_score = 1.0 if finding_ids & source_ids else 0.0
                role_score = (
                    len(role_tokens & source_tokens) / max(1, len(role_tokens))
                    if role_tokens
                    else 0.5
                )
                expected_sign: Optional[float] = None
                for key in ("coefficient", "estimate", "effect", "weight"):
                    try:
                        if finding.get(key) is not None:
                            expected_sign = float(finding[key])
                            break
                    except (TypeError, ValueError):
                        continue
                direction_score, relation = _direction_relation(
                    edge_source,
                    edge_target,
                    source_text,
                    expected_sign=expected_sign,
                )

                source_population = str(
                    source.metadata.get("population")
                    or source.metadata.get("cohort")
                    or source.abstract
                    or ""
                )
                population_score, population_mismatch = _component_overlap(
                    expected_population, source_population
                )
                source_context = str(
                    source.metadata.get("context")
                    or source.metadata.get("setting")
                    or source.venue
                    or source.abstract
                    or ""
                )
                context_score, context_mismatch = _component_overlap(
                    expected_context, source_context
                )
                time_score, time_mismatch = _time_overlap(
                    expected_period, str(source.date or "")
                )
                design_hits = len(
                    set(normalize_tokens(source_text)) & set(_DESIGN_TERMS)
                )
                design_score = min(1.0, 0.25 * design_hits)
                if not source.evidence_text:
                    design_score *= 0.5

                weights = {
                    "lexical": 0.28,
                    "identifier": 0.10,
                    "role": 0.12,
                    "direction": 0.12,
                    "population": 0.12,
                    "context": 0.10,
                    "time": 0.06,
                    "design": 0.10,
                }
                overall = (
                    lexical_alias * weights["lexical"]
                    + identifier_score * weights["identifier"]
                    + role_score * weights["role"]
                    + direction_score * weights["direction"]
                    + population_score * weights["population"]
                    + context_score * weights["context"]
                    + time_score * weights["time"]
                    + design_score * weights["design"]
                )
                warnings: list[str] = []
                if population_mismatch:
                    warnings.append(
                        "population terms do not overlap; context transfer is weak"
                    )
                    overall *= 0.75
                if context_mismatch:
                    warnings.append(
                        "study setting/context does not overlap the handoff context"
                    )
                    overall *= 0.85
                if time_mismatch:
                    warnings.append("study period may not match the handoff period")
                    overall *= 0.95
                if relation == "contradicts":
                    warnings.append(
                        "text contains negation/inconsistency near matched concepts"
                    )
                    # Contradiction is relevant but not comparable confirmation.
                    overall *= 0.90
                if source.availability in ("metadata", "snippet"):
                    warnings.append(f"{source.availability}-level evidence only")
                if overall < minimum_relevance and identifier_score == 0:
                    continue

                reasons = [
                    MatchReason(
                        code="tfidf_alias",
                        detail="Normalized token/alias TF-IDF relevance",
                        component="lexical_alias",
                        score=lexical_alias,
                    ),
                    MatchReason(
                        code="identifier",
                        detail="Exact DOI/arXiv/stable identifier overlap",
                        component="identifier",
                        score=identifier_score,
                    ),
                    MatchReason(
                        code="role_compatibility",
                        detail="Treatment/outcome/instrument/confounder term overlap",
                        component="role_compatibility",
                        score=role_score,
                    ),
                    MatchReason(
                        code="direction",
                        detail="Deterministic direction/negation scan",
                        component="direction_agreement",
                        score=direction_score,
                    ),
                ]
                comparability = ComparabilityScore(
                    lexical_alias=lexical_alias,
                    identifier=identifier_score,
                    role_compatibility=role_score,
                    direction_agreement=direction_score,
                    population_overlap=population_score,
                    context_overlap=context_score,
                    time_overlap=time_score,
                    design_relevance=design_score,
                    overall=overall,
                    warnings=warnings,
                )

                # SLM can adjudicate only this deterministic candidate. It may
                # clarify relation/relevance, never create a source or causal proof.
                if use_slm and self.slm_backend is not None:
                    try:
                        raw = self.slm_backend.adjudicate_match(
                            {
                                "finding_id": finding_id,
                                "finding": finding,
                                "source": source.to_dict(),
                                "components": comparability.to_dict(),
                                "allowed_relations": [
                                    "supports",
                                    "contradicts",
                                    "contextualizes",
                                    "insufficient",
                                ],
                            }
                        )
                        if isinstance(raw, Mapping):
                            proposed = str(raw.get("relation") or "")
                            semantic = float(raw.get("semantic_relevance") or 0.0)
                            if proposed in (
                                "supports",
                                "contradicts",
                                "contextualizes",
                                "insufficient",
                            ):
                                # Semantic adjudication may weaken a relation,
                                # but it cannot create causal confirmation or
                                # erase a deterministic contradiction signal.
                                if proposed == relation or proposed in (
                                    "contextualizes",
                                    "insufficient",
                                ):
                                    relation = proposed
                            reasons.append(
                                MatchReason(
                                    code="slm_semantic_adjudication",
                                    detail=(
                                        "Validated semantic adjudication after "
                                        "deterministic candidate generation; "
                                        "relevance only, never causal confirmation"
                                    ),
                                    component="semantic_relevance",
                                    score=min(1.0, max(0.0, semantic)),
                                )
                            )
                    except Exception:
                        pass

                matched = sorted(
                    (
                        set(normalize_tokens(edge_source + " " + edge_target))
                        & source_tokens
                    )
                    | (alias_tokens & source_tokens)
                )
                matches.append(
                    CrossMatch(
                        finding_id=finding_id,
                        source_id=source.source_id,
                        claim_id=None,
                        reasons=reasons,
                        comparability=comparability,
                        relation=relation,  # type: ignore[arg-type]
                        matched_concepts=matched,
                    )
                )
        matches.sort(
            key=lambda item: (
                item.finding_id,
                -item.comparability.overall,
                item.source_id,
            )
        )
        return matches

    @staticmethod
    def build_graph(
        handoff: ResearchHandoff,
        sources: Sequence[SourceRecord],
        claims: Sequence[ResearchClaim],
        matches: Sequence[CrossMatch],
    ) -> ClaimEvidenceGraph:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for finding in handoff.findings:
            nodes.append(
                {
                    "id": str(finding.get("id")),
                    "kind": "autocausal_finding",
                    "label": str(finding.get("summary") or finding.get("id")),
                }
            )
        for source in sources:
            nodes.append(
                {
                    "id": source.source_id,
                    "kind": "source",
                    "label": source.title,
                    "provider": source.provider,
                }
            )
            for reference in source.references:
                edges.append(
                    {
                        "source": source.source_id,
                        "target": str(reference),
                        "relation": "references",
                        "target_may_be_unretrieved": True,
                    }
                )
        for claim in claims:
            nodes.append(
                {
                    "id": claim.claim_id,
                    "kind": "research_claim",
                    "label": claim.normalized_claim,
                }
            )
            for finding_id in claim.linked_finding_ids:
                edges.append(
                    {
                        "source": finding_id,
                        "target": claim.claim_id,
                        "relation": "external_context_claim",
                    }
                )
            for span in claim.evidence_spans:
                nodes.append(
                    {
                        "id": span.span_id,
                        "kind": "evidence_span",
                        "label": span.exact_text[:160],
                    }
                )
                edges.append(
                    {
                        "source": source_id_for(span.source_id),
                        "target": span.span_id,
                        "relation": "contains_exact_span",
                    }
                )
                edges.append(
                    {
                        "source": span.span_id,
                        "target": claim.claim_id,
                        "relation": span.claim_relation,
                    }
                )
        for match in matches:
            edges.append(
                {
                    "source": match.finding_id,
                    "target": match.source_id,
                    "relation": "cross_match",
                    "match_id": match.match_id,
                    "comparability": match.comparability.to_dict(),
                }
            )
        # Keep one node per id while preserving deterministic order.
        unique_nodes: dict[str, dict[str, Any]] = {}
        for node in nodes:
            unique_nodes.setdefault(str(node["id"]), node)
        return ClaimEvidenceGraph(nodes=list(unique_nodes.values()), edges=edges)


def source_id_for(value: str) -> str:
    """Identity helper kept explicit for graph readability/type checking."""

    return str(value)


class EvidenceExtractionBackend(Protocol):
    def extract_evidence(self, payload: Mapping[str, Any]) -> Any: ...


def _sentences(text: str) -> list[str]:
    # Returned values remain exact substrings after edge whitespace stripping.
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(item.strip()) >= 20
    ]


class EvidenceExtractor:
    """Extract exact abstract/snippet spans; reject generated quotation drift."""

    def __init__(
        self, *, slm_backend: Optional[EvidenceExtractionBackend] = None
    ) -> None:
        self.slm_backend = slm_backend

    def extract(
        self,
        handoff: ResearchHandoff,
        questions: Sequence[ResearchQuestion],
        sources: Sequence[SourceRecord],
        matches: Sequence[CrossMatch],
        *,
        policy: ResearchPolicy,
        use_slm: bool = False,
    ) -> tuple[list[ResearchClaim], list[dict[str, Any]]]:
        source_map = {source.source_id: source for source in sources}
        finding_map = {str(edge.get("finding_id")): edge for edge in handoff.edges}
        matches_by_finding: dict[str, list[CrossMatch]] = defaultdict(list)
        for match in matches:
            matches_by_finding[match.finding_id].append(match)
        claims: list[ResearchClaim] = []
        failures: list[dict[str, Any]] = []

        finding_ids = list(finding_map)
        if not finding_ids:
            finding_ids = [
                str(item.get("id")) for item in handoff.findings if item.get("id")
            ]
        for finding_id in finding_ids:
            edge = finding_map.get(finding_id) or {}
            source_label = str(edge.get("source") or "exposure")
            target_label = str(edge.get("target") or "outcome")
            claim_text = (
                f"External literature concerning {source_label} and {target_label}"
            )
            spans: list[EvidenceSpan] = []
            for match in matches_by_finding.get(finding_id, []):
                source = source_map.get(match.source_id)
                if source is None or not source.evidence_text:
                    continue
                target_tokens = set(
                    normalize_tokens(
                        f"{source_label} {target_label} "
                        + " ".join(match.matched_concepts)
                    )
                )
                candidates: list[tuple[float, str]] = []
                for sentence in _sentences(source.evidence_text):
                    sentence_tokens = set(normalize_tokens(sentence))
                    relevance = len(target_tokens & sentence_tokens) / max(
                        1, len(target_tokens)
                    )
                    if relevance > 0:
                        candidates.append((relevance, sentence))
                candidates.sort(key=lambda row: (-row[0], row[1]))
                if candidates:
                    exact = candidates[0][1]
                    if exact not in source.evidence_text:
                        failures.append(
                            {
                                "source_id": source.source_id,
                                "finding_id": finding_id,
                                "reason": "rule exact-span mismatch",
                            }
                        )
                        continue
                    spans.append(
                        EvidenceSpan(
                            source_id=source.source_id,
                            exact_text=exact,
                            claim_relation=match.relation,
                            extraction_method=(f"rule:{source.availability}"),
                            confidence=min(
                                0.90,
                                0.40 + 0.50 * match.comparability.overall,
                            ),
                            structured_field=(
                                "full_text"
                                if source.full_text
                                else "abstract"
                                if source.abstract
                                else "snippet"
                            ),
                        )
                    )

            if use_slm and self.slm_backend is not None:
                try:
                    raw = self.slm_backend.extract_evidence(
                        {
                            "finding_id": finding_id,
                            "claim": claim_text,
                            "sources": [
                                {
                                    "source_id": source.source_id,
                                    "text": source.evidence_text,
                                    "availability": source.availability,
                                }
                                for source in sources
                                if source.evidence_text
                            ],
                            "instruction": (
                                "Return exact text copied from a supplied source; "
                                "never cite another identifier."
                            ),
                        }
                    )
                    items = raw.get("spans") if isinstance(raw, Mapping) else []
                    for item in items or []:
                        if not isinstance(item, Mapping):
                            continue
                        source_id = str(item.get("source_id") or "")
                        exact = str(item.get("exact_text") or "").strip()
                        relation = str(item.get("claim_relation") or "")
                        source = source_map.get(source_id)
                        if (
                            source is None
                            or not exact
                            or exact not in source.evidence_text
                            or relation
                            not in (
                                "supports",
                                "contradicts",
                                "contextualizes",
                                "insufficient",
                            )
                        ):
                            failures.append(
                                {
                                    "source_id": source_id,
                                    "finding_id": finding_id,
                                    "reason": "SLM citation/span validation rejected",
                                }
                            )
                            continue
                        span = EvidenceSpan(
                            source_id=source_id,
                            exact_text=exact,
                            claim_relation=relation,  # type: ignore[arg-type]
                            extraction_method="slm_validated_exact_span",
                            confidence=min(
                                1.0, max(0.0, float(item.get("confidence") or 0.5))
                            ),
                            structured_field=str(item.get("structured_field") or "")
                            or None,
                        )
                        if not any(
                            existing.span_id == span.span_id for existing in spans
                        ):
                            spans.append(span)
                except Exception as exc:
                    failures.append(
                        {
                            "finding_id": finding_id,
                            "reason": f"SLM extraction fallback: {type(exc).__name__}",
                        }
                    )

            claim = ResearchClaim(
                normalized_claim=claim_text,
                linked_finding_ids=[finding_id],
                linked_edge=dict(edge) if edge else None,
                evidence_spans=spans,
            )
            claims.append(claim)

        # Attach claim ids to matching edges.
        claim_by_finding = {
            finding_id: claim.claim_id
            for claim in claims
            for finding_id in claim.linked_finding_ids
        }
        for match in matches:
            match.claim_id = claim_by_finding.get(match.finding_id)
        label_claims(
            claims,
            sources=sources,
            minimum_independent_sources=policy.minimum_independent_sources,
        )
        return claims, failures


def label_claims(
    claims: Sequence[ResearchClaim],
    *,
    sources: Sequence[SourceRecord],
    minimum_independent_sources: int,
) -> None:
    groups = source_independence_groups(sources)
    source_to_group = {
        source_id: group
        for group, source_ids in groups.items()
        for source_id in source_ids
    }
    for claim in claims:
        support_groups = {
            source_to_group.get(span.source_id, span.source_id)
            for span in claim.evidence_spans
            if span.claim_relation == "supports"
        }
        contradict_groups = {
            source_to_group.get(span.source_id, span.source_id)
            for span in claim.evidence_spans
            if span.claim_relation == "contradicts"
        }
        all_groups = (
            support_groups
            | contradict_groups
            | {
                source_to_group.get(span.source_id, span.source_id)
                for span in claim.evidence_spans
                if span.claim_relation == "contextualizes"
            }
        )
        claim.independent_source_count = len(all_groups)
        if support_groups and contradict_groups:
            claim.contradiction_status = "mixed"
            claim.literature_label = "mixed"
        elif contradict_groups and not support_groups:
            claim.contradiction_status = "contradicted"
            claim.literature_label = "contradicted"
        elif len(support_groups) >= minimum_independent_sources:
            claim.contradiction_status = "none_observed"
            claim.literature_label = "supported_literature_context"
        elif claim.evidence_spans:
            claim.contradiction_status = "unresolved"
            claim.literature_label = "insufficient_independent_sources"
        else:
            claim.contradiction_status = "unresolved"
            claim.literature_label = "unresolved"


def contradiction_records(
    claims: Sequence[ResearchClaim],
    sources: Sequence[SourceRecord],
) -> list[dict[str, Any]]:
    source_map = {source.source_id: source for source in sources}
    out: list[dict[str, Any]] = []
    for claim in claims:
        supports = [
            span for span in claim.evidence_spans if span.claim_relation == "supports"
        ]
        contradicts = [
            span
            for span in claim.evidence_spans
            if span.claim_relation == "contradicts"
        ]
        if not contradicts:
            continue
        dates = {
            span.source_id: source_map[span.source_id].date
            for span in [*supports, *contradicts]
            if span.source_id in source_map
        }
        contexts = {
            span.source_id: (
                source_map[span.source_id].metadata.get("population")
                or source_map[span.source_id].metadata.get("context")
                or source_map[span.source_id].venue
            )
            for span in [*supports, *contradicts]
            if span.source_id in source_map
        }
        out.append(
            {
                "claim_id": claim.claim_id,
                "status": claim.contradiction_status,
                "detail": (
                    f"{len(contradicts)} contradicting and {len(supports)} "
                    "supporting exact evidence span(s)"
                ),
                "supporting_sources": sorted({span.source_id for span in supports}),
                "contradicting_sources": sorted(
                    {span.source_id for span in contradicts}
                ),
                "publication_dates": dates,
                "contexts": contexts,
            }
        )
    return out


def citation_integrity_errors(
    claims: Sequence[ResearchClaim], sources: Sequence[SourceRecord]
) -> list[str]:
    source_map = {source.source_id: source for source in sources}
    errors: list[str] = []
    for claim in claims:
        for span in claim.evidence_spans:
            source = source_map.get(span.source_id)
            if source is None:
                errors.append(
                    f"{claim.claim_id}: source {span.source_id} was not retrieved"
                )
            elif (
                not source.evidence_text or span.exact_text not in source.evidence_text
            ):
                errors.append(
                    f"{claim.claim_id}: quotation mismatch for {span.source_id}"
                )
    return errors


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
_ARXIV_RE = re.compile(
    r"\b(?:arXiv:)?(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+/\d{7})\b",
    re.I,
)


def extract_reference_identifiers(references: Sequence[str]) -> list[dict[str, str]]:
    """Parse DOI / arXiv identifiers from related-work reference strings."""

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for reference in references:
        text = str(reference or "").strip()
        if not text:
            continue
        doi = _DOI_RE.search(text)
        if doi:
            value = doi.group(0).lower().rstrip(".")
            key = f"doi:{value}"
            if key not in seen:
                seen.add(key)
                out.append({"kind": "doi", "value": value, "query": f"doi:{value}"})
            continue
        arxiv = _ARXIV_RE.search(text)
        if arxiv:
            value = arxiv.group(0)
            value = re.sub(r"^arxiv:", "", value, flags=re.I)
            key = f"arxiv:{value.lower()}"
            if key not in seen:
                seen.add(key)
                out.append(
                    {
                        "kind": "arxiv",
                        "value": value,
                        "query": f"arXiv:{value}",
                    }
                )
            continue
        # Keep short bibliographic fingerprints as lexical related-work queries.
        cleaned = " ".join(text.split())
        if 12 <= len(cleaned) <= 180 and "://" not in cleaned:
            key = f"title:{cleaned.lower()}"
            if key not in seen:
                seen.add(key)
                out.append(
                    {
                        "kind": "title",
                        "value": cleaned,
                        "query": f'"{cleaned}" related work',
                    }
                )
    return out


def expand_related_work_queries(
    sources: Sequence[SourceRecord],
    *,
    question_id: str = "related_work",
    limit: int = 8,
    prefer_identifiers: bool = True,
) -> list[tuple[str, str]]:
    """Build deepen-round queries from fetched source references.

    Identifier queries (DOI/arXiv) are preferred because they can be resolved
    without inventing citations. Title fingerprints remain lexical-only.
    """

    references = [
        reference for source in sources for reference in (source.references or [])
    ]
    identifiers = extract_reference_identifiers(references)
    if prefer_identifiers:
        identifiers = sorted(
            identifiers,
            key=lambda item: 0 if item["kind"] in {"doi", "arxiv"} else 1,
        )
    queries: list[tuple[str, str]] = []
    for item in identifiers[: max(0, int(limit))]:
        queries.append((question_id, item["query"]))
    return list(dict.fromkeys(queries))


def match_prior_sources(
    handoff: ResearchHandoff,
    sources: Sequence[SourceRecord],
    *,
    prior_sources: Sequence[SourceRecord] = (),
    episode_sources: Sequence[SourceRecord] = (),
    public_sources: Sequence[SourceRecord] = (),
    use_slm: bool = False,
    slm_backend: Optional[MatchAdjudicator] = None,
    minimum_relevance: float = 0.08,
) -> list[CrossMatch]:
    """Cross-match findings against current plus prior/episode/public corpora."""

    engine = CrossMatchEngine(slm_backend=slm_backend)
    combined: list[SourceRecord] = []
    provenance: dict[str, str] = {}
    for label, bucket in (
        ("retrieved", sources),
        ("prior_research", prior_sources),
        ("episode", episode_sources),
        ("public_corpus", public_sources),
    ):
        for source in bucket:
            if not isinstance(source, SourceRecord):
                continue
            combined.append(source)
            # Prefer non-retrieved labels when the same id appears in multiple
            # corpora so prior/episode/public boosts remain inspectable.
            if source.source_id not in provenance or label != "retrieved":
                provenance[source.source_id] = label
    deduped, _, aliases = deduplicate_sources(combined)
    for old_id, winner_id in aliases.items():
        if old_id in provenance and winner_id not in provenance:
            provenance[winner_id] = provenance[old_id]
    matches = engine.match(
        handoff,
        deduped,
        use_slm=use_slm,
        minimum_relevance=minimum_relevance,
    )
    for match in matches:
        origin = provenance.get(match.source_id)
        if origin and origin != "retrieved":
            match.reasons.append(
                MatchReason(
                    code="prior_corpus",
                    detail=f"Matched against {origin.replace('_', ' ')} corpus",
                    component="prior_corpus",
                    score=0.7,
                )
            )
            # Recompute overall with a mild prior-corpus boost already reflected
            # in reasons; keep component scores inspectable.
            match.comparability.overall = min(
                1.0, float(match.comparability.overall) + 0.03
            )
    matches.sort(
        key=lambda item: (
            item.finding_id,
            -item.comparability.overall,
            item.source_id,
        )
    )
    return matches


__all__ = [
    "CrossMatchEngine",
    "EvidenceExtractor",
    "citation_integrity_errors",
    "contradiction_records",
    "deduplicate_sources",
    "expand_related_work_queries",
    "extract_reference_identifiers",
    "label_claims",
    "match_prior_sources",
    "normalize_tokens",
    "source_dedup_key",
    "source_independence_groups",
    "title_fingerprint",
]
