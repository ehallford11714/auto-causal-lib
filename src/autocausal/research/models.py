"""Typed contracts for privacy-safe, citation-grounded deep research.

The models in this module intentionally depend only on the Python standard
library.  They are suitable for JSON handoff between AutoCausal, local agents,
and external retrieval workers without including a raw dataframe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Optional


ClaimRelation = Literal["supports", "contradicts", "contextualizes", "insufficient"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _bounded_metadata(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated]"
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in list(value.items())[:100]:
            key_s = str(key)
            low = key_s.lower()
            if any(
                token in low
                for token in (
                    "password",
                    "secret",
                    "api_key",
                    "access_token",
                    "authorization",
                    "cookie",
                    "credential",
                )
            ):
                out[key_s] = "[redacted-secret]"
            elif low in ("rows", "records", "raw_frame", "dataframe", "sample_values"):
                out[key_s] = "[raw-content-omitted]"
            else:
                out[key_s] = _bounded_metadata(item, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_bounded_metadata(item, depth=depth + 1) for item in list(value)[:100]]
    if isinstance(value, str):
        return value[:4_000]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return str(value)[:1_000]


class SearchIntensity(str, Enum):
    """Named research depth.  It is a budget profile, never a policy bypass."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"

    @classmethod
    def parse(cls, value: "SearchIntensity | str") -> "SearchIntensity":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(
                "intensity must be quick, standard, deep, or exhaustive"
            ) from exc

    @property
    def rank(self) -> int:
        return {
            self.QUICK: 0,
            self.STANDARD: 1,
            self.DEEP: 2,
            self.EXHAUSTIVE: 3,
        }[self]


@dataclass
class ResearchBudget:
    """Planned limits for one research run."""

    max_questions: int
    queries_per_question: int
    sources_per_provider: int
    max_sources: int
    max_providers: int
    max_rounds: int
    wall_time_seconds: float
    max_tokens: int
    max_bytes: int
    publication_year_min: Optional[int] = None
    publication_year_max: Optional[int] = None
    languages: list[str] = field(default_factory=lambda: ["en"])

    def __post_init__(self) -> None:
        for name in (
            "max_questions",
            "queries_per_question",
            "sources_per_provider",
            "max_sources",
            "max_providers",
            "max_rounds",
            "max_tokens",
            "max_bytes",
        ):
            setattr(self, name, max(1, int(getattr(self, name))))
        self.wall_time_seconds = max(0.01, float(self.wall_time_seconds))
        self.languages = list(
            dict.fromkeys(str(item).strip().lower() for item in self.languages if item)
        ) or ["en"]
        if (
            self.publication_year_min is not None
            and self.publication_year_max is not None
            and self.publication_year_min > self.publication_year_max
        ):
            raise ValueError("publication_year_min cannot exceed publication_year_max")

    @classmethod
    def for_intensity(cls, intensity: SearchIntensity | str) -> "ResearchBudget":
        level = SearchIntensity.parse(intensity)
        presets = {
            SearchIntensity.QUICK: dict(
                max_questions=2,
                queries_per_question=1,
                sources_per_provider=4,
                max_sources=6,
                max_providers=1,
                max_rounds=1,
                wall_time_seconds=30.0,
                max_tokens=2_000,
                max_bytes=1_000_000,
            ),
            SearchIntensity.STANDARD: dict(
                max_questions=5,
                queries_per_question=3,
                sources_per_provider=8,
                max_sources=24,
                max_providers=2,
                max_rounds=2,
                wall_time_seconds=120.0,
                max_tokens=8_000,
                max_bytes=5_000_000,
            ),
            SearchIntensity.DEEP: dict(
                max_questions=8,
                queries_per_question=5,
                sources_per_provider=15,
                max_sources=60,
                max_providers=3,
                max_rounds=4,
                wall_time_seconds=300.0,
                max_tokens=24_000,
                max_bytes=20_000_000,
            ),
            SearchIntensity.EXHAUSTIVE: dict(
                max_questions=15,
                queries_per_question=8,
                sources_per_provider=30,
                max_sources=150,
                max_providers=6,
                max_rounds=8,
                wall_time_seconds=900.0,
                max_tokens=64_000,
                max_bytes=75_000_000,
            ),
        }
        return cls(**presets[level])

    def with_overrides(self, **overrides: Any) -> "ResearchBudget":
        payload = self.to_dict()
        aliases = {
            "max_queries_per_question": "queries_per_question",
            "max_sources_per_provider": "sources_per_provider",
            "rounds": "max_rounds",
            "max_wall_time": "wall_time_seconds",
            "max_wall_time_seconds": "wall_time_seconds",
            "token_budget": "max_tokens",
            "max_token_budget": "max_tokens",
            "byte_budget": "max_bytes",
            "max_byte_budget": "max_bytes",
            "from_year": "publication_year_min",
            "min_year": "publication_year_min",
            "to_year": "publication_year_max",
            "max_year": "publication_year_max",
            "language_filters": "languages",
        }
        normalized: dict[str, Any] = {}
        for key, value in overrides.items():
            canonical = aliases.get(key, key)
            if canonical in normalized and normalized[canonical] != value:
                raise ValueError(
                    f"conflicting research budget override for {canonical!r}"
                )
            normalized[canonical] = value
        overrides = normalized
        unknown = set(overrides) - set(payload)
        if unknown:
            raise KeyError(f"unknown research budget field(s): {sorted(unknown)}")
        payload.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        return ResearchBudget(**payload)

    def capped_by(self, maximum: "ResearchBudget") -> "ResearchBudget":
        """Return a budget no larger than policy maxima."""

        return ResearchBudget(
            max_questions=min(self.max_questions, maximum.max_questions),
            queries_per_question=min(
                self.queries_per_question, maximum.queries_per_question
            ),
            sources_per_provider=min(
                self.sources_per_provider, maximum.sources_per_provider
            ),
            max_sources=min(self.max_sources, maximum.max_sources),
            max_providers=min(self.max_providers, maximum.max_providers),
            max_rounds=min(self.max_rounds, maximum.max_rounds),
            wall_time_seconds=min(self.wall_time_seconds, maximum.wall_time_seconds),
            max_tokens=min(self.max_tokens, maximum.max_tokens),
            max_bytes=min(self.max_bytes, maximum.max_bytes),
            publication_year_min=(
                self.publication_year_min
                if maximum.publication_year_min is None
                else max(
                    self.publication_year_min or maximum.publication_year_min,
                    maximum.publication_year_min,
                )
            ),
            publication_year_max=(
                self.publication_year_max
                if maximum.publication_year_max is None
                else min(
                    self.publication_year_max or maximum.publication_year_max,
                    maximum.publication_year_max,
                )
            ),
            languages=(
                list(self.languages)
                if "*" in maximum.languages
                else [
                    language
                    for language in self.languages
                    if language in set(maximum.languages)
                ]
                or list(maximum.languages)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchBudget":
        baseline = cls.for_intensity("standard")
        return baseline.with_overrides(**dict(value))

    @property
    def max_queries_per_question(self) -> int:
        return self.queries_per_question

    @property
    def max_sources_per_provider(self) -> int:
        return self.sources_per_provider

    @property
    def max_wall_time_seconds(self) -> float:
        return self.wall_time_seconds


@dataclass
class BudgetUsage:
    """Actual measured consumption, included in every report."""

    questions: int = 0
    queries: int = 0
    providers: int = 0
    sources_fetched: int = 0
    sources_retained: int = 0
    rounds: int = 0
    tokens: int = 0
    bytes: int = 0
    wall_time_seconds: float = 0.0
    cache_hits: int = 0
    failed_queries: int = 0
    stopped_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BudgetUsage":
        return cls(**dict(value))


@dataclass
class ResearchPolicy:
    """Security, retrieval, citation, cost, and escalation policy."""

    allowed_providers: tuple[str, ...] = (
        "local",
        "arxiv",
        "crossref",
        "openalex",
    )
    allow_network: bool = False
    external_network_consent: bool = False
    allow_generic_web: bool = False
    provider_domains: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "arxiv": ("export.arxiv.org",),
            "crossref": ("api.crossref.org",),
            "openalex": ("api.openalex.org",),
            "semantic_scholar": ("api.semanticscholar.org",),
        }
    )
    maximum_budget: ResearchBudget = field(
        default_factory=lambda: ResearchBudget.for_intensity(
            "exhaustive"
        ).with_overrides(languages=["*"])
    )
    require_citations: bool = True
    minimum_independent_sources: int = 2
    minimum_comparability: float = 0.35
    require_exact_evidence_spans: bool = True
    redact_variable_labels: bool = True
    redact_context: bool = True
    allow_raw_frames: bool = False
    allow_raw_text_columns: bool = False
    production_policy: Optional[dict[str, Any]] = None
    cache_dir: Optional[str] = None
    request_timeout_seconds: float = 12.0
    response_size_limit_bytes: int = 2_000_000
    retry_attempts: int = 2
    user_agent: str = (
        "AutoCausalLib-Research/0.1 (metadata research; contact=autocausal-library)"
    )
    production_mode: bool = False
    high_impact_domains: tuple[str, ...] = (
        "health",
        "medical",
        "clinical",
        "finance",
        "employment",
        "education",
        "criminal_justice",
    )
    require_human_for_high_impact: bool = True
    require_human_for_exhaustive: bool = True
    approval_granted: bool = False
    escalate_on_contradiction: bool = True
    escalate_on_insufficient_sources: bool = True
    stop_on_privacy_gate: bool = True
    schema: str = "AutoCausalResearchPolicy.v1"

    def __post_init__(self) -> None:
        self.allowed_providers = tuple(
            dict.fromkeys(str(item).strip().lower() for item in self.allowed_providers)
        )
        if isinstance(self.maximum_budget, Mapping):
            self.maximum_budget = ResearchBudget.for_intensity(
                "exhaustive"
            ).with_overrides(**dict(self.maximum_budget))
        if self.production_policy is not None:
            if hasattr(self.production_policy, "to_dict"):
                self.production_policy = dict(self.production_policy.to_dict())  # type: ignore[union-attr]
            elif isinstance(self.production_policy, Mapping):
                self.production_policy = dict(self.production_policy)
            else:
                raise TypeError(
                    "production_policy must be a ProductionPolicy snapshot or mapping"
                )
            profile = str(self.production_policy.get("profile") or "").lower()
            if profile == "production":
                self.production_mode = True
            operational = self.production_policy.get("operational") or {}
            max_rounds = operational.get("max_rounds") or self.production_policy.get(
                "max_rounds"
            )
            max_seconds = operational.get("max_seconds") or self.production_policy.get(
                "max_seconds"
            )
            if max_rounds is not None or max_seconds is not None:
                cap = self.maximum_budget.to_dict()
                if max_rounds is not None:
                    cap["max_rounds"] = min(cap["max_rounds"], int(max_rounds))
                if max_seconds is not None:
                    cap["wall_time_seconds"] = min(
                        cap["wall_time_seconds"], float(max_seconds)
                    )
                self.maximum_budget = ResearchBudget.from_dict(cap)
        self.minimum_independent_sources = max(1, int(self.minimum_independent_sources))
        self.minimum_comparability = min(
            1.0, max(0.0, float(self.minimum_comparability))
        )
        self.request_timeout_seconds = max(0.1, float(self.request_timeout_seconds))
        self.response_size_limit_bytes = max(1, int(self.response_size_limit_bytes))
        self.retry_attempts = max(0, int(self.retry_attempts))
        if self.allow_raw_frames or self.allow_raw_text_columns:
            raise ValueError(
                "deep-research handoffs never permit raw frames or raw text columns"
            )
        if self.allow_generic_web and "generic_web" not in self.allowed_providers:
            raise ValueError(
                "allow_generic_web=True also requires generic_web in allowed_providers"
            )

    def budget_for(
        self,
        intensity: SearchIntensity | str,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> ResearchBudget:
        budget = ResearchBudget.for_intensity(intensity)
        if overrides:
            budget = budget.with_overrides(**dict(overrides))
        return budget.capped_by(self.maximum_budget)

    def permits_provider(self, name: str, *, network: bool) -> bool:
        normalized = str(name).strip().lower()
        if normalized not in self.allowed_providers:
            return False
        if normalized == "generic_web" and not self.allow_generic_web:
            return False
        if network and not (self.allow_network and self.external_network_consent):
            return False
        return True

    def approval_reasons(
        self,
        intensity: SearchIntensity | str,
        *,
        domain: str = "",
        external: bool = False,
    ) -> list[str]:
        level = SearchIntensity.parse(intensity)
        reasons: list[str] = []
        high_impact = any(
            token in str(domain).lower() for token in self.high_impact_domains
        )
        if (
            level is SearchIntensity.EXHAUSTIVE
            and self.require_human_for_exhaustive
            and (self.production_mode or high_impact)
        ):
            reasons.append(
                "exhaustive research in production/high-impact context "
                "requires explicit human approval"
            )
        if high_impact and self.require_human_for_high_impact:
            reasons.append(f"high-impact domain requires human review: {domain}")
        if (
            self.production_mode
            and external
            and level.rank >= SearchIntensity.DEEP.rank
        ):
            reasons.append(
                "deep/high-cost external research in production requires approval"
            )
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "allowed_providers": list(self.allowed_providers),
            "allow_network": self.allow_network,
            "external_network_consent": self.external_network_consent,
            "allow_generic_web": self.allow_generic_web,
            "provider_domains": {
                key: list(value) for key, value in self.provider_domains.items()
            },
            "maximum_budget": self.maximum_budget.to_dict(),
            "require_citations": self.require_citations,
            "minimum_independent_sources": self.minimum_independent_sources,
            "minimum_comparability": self.minimum_comparability,
            "require_exact_evidence_spans": self.require_exact_evidence_spans,
            "redact_variable_labels": self.redact_variable_labels,
            "redact_context": self.redact_context,
            "allow_raw_frames": False,
            "allow_raw_text_columns": False,
            "production_policy": _json(self.production_policy),
            "cache_dir": self.cache_dir,
            "request_timeout_seconds": self.request_timeout_seconds,
            "response_size_limit_bytes": self.response_size_limit_bytes,
            "retry_attempts": self.retry_attempts,
            "user_agent": self.user_agent,
            "production_mode": self.production_mode,
            "high_impact_domains": list(self.high_impact_domains),
            "require_human_for_high_impact": self.require_human_for_high_impact,
            "require_human_for_exhaustive": self.require_human_for_exhaustive,
            "approval_granted": self.approval_granted,
            "escalate_on_contradiction": self.escalate_on_contradiction,
            "escalate_on_insufficient_sources": self.escalate_on_insufficient_sources,
            "stop_on_privacy_gate": self.stop_on_privacy_gate,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchPolicy":
        payload = dict(value)
        payload.pop("schema", None)
        if isinstance(payload.get("provider_domains"), Mapping):
            payload["provider_domains"] = {
                str(key): tuple(items)
                for key, items in payload["provider_domains"].items()
            }
        for key in ("allowed_providers", "high_impact_domains"):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> "ResearchPolicy":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise TypeError("ResearchPolicy JSON must contain an object")
        return cls.from_dict(payload)

    @classmethod
    def from_production_policy(
        cls,
        production_policy: Any,
        **overrides: Any,
    ) -> "ResearchPolicy":
        """Reference an AutoCausal ``ProductionPolicy`` without importing it."""

        snapshot = (
            production_policy.to_dict()
            if hasattr(production_policy, "to_dict")
            else dict(production_policy)
            if isinstance(production_policy, Mapping)
            else None
        )
        if snapshot is None:
            raise TypeError("production_policy must expose to_dict() or be a mapping")
        return cls(production_policy=dict(snapshot), **overrides)


@dataclass
class ResearchHandoff:
    """Privacy-safe summary of AutoCausal findings; contains no raw observations."""

    run_id: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    evidence_grades: dict[str, str] = field(default_factory=dict)
    gate_failures: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: list[dict[str, Any]] = field(default_factory=list)
    candidate_roles: dict[str, list[str]] = field(default_factory=dict)
    domain: str = "general"
    context: dict[str, Any] = field(default_factory=dict)
    variable_labels: dict[str, str] = field(default_factory=dict)
    recommended_experiments: list[dict[str, Any]] = field(default_factory=list)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    mode: str = "exploratory"
    source_type: str = "unknown"
    provenance: dict[str, Any] = field(default_factory=dict)
    schema: str = "AutoCausalResearchHandoff.v1"

    def __post_init__(self) -> None:
        self.run_id = str(self.run_id or f"handoff-{_stable_hash(utc_now())}")
        self.mode = str(self.mode or "exploratory").lower()
        if self.mode not in ("exploratory", "production"):
            raise ValueError("ResearchHandoff.mode must be exploratory or production")
        self.provenance.setdefault("contains_raw_frame", False)
        self.provenance.setdefault("contains_raw_text_columns", False)
        if self.provenance.get("contains_raw_frame") or self.provenance.get(
            "contains_raw_text_columns"
        ):
            raise ValueError("ResearchHandoff cannot contain raw frames/text columns")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "findings": _json(self.findings),
            "edges": _json(self.edges),
            "evidence_grades": dict(self.evidence_grades),
            "gate_failures": _json(self.gate_failures),
            "uncertainty": _json(self.uncertainty),
            "candidate_roles": {
                str(key): list(value) for key, value in self.candidate_roles.items()
            },
            "domain": self.domain,
            "context": _json(self.context),
            "variable_labels": dict(self.variable_labels),
            "recommended_experiments": _json(self.recommended_experiments),
            "aliases": {str(key): list(value) for key, value in self.aliases.items()},
            "mode": self.mode,
            "source_type": self.source_type,
            "provenance": _json(self.provenance),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchHandoff":
        payload = dict(value)
        payload.pop("schema", None)
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> "ResearchHandoff":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise TypeError("ResearchHandoff JSON must contain an object")
        return cls.from_dict(payload)


@dataclass
class ResearchQuestion:
    id: str
    priority: int
    finding_ids: list[str]
    question: str
    rationale: str
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    query_variants: list[str] = field(default_factory=list)
    population: Optional[str] = None
    exposure: Optional[str] = None
    comparator: Optional[str] = None
    outcome: Optional[str] = None
    context: Optional[str] = None
    source: str = "rule"

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        self.priority = min(100, max(0, int(self.priority)))
        self.question = str(self.question).strip()
        self.rationale = str(self.rationale).strip()
        self.finding_ids = list(dict.fromkeys(str(item) for item in self.finding_ids))
        self.query_variants = list(
            dict.fromkeys(
                str(item).strip() for item in self.query_variants if str(item).strip()
            )
        )
        if not self.id or not self.question or not self.rationale:
            raise ValueError("ResearchQuestion requires id, question, and rationale")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchQuestion":
        return cls(**dict(value))


@dataclass
class SourceRecord:
    """One actually retrieved or user-supplied source."""

    provider: str
    stable_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    date: Optional[str] = None
    abstract: Optional[str] = None
    snippet: Optional[str] = None
    full_text: Optional[str] = None
    retrieval_timestamp: str = field(default_factory=utc_now)
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    license: Optional[str] = None
    availability: str = "metadata"
    language: Optional[str] = None
    venue: Optional[str] = None
    references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_id: str = ""

    def __post_init__(self) -> None:
        self.provider = str(self.provider).strip().lower()
        self.title = " ".join(str(self.title).split())
        self.doi = (
            str(self.doi)
            .lower()
            .replace("https://doi.org/", "")
            .replace("http://doi.org/", "")
            .strip()
            or None
            if self.doi
            else None
        )
        self.arxiv_id = (
            str(self.arxiv_id).replace("arXiv:", "").strip() or None
            if self.arxiv_id
            else None
        )
        stable = str(self.stable_id or self.doi or self.arxiv_id or self.url or "")
        if not stable and self.title:
            stable = f"title:{_stable_hash(self.title.lower())}"
        if not self.provider or not stable or not self.title:
            raise ValueError(
                "SourceRecord requires provider, a stable identifier, and title"
            )
        self.stable_id = stable
        if not self.source_id:
            self.source_id = f"{self.provider}:{self.stable_id}"
        self.authors = list(dict.fromkeys(str(item) for item in self.authors if item))
        self.references = list(
            dict.fromkeys(str(item) for item in self.references if item)
        )
        bounded_metadata = _bounded_metadata(self.metadata)
        self.metadata = bounded_metadata if isinstance(bounded_metadata, dict) else {}
        self.availability = str(self.availability or "metadata").lower()
        if self.availability not in (
            "metadata",
            "abstract",
            "full_text",
            "snippet",
            "user_supplied",
        ):
            self.availability = "metadata"
        if self.full_text:
            self.availability = "full_text"
        elif self.abstract and self.availability == "metadata":
            self.availability = "abstract"
        elif self.snippet and self.availability == "metadata":
            self.availability = "snippet"

    @property
    def evidence_text(self) -> str:
        return str(self.full_text or self.abstract or self.snippet or "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceRecord":
        return cls(**dict(value))


@dataclass
class EvidenceSpan:
    source_id: str
    exact_text: str
    claim_relation: ClaimRelation
    extraction_method: str
    confidence: float
    structured_field: Optional[str] = None
    span_id: str = ""

    def __post_init__(self) -> None:
        self.source_id = str(self.source_id)
        self.exact_text = str(self.exact_text).strip()
        if self.claim_relation not in (
            "supports",
            "contradicts",
            "contextualizes",
            "insufficient",
        ):
            raise ValueError(f"invalid claim relation {self.claim_relation!r}")
        self.confidence = min(1.0, max(0.0, float(self.confidence)))
        if not self.source_id or not self.exact_text:
            raise ValueError("EvidenceSpan requires source_id and exact_text")
        if not self.span_id:
            self.span_id = "span:" + _stable_hash(
                f"{self.source_id}|{self.exact_text}|{self.claim_relation}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceSpan":
        return cls(**dict(value))


@dataclass
class ResearchClaim:
    normalized_claim: str
    linked_finding_ids: list[str] = field(default_factory=list)
    linked_edge: Optional[dict[str, Any]] = None
    evidence_spans: list[EvidenceSpan] = field(default_factory=list)
    contradiction_status: str = "unresolved"
    literature_label: str = "unresolved"
    independent_source_count: int = 0
    claim_id: str = ""

    def __post_init__(self) -> None:
        self.normalized_claim = " ".join(str(self.normalized_claim).split())
        if not self.normalized_claim:
            raise ValueError("ResearchClaim.normalized_claim cannot be empty")
        if not self.claim_id:
            self.claim_id = "claim:" + _stable_hash(self.normalized_claim.lower())
        self.evidence_spans = [
            item if isinstance(item, EvidenceSpan) else EvidenceSpan.from_dict(item)
            for item in self.evidence_spans
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "normalized_claim": self.normalized_claim,
            "linked_finding_ids": list(self.linked_finding_ids),
            "linked_edge": _json(self.linked_edge),
            "evidence_spans": [item.to_dict() for item in self.evidence_spans],
            "contradiction_status": self.contradiction_status,
            "literature_label": self.literature_label,
            "independent_source_count": self.independent_source_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchClaim":
        payload = dict(value)
        payload["evidence_spans"] = [
            item if isinstance(item, EvidenceSpan) else EvidenceSpan.from_dict(item)
            for item in payload.get("evidence_spans") or []
        ]
        return cls(**payload)


@dataclass
class MatchReason:
    code: str
    detail: str
    component: str
    score: float

    def __post_init__(self) -> None:
        self.score = min(1.0, max(0.0, float(self.score)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatchReason":
        return cls(**dict(value))


@dataclass
class ComparabilityScore:
    """Transparent component scores; similarity is relevance, not confirmation."""

    lexical_alias: float = 0.0
    identifier: float = 0.0
    role_compatibility: float = 0.0
    direction_agreement: float = 0.0
    population_overlap: float = 0.0
    context_overlap: float = 0.0
    time_overlap: float = 0.0
    design_relevance: float = 0.0
    overall: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in (
            "lexical_alias",
            "identifier",
            "role_compatibility",
            "direction_agreement",
            "population_overlap",
            "context_overlap",
            "time_overlap",
            "design_relevance",
            "overall",
        ):
            setattr(self, name, min(1.0, max(0.0, float(getattr(self, name)))))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComparabilityScore":
        return cls(**dict(value))


@dataclass
class CrossMatch:
    finding_id: str
    source_id: str
    claim_id: Optional[str]
    reasons: list[MatchReason]
    comparability: ComparabilityScore
    relation: ClaimRelation = "contextualizes"
    matched_concepts: list[str] = field(default_factory=list)
    match_id: str = ""

    def __post_init__(self) -> None:
        self.reasons = [
            item if isinstance(item, MatchReason) else MatchReason.from_dict(item)
            for item in self.reasons
        ]
        if not isinstance(self.comparability, ComparabilityScore):
            self.comparability = ComparabilityScore.from_dict(self.comparability)
        if not self.match_id:
            self.match_id = "match:" + _stable_hash(
                f"{self.finding_id}|{self.source_id}|{self.claim_id or ''}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "finding_id": self.finding_id,
            "source_id": self.source_id,
            "claim_id": self.claim_id,
            "reasons": [item.to_dict() for item in self.reasons],
            "comparability": self.comparability.to_dict(),
            "relation": self.relation,
            "matched_concepts": list(self.matched_concepts),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CrossMatch":
        payload = dict(value)
        payload["reasons"] = [
            MatchReason.from_dict(item) for item in payload.get("reasons") or []
        ]
        payload["comparability"] = ComparabilityScore.from_dict(
            payload.get("comparability") or {}
        )
        return cls(**payload)


@dataclass
class ClaimEvidenceGraph:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    schema: str = "AutoCausalClaimEvidenceGraph.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "nodes": _json(self.nodes),
            "edges": _json(self.edges),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClaimEvidenceGraph":
        return cls(
            nodes=list(value.get("nodes") or []),
            edges=list(value.get("edges") or []),
            schema=str(value.get("schema") or "AutoCausalClaimEvidenceGraph.v1"),
        )


@dataclass
class IntensityRecommendation:
    selected: SearchIntensity
    recommended: SearchIntensity
    reasons: list[str] = field(default_factory=list)
    approval_required: bool = False
    decision: str = "proceed"
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.selected = SearchIntensity.parse(self.selected)
        self.recommended = SearchIntensity.parse(self.recommended)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.value,
            "recommended": self.recommended.value,
            "reasons": list(self.reasons),
            "approval_required": self.approval_required,
            "decision": self.decision,
            "metrics": _json(self.metrics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IntensityRecommendation":
        return cls(**dict(value))


@dataclass
class ResearchReport:
    """Complete research result with verified source-backed citation graph."""

    handoff_run_id: str
    agenda: list[ResearchQuestion] = field(default_factory=list)
    sources: list[SourceRecord] = field(default_factory=list)
    claims: list[ResearchClaim] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    experiment_recommendations: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    costs_limits: dict[str, Any] = field(default_factory=dict)
    selected_intensity: SearchIntensity = SearchIntensity.STANDARD
    recommended_intensity: SearchIntensity = SearchIntensity.STANDARD
    intensity_rationale: list[str] = field(default_factory=list)
    budget_planned: Optional[ResearchBudget] = None
    budget_used: BudgetUsage = field(default_factory=BudgetUsage)
    cross_matches: list[CrossMatch] = field(default_factory=list)
    claim_graph: ClaimEvidenceGraph = field(default_factory=ClaimEvidenceGraph)
    unresolved_evidence_gaps: list[dict[str, Any]] = field(default_factory=list)
    source_independence_groups: dict[str, list[str]] = field(default_factory=dict)
    saturation_curve: list[dict[str, Any]] = field(default_factory=list)
    round_history: list[dict[str, Any]] = field(default_factory=list)
    context_transfer_warnings: list[str] = field(default_factory=list)
    handback_recommendations: list[dict[str, Any]] = field(default_factory=list)
    query_failures: list[dict[str, Any]] = field(default_factory=list)
    screening_log: list[dict[str, Any]] = field(default_factory=list)
    status: str = "complete"
    stop_reason: str = ""
    schema: str = "AutoCausalResearchReport.v1"
    _suite: Any = field(default=None, repr=False, compare=False)
    _handoff: Optional[ResearchHandoff] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.selected_intensity = SearchIntensity.parse(self.selected_intensity)
        self.recommended_intensity = SearchIntensity.parse(self.recommended_intensity)
        self.agenda = [
            item
            if isinstance(item, ResearchQuestion)
            else ResearchQuestion.from_dict(item)
            for item in self.agenda
        ]
        self.sources = [
            item if isinstance(item, SourceRecord) else SourceRecord.from_dict(item)
            for item in self.sources
        ]
        self.claims = [
            item if isinstance(item, ResearchClaim) else ResearchClaim.from_dict(item)
            for item in self.claims
        ]
        self.cross_matches = [
            item if isinstance(item, CrossMatch) else CrossMatch.from_dict(item)
            for item in self.cross_matches
        ]
        if not isinstance(self.claim_graph, ClaimEvidenceGraph):
            self.claim_graph = ClaimEvidenceGraph.from_dict(self.claim_graph)
        if isinstance(self.budget_planned, Mapping):
            self.budget_planned = ResearchBudget.from_dict(self.budget_planned)
        if not isinstance(self.budget_used, BudgetUsage):
            self.budget_used = BudgetUsage.from_dict(self.budget_used)
        default_caveats = [
            "Retrieved literature is external context and does not increase AutoCausal identification grade.",
            "Only fetched or user-supplied SourceRecord objects may be cited.",
            "Abstract/snippet evidence is weaker than verified full-text evidence.",
            "Semantic similarity indicates retrieval relevance, not causal confirmation.",
        ]
        for caveat in default_caveats:
            if caveat not in self.caveats:
                self.caveats.append(caveat)

    def source_map(self) -> dict[str, SourceRecord]:
        return {source.source_id: source for source in self.sources}

    def validate_citations(self, *, strict: bool = True) -> list[str]:
        """Return integrity errors and optionally raise.

        A span is valid only if its SourceRecord exists and its exact text occurs
        in the fetched abstract/snippet (unless the source is metadata-only, in
        which case no evidence span is allowed).
        """

        source_map = self.source_map()
        errors: list[str] = []
        for claim in self.claims:
            valid_spans: list[EvidenceSpan] = []
            for span in claim.evidence_spans:
                source = source_map.get(span.source_id)
                if source is None:
                    errors.append(
                        f"{claim.claim_id}: missing SourceRecord {span.source_id}"
                    )
                    continue
                evidence = source.evidence_text
                if not evidence or span.exact_text not in evidence:
                    errors.append(
                        f"{claim.claim_id}: evidence text mismatch for {span.source_id}"
                    )
                    continue
                valid_spans.append(span)
            if not strict:
                claim.evidence_spans = valid_spans
        if errors and strict:
            raise ValueError("citation integrity failed: " + "; ".join(errors))
        return errors

    def deepen(
        self,
        intensity: SearchIntensity | str = SearchIntensity.DEEP,
        *,
        budget_overrides: Optional[Mapping[str, Any]] = None,
        approval_granted: Optional[bool] = None,
    ) -> "ResearchReport":
        """Resume unresolved work using the suite/cache that produced this report."""

        if self._suite is None or self._handoff is None:
            raise RuntimeError(
                "This deserialized/detached report has no live research suite. "
                "Use DeepResearchSuite.resume(report, handoff=...)."
            )
        return self._suite.run(
            self._handoff,
            intensity=intensity,
            budget_overrides=budget_overrides,
            resume_from=self,
            approval_granted=approval_granted,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "handoff_run_id": self.handoff_run_id,
            "agenda": [item.to_dict() for item in self.agenda],
            "sources": [item.to_dict() for item in self.sources],
            "claims": [item.to_dict() for item in self.claims],
            "contradictions": _json(self.contradictions),
            "unresolved_questions": list(self.unresolved_questions),
            "experiment_recommendations": _json(self.experiment_recommendations),
            "caveats": list(self.caveats),
            "provenance": _json(self.provenance),
            "costs_limits": _json(self.costs_limits),
            "selected_intensity": self.selected_intensity.value,
            "recommended_intensity": self.recommended_intensity.value,
            "intensity_rationale": list(self.intensity_rationale),
            "budget_planned": (
                self.budget_planned.to_dict() if self.budget_planned else None
            ),
            "budget_used": self.budget_used.to_dict(),
            "cross_matches": [item.to_dict() for item in self.cross_matches],
            "claim_graph": self.claim_graph.to_dict(),
            "unresolved_evidence_gaps": _json(self.unresolved_evidence_gaps),
            "source_independence_groups": {
                key: list(value)
                for key, value in self.source_independence_groups.items()
            },
            "saturation_curve": _json(self.saturation_curve),
            "round_history": _json(self.round_history),
            "context_transfer_warnings": list(self.context_transfer_warnings),
            "handback_recommendations": _json(self.handback_recommendations),
            "query_failures": _json(self.query_failures),
            "screening_log": _json(self.screening_log),
            "status": self.status,
            "stop_reason": self.stop_reason,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchReport":
        payload = dict(value)
        payload.pop("schema", None)
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> "ResearchReport":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise TypeError("ResearchReport JSON must contain an object")
        return cls.from_dict(payload)

    def to_markdown(self) -> str:
        """Render four clearly separated evidence layers."""

        self.validate_citations(strict=True)
        lines = [
            "# AutoCausal deep research report",
            "",
            f"- handoff run: `{self.handoff_run_id}`",
            f"- status: `{self.status}`",
            f"- intensity: `{self.selected_intensity.value}` "
            f"(recommended `{self.recommended_intensity.value}`)",
            f"- sources: {len(self.sources)}; claims: {len(self.claims)}",
            f"- stop: {self.stop_reason or 'completed'}",
            "",
            "## AutoCausal empirical findings",
            "",
            "These findings originated in AutoCausal and remain exploratory unless "
            "their original design established otherwise. Literature does not upgrade "
            "their statistical or causal-identification grade.",
            "",
        ]
        handoff_edges = self.provenance.get("handoff_edges") or []
        if handoff_edges:
            for edge in handoff_edges[:30]:
                lines.append(
                    f"- `{edge.get('source')}` → `{edge.get('target')}` "
                    f"(grade={edge.get('evidence_grade', 'unverified')}; "
                    f"stability={edge.get('stability', 'n/a')})"
                )
        else:
            lines.append("- See the linked handoff; no raw observations are included.")
        lines.extend(["", "## Research agenda", ""])
        for question in self.agenda:
            lines.append(
                f"- **{question.priority}** `{question.id}` — {question.question}"
            )
            lines.append(f"  - Rationale: {question.rationale}")
        if not self.agenda:
            lines.append("- No eligible research questions.")

        lines.extend(["", "## Retrieved literature evidence", ""])
        for claim in self.claims:
            lines.append(
                f"### {claim.normalized_claim} "
                f"({claim.literature_label}; independent sources="
                f"{claim.independent_source_count})"
            )
            lines.append("")
            if not claim.evidence_spans:
                lines.append("- Unresolved: no verified evidence span.")
            for span in claim.evidence_spans:
                lines.append(
                    f"- **{span.claim_relation}** "
                    f"[{span.source_id}]: “{span.exact_text}” "
                    f"({span.extraction_method}, confidence={span.confidence:.2f})"
                )
            lines.append("")

        lines.extend(["## Cross-match comparability", ""])
        if not self.cross_matches:
            lines.append("- No deterministic finding/source candidate passed the relevance floor.")
        for match in self.cross_matches[:50]:
            score = match.comparability
            lines.append(
                f"- `{match.finding_id}` ↔ [{match.source_id}] "
                f"relation=`{match.relation}`, overall={score.overall:.2f}; "
                f"lexical={score.lexical_alias:.2f}, role={score.role_compatibility:.2f}, "
                f"direction={score.direction_agreement:.2f}, "
                f"population={score.population_overlap:.2f}, "
                f"context={score.context_overlap:.2f}, design={score.design_relevance:.2f}"
            )
            for warning in score.warnings:
                lines.append(f"  - Warning: {warning}")
        lines.append("")

        lines.extend(["## SLM synthesis", ""])
        synthesis = str(self.provenance.get("slm_synthesis") or "")
        lines.append(
            synthesis
            or "No SLM synthesis was accepted; deterministic evidence summaries are shown."
        )
        lines.extend(["", "## Contradictions and unresolved assumptions", ""])
        if self.contradictions:
            for item in self.contradictions:
                lines.append(
                    f"- {item.get('claim_id', 'claim')}: "
                    f"{item.get('detail', item.get('status', 'contradiction'))}"
                )
        if self.unresolved_evidence_gaps:
            for gap in self.unresolved_evidence_gaps:
                lines.append(f"- {gap.get('detail') or gap.get('question') or gap}")
        if not self.contradictions and not self.unresolved_evidence_gaps:
            lines.append("- None recorded within the configured search budget.")

        lines.extend(["", "## Handback recommendations", ""])
        for item in self.handback_recommendations:
            lines.append(
                f"- `{item.get('action', 'human_review')}` — {item.get('detail', '')}"
            )
        if not self.handback_recommendations:
            lines.append("- Human review of design assumptions remains required.")

        lines.extend(["", "## Sources", ""])
        for source in self.sources:
            identifier = source.doi or source.arxiv_id or source.stable_id
            target = source.url or (
                f"https://doi.org/{source.doi}" if source.doi else ""
            )
            suffix = f" — {target}" if target else ""
            lines.append(
                f"- [{source.source_id}] {source.title} "
                f"({source.date or 'date unavailable'}; {source.availability}; "
                f"id={identifier}){suffix}"
            )
        if not self.sources:
            lines.append("- No sources were retrieved; no citations are emitted.")

        lines.extend(["", "## Costs, limits, and caveats", ""])
        lines.append(
            "- Budget used: "
            + json.dumps(self.budget_used.to_dict(), sort_keys=True, default=str)
        )
        lines.append(
            "- Source independence groups: "
            + json.dumps(self.source_independence_groups, sort_keys=True)
        )
        lines.append(
            "- Saturation curve: "
            + json.dumps(self.saturation_curve, sort_keys=True, default=str)
        )
        for caveat in self.caveats:
            lines.append(f"- {caveat}")
        for warning in self.context_transfer_warnings:
            lines.append(f"- Context-transfer warning: {warning}")
        return "\n".join(lines) + "\n"

    def report(self, *, as_markdown: bool = True) -> str:
        return self.to_markdown() if as_markdown else self.to_json()

    def write(self, path: str | Path, *, fmt: Optional[str] = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        selected = (fmt or target.suffix.lstrip(".") or "md").lower()
        if selected == "json":
            target.write_text(self.to_json(), encoding="utf-8")
        else:
            if not target.suffix:
                target = target.with_suffix(".md")
            target.write_text(self.to_markdown(), encoding="utf-8")
        return target


__all__ = [
    "BudgetUsage",
    "ClaimEvidenceGraph",
    "ClaimRelation",
    "ComparabilityScore",
    "CrossMatch",
    "EvidenceSpan",
    "IntensityRecommendation",
    "MatchReason",
    "ResearchBudget",
    "ResearchClaim",
    "ResearchHandoff",
    "ResearchPolicy",
    "ResearchQuestion",
    "ResearchReport",
    "SearchIntensity",
    "SourceRecord",
    "utc_now",
]
