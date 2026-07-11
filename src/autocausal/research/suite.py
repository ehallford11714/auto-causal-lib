"""Library-first, resumable SLM-guided deep-research workflow."""

from __future__ import annotations

import copy
import json
import re
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from autocausal.research.evidence import (
    CrossMatchEngine,
    EvidenceExtractor,
    citation_integrity_errors,
    contradiction_records,
    deduplicate_sources,
    expand_related_work_queries,
    match_prior_sources,
    source_independence_groups,
)
from autocausal.research.handoff import safe_variable_label, to_research_handoff
from autocausal.research.models import (
    BudgetUsage,
    IntensityRecommendation,
    ResearchBudget,
    ResearchClaim,
    ResearchHandoff,
    ResearchPolicy,
    ResearchQuestion,
    ResearchReport,
    SearchIntensity,
    SourceRecord,
    utc_now,
)
from autocausal.research.planning import AgendaPlanner, IntensityRouter
from autocausal.research.providers import (
    LocalDocumentProvider,
    ProviderQuery,
    ResearchCache,
    ResearchProvider,
    default_provider,
)
from autocausal.research.slm import resolve_research_slm


WORKFLOW_NODES = (
    "prepare_handoff",
    "prioritize_uncertainty",
    "plan_queries",
    "retrieve",
    "deduplicate_screen",
    "extract_evidence",
    "cross_check_contradiction",
    "synthesize",
    "recommend_experiments",
    "handback",
)

DEEPEN_NODES = (
    "gap_analysis",
    "intensity_route",
    "query_expand",
    "retrieve",
    "cross_match",
    "contradiction_probe",
    "source_independence_check",
    "evidence_saturation",
    "route",
)


class ResearchPolicyError(RuntimeError):
    pass


class PrivacyGateError(ResearchPolicyError):
    pass


class ResearchApprovalRequired(ResearchPolicyError):
    def __init__(
        self, message: str, *, recommendation: IntensityRecommendation
    ) -> None:
        super().__init__(message)
        self.recommendation = recommendation


class ResearchLimitStop(RuntimeError):
    pass


def _provider_name(provider: Any) -> str:
    return str(getattr(provider, "name", type(provider).__name__)).lower()


def _provider_network(provider: Any) -> bool:
    return bool(getattr(provider, "network", False))


def _trace_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Bound observability data and refuse row/text payloads."""

    out: dict[str, Any] = {}
    for key, item in value.items():
        if key.lower() in ("frame", "rows", "records", "raw_text", "document"):
            out[str(key)] = "[omitted]"
        elif isinstance(item, (str, int, float, bool)) or item is None:
            out[str(key)] = str(item)[:300] if isinstance(item, str) else item
        elif isinstance(item, (list, tuple, set)):
            out[str(key)] = [str(value)[:100] for value in list(item)[:20]]
        elif isinstance(item, Mapping):
            out[str(key)] = {
                str(k): v
                for k, v in list(item.items())[:20]
                if isinstance(v, (str, int, float, bool)) or v is None
            }
        else:
            out[str(key)] = type(item).__name__
    return out


@contextmanager
def _stage(
    traces: list[dict[str, Any]],
    name: str,
    **metadata: Any,
) -> Iterator[dict[str, Any]]:
    started_wall = utc_now()
    started = time.perf_counter()
    row: dict[str, Any] = {
        "stage": name,
        "status": "running",
        "started_at": started_wall,
        "metadata": _trace_metadata(metadata),
    }
    traces.append(row)
    try:
        yield row
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        row["ended_at"] = utc_now()
        row["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)


def _forbidden_payload_paths(
    value: Any, *, prefix: str = "", depth: int = 0
) -> list[str]:
    if depth > 6:
        return []
    forbidden: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_s = str(key)
            path = f"{prefix}.{key_s}" if prefix else key_s
            if key_s.lower() in (
                "frame",
                "dataframe",
                "raw_frame",
                "raw_text_column",
                "sample_values",
                "records",
                "rows",
            ) and item not in (None, False, [], {}, "[raw-content-omitted]"):
                forbidden.append(path)
            else:
                forbidden.extend(
                    _forbidden_payload_paths(item, prefix=path, depth=depth + 1)
                )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value[:100]):
            forbidden.extend(
                _forbidden_payload_paths(
                    item, prefix=f"{prefix}[{index}]", depth=depth + 1
                )
            )
    return forbidden


def _question_query_plan(
    questions: Sequence[ResearchQuestion],
    *,
    budget: ResearchBudget,
    round_index: int,
    unresolved_finding_ids: Optional[set[str]] = None,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for question in questions[: budget.max_questions]:
        if unresolved_finding_ids is not None and question.finding_ids:
            if not (set(question.finding_ids) & unresolved_finding_ids):
                continue
        variants = list(question.query_variants)
        if not variants:
            variants = [question.question]
        if round_index > 0:
            suffixes = (
                "systematic review meta analysis",
                "contradictory findings limitations",
                "population context external validity",
                "negative control sensitivity",
                "replication study design",
            )
            suffix = suffixes[min(round_index - 1, len(suffixes) - 1)]
            variants = [
                *variants,
                f"{variants[0]} {suffix}",
            ]
        for query in variants[: budget.queries_per_question]:
            normalized = " ".join(str(query).split())[:500]
            if normalized:
                out.append((question.id, normalized))
    return list(dict.fromkeys(out))


def _screen_sources(
    sources: Sequence[SourceRecord],
    questions: Sequence[ResearchQuestion],
    *,
    slm_backend: Any = None,
    use_slm: bool = False,
    systematic: bool = False,
) -> tuple[list[SourceRecord], list[dict[str, Any]]]:
    query_tokens = {
        token
        for question in questions
        for text in [question.question, *question.query_variants]
        for token in re.split(r"[^a-z0-9]+", text.lower())
        if len(token) > 2
    }
    retained: list[SourceRecord] = []
    log: list[dict[str, Any]] = []
    for source in sources:
        text_tokens = {
            token
            for token in re.split(
                r"[^a-z0-9]+",
                f"{source.title} {source.abstract or ''} {source.snippet or ''}".lower(),
            )
            if len(token) > 2
        }
        overlap = len(query_tokens & text_tokens)
        keep = overlap > 0 or not query_tokens
        reason = (
            f"deterministic token overlap={overlap}"
            if keep
            else "no deterministic question/source term overlap"
        )
        if keep:
            retained.append(source)
        if systematic or not keep:
            log.append(
                {
                    "source_id": source.source_id,
                    "decision": "include" if keep else "exclude",
                    "reason": reason,
                    "stage": "deterministic_screen",
                }
            )

    if use_slm and slm_backend is not None and retained:
        method = getattr(slm_backend, "screen_sources", None)
        if callable(method):
            try:
                raw = method(
                    {
                        "questions": [question.to_dict() for question in questions],
                        "sources": [
                            {
                                "source_id": source.source_id,
                                "title": source.title,
                                "abstract": source.abstract,
                                "snippet": source.snippet,
                            }
                            for source in retained
                        ],
                    }
                )
                known = {source.source_id: source for source in retained}
                requested = (
                    set(str(item) for item in raw.get("keep_source_ids") or [])
                    if isinstance(raw, Mapping)
                    else set()
                )
                # Empty output is not allowed to erase deterministic screening.
                if requested:
                    accepted = requested & set(known)
                    rejected_unknown = requested - set(known)
                    if accepted:
                        retained = [
                            known[source_id] for source_id in sorted(accepted)
                        ]
                    log.append(
                        {
                            "stage": "slm_screen",
                            "decision": (
                                "validated"
                                if accepted
                                else "rule_fallback"
                            ),
                            "retained_ids": sorted(accepted),
                            "rejected_unknown_ids": sorted(rejected_unknown),
                        }
                    )
            except Exception as exc:
                log.append(
                    {
                        "stage": "slm_screen",
                        "decision": "rule_fallback",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    return retained, log


def _coverage_gaps(
    questions: Sequence[ResearchQuestion],
    claims: Sequence[ResearchClaim],
    *,
    policy: ResearchPolicy,
    matches: Sequence[Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    claims_by_finding = {
        finding_id: claim for claim in claims for finding_id in claim.linked_finding_ids
    }
    comparison_by_finding: dict[str, list[float]] = {}
    for match in matches:
        comparison_by_finding.setdefault(match.finding_id, []).append(
            match.comparability.overall
        )
    gaps: list[dict[str, Any]] = []
    unresolved: set[str] = set()
    for question in questions:
        ids = question.finding_ids or [question.id]
        question_resolved = True
        for finding_id in ids:
            claim = claims_by_finding.get(finding_id)
            scores = comparison_by_finding.get(finding_id) or []
            reasons: list[str] = []
            if claim is None or not claim.evidence_spans:
                reasons.append("no verified exact evidence span")
            elif claim.independent_source_count < policy.minimum_independent_sources:
                reasons.append("minimum independent-source requirement not met")
            if claim is not None and claim.contradiction_status in (
                "mixed",
                "contradicted",
            ):
                reasons.append("contradictory evidence remains")
            if scores and max(scores) < policy.minimum_comparability:
                reasons.append("insufficient population/context comparability")
            if not scores:
                reasons.append("no cross-match candidate")
            if reasons:
                question_resolved = False
                unresolved.add(finding_id)
                gaps.append(
                    {
                        "question_id": question.id,
                        "finding_id": finding_id,
                        "question": question.question,
                        "detail": "; ".join(dict.fromkeys(reasons)),
                    }
                )
        if not question_resolved and not question.finding_ids:
            unresolved.add(question.id)
    unique_gaps: dict[tuple[str, str], dict[str, Any]] = {}
    for gap in gaps:
        key = (
            str(gap.get("question_id") or ""),
            str(gap.get("finding_id") or ""),
        )
        unique_gaps.setdefault(key, gap)
    return list(unique_gaps.values()), unresolved


def _citation_references(text: str) -> set[str]:
    bracketed = {
        item.strip()
        for item in re.findall(r"\[([^\[\]]{2,160})\]", str(text))
    }
    dois = set(
        match.lower().rstrip(".,;)")
        for match in re.findall(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", str(text), re.I)
    )
    urls = {
        match.rstrip(".,;)")
        for match in re.findall(r"https://[^\s\]\)>]+", str(text), re.I)
    }
    return bracketed | dois | urls


def _synthesize(
    claims: Sequence[ResearchClaim],
    sources: Sequence[SourceRecord],
    *,
    slm_backend: Any = None,
    use_slm: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    deterministic = (
        f"Retrieved {len(sources)} unique source(s). "
        f"{sum(c.literature_label == 'supported_literature_context' for c in claims)} "
        "claim(s) met the configured independent-source literature threshold; "
        f"{sum(c.contradiction_status in ('mixed', 'contradicted') for c in claims)} "
        "claim(s) contain contradictory evidence. Literature remains external context."
    )
    if not use_slm or slm_backend is None:
        return deterministic, traces
    method = getattr(slm_backend, "synthesize", None)
    if not callable(method):
        return deterministic, traces
    try:
        raw = method(
            {
                "claims": [claim.to_dict() for claim in claims],
                "sources": [
                    {
                        "source_id": source.source_id,
                        "title": source.title,
                        "date": source.date,
                    }
                    for source in sources
                ],
                "instruction": (
                    "Use only supplied source_id values for references. "
                    "Do not claim literature changes causal identification grade."
                ),
            }
        )
        narrative = (
            str(raw.get("narrative") or "").strip() if isinstance(raw, Mapping) else ""
        )
        allowed = {source.source_id for source in sources}
        for source in sources:
            if source.doi:
                allowed.update(
                    {
                        source.doi,
                        f"doi:{source.doi}",
                        f"https://doi.org/{source.doi}",
                    }
                )
            if source.arxiv_id:
                allowed.update(
                    {
                        source.arxiv_id,
                        f"arxiv:{source.arxiv_id}",
                        f"https://arxiv.org/abs/{source.arxiv_id}",
                    }
                )
            if source.url:
                allowed.add(source.url)
        references = _citation_references(narrative)
        unsupported = sorted(
            reference for reference in references if reference not in allowed
        )
        if unsupported:
            traces.append(
                {
                    "stage": "slm_synthesis",
                    "ok": False,
                    "reason": "unsupported citation/reference blocked",
                    "unsupported": unsupported,
                }
            )
            return deterministic, traces
        if narrative:
            traces.append(
                {
                    "stage": "slm_synthesis",
                    "ok": True,
                    "validated_citations": True,
                }
            )
            return narrative, traces
    except Exception as exc:
        traces.append(
            {
                "stage": "slm_synthesis",
                "ok": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "fallback": "deterministic",
            }
        )
    return deterministic, traces


def _handback(
    handoff: ResearchHandoff,
    gaps: Sequence[Mapping[str, Any]],
    claims: Sequence[ResearchClaim],
    contradictions: Sequence[Mapping[str, Any]],
    recommendation: IntensityRecommendation,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    uncertainty_kinds = {str(item.get("kind") or "") for item in handoff.uncertainty}
    if uncertainty_kinds & {
        "low_overlap_positivity",
        "confounder_uncertainty",
        "surprising_subgroup_effect",
    }:
        out.append(
            {
                "action": "collect_data",
                "detail": (
                    "Collect overlap/confounder/population variables identified in "
                    "the unresolved assumptions; do not infer them from literature."
                ),
            }
        )
    if uncertainty_kinds & {
        "refuted_estimate",
        "sensitivity_instability",
        "low_bootstrap_stability",
    }:
        out.append(
            {
                "action": "run_refutation",
                "detail": (
                    "Run pre-specified placebo, negative-control, bootstrap, and "
                    "sensitivity checks on the AutoCausal data."
                ),
            }
        )
    if uncertainty_kinds & {
        "unsupported_orientation",
        "generative_only_claim",
        "weak_instrument",
        "invalid_iv_assumption",
    }:
        out.append(
            {
                "action": "revise_roles",
                "detail": (
                    "Human-review treatment/outcome/instrument/confounder roles and "
                    "direction; literature similarity is not role validation."
                ),
            }
        )
    if gaps and recommendation.recommended.rank > recommendation.selected.rank:
        out.append(
            {
                "action": "search_deeper",
                "detail": (
                    f"Consider `{recommendation.recommended.value}` intensity for "
                    f"{len(gaps)} unresolved evidence gap(s), subject to policy/approval."
                ),
            }
        )
    if contradictions or any(claim.independent_source_count == 0 for claim in claims):
        out.append(
            {
                "action": "human_review",
                "detail": (
                    "A domain expert should review contradictions, source independence, "
                    "and context transfer before decisions."
                ),
            }
        )
    if not out:
        out.append(
            {
                "action": "human_review",
                "detail": "Review external-context findings alongside the original design assumptions.",
            }
        )
    return out


class DeepResearchSuite:
    """Policy-bounded research FSM with deterministic fallback and resume."""

    workflow_nodes = WORKFLOW_NODES
    deepen_nodes = DEEPEN_NODES

    def __init__(
        self,
        *,
        policy: Optional[ResearchPolicy | Mapping[str, Any]] = None,
        use_slm: bool = False,
        model_name: Optional[str] = None,
        slm_backend: Any = None,
        providers: Optional[
            Sequence[ResearchProvider] | Mapping[str, ResearchProvider]
        ] = None,
        local_records: Optional[Iterable[SourceRecord | Mapping[str, Any]]] = None,
        prior_sources: Optional[Iterable[SourceRecord | Mapping[str, Any]]] = None,
        episode_sources: Optional[Iterable[SourceRecord | Mapping[str, Any]]] = None,
        public_sources: Optional[Iterable[SourceRecord | Mapping[str, Any]]] = None,
        cache: Optional[ResearchCache] = None,
    ) -> None:
        if policy is None:
            self.policy = ResearchPolicy()
        elif isinstance(policy, ResearchPolicy):
            self.policy = ResearchPolicy.from_dict(policy.to_dict())
        else:
            self.policy = ResearchPolicy.from_dict(policy)
        self.use_slm = bool(use_slm)
        self.model_name = model_name
        self.slm_backend = resolve_research_slm(
            use_slm=self.use_slm,
            backend=slm_backend,
            model_name=model_name,
        )
        self.cache = cache or ResearchCache(self.policy.cache_dir)
        self.providers: dict[str, ResearchProvider] = {}
        if isinstance(providers, Mapping):
            for name, provider in providers.items():
                self.providers[str(name).lower()] = provider
        elif providers is not None:
            for provider in providers:
                self.providers[_provider_name(provider)] = provider
        else:
            local = LocalDocumentProvider(local_records)
            self.providers["local"] = local
            for name in self.policy.allowed_providers:
                if name == "local" or name == "generic_web":
                    continue
                try:
                    self.providers[name] = default_provider(name)
                except KeyError:
                    continue
        if local_records is not None and "local" not in self.providers:
            self.providers["local"] = LocalDocumentProvider(local_records)

        def _coerce_sources(
            rows: Optional[Iterable[SourceRecord | Mapping[str, Any]]],
        ) -> list[SourceRecord]:
            out: list[SourceRecord] = []
            for row in rows or ():
                if isinstance(row, SourceRecord):
                    out.append(row)
                elif isinstance(row, Mapping):
                    out.append(SourceRecord.from_dict(row))
            return out

        self.prior_sources = _coerce_sources(prior_sources)
        self.episode_sources = _coerce_sources(episode_sources)
        self.public_sources = _coerce_sources(public_sources)
        self.planner = AgendaPlanner(slm_backend=self.slm_backend)
        self.router = IntensityRouter()
        self.cross_matcher = CrossMatchEngine(slm_backend=self.slm_backend)
        self.extractor = EvidenceExtractor(slm_backend=self.slm_backend)
        self.last_report: Optional[ResearchReport] = None
        self._reports: dict[str, ResearchReport] = {}

    def _effective_policy(
        self,
        handoff: ResearchHandoff,
        *,
        approval_granted: Optional[bool],
    ) -> ResearchPolicy:
        payload = self.policy.to_dict()
        payload["production_mode"] = bool(
            self.policy.production_mode or handoff.mode == "production"
        )
        if approval_granted is not None:
            payload["approval_granted"] = bool(approval_granted)
        return ResearchPolicy.from_dict(payload)

    def _ordered_providers(
        self, policy: ResearchPolicy, budget: ResearchBudget
    ) -> tuple[list[ResearchProvider], list[dict[str, Any]]]:
        allowed: list[ResearchProvider] = []
        skipped: list[dict[str, Any]] = []
        for name in policy.allowed_providers:
            provider = self.providers.get(name)
            if provider is None:
                skipped.append(
                    {
                        "provider": name,
                        "reason": "provider adapter not configured",
                    }
                )
                continue
            network = _provider_network(provider)
            if not policy.permits_provider(name, network=network):
                skipped.append(
                    {
                        "provider": name,
                        "reason": (
                            "provider/network denied or explicit external consent missing"
                        ),
                    }
                )
                continue
            allowed.append(provider)
        # Prefer populated local stores; otherwise an enabled network metadata
        # provider should not lose a quick run to an empty local adapter.
        allowed.sort(
            key=lambda provider: (
                0
                if _provider_name(provider) == "local"
                and (
                    bool(getattr(provider, "records", []))
                    or getattr(provider, "vector_store", None) is not None
                )
                else 1
                if _provider_network(provider)
                else 2,
                list(policy.allowed_providers).index(_provider_name(provider))
                if _provider_name(provider) in policy.allowed_providers
                else 999,
            )
        )
        return allowed[: budget.max_providers], skipped

    def _check_limits(
        self,
        usage: BudgetUsage,
        budget: ResearchBudget,
        *,
        started: float,
    ) -> None:
        usage.wall_time_seconds = round(time.perf_counter() - started, 4)
        if usage.wall_time_seconds >= budget.wall_time_seconds:
            usage.stopped_by = "wall_time"
            raise ResearchLimitStop("wall-time budget exhausted")
        if usage.sources_retained >= budget.max_sources:
            usage.stopped_by = "source_limit"
            raise ResearchLimitStop("source budget exhausted")
        if usage.tokens >= budget.max_tokens:
            usage.stopped_by = "token_limit"
            raise ResearchLimitStop("token budget exhausted")
        if usage.bytes >= budget.max_bytes:
            usage.stopped_by = "byte_limit"
            raise ResearchLimitStop("byte budget exhausted")

    def plan(
        self,
        handoff: ResearchHandoff | Any,
        *,
        intensity: SearchIntensity | str = SearchIntensity.STANDARD,
        budget_overrides: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        resolved = (
            handoff
            if isinstance(handoff, ResearchHandoff)
            else to_research_handoff(handoff, policy=self.policy)
        )
        policy = self._effective_policy(resolved, approval_granted=None)
        level = SearchIntensity.parse(intensity)
        budget = policy.budget_for(level, budget_overrides)
        agenda, trace = self.planner.plan(resolved, budget=budget, use_slm=self.use_slm)
        providers, skipped = self._ordered_providers(policy, budget)
        route = self.router.recommend(
            resolved,
            selected=level,
            policy=policy,
            external=any(_provider_network(item) for item in providers),
        )
        return {
            "schema": "AutoCausalResearchPlan.v1",
            "handoff_run_id": resolved.run_id,
            "selected_intensity": level.value,
            "recommended_intensity": route.recommended.value,
            "intensity_rationale": route.reasons,
            "approval_required": route.approval_required,
            "budget": budget.to_dict(),
            "agenda": [question.to_dict() for question in agenda],
            "providers": [_provider_name(item) for item in providers],
            "provider_skips": skipped,
            "trace": trace,
            "contains_raw_values": False,
        }

    def run(
        self,
        handoff: ResearchHandoff | Any,
        *,
        intensity: SearchIntensity | str = SearchIntensity.STANDARD,
        budget_overrides: Optional[Mapping[str, Any]] = None,
        resume_from: Optional[ResearchReport] = None,
        approval_granted: Optional[bool] = None,
    ) -> ResearchReport:
        traces: list[dict[str, Any]] = []
        tool_traces: list[dict[str, Any]] = []
        started = time.perf_counter()
        slm_token_start = int(
            getattr(self.slm_backend, "tokens_used", 0) or 0
        )
        level = SearchIntensity.parse(intensity)

        with _stage(traces, "prepare_handoff", intensity=level.value):
            resolved = (
                handoff
                if isinstance(handoff, ResearchHandoff)
                else to_research_handoff(handoff, policy=self.policy)
            )
            forbidden = _forbidden_payload_paths(resolved.to_dict())
            if forbidden:
                raise PrivacyGateError(
                    "research handoff contains forbidden raw payload fields: "
                    + ", ".join(forbidden[:10])
                )
            policy = self._effective_policy(resolved, approval_granted=approval_granted)
            if policy.redact_variable_labels:
                labels_to_check = [
                    str(edge.get(key) or "")
                    for edge in resolved.edges
                    for key in ("source", "target", "instrument")
                    if edge.get(key)
                ]
                labels_to_check.extend(
                    str(value)
                    for values in resolved.aliases.values()
                    for value in values
                )
                if any(
                    safe_variable_label(label).startswith("private_variable_")
                    for label in labels_to_check
                ):
                    raise PrivacyGateError(
                        "ResearchHandoff contains an unredacted PII-like "
                        "variable label; rebuild it with to_research_handoff()."
                    )
            budget = policy.budget_for(level, budget_overrides)
            providers, provider_skips = self._ordered_providers(policy, budget)
            external = any(_provider_network(item) for item in providers)
            if (
                external
                and policy.production_mode
                and not policy.external_network_consent
            ):
                raise PrivacyGateError(
                    "production external retrieval requires explicit network consent"
                )

        with _stage(
            traces,
            "prioritize_uncertainty",
            n_uncertainty=len(resolved.uncertainty),
            n_gate_failures=len(resolved.gate_failures),
        ):
            route = self.router.recommend(
                resolved,
                selected=level,
                policy=policy,
                report=resume_from,
                external=external,
            )
            if route.approval_required and not policy.approval_granted:
                raise ResearchApprovalRequired(
                    "; ".join(route.reasons),
                    recommendation=route,
                )

        with _stage(traces, "plan_queries", resume=resume_from is not None):
            agenda, planner_trace = self.planner.plan(
                resolved, budget=budget, use_slm=self.use_slm
            )
            if resume_from is not None:
                existing = {question.id: question for question in resume_from.agenda}
                for question in agenda:
                    existing.setdefault(question.id, question)
                agenda = sorted(
                    existing.values(), key=lambda item: (-item.priority, item.id)
                )[: budget.max_questions]

        previous_sources = (
            [copy.deepcopy(source) for source in resume_from.sources]
            if resume_from is not None
            else []
        )
        previous_usage = (
            BudgetUsage.from_dict(resume_from.budget_used.to_dict())
            if resume_from is not None
            else BudgetUsage()
        )
        usage = previous_usage
        prior_token_usage = usage.tokens
        prior_wall_usage = usage.wall_time_seconds
        usage.questions = len(agenda)
        usage.providers = len(providers)
        completed_queries = set(
            str(item)
            for item in (
                (resume_from.provenance.get("completed_queries") or [])
                if resume_from is not None
                else []
            )
        )
        query_log = list(
            (resume_from.provenance.get("query_log") or [])
            if resume_from is not None
            else []
        )
        query_failures = list(
            resume_from.query_failures if resume_from is not None else []
        )
        query_failures.extend(
            {
                "provider": item["provider"],
                "query": None,
                "error": item["reason"],
                "soft": True,
            }
            for item in provider_skips
        )
        screening_log = list(
            resume_from.screening_log if resume_from is not None else []
        )
        round_history = list(
            resume_from.round_history if resume_from is not None else []
        )
        saturation_curve = list(
            resume_from.saturation_curve if resume_from is not None else []
        )
        provider_counts = dict(
            (resume_from.provenance.get("provider_source_counts") or {})
            if resume_from is not None
            else {}
        )
        all_sources = list(previous_sources)
        dedupe_aliases: dict[str, str] = {}
        dedupe_log: list[dict[str, Any]] = []
        claims: list[ResearchClaim] = (
            [copy.deepcopy(claim) for claim in resume_from.claims]
            if resume_from is not None
            else []
        )
        matches = (
            [copy.deepcopy(match) for match in resume_from.cross_matches]
            if resume_from is not None
            else []
        )
        contradictions = list(
            resume_from.contradictions if resume_from is not None else []
        )
        gaps, unresolved_ids = _coverage_gaps(
            agenda, claims, policy=policy, matches=matches
        )
        if not unresolved_ids:
            unresolved_ids = {
                str(edge.get("finding_id"))
                for edge in resolved.edges
                if edge.get("finding_id")
            }

        base_round = (
            max(
                [int(item.get("round") or 0) for item in round_history],
                default=-1,
            )
            + 1
        )
        stop_reason = ""
        status = "complete"
        max_query_calls = (
            budget.max_questions
            * budget.queries_per_question
            * budget.max_rounds
            * max(1, budget.max_providers)
        )

        for local_round in range(budget.max_rounds):
            round_index = base_round + local_round
            usage.rounds = max(usage.rounds, round_index + 1)
            before_unique = len(all_sources)
            round_queries = _question_query_plan(
                agenda,
                budget=budget,
                round_index=local_round,
                unresolved_finding_ids=unresolved_ids
                if resume_from is not None or local_round > 0
                else None,
            )
            if local_round > 0:
                with _stage(
                    traces,
                    "gap_analysis",
                    round=round_index,
                    gaps=len(gaps),
                ):
                    pass
                with _stage(
                    traces,
                    "intensity_route",
                    round=round_index,
                    selected=level.value,
                ):
                    route = self.router.recommend(
                        resolved,
                        selected=level,
                        policy=policy,
                        report=None,
                        external=external,
                    )
                with _stage(
                    traces,
                    "query_expand",
                    round=round_index,
                    deterministic_queries=len(round_queries),
                ):
                    # Citation/related-work expansion is bounded and uses only
                    # references already present in fetched SourceRecords.
                    if level.rank >= SearchIntensity.DEEP.rank:
                        related = expand_related_work_queries(
                            all_sources,
                            question_id=agenda[0].id if agenda else "related_work",
                            limit=budget.queries_per_question
                            * (2 if level is SearchIntensity.EXHAUSTIVE else 1),
                            prefer_identifiers=True,
                        )
                        round_queries.extend(related)
                        tool_traces.append(
                            {
                                "stage": "query_expand",
                                "tool": "related_work_identifiers",
                                "ok": True,
                                "added_queries": len(related),
                            }
                        )
                    expand = getattr(self.slm_backend, "expand_queries", None)
                    if self.use_slm and callable(expand) and gaps:
                        try:
                            raw = expand(
                                {
                                    "gaps": gaps[: budget.max_questions],
                                    "existing_queries": [
                                        query for _, query in round_queries
                                    ],
                                    "max_new_queries": budget.queries_per_question,
                                }
                            )
                            proposed = (
                                raw.get("queries") if isinstance(raw, Mapping) else []
                            )
                            # Query text is not evidence/citation. Keep bounded,
                            # reject URL-shaped or identifier-inventing strings.
                            for query in proposed or []:
                                text = " ".join(str(query).split())[:500]
                                if (
                                    text
                                    and "://" not in text
                                    and not re.search(r"\b10\.\d{4,9}/\S+", text)
                                ):
                                    qid = agenda[0].id if agenda else "slm_expand"
                                    round_queries.append((qid, text))
                        except Exception as exc:
                            tool_traces.append(
                                {
                                    "stage": "query_expand",
                                    "tool": "slm",
                                    "ok": False,
                                    "fallback": "rule",
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                    round_queries = list(dict.fromkeys(round_queries))

            if not providers:
                stop_reason = (
                    "No provider was permitted/configured. Claims remain unresolved."
                )
                status = "insufficient_sources"
                break
            if not round_queries:
                stop_reason = "No unresolved or uncompleted query remained."
                break

            newly_fetched: list[SourceRecord] = []
            try:
                with _stage(
                    traces,
                    "retrieve",
                    round=round_index,
                    providers=[_provider_name(item) for item in providers],
                    planned_queries=len(round_queries),
                ):
                    for provider in providers:
                        name = _provider_name(provider)
                        remaining_provider = budget.sources_per_provider - int(
                            provider_counts.get(name, 0)
                        )
                        if remaining_provider <= 0:
                            continue
                        for question_id, query in round_queries:
                            usage.tokens = prior_token_usage + max(
                                0,
                                int(
                                    getattr(self.slm_backend, "tokens_used", 0)
                                    or 0
                                )
                                - slm_token_start,
                            )
                            self._check_limits(
                                usage,
                                budget,
                                started=started - prior_wall_usage,
                            )
                            if usage.queries >= max_query_calls:
                                usage.stopped_by = "query_limit"
                                raise ResearchLimitStop(
                                    "derived query budget exhausted"
                                )
                            request = ProviderQuery(
                                query=query,
                                limit=min(
                                    budget.sources_per_provider,
                                    remaining_provider,
                                    max(
                                        1,
                                        budget.max_sources - usage.sources_retained,
                                    ),
                                ),
                                publication_year_min=budget.publication_year_min,
                                publication_year_max=budget.publication_year_max,
                                languages=list(budget.languages),
                                question_id=question_id,
                                round_index=round_index,
                            )
                            signature = f"{name}|" + json.dumps(
                                {
                                    "query": " ".join(request.query.lower().split()),
                                    "publication_year_min": (
                                        request.publication_year_min
                                    ),
                                    "publication_year_max": (
                                        request.publication_year_max
                                    ),
                                    "languages": sorted(request.languages),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            if signature in completed_queries:
                                tool_traces.append(
                                    {
                                        "stage": "retrieve",
                                        "provider": name,
                                        "query": query,
                                        "ok": True,
                                        "resume_skip": True,
                                    }
                                )
                                continue
                            usage.queries += 1
                            cached = self.cache.get(name, request)
                            cache_hit = cached is not None
                            if cache_hit:
                                usage.cache_hits += 1
                            try:
                                fetched = (
                                    cached
                                    if cached is not None
                                    else provider.search(request, policy=policy)
                                )
                                if cached is None:
                                    self.cache.put(name, request, fetched)
                                fetched = list(fetched)[
                                    : min(request.limit, remaining_provider)
                                ]
                                serialized_bytes = len(
                                    json.dumps(
                                        [item.to_dict() for item in fetched],
                                        default=str,
                                    ).encode("utf-8")
                                )
                                bytes_used = (
                                    serialized_bytes
                                    if cache_hit
                                    else max(
                                        serialized_bytes,
                                        int(
                                            getattr(
                                                provider,
                                                "last_response_bytes",
                                                0,
                                            )
                                            or 0
                                        ),
                                    )
                                )
                                if usage.bytes + bytes_used > budget.max_bytes:
                                    usage.stopped_by = "byte_limit"
                                    raise ResearchLimitStop(
                                        "retrieval would exceed byte budget"
                                    )
                                usage.bytes += bytes_used
                                usage.sources_fetched += len(fetched)
                                newly_fetched.extend(fetched)
                                provider_counts[name] = int(
                                    provider_counts.get(name, 0)
                                ) + len(fetched)
                                remaining_provider -= len(fetched)
                                completed_queries.add(signature)
                                query_log.append(
                                    {
                                        "provider": name,
                                        "question_id": question_id,
                                        "query": query,
                                        "round": round_index,
                                        "limit": request.limit,
                                        "returned": len(fetched),
                                        "cache_hit": cache_hit,
                                        "timestamp": utc_now(),
                                    }
                                )
                                tool_traces.append(
                                    {
                                        "stage": "retrieve",
                                        "provider": name,
                                        "query": query,
                                        "ok": True,
                                        "returned": len(fetched),
                                        "cache_hit": cache_hit,
                                    }
                                )
                            except ResearchLimitStop:
                                raise
                            except Exception as exc:
                                usage.failed_queries += 1
                                failure = {
                                    "provider": name,
                                    "question_id": question_id,
                                    "query": query,
                                    "round": round_index,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                                query_failures.append(failure)
                                tool_traces.append(
                                    {
                                        "stage": "retrieve",
                                        "provider": name,
                                        "query": query,
                                        "ok": False,
                                        "error": failure["error"],
                                    }
                                )
                            if (
                                remaining_provider <= 0
                                or usage.sources_fetched >= budget.max_sources * 3
                            ):
                                break
            except ResearchLimitStop as exc:
                stop_reason = str(exc)
                status = "policy_limit"

            with _stage(
                traces,
                "deduplicate_screen",
                round=round_index,
                fetched=len(newly_fetched),
            ):
                combined, round_dedupe, aliases = deduplicate_sources(
                    [*all_sources, *newly_fetched]
                )
                dedupe_log.extend(round_dedupe)
                dedupe_aliases.update(aliases)
                screened, round_screen = _screen_sources(
                    combined,
                    agenda,
                    slm_backend=self.slm_backend,
                    use_slm=self.use_slm,
                    systematic=level is SearchIntensity.EXHAUSTIVE,
                )
                all_sources = screened[: budget.max_sources]
                screening_log.extend(round_screen)
                usage.sources_retained = len(all_sources)

            cross_stage = "cross_match" if local_round > 0 else "cross_match"
            with _stage(
                traces,
                cross_stage,
                round=round_index,
                sources=len(all_sources),
                prior_sources=len(self.prior_sources),
                episode_sources=len(self.episode_sources),
                public_sources=len(self.public_sources),
            ):
                matches = match_prior_sources(
                    resolved,
                    all_sources,
                    prior_sources=self.prior_sources,
                    episode_sources=self.episode_sources,
                    public_sources=self.public_sources,
                    use_slm=self.use_slm,
                    slm_backend=self.slm_backend,
                )

            with _stage(
                traces,
                "extract_evidence",
                round=round_index,
                matches=len(matches),
            ):
                claims, extraction_failures = self.extractor.extract(
                    resolved,
                    agenda,
                    all_sources,
                    matches,
                    policy=policy,
                    use_slm=self.use_slm,
                )
                tool_traces.extend(
                    {
                        "stage": "extract_evidence",
                        "tool": "citation_guard",
                        "ok": False,
                        **failure,
                    }
                    for failure in extraction_failures
                )
                integrity = citation_integrity_errors(claims, all_sources)
                if integrity:
                    raise ResearchPolicyError(
                        "citation integrity failed: " + "; ".join(integrity)
                    )

            contradiction_stage = (
                "contradiction_probe"
                if local_round > 0
                else "cross_check_contradiction"
            )
            with _stage(
                traces,
                contradiction_stage,
                round=round_index,
            ):
                contradictions = contradiction_records(claims, all_sources)

            with _stage(
                traces,
                "source_independence_check",
                round=round_index,
            ):
                independence = source_independence_groups(all_sources)
                gaps, unresolved_ids = _coverage_gaps(
                    agenda, claims, policy=policy, matches=matches
                )

            new_unique = max(0, len(all_sources) - before_unique)
            saturation_ratio = new_unique / max(1, len(newly_fetched))
            saturation_row = {
                "round": round_index,
                "queries": len(round_queries),
                "fetched": len(newly_fetched),
                "new_unique_sources": new_unique,
                "retained_sources": len(all_sources),
                "marginal_unique_ratio": round(saturation_ratio, 4),
                "unresolved_gaps": len(gaps),
                "contradictions": len(contradictions),
            }
            with _stage(
                traces,
                "evidence_saturation",
                **saturation_row,
            ):
                saturation_curve.append(saturation_row)

            round_history.append(
                {
                    **saturation_row,
                    "provider_counts": dict(provider_counts),
                    "budget_used": usage.to_dict(),
                }
            )

            with _stage(
                traces,
                "route",
                round=round_index,
                unresolved=len(gaps),
            ):
                if status == "policy_limit":
                    break
                if not gaps and not contradictions:
                    stop_reason = (
                        "Minimum independent-source coverage met with no "
                        "unresolved contradiction."
                    )
                    break
                if new_unique == 0 and (
                    local_round > 0 or resume_from is not None or not newly_fetched
                ):
                    stop_reason = (
                        "Evidence saturation/diminishing returns: no new unique "
                        "source in the latest round."
                    )
                    status = "human_review" if contradictions or gaps else "complete"
                    break
                if local_round >= budget.max_rounds - 1:
                    stop_reason = (
                        f"Completed configured max_rounds={budget.max_rounds}; "
                        f"{len(gaps)} gap(s) remain."
                    )
                    if gaps or contradictions:
                        status = "human_review"
                    break
                if level is SearchIntensity.QUICK:
                    stop_reason = "Quick intensity completed one bounded round."
                    if gaps:
                        status = "insufficient_sources"
                    break

        usage.wall_time_seconds = round(
            prior_wall_usage + time.perf_counter() - started, 4
        )
        usage.tokens = prior_token_usage + max(
            0,
            int(getattr(self.slm_backend, "tokens_used", 0) or 0)
            - slm_token_start,
        )
        if usage.tokens > budget.max_tokens:
            usage.stopped_by = "token_limit"
            status = "policy_limit"
            stop_reason = "SLM token budget exhausted."

        with _stage(traces, "synthesize", sources=len(all_sources)):
            synthesis, synthesis_trace = _synthesize(
                claims,
                all_sources,
                slm_backend=self.slm_backend,
                use_slm=self.use_slm,
            )
            tool_traces.extend(synthesis_trace)

        with _stage(
            traces,
            "recommend_experiments",
            unresolved=len(gaps),
        ):
            experiments = list(resolved.recommended_experiments)
            for gap in gaps:
                experiments.append(
                    {
                        "kind": "evidence_gap_follow_up",
                        "priority": "high",
                        "title": str(gap.get("question") or "Resolve evidence gap"),
                        "rationale": str(gap.get("detail") or ""),
                        "finding_id": gap.get("finding_id"),
                    }
                )
            # Deduplicate stable serialized recommendations.
            unique_experiments: dict[str, dict[str, Any]] = {}
            for experiment in experiments:
                key = json.dumps(experiment, sort_keys=True, default=str)
                unique_experiments.setdefault(key, experiment)
            experiments = list(unique_experiments.values())[:50]

        route = self.router.recommend(
            resolved,
            selected=level,
            policy=policy,
            report=None,
            external=external,
        )
        if gaps and route.recommended.rank <= level.rank:
            # The current run's gaps, contradictions, and comparability can
            # recommend a next intensity without automatically escalating it.
            if level.rank < SearchIntensity.EXHAUSTIVE.rank:
                route.recommended = SearchIntensity(
                    [
                        "quick",
                        "standard",
                        "deep",
                        "exhaustive",
                    ][level.rank + 1]
                )
                route.reasons.append(
                    f"{len(gaps)} unresolved gap(s) remain after current budget"
                )

        context_warnings = sorted(
            {
                warning
                for match in matches
                for warning in match.comparability.warnings
                if "population" in warning
                or "context" in warning
                or "period" in warning
            }
        )
        with _stage(traces, "handback", status=status):
            handback = _handback(resolved, gaps, claims, contradictions, route)

        graph = self.cross_matcher.build_graph(resolved, all_sources, claims, matches)
        report = ResearchReport(
            handoff_run_id=resolved.run_id,
            agenda=agenda,
            sources=all_sources,
            claims=claims,
            contradictions=contradictions,
            unresolved_questions=[
                str(item.get("question") or item.get("detail") or "") for item in gaps
            ],
            experiment_recommendations=experiments,
            caveats=[],
            provenance={
                "workflow": list(WORKFLOW_NODES),
                "deepen_workflow": list(DEEPEN_NODES),
                "agent_spans": traces,
                "tool_traces": tool_traces,
                "planner_trace": planner_trace,
                "query_log": query_log,
                "completed_queries": sorted(completed_queries),
                "provider_source_counts": provider_counts,
                "provider_skips": provider_skips,
                "dedupe_log": dedupe_log,
                "dedupe_aliases": dedupe_aliases,
                "handoff_source_type": resolved.source_type,
                "handoff_edges": resolved.edges,
                "handoff_contains_raw_values": False,
                "slm_used": bool(self.use_slm and self.slm_backend is not None),
                "slm_synthesis": synthesis,
                "resumed": resume_from is not None,
                "resumed_from_intensity": (
                    resume_from.selected_intensity.value
                    if resume_from is not None
                    else None
                ),
            },
            costs_limits={
                "policy": policy.to_dict(),
                "planned": budget.to_dict(),
                "used": usage.to_dict(),
                "limits_enforced": True,
            },
            selected_intensity=level,
            recommended_intensity=route.recommended,
            intensity_rationale=route.reasons,
            budget_planned=budget,
            budget_used=usage,
            cross_matches=matches,
            claim_graph=graph,
            unresolved_evidence_gaps=gaps,
            source_independence_groups=independence if all_sources else {},
            saturation_curve=saturation_curve,
            round_history=round_history,
            context_transfer_warnings=context_warnings,
            handback_recommendations=handback,
            query_failures=query_failures,
            screening_log=screening_log,
            status=status,
            stop_reason=stop_reason
            or "Research workflow completed within configured policy.",
        )
        report._suite = self
        report._handoff = resolved
        # Final fail-closed guard strips nothing silently; any mismatch is a bug.
        report.validate_citations(strict=True)
        self.last_report = report
        self._reports[resolved.run_id] = report
        return report

    def resume(
        self,
        report: ResearchReport,
        *,
        handoff: Optional[ResearchHandoff] = None,
        intensity: SearchIntensity | str = SearchIntensity.DEEP,
        budget_overrides: Optional[Mapping[str, Any]] = None,
        approval_granted: Optional[bool] = None,
    ) -> ResearchReport:
        resolved = handoff or report._handoff
        if resolved is None:
            raise ValueError(
                "resume requires the original ResearchHandoff for a detached report"
            )
        return self.run(
            resolved,
            intensity=intensity,
            budget_overrides=budget_overrides,
            resume_from=report,
            approval_granted=approval_granted,
        )

    def status(self, run_id: Optional[str] = None) -> dict[str, Any]:
        if run_id is not None:
            report = self._reports.get(str(run_id))
            return {
                "run_id": str(run_id),
                "found": report is not None,
                "status": report.status if report else "unknown",
                "stop_reason": report.stop_reason if report else "",
                "budget_used": report.budget_used.to_dict() if report else None,
            }
        return {
            "runs": [
                {
                    "run_id": key,
                    "status": report.status,
                    "intensity": report.selected_intensity.value,
                    "sources": len(report.sources),
                    "claims": len(report.claims),
                }
                for key, report in sorted(self._reports.items())
            ]
        }


__all__ = [
    "DEEPEN_NODES",
    "WORKFLOW_NODES",
    "DeepResearchSuite",
    "PrivacyGateError",
    "ResearchApprovalRequired",
    "ResearchLimitStop",
    "ResearchPolicyError",
]
