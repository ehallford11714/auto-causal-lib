"""Safe adapters from AutoCausal result objects to :class:`ResearchHandoff`."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Optional, Sequence

from autocausal.research.models import ResearchHandoff, ResearchPolicy


_PII_RE = re.compile(
    r"(^|[_\-\s])(ssn|social.?security|email|phone|mobile|address|"
    r"first.?name|last.?name|full.?name|dob|birth|passport|credit.?card|"
    r"account.?number|ip.?address|patient.?id|customer.?id|user.?id|"
    r"member.?id|person.?id|employee.?id|student.?id|identifier)($|[_\-\s])",
    re.I,
)
_SECRET_RE = re.compile(
    r"(api[_-]?key|access[_-]?token|secret|password|authorization|bearer)",
    re.I,
)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b")


def _hash(value: str, length: int = 10) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        data = value.to_dict()
        return dict(data) if isinstance(data, Mapping) else {}
    try:
        return dict(vars(value))
    except Exception:
        return {}


def _safe_scalar(value: Any, *, max_chars: int = 500) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _PHONE_RE.sub("[redacted-phone]", text)
    text = _LONG_TOKEN_RE.sub("[redacted-token]", text)
    if _SECRET_RE.search(text):
        return "[redacted-secret]"
    return text[:max_chars]


def redact_context(value: Any, *, max_depth: int = 4) -> Any:
    """Redact likely identifiers/secrets and bound nested context size."""

    if max_depth <= 0:
        return "[truncated]"
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in list(value.items())[:50]:
            key_s = str(key)
            if _SECRET_RE.search(key_s):
                out[key_s] = "[redacted-secret]"
            elif key_s.lower() in ("frame", "dataframe", "rows", "records", "raw_text"):
                out[key_s] = "[raw-content-omitted]"
            else:
                out[key_s] = redact_context(item, max_depth=max_depth - 1)
        return out
    if isinstance(value, (list, tuple, set)):
        return [
            redact_context(item, max_depth=max_depth - 1) for item in list(value)[:50]
        ]
    return _safe_scalar(value)


def safe_variable_label(
    name: Any,
    *,
    redact: bool = True,
) -> str:
    """Return a query-safe schema label, pseudonymizing PII-like names."""

    raw = str(name or "unknown").strip()
    if redact and (_PII_RE.search(raw) or _SECRET_RE.search(raw)):
        return f"private_variable_{_hash(raw)}"
    cleaned = re.sub(r"[^A-Za-z0-9_\-\s]", " ", raw)
    cleaned = re.sub(r"[_\-\s]+", " ", cleaned).strip().lower()
    if not cleaned:
        return f"variable_{_hash(raw)}"
    return cleaned[:80]


def _finding_id(edge: Mapping[str, Any], index: int) -> str:
    src = edge.get("source") or edge.get("from") or "unknown"
    tgt = edge.get("target") or edge.get("to") or "unknown"
    method = edge.get("method") or edge.get("orientation") or ""
    return f"edge:{_hash(f'{src}|{tgt}|{method}|{index}', 14)}"


def _gate_failures(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    candidates.extend(payload.get("evidence_gates") or [])
    candidates.extend(payload.get("gate_failures") or [])
    gates = payload.get("gates") or []
    if isinstance(gates, Mapping):
        gates = gates.get("results") or []
    candidates.extend(gates)
    manifest = payload.get("manifest") or {}
    if isinstance(manifest, Mapping):
        candidates.extend(manifest.get("gates") or [])
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        gate = _payload(item)
        status = str(gate.get("status") or "").lower()
        failed = gate.get("ok") is False or status in ("fail", "escalate")
        if not failed:
            continue
        gate_id = str(gate.get("id") or gate.get("code") or "gate_failure")
        key = gate_id + "|" + str(gate.get("detail") or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "id": gate_id,
                "status": status or "fail",
                "detail": _safe_scalar(gate.get("detail") or gate.get("message") or ""),
                "metric": _safe_scalar(gate.get("metric")),
                "threshold": _safe_scalar(gate.get("threshold")),
                "recommendation": _safe_scalar(
                    gate.get("recommendation") or gate.get("remediation") or ""
                ),
                "stage": _safe_scalar(gate.get("stage") or "unspecified"),
            }
        )
    return out


def _uncertainties(
    edges: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    gate_failures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    uncertainties: list[dict[str, Any]] = []

    def add(
        kind: str,
        detail: str,
        *,
        finding_id: Optional[str] = None,
        severity: str = "medium",
        metric: Any = None,
    ) -> None:
        row = {
            "kind": kind,
            "detail": detail,
            "severity": severity,
        }
        if finding_id:
            row["finding_id"] = finding_id
        if metric is not None:
            row["metric"] = metric
        if row not in uncertainties:
            uncertainties.append(row)

    for index, edge in enumerate(edges):
        finding_id = str(edge.get("finding_id") or _finding_id(edge, index))
        stability = edge.get("stability")
        if stability is not None:
            try:
                if float(stability) < 0.60:
                    add(
                        "low_bootstrap_stability",
                        f"Bootstrap stability {float(stability):.3f} is below 0.60.",
                        finding_id=finding_id,
                        severity="high",
                        metric=float(stability),
                    )
            except (TypeError, ValueError):
                pass
        agreement = edge.get("agreement")
        if agreement is not None:
            try:
                if float(agreement) < 0.67:
                    add(
                        "engine_disagreement",
                        f"Discovery method agreement is {float(agreement):.3f}.",
                        finding_id=finding_id,
                        severity="high",
                        metric=float(agreement),
                    )
            except (TypeError, ValueError):
                pass
        methods = list(edge.get("methods") or [])
        if edge.get("n_methods") is not None and int(edge.get("n_methods") or 0) < 2:
            add(
                "engine_disagreement",
                "Fewer than two independent discovery methods support the edge.",
                finding_id=finding_id,
            )
        if str(edge.get("type") or "").lower() == "iv_2sls" or edge.get("instrument"):
            f_value = edge.get("first_stage_f")
            synthetic = edge.get("synthetic") or edge.get("auto_instrument")
            if synthetic:
                add(
                    "invalid_iv_assumption",
                    "The instrument is synthetic/demo-only and cannot identify an effect.",
                    finding_id=finding_id,
                    severity="high",
                )
            if f_value is None:
                add(
                    "unverified_iv_assumptions",
                    "IV relevance, exclusion, and independence are not verified.",
                    finding_id=finding_id,
                    severity="high",
                )
            else:
                try:
                    if float(f_value) < 10.0:
                        add(
                            "weak_instrument",
                            f"First-stage F={float(f_value):.3f} is below 10.",
                            finding_id=finding_id,
                            severity="high",
                            metric=float(f_value),
                        )
                except (TypeError, ValueError):
                    pass
        orientation = str(edge.get("orientation") or "").lower()
        method = str(edge.get("method") or "").lower()
        if orientation in ("score_r2", "unknown", "") or method in (
            "grail_boost",
            "slm",
            "nlp",
        ):
            add(
                "unsupported_orientation",
                "Direction/orientation is heuristic and needs external/design support.",
                finding_id=finding_id,
                severity="medium",
            )
        if method in ("grail_boost", "slm", "nlp") or (
            methods
            and all(str(item).lower() in ("grail", "slm", "nlp") for item in methods)
        ):
            add(
                "generative_only_claim",
                "Claim originated only from SLM/GRAIL/NLP hypothesis generation.",
                finding_id=finding_id,
                severity="high",
            )
        if edge.get("refuted") or str(edge.get("evidence_grade")) == "refuted":
            add(
                "refuted_estimate",
                "A refutation result conflicts with the empirical edge/estimate.",
                finding_id=finding_id,
                severity="high",
            )
        if edge.get("subgroup") or edge.get("heterogeneity"):
            add(
                "surprising_subgroup_effect",
                "Subgroup/heterogeneous effect requires population-specific follow-up.",
                finding_id=finding_id,
            )

    for gate in gate_failures:
        gate_id = str(gate.get("id") or "").lower()
        detail = str(gate.get("detail") or "")
        kind = "production_gate_failure"
        if "overlap" in gate_id or "positiv" in gate_id:
            kind = "low_overlap_positivity"
        elif "confound" in gate_id:
            kind = "confounder_uncertainty"
        elif "collider" in gate_id or "selection" in gate_id:
            kind = "collider_selection_bias"
        elif "instrument" in gate_id or "iv" in gate_id:
            kind = "weak_iv_assumption"
        add(kind, detail or f"Gate {gate_id} failed.", severity="high")

    notes = " ".join(str(item) for item in payload.get("notes") or []).lower()
    for needle, kind, detail in (
        (
            "positivity",
            "low_overlap_positivity",
            "Notes indicate a possible positivity/overlap problem.",
        ),
        (
            "overlap",
            "low_overlap_positivity",
            "Notes indicate a possible positivity/overlap problem.",
        ),
        (
            "collider",
            "collider_selection_bias",
            "Notes indicate possible collider adjustment.",
        ),
        (
            "selection bias",
            "collider_selection_bias",
            "Notes indicate possible selection bias.",
        ),
        (
            "confound",
            "confounder_uncertainty",
            "Potential unmeasured or uncertain confounding remains.",
        ),
    ):
        if needle in notes:
            add(kind, detail)

    sensitivity = payload.get("sensitivity") or payload.get("sensitivity_report") or {}
    if isinstance(sensitivity, Mapping):
        for metric in sensitivity.get("metrics") or []:
            row = _payload(metric)
            name = str(row.get("name") or "").lower()
            value = row.get("value")
            if "stability" in name and value is not None:
                try:
                    if float(value) < 0.60:
                        add(
                            "sensitivity_instability",
                            f"Sensitivity stability metric is {float(value):.3f}.",
                            severity="high",
                            metric=float(value),
                        )
                except (TypeError, ValueError):
                    pass
            if "overlap" in name and value is not None:
                try:
                    if float(value) < 0.90:
                        add(
                            "low_overlap_positivity",
                            f"Overlap metric is {float(value):.3f}.",
                            severity="high",
                            metric=float(value),
                        )
                except (TypeError, ValueError):
                    pass
    return uncertainties


def to_research_handoff(
    result: Any,
    *,
    domain: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
    policy: Optional[ResearchPolicy] = None,
    recommended_experiments: Optional[Sequence[Mapping[str, Any]]] = None,
) -> ResearchHandoff:
    """Convert Discovery/Auto/Insight/Agentic/Gate results without raw data."""

    research_policy = policy or ResearchPolicy()
    root = _payload(result)
    source_type = type(result).__name__

    discovery = root.get("discovery")
    if isinstance(discovery, Mapping):
        base = dict(discovery)
        # AutoResult/InsightReport may keep important context at the outer layer.
        for key in (
            "nlp_hints",
            "sensitivity",
            "sensitivity_report",
            "notes",
            "guide",
            "grail",
            "manifest",
        ):
            if key not in base and root.get(key) is not None:
                base[key] = root.get(key)
    else:
        base = dict(root)

    raw_edges = list(
        base.get("edges") or base.get("key_edges") or root.get("key_edges") or []
    )
    raw_edges.extend(base.get("rejected_edges") or [])
    if not raw_edges and root.get("handles"):
        raw_edges = list(root.get("key_edges") or [])

    variables: list[str] = []
    for edge in raw_edges:
        row = _payload(edge)
        for key in ("source", "target", "instrument"):
            if row.get(key) is not None:
                variables.append(str(row[key]))
    raw_candidates = (
        base.get("candidates")
        or root.get("candidate_roles")
        or root.get("role_hypotheses")
        or {}
    )
    if isinstance(raw_candidates, Mapping):
        for values in raw_candidates.values():
            if isinstance(values, (list, tuple, set)):
                variables.extend(str(item) for item in values)

    labels = {
        name: safe_variable_label(name, redact=research_policy.redact_variable_labels)
        for name in dict.fromkeys(variables)
    }
    safe_edges: list[dict[str, Any]] = []
    evidence_grades: dict[str, str] = {}
    findings: list[dict[str, Any]] = []
    for index, item in enumerate(raw_edges):
        raw = _payload(item)
        edge = {
            str(key): redact_context(value)
            for key, value in raw.items()
            if key
            not in (
                "frame",
                "data",
                "rows",
                "records",
                "sample_values",
                "raw_text",
            )
        }
        for key in ("source", "target", "instrument"):
            if raw.get(key) is not None:
                edge[key] = labels.get(str(raw[key]), safe_variable_label(raw[key]))
        finding_id = str(raw.get("finding_id") or _finding_id(raw, index))
        edge["finding_id"] = finding_id
        grade = str(raw.get("evidence_grade") or "unverified")
        evidence_grades[finding_id] = grade
        safe_edges.append(edge)
        findings.append(
            {
                "id": finding_id,
                "kind": "edge",
                "summary": (
                    f"{edge.get('source', 'unknown')} -> "
                    f"{edge.get('target', 'unknown')}"
                ),
                "evidence_grade": grade,
                "method": edge.get("method"),
                "production_eligible": edge.get("production_eligible"),
            }
        )

    candidate_roles: dict[str, list[str]] = {}
    if isinstance(raw_candidates, Mapping):
        key_aliases = {
            "treatment_X": "treatment",
            "outcome_Y": "outcome",
            "instrument_Z": "instrument",
            "confounder_W": "confounder",
            "treatments": "treatment",
            "outcomes": "outcome",
            "instruments": "instrument",
            "confounders": "confounder",
        }
        for key, values in raw_candidates.items():
            role = key_aliases.get(str(key), str(key))
            if not isinstance(values, (list, tuple, set)):
                continue
            candidate_roles[role] = [
                labels.get(str(item), safe_variable_label(item)) for item in values
            ]

    gates = _gate_failures({**root, **base})
    uncertainty = _uncertainties(safe_edges, {**root, **base}, gates)
    run_id = str(
        base.get("run_id")
        or root.get("run_id")
        or (_payload(base.get("manifest")).get("run_id"))
        or f"handoff-{_hash(repr(sorted(evidence_grades.items())))}"
    )
    mode = str(base.get("mode") or root.get("mode") or "exploratory")
    inferred_domain = str(
        domain
        or _payload(base.get("sensitivity") or base.get("sensitivity_report")).get(
            "domain_hint"
        )
        or root.get("domain")
        or "general"
    )

    experiments: list[dict[str, Any]] = []
    for item in (
        list(recommended_experiments or [])
        + list(root.get("experiments_recommended") or [])
        + list(root.get("experiments") or [])
    ):
        experiments.append(redact_context(_payload(item)))

    safe_context = redact_context(dict(context or {}))
    if research_policy.redact_context:
        safe_context = redact_context(safe_context)

    aliases: dict[str, list[str]] = {}
    for original, label in labels.items():
        if label.startswith("private_variable_"):
            aliases[label] = []
        else:
            tokens = [
                token
                for token in re.split(r"[^a-z0-9]+", label.lower())
                if len(token) > 1
            ]
            aliases[label] = list(dict.fromkeys([label, *tokens]))

    return ResearchHandoff(
        run_id=run_id,
        findings=findings,
        edges=safe_edges,
        evidence_grades=evidence_grades,
        gate_failures=gates,
        uncertainty=uncertainty,
        candidate_roles=candidate_roles,
        domain=inferred_domain,
        context=safe_context if isinstance(safe_context, dict) else {},
        variable_labels={label: label for label in dict.fromkeys(labels.values())},
        recommended_experiments=experiments,
        aliases=aliases,
        mode=mode if mode in ("exploratory", "production") else "exploratory",
        source_type=source_type,
        provenance={
            "adapter": "autocausal.research.handoff",
            "source_type": source_type,
            "contains_raw_frame": False,
            "contains_raw_text_columns": False,
            "variable_labels_redacted": research_policy.redact_variable_labels,
            "context_redacted": research_policy.redact_context,
        },
    )


def handoff_from_gate_report(
    report: Any,
    *,
    run_id: str = "",
    domain: str = "general",
    context: Optional[Mapping[str, Any]] = None,
    policy: Optional[ResearchPolicy] = None,
) -> ResearchHandoff:
    payload = _payload(report)
    payload.setdefault("run_id", run_id)
    payload.setdefault("mode", "production")
    return to_research_handoff(payload, domain=domain, context=context, policy=policy)


__all__ = [
    "handoff_from_gate_report",
    "redact_context",
    "safe_variable_label",
    "to_research_handoff",
]
