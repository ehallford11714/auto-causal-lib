"""Lazy, privacy-preserving adapters for AutoCausal report families.

Adapters use class names, schemas, and duck typing instead of importing
optional modules.  This keeps ``autocausal.reporting`` importable while
concurrent/optional AutoML, NLP, visualization, and deep-research packages are
absent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .models import (
    ChartSpec,
    ReportCitation,
    ReportFact,
    ReportPolicy,
    ReportSafetyError,
    ReportSource,
    ReportTable,
    ensure_unique_ids,
)


ASSOCIATION_CAVEAT = (
    "Correlations and associations are descriptive and explicitly non-causal."
)
CAUSAL_CAVEAT = (
    "AutoCausal discovery and estimation outputs require an identification "
    "strategy and human review; exploratory edges are not identified effects."
)
SLM_CAVEAT = (
    "SLM-generated narrative is generative assistance over normalized facts, "
    "not causal identification."
)
NLP_PRIVACY_CAVEAT = (
    "NLP and behavioral summaries can encode sensitive attributes; raw text and "
    "sample-level payloads are omitted."
)
SYNTHETIC_IV_CAVEAT = (
    "Synthetic instruments are demo plumbing with identification=none and are "
    "excluded from production evidence."
)

_SECRET_KEY_RE = re.compile(
    r"(^|_)(password|passwd|secret|token|api_?key|access_?key|private_?key|"
    r"authorization|cookie|credential)($|_)",
    re.I,
)
_PII_KEY_RE = re.compile(
    r"(^|_)(ssn|social_?security|email|e_?mail|phone|mobile|address|"
    r"first_?name|last_?name|full_?name|dob|date_?of_?birth|passport|"
    r"credit_?card|account_?number|ip_?address)($|_)",
    re.I,
)
_RAW_KEY_RE = re.compile(
    r"(^|_)(raw|raw_?data|raw_?text|rows|records|samples?|sample_?values|"
    r"documents?|corpus|frame|dataframe|predictions?|residuals?)($|_)",
    re.I,
)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_LONG_SECRET_RE = re.compile(r"\b(?:sk|pk|ghp|hf|xox[baprs])_[A-Za-z0-9_-]{12,}\b")
_NUMBER = (int, float)


def _slug(value: str, *, limit: int = 64) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return (text or "item")[:limit]


def _class_name(value: Any) -> str:
    return type(value).__name__


def _module_name(value: Any) -> str:
    return str(getattr(type(value), "__module__", ""))


def _is_dataframe_like(value: Any) -> bool:
    return (
        _class_name(value) in {"DataFrame", "Series"}
        and _module_name(value).startswith(("pandas", "polars"))
    )


def _object_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            payload = value.to_dict()
            if isinstance(payload, Mapping):
                return dict(payload)
        except Exception:
            pass
    if is_dataclass(value):
        try:
            return dict(asdict(value))
        except Exception:
            pass
    return {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    return [value]


def _is_blank(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _numeric(value: Any) -> bool:
    return isinstance(value, _NUMBER) and not isinstance(value, bool)


def _coalesce(*values: Any) -> Any:
    """Return the first non-None value while preserving numeric zero."""
    for value in values:
        if value is not None:
            return value
    return None


def _synthetic_instrument(value: Any, payload: Mapping[str, Any] | None = None) -> bool:
    text = str(value or "").lower()
    source = payload or {}
    return bool(
        source.get("synthetic")
        or source.get("auto_instrument")
        or source.get("identification") == "none"
        or str(source.get("instrument_provenance") or "").lower() == "synthetic"
        or text.startswith("auto_instrument")
        or "synthetic" in text
    )


class AdapterContext:
    """Mutable state shared by one normalization pass."""

    def __init__(self, policy: ReportPolicy) -> None:
        self.policy = policy
        self.counters: dict[str, int] = {}
        self.warnings: list[str] = []
        self.unsafe_findings: list[str] = []
        self.seen_objects: set[int] = set()

    def next_source_id(self, family: str, payload: Mapping[str, Any]) -> str:
        self.counters[family] = self.counters.get(family, 0) + 1
        stable_hint = (
            payload.get("run_id")
            or payload.get("id")
            or payload.get("schema")
            or payload.get("source")
            or ""
        )
        digest = hashlib.sha256(
            f"{family}:{stable_hint}:{self.counters[family]}".encode("utf-8")
        ).hexdigest()[:8]
        return f"{_slug(family)}-{self.counters[family]:02d}-{digest}"

    def note_redaction(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def note_unsafe(self, message: str) -> None:
        if message not in self.unsafe_findings:
            self.unsafe_findings.append(message)


def _safe_key(key: Any, context: AdapterContext) -> str:
    text = str(key)
    if context.policy.redact_secrets and _SECRET_KEY_RE.search(text):
        context.note_redaction(f"Secret-like key `{text}` was redacted.")
        return "[REDACTED_SECRET_KEY]"
    if context.policy.redact_pii and _PII_KEY_RE.search(text):
        context.note_redaction(f"PII-like key `{text}` was redacted.")
        return "[REDACTED_PII_KEY]"
    return text


def _safe_string(value: str, context: AdapterContext, *, limit: int = 500) -> str:
    text = str(value)
    original = text
    if context.policy.redact_pii:
        text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
        text = _SSN_RE.sub("[REDACTED_SSN]", text)
    if context.policy.redact_secrets:
        text = _LONG_SECRET_RE.sub("[REDACTED_SECRET]", text)
    if text != original:
        context.note_redaction("Sensitive string patterns were redacted.")
    if len(text) > limit:
        context.note_redaction(
            f"A text field was truncated to {limit} characters for reporting."
        )
        text = text[: max(0, limit - 3)] + "..."
    return text


def _safe_value(
    value: Any,
    context: AdapterContext,
    *,
    key: str = "",
    depth: int = 0,
) -> Any:
    """Bound and redact values retained in normalized evidence."""
    if _is_dataframe_like(value):
        context.note_unsafe(f"Raw {_class_name(value)} payload was omitted.")
        return "[OMITTED_RAW_FRAME]"
    if context.policy.redact_secrets and _SECRET_KEY_RE.search(key):
        context.note_redaction(f"Secret-like field `{key}` was redacted.")
        return "[REDACTED_SECRET]"
    if context.policy.redact_pii and _PII_KEY_RE.search(key):
        context.note_redaction(f"PII-like field `{key}` was redacted.")
        return "[REDACTED_PII]"
    if context.policy.raw_data_prohibited and _RAW_KEY_RE.search(key):
        context.note_unsafe(f"Raw/sample field `{key}` was omitted.")
        return "[OMITTED_RAW_PAYLOAD]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _safe_string(value, context)
    if depth >= 4:
        return f"[{_class_name(value)} omitted at depth limit]"
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for index, (child_key, child_value) in enumerate(
            sorted(value.items(), key=lambda pair: str(pair[0]))
        ):
            if index >= 80:
                out["__truncated__"] = True
                break
            safe_key = _safe_key(child_key, context)
            out[safe_key] = _safe_value(
                child_value,
                context,
                key=str(child_key),
                depth=depth + 1,
            )
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        result = [
            _safe_value(item, context, key=key, depth=depth + 1)
            for item in values[:80]
        ]
        if len(values) > 80:
            result.append(f"[{len(values) - 80} additional items omitted]")
        return result
    if hasattr(value, "to_dict"):
        return _safe_value(_object_mapping(value), context, key=key, depth=depth + 1)
    return _safe_string(str(value), context)


class SourceBuilder:
    """Convenience builder that guarantees source/fact provenance ids."""

    def __init__(
        self,
        context: AdapterContext,
        *,
        family: str,
        title: str,
        payload: Mapping[str, Any],
    ) -> None:
        self.context = context
        self.family = family
        self.payload = dict(payload)
        self.id = context.next_source_id(family, payload)
        self.title = title
        self.facts: list[ReportFact] = []
        self.tables: list[ReportTable] = []
        self.charts: list[ChartSpec] = []
        self.citations: list[ReportCitation] = []
        self.caveats: list[str] = []
        self.warnings: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.contains_raw_data = False
        self._fact_keys: dict[str, int] = {}
        self._table_keys: dict[str, int] = {}
        self._chart_keys: dict[str, int] = {}

    def _unique(self, bucket: dict[str, int], key: str) -> str:
        slug = _slug(key)
        bucket[slug] = bucket.get(slug, 0) + 1
        suffix = f"-{bucket[slug]}" if bucket[slug] > 1 else ""
        return slug + suffix

    def fact(
        self,
        key: str,
        label: str,
        value: Any,
        category: str,
        *,
        unit: str = "",
        priority: int = 50,
        citation_ids: Sequence[str] = (),
        caveat: str = "",
        evidence_eligible: bool = True,
        attributes: Mapping[str, Any] | None = None,
        provenance_id: str = "",
    ) -> ReportFact | None:
        if _is_blank(value):
            return None
        safe = _safe_value(value, self.context, key=key)
        fact_key = self._unique(self._fact_keys, key)
        fact = ReportFact(
            id=f"{self.id}:fact:{fact_key}",
            source_id=self.id,
            provenance_id=provenance_id or f"{self.id}:{fact_key}",
            label=_safe_string(label, self.context, limit=180),
            value=safe,
            category=category,
            unit=unit,
            priority=priority,
            citation_ids=list(citation_ids),
            caveat=caveat,
            sensitive=(
                "[REDACTED" in str(safe)
                or "[OMITTED_RAW" in str(safe)
            ),
            evidence_eligible=evidence_eligible,
            attributes=_safe_value(
                dict(attributes or {}), self.context, key="attributes"
            ),
        )
        self.facts.append(fact)
        return fact

    def table(
        self,
        key: str,
        title: str,
        columns: Sequence[str],
        rows: Iterable[Mapping[str, Any]],
        category: str,
        *,
        provenance_ids: Sequence[str] = (),
        footnote: str = "",
    ) -> ReportTable:
        table_key = self._unique(self._table_keys, key)
        safe_columns = [_safe_key(column, self.context) for column in columns]
        safe_rows: list[dict[str, Any]] = []
        source_rows = list(rows)
        limit = self.context.policy.max_rows_per_table
        for row in source_rows[:limit]:
            normalized: dict[str, Any] = {}
            for original_column, safe_column in zip(columns, safe_columns):
                normalized[safe_column] = _safe_value(
                    row.get(original_column),
                    self.context,
                    key=str(original_column),
                )
            safe_rows.append(normalized)
        if len(source_rows) > limit:
            self.warnings.append(
                f"Table `{title}` limited to {limit} of {len(source_rows)} rows."
            )
            if footnote:
                footnote += " "
            footnote += f"Showing {limit} of {len(source_rows)} rows by policy."
        table = ReportTable(
            id=f"{self.id}:table:{table_key}",
            title=_safe_string(title, self.context, limit=180),
            columns=safe_columns,
            rows=safe_rows,
            provenance_ids=list(provenance_ids) or [f"{self.id}:{table_key}"],
            footnote=footnote,
            category=category,
        )
        self.tables.append(table)
        return table

    def chart(
        self,
        key: str,
        *,
        chart_type: str,
        title: str,
        alt_text: str,
        category: str = "visualizations",
        source_fact_ids: Sequence[str] = (),
        source_table_id: str = "",
        image_path: str = "",
        spec: Mapping[str, Any] | None = None,
        provenance_ids: Sequence[str] = (),
        priority: int = 50,
        caption: str = "",
        runtime_artifact: Any = None,
    ) -> ChartSpec:
        chart_key = self._unique(self._chart_keys, key)
        chart = ChartSpec(
            id=f"{self.id}:chart:{chart_key}",
            chart_type=_safe_string(chart_type, self.context, limit=80),
            title=_safe_string(title, self.context, limit=180),
            alt_text=_safe_string(alt_text, self.context, limit=300),
            source_fact_ids=list(source_fact_ids),
            source_table_id=source_table_id,
            image_path=_safe_string(image_path, self.context, limit=500)
            if image_path
            else "",
            spec=_safe_value(dict(spec or {}), self.context, key="chart_spec"),
            provenance_ids=list(provenance_ids) or [f"{self.id}:{chart_key}"],
            priority=priority,
            caption=_safe_string(caption, self.context, limit=400)
            if caption
            else "",
            category=category,
            runtime_artifact=runtime_artifact,
        )
        self.charts.append(chart)
        return chart

    def citation(self, value: Any) -> ReportCitation | None:
        payload = _object_mapping(value)
        citation_id = (
            payload.get("id")
            or payload.get("source_id")
            or payload.get("record_id")
            or payload.get("citation_id")
        )
        if not citation_id:
            self.metadata["unsupported_citations"] = int(
                self.metadata.get("unsupported_citations") or 0
            ) + 1
            self.warnings.append(
                "A citation/reference without a SourceRecord id was excluded."
            )
            return None
        verification = payload.get("verification")
        verification_status = str(
            payload.get("verification_status")
            or (
                verification.get("status")
                if isinstance(verification, Mapping)
                else verification or ""
            )
            or ""
        )
        source_record_backed = bool(
            payload.get("source_id")
            and payload.get("stable_id")
            and payload.get("provider")
            and payload.get("title")
        )
        verified = bool(payload.get("verified")) or verification_status.lower() in {
            "verified",
            "valid",
            "passed",
            "ok",
        } or source_record_backed
        if source_record_backed and not verification_status:
            verification_status = "source_record"
        authors_raw = payload.get("authors") or payload.get("author") or []
        authors = [
            _safe_string(str(author), self.context, limit=120)
            for author in _as_list(authors_raw)
        ]
        citation = ReportCitation(
            id=_safe_string(str(citation_id), self.context, limit=160),
            title=_safe_string(
                str(payload.get("title") or payload.get("name") or ""),
                self.context,
                limit=300,
            ),
            url=_safe_string(
                str(
                    payload.get("url")
                    or payload.get("doi_url")
                    or payload.get("source_url")
                    or ""
                ),
                self.context,
                limit=500,
            ),
            authors=authors,
            year=str(
                payload.get("year")
                or payload.get("published_year")
                or payload.get("date")
                or ""
            ),
            verified=verified,
            verification_status=verification_status,
            supplied_by=str(payload.get("supplied_by") or "source_record"),
            metadata={
                key: _safe_value(payload.get(key), self.context, key=key)
                for key in (
                    "doi",
                    "provider",
                    "retrieved_at",
                    "retrieval_timestamp",
                    "source_type",
                    "availability",
                )
                if payload.get(key) is not None
            },
        )
        self.citations.append(citation)
        return citation

    def finish(self) -> ReportSource:
        common_meta = {}
        for key in (
            "schema",
            "run_id",
            "mode",
            "source",
            "backend",
            "produced_by",
            "produced_at",
            "package_version",
            "status",
            "profile",
        ):
            if self.payload.get(key) is not None:
                common_meta[key] = _safe_value(
                    self.payload[key], self.context, key=key
                )
        common_meta.update(self.metadata)
        return ReportSource(
            id=self.id,
            family=self.family,
            title=self.title,
            facts=self.facts,
            tables=self.tables,
            charts=self.charts,
            citations=self.citations,
            caveats=list(dict.fromkeys(self.caveats)),
            metadata=common_meta,
            warnings=list(dict.fromkeys(self.warnings)),
            contains_raw_data=self.contains_raw_data,
        )


class ReportSourceAdapter(Protocol):
    """Protocol for lazy report-source adapters."""

    def matches(self, value: Any, family_hint: str | None = None) -> bool: ...

    def adapt(
        self,
        value: Any,
        context: AdapterContext,
        family_hint: str | None = None,
    ) -> list[ReportSource]: ...


class AdapterRegistry:
    """Ordered source adapter registry with no optional imports."""

    def __init__(self, adapters: Sequence[ReportSourceAdapter] | None = None) -> None:
        self.adapters = list(adapters or [])

    def register(self, adapter: ReportSourceAdapter) -> None:
        self.adapters.append(adapter)

    def adapt(
        self,
        value: Any,
        context: AdapterContext,
        family_hint: str | None = None,
    ) -> list[ReportSource]:
        for adapter in self.adapters:
            if adapter.matches(value, family_hint):
                return adapter.adapt(value, context, family_hint)
        raise TypeError(
            f"No reporting adapter for {_module_name(value)}.{_class_name(value)}"
        )


def _family_for(value: Any, payload: Mapping[str, Any]) -> str:
    class_name = _class_name(value).lower()
    schema = str(payload.get("schema") or "").lower()
    produced_by = str(payload.get("produced_by") or "").lower()

    exact = {
        "discoveryresult": "discovery",
        "autoresult": "auto_result",
        "miningreport": "mining",
        "minereport": "mine",
        "qcreport": "qc",
        "cleansereport": "cleanse",
        "edareport": "eda",
        "imputationreport": "cleanse",
        "estimateresult": "estimate",
        "causalinferenceresult": "estimate",
        "causalspec": "estimate",
        "refuteresult": "refute",
        "sensitivityreport": "sensitivity",
        "gatereport": "gate",
        "runmanifest": "manifest",
        "insightreport": "insight",
        "agenticloopreport": "agentic",
        "slmchainreport": "agentic",
        "grailreport": "grail",
        "fitreport": "automl",
        "kpiloopresult": "automl",
        "modelconstructplan": "automl",
        "behavioralreport": "nlp_behavioral",
        "behavioralmineresult": "nlp_behavioral",
        "textcausalhints": "nlp",
        "sentimentresult": "nlp",
        "autovizreport": "autoviz",
        "vizplan": "autoviz",
        "renderedchart": "autochart",
        "deepresearchreport": "deep_research",
        "researchreport": "deep_research",
        "sourcerecord": "source_record",
        "publiccausalreport": "public_causal",
        "validationreport": "validation",
        "groundingreport": "insight",
    }
    if class_name in exact:
        return exact[class_name]
    if "deepresearch" in class_name or "deep_research" in schema:
        return "deep_research"
    if "autochart" in class_name or "chartartifact" in class_name:
        return "autochart"
    if "automl" in class_name or "automl" in schema or "autocausal.ml" in produced_by:
        return "automl"
    if "autonlp" in class_name or "nlp" in schema:
        return "nlp"
    if "autoviz" in class_name or "viz" in schema:
        return "autoviz"
    if "manifest" in schema:
        return "manifest"
    if "gate" in schema:
        return "gate"
    if "sensitivity" in schema:
        return "sensitivity"
    if "fitreport" in schema:
        return "automl"
    if "discovery" in schema or {"edges", "graph", "candidates"} <= set(payload):
        return "discovery"
    if "associations" in payload and "columns" in payload:
        return "mining"
    if "citations" in payload or "source_records" in payload:
        return "deep_research"
    return "generic"


def _source_title(family: str) -> str:
    return {
        "autocausal": "AutoCausal session",
        "auto_result": "AutoCausal pipeline result",
        "discovery": "Causal discovery result",
        "mining": "Mining report",
        "mine": "AutoMine report",
        "qc": "Quality-control report",
        "cleanse": "Data cleansing report",
        "eda": "Exploratory analysis report",
        "estimate": "Causal estimate",
        "refute": "Refutation result",
        "sensitivity": "Sensitivity report",
        "gate": "Production gate report",
        "manifest": "Run manifest",
        "insight": "Insight report",
        "agentic": "Agentic report",
        "grail": "GRAIL report",
        "automl": "AutoML report",
        "nlp": "NLP report",
        "nlp_behavioral": "Behavioral findings",
        "autoviz": "AutoViz plan",
        "autochart": "AutoChart artifact",
        "deep_research": "Deep research report",
        "source_record": "Reference source record",
        "public_causal": "Public causal report",
        "validation": "Validation report",
        "generic": "Structured report artifact",
    }.get(family, family.replace("_", " ").title())


def _common_notes(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    for key in ("warnings", "notes", "caveats"):
        for note in _as_list(payload.get(key)):
            if isinstance(note, Mapping):
                note = note.get("message") or note.get("detail") or json.dumps(
                    dict(note), sort_keys=True, default=str
                )
            if note:
                safe = _safe_string(str(note), builder.context, limit=500)
                if key == "caveats":
                    builder.caveats.append(safe)
                else:
                    builder.warnings.append(safe)


def _add_shape(
    builder: SourceBuilder,
    payload: Mapping[str, Any],
    category: str = "scope_provenance",
) -> None:
    rows = _coalesce(
        payload.get("n_rows"),
        payload.get("n_rows_out"),
        payload.get("row_count"),
    )
    cols = _coalesce(
        payload.get("n_cols"),
        payload.get("n_cols_out"),
        payload.get("n_columns"),
        payload.get("column_count"),
    )
    builder.fact("row_count", "Rows", rows, category, priority=65)
    builder.fact("column_count", "Columns", cols, category, priority=65)


def _adapt_autocausal(value: Any, context: AdapterContext) -> ReportSource:
    payload: dict[str, Any] = {
        "source": getattr(value, "source", ""),
        "mode": getattr(value, "mode", "exploratory"),
        "run_id": getattr(value, "run_id", ""),
    }
    frame = getattr(value, "_df", None)
    if frame is None:
        try:
            frame = getattr(value, "df", None)
        except Exception:
            frame = None
    if frame is not None:
        try:
            payload["n_rows"] = int(len(frame))
            payload["n_cols"] = int(len(frame.columns))
        except Exception:
            pass
    policy = getattr(value, "policy", None)
    if policy is not None and hasattr(policy, "to_dict"):
        try:
            policy = policy.to_dict()
        except Exception:
            policy = None
    if isinstance(policy, Mapping):
        payload["profile"] = policy.get("profile")
        payload["run_policy"] = {
            key: policy.get(key)
            for key in (
                "profile",
                "qc",
                "stability",
                "bootstrap_n",
                "ensemble",
                "require_observed_instrument",
                "required_evidence",
            )
            if key in policy
        }
    builder = SourceBuilder(
        context,
        family="autocausal",
        title="AutoCausal session",
        payload=payload,
    )
    builder.fact("source", "Source", payload.get("source"), "scope_provenance", priority=80)
    builder.fact("mode", "Run mode", payload.get("mode"), "scope_provenance", priority=90)
    builder.fact("run_id", "Run id", payload.get("run_id"), "scope_provenance", priority=85)
    _add_shape(builder, payload)
    builder.fact(
        "run_policy",
        "Run policy",
        payload.get("run_policy"),
        "scope_provenance",
        priority=80,
    )
    builder.caveats.append(CAUSAL_CAVEAT)
    return builder.finish()


def _adapt_auto_result(
    builder: SourceBuilder, payload: Mapping[str, Any]
) -> None:
    builder.fact("source", "Source", payload.get("source"), "scope_provenance", priority=75)
    builder.fact(
        "pipeline_notes_count",
        "Pipeline notes",
        len(_as_list(payload.get("notes"))),
        "technical_appendix",
    )
    builder.fact(
        "public_joins",
        "Public joins",
        len(_as_list(payload.get("join_log"))),
        "scope_provenance",
    )
    ping = payload.get("ping")
    if isinstance(ping, Mapping):
        builder.fact("source_ping_ok", "Source connectivity", ping.get("ok"), "scope_provenance")
    builder.caveats.append(CAUSAL_CAVEAT)


def _edge_rows_and_facts(
    builder: SourceBuilder,
    edges: Sequence[Any],
    *,
    rejected: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    discovery_rows: list[dict[str, Any]] = []
    observed_iv_rows: list[dict[str, Any]] = []
    synthetic_iv_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(edges):
        edge = _object_mapping(raw)
        source = edge.get("source")
        target = edge.get("target")
        if not source and not target:
            continue
        instrument = edge.get("instrument")
        is_iv = str(edge.get("type") or "").lower() in {
            "iv",
            "iv_2sls",
            "2sls",
        } or instrument is not None
        synthetic = is_iv and _synthetic_instrument(instrument, edge)
        category = "iv_evidence" if is_iv else "discovery"
        evidence_eligible = not rejected and not synthetic
        status = (
            "REJECTED"
            if rejected
            else ("EXCLUDED_SYNTHETIC" if synthetic else "reported")
        )
        value = {
            "source": source,
            "target": target,
            "type": edge.get("type") or "association",
            "score": edge.get("score"),
            "confidence": edge.get("confidence"),
            "pvalue": _coalesce(edge.get("pvalue"), edge.get("p_value")),
            "stability": edge.get("stability"),
            "evidence_grade": edge.get("evidence_grade"),
            "identification": "none" if synthetic else edge.get("identification"),
            "instrument": instrument,
            "status": status,
        }
        fact = builder.fact(
            f"{'rejected-' if rejected else ''}edge-{index + 1}",
            f"{'Rejected ' if rejected else ''}{'IV ' if is_iv else ''}edge "
            f"{source} → {target}",
            value,
            category,
            priority=90 if rejected or synthetic else 70,
            caveat=SYNTHETIC_IV_CAVEAT if synthetic else CAUSAL_CAVEAT,
            evidence_eligible=evidence_eligible,
            attributes={
                "edge_pair": [source, target],
                "synthetic_iv": synthetic,
                "rejected": rejected,
                "failed_gates": edge.get("failed_gates") or [],
            },
            provenance_id=str(
                (edge.get("provenance") or {}).get("id")
                if isinstance(edge.get("provenance"), Mapping)
                else ""
            )
            or f"{builder.id}:edge:{index + 1}",
        )
        row = {
            "fact_id": fact.id if fact else "",
            "source": source,
            "target": target,
            "type": edge.get("type") or "",
            "instrument": instrument or "",
            "score": edge.get("score"),
            "confidence": edge.get("confidence"),
            "p_value": _coalesce(edge.get("pvalue"), edge.get("p_value")),
            "stability": edge.get("stability"),
            "methods": _coalesce(
                edge.get("n_methods"), len(_as_list(edge.get("methods")))
            ),
            "evidence": edge.get("evidence_grade") or "exploratory",
            "identification": "none" if synthetic else edge.get("identification") or "unverified",
            "status": status,
        }
        if synthetic:
            synthetic_iv_rows.append(row)
        elif is_iv:
            observed_iv_rows.append(row)
        else:
            discovery_rows.append(row)
    return discovery_rows, observed_iv_rows, synthetic_iv_rows


def _adapt_discovery(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    builder.fact("method", "Discovery method", payload.get("method"), "discovery", priority=75)
    builder.fact("mode", "Run mode", payload.get("mode"), "scope_provenance", priority=85)
    builder.fact("run_id", "Run id", payload.get("run_id"), "scope_provenance", priority=65)
    builder.fact(
        "stability_enabled",
        "Bootstrap stability enabled",
        payload.get("stability_enabled"),
        "discovery",
        priority=80,
    )
    builder.fact(
        "bootstrap_n",
        "Bootstrap repetitions",
        payload.get("bootstrap_n"),
        "discovery",
        priority=70,
    )
    builder.fact(
        "ensemble_methods",
        "Discovery ensemble methods",
        payload.get("ensemble_methods"),
        "discovery",
        priority=70,
    )
    roles = payload.get("roles")
    if isinstance(roles, Mapping):
        builder.fact("roles", "Column role hypotheses", roles, "causal_estimates", priority=65)
    candidates = payload.get("candidates")
    if isinstance(candidates, Mapping):
        builder.fact(
            "candidates",
            "Causal role candidates",
            candidates,
            "causal_estimates",
            priority=75,
            caveat="Role candidates are hypotheses, not validated causal assignments.",
        )

    edges = _as_list(payload.get("edges"))
    rejected = _as_list(payload.get("rejected_edges"))
    accepted_rows, observed_iv, synthetic_iv = _edge_rows_and_facts(builder, edges)
    rejected_rows, rejected_observed, rejected_synthetic = _edge_rows_and_facts(
        builder, rejected, rejected=True
    )
    builder.fact("accepted_edges", "Accepted discovery edges", len(edges), "discovery", priority=85)
    builder.fact(
        "rejected_edges",
        "Rejected discovery edges",
        len(rejected),
        "refutations_sensitivity",
        priority=95 if rejected else 60,
    )
    if accepted_rows:
        builder.table(
            "discovery_edges",
            "Discovery edges",
            (
                "fact_id",
                "source",
                "target",
                "type",
                "score",
                "confidence",
                "p_value",
                "stability",
                "methods",
                "evidence",
                "status",
            ),
            accepted_rows,
            "discovery",
            footnote=CAUSAL_CAVEAT,
        )
    if observed_iv or rejected_observed:
        builder.table(
            "observed_iv_edges",
            "Observed-instrument IV edges",
            (
                "fact_id",
                "source",
                "target",
                "instrument",
                "score",
                "confidence",
                "evidence",
                "identification",
                "status",
            ),
            observed_iv + rejected_observed,
            "iv_evidence",
            footnote=(
                "Observed Z does not by itself establish relevance, exclusion, or "
                "causal identification."
            ),
        )
    if synthetic_iv or rejected_synthetic:
        builder.table(
            "synthetic_iv_edges",
            "Synthetic IV audit ledger (excluded from evidence)",
            (
                "fact_id",
                "source",
                "target",
                "instrument",
                "score",
                "confidence",
                "evidence",
                "identification",
                "status",
            ),
            synthetic_iv + rejected_synthetic,
            "iv_evidence",
            footnote=SYNTHETIC_IV_CAVEAT,
        )
        builder.caveats.append(SYNTHETIC_IV_CAVEAT)
    if rejected_rows:
        builder.table(
            "rejected_edges",
            "Edges rejected by evidence gates",
            (
                "fact_id",
                "source",
                "target",
                "type",
                "score",
                "stability",
                "evidence",
                "status",
            ),
            rejected_rows,
            "refutations_sensitivity",
            footnote="Retained for audit only; not production-eligible evidence.",
        )

    gates = _as_list(payload.get("evidence_gates"))
    gate_rows: list[dict[str, Any]] = []
    failed = 0
    for gate in gates:
        row = _object_mapping(gate)
        ok = row.get("ok")
        status = row.get("status") or ("pass" if ok else "fail")
        if status in ("fail", "escalate") or ok is False:
            failed += 1
        gate_rows.append(
            {
                "id": row.get("id"),
                "status": status,
                "detail": row.get("detail") or row.get("message"),
                "metric": row.get("metric"),
                "threshold": row.get("threshold"),
                "recommendation": row.get("recommendation") or row.get("remediation"),
            }
        )
    if gates:
        builder.fact(
            "failed_evidence_gates",
            "Failed or escalated evidence gates",
            failed,
            "refutations_sensitivity",
            priority=100 if failed else 70,
        )
        builder.table(
            "evidence_gates",
            "Evidence gates",
            ("id", "status", "detail", "metric", "threshold", "recommendation"),
            gate_rows,
            "refutations_sensitivity",
        )

    method_edges = payload.get("method_edges")
    if isinstance(method_edges, Mapping):
        agreement_rows = [
            {"method": method, "edge_count": len(_as_list(method_value))}
            for method, method_value in method_edges.items()
        ]
        builder.table(
            "method_agreement",
            "Discovery method coverage",
            ("method", "edge_count"),
            agreement_rows,
            "discovery",
        )
    builder.caveats.append(CAUSAL_CAVEAT)


def _adapt_mining(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    _add_shape(builder, payload, "eda_readiness")
    columns = _as_list(payload.get("columns"))
    associations = _as_list(payload.get("associations"))
    suggestions = _as_list(payload.get("suggestions"))
    builder.fact(
        "profiled_columns",
        "Profiled columns",
        len(columns) or payload.get("n_cols"),
        "eda_readiness",
        priority=65,
    )
    builder.fact(
        "association_count",
        "Reported associations",
        len(associations),
        "associations",
        priority=75,
        caveat=ASSOCIATION_CAVEAT,
    )
    builder.fact(
        "kpis",
        "Suggested KPIs",
        payload.get("kpis"),
        "eda_readiness",
        caveat="KPI labels are heuristic and not causal outcomes by default.",
    )
    column_rows = []
    for item in columns:
        row = _object_mapping(item)
        column_rows.append(
            {
                "column": row.get("name") or row.get("column"),
                "dtype": row.get("dtype"),
                "role": row.get("role"),
                "missing_pct": _coalesce(
                    row.get("null_pct"), row.get("missing_pct")
                ),
                "unique": _coalesce(
                    row.get("nunique"), row.get("cardinality")
                ),
                "skew": row.get("skew"),
            }
        )
        if row.get("top_values") or row.get("sample_values"):
            builder.warnings.append(
                "Sample/top values from mining profiles were omitted by report policy."
            )
    if column_rows:
        builder.table(
            "column_profiles",
            "Column profiles (sample values omitted)",
            ("column", "dtype", "role", "missing_pct", "unique", "skew"),
            column_rows,
            "eda_readiness",
        )
    association_rows = []
    for index, item in enumerate(associations):
        row = _object_mapping(item)
        fact = builder.fact(
            f"association-{index + 1}",
            f"Association {row.get('a')} ↔ {row.get('b')}",
            {
                "a": row.get("a"),
                "b": row.get("b"),
                "metric": row.get("metric"),
                "score": row.get("score"),
            },
            "associations",
            priority=70,
            caveat=ASSOCIATION_CAVEAT,
            attributes={"association_pair": [row.get("a"), row.get("b")]},
        )
        association_rows.append(
            {
                "fact_id": fact.id if fact else "",
                "a": row.get("a"),
                "b": row.get("b"),
                "metric": row.get("metric"),
                "score": row.get("score"),
            }
        )
    if association_rows:
        builder.table(
            "associations",
            "Top associations",
            ("fact_id", "a", "b", "metric", "score"),
            association_rows,
            "associations",
            footnote=ASSOCIATION_CAVEAT,
        )
    suggestion_rows = []
    for item in suggestions:
        row = _object_mapping(item)
        suggestion_rows.append(
            {
                "source": row.get("source"),
                "target": row.get("target"),
                "score": row.get("score"),
                "reason": row.get("reason"),
            }
        )
    if suggestion_rows:
        builder.table(
            "relationship_suggestions",
            "Suggested relationships (hypotheses)",
            ("source", "target", "score", "reason"),
            suggestion_rows,
            "associations",
            footnote=ASSOCIATION_CAVEAT,
        )
    builder.caveats.append(ASSOCIATION_CAVEAT)


def _adapt_qc(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    _add_shape(builder, payload, "data_quality")
    builder.fact("qc_ok", "QC passed", payload.get("ok"), "data_quality", priority=90)
    builder.fact(
        "qc_blocked", "QC blocked analysis", payload.get("blocked"), "data_quality", priority=100
    )
    issues = _as_list(payload.get("issues"))
    builder.fact(
        "issue_count", "QC issues", len(issues), "data_quality", priority=85
    )
    rows = []
    for issue in issues:
        row = _object_mapping(issue)
        rows.append(
            {
                "code": row.get("code") or row.get("id"),
                "severity": row.get("severity") or row.get("status"),
                "message": row.get("message") or row.get("detail"),
                "columns": row.get("columns"),
            }
        )
    if rows:
        builder.table(
            "qc_issues",
            "Quality-control issues",
            ("code", "severity", "message", "columns"),
            rows,
            "data_quality",
        )


def _adapt_cleanse(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    for key, label in (
        ("n_rows_in", "Input rows"),
        ("n_rows_out", "Output rows"),
        ("n_cols_in", "Input columns"),
        ("n_cols_out", "Output columns"),
    ):
        builder.fact(key, label, payload.get(key), "data_quality", priority=70)
    operations = _as_list(payload.get("operations"))
    builder.fact(
        "operation_count",
        "Cleanse operations",
        len(operations),
        "data_quality",
        priority=80,
    )
    rows = []
    for operation in operations:
        row = _object_mapping(operation)
        rows.append(
            {
                "operation": row.get("op") or row.get("operation"),
                "detail": row.get("detail"),
                "columns": row.get("columns"),
                "affected": row.get("n_affected"),
            }
        )
    if rows:
        builder.table(
            "cleanse_ledger",
            "AutoCleanse operation ledger",
            ("operation", "detail", "columns", "affected"),
            rows,
            "data_quality",
        )
    builder.fact(
        "dropped_columns",
        "Dropped columns",
        payload.get("dropped_columns"),
        "data_quality",
        caveat="Review all destructive transformations before causal analysis.",
    )
    imputation = payload.get("imputation")
    if isinstance(imputation, Mapping):
        builder.fact(
            "imputation",
            "Imputation summary",
            {
                key: imputation.get(key)
                for key in (
                    "method",
                    "total_missing_before",
                    "total_missing_after",
                )
                if key in imputation
            },
            "data_quality",
            priority=80,
        )


def _adapt_eda(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    _add_shape(builder, payload, "eda_readiness")
    builder.fact(
        "readiness_score",
        "Causal-readiness score",
        payload.get("readiness_score"),
        "eda_readiness",
        priority=90,
        caveat="Readiness is a heuristic screening score, not identification.",
    )
    roles = payload.get("roles")
    if roles:
        builder.fact(
            "role_proposals",
            "Role proposals",
            _object_mapping(roles) or roles,
            "eda_readiness",
            priority=80,
            caveat="Proposed roles are hypotheses, not ground truth.",
        )
    missingness = payload.get("missingness")
    cardinality = payload.get("cardinality")
    if isinstance(missingness, Mapping):
        rows = [
            {
                "column": column,
                "missing_fraction": fraction,
                "cardinality": (
                    cardinality.get(column)
                    if isinstance(cardinality, Mapping)
                    else None
                ),
            }
            for column, fraction in sorted(
                missingness.items(),
                key=lambda pair: float(pair[1] or 0),
                reverse=True,
            )
        ]
        builder.table(
            "missingness",
            "Missingness profile",
            ("column", "missing_fraction", "cardinality"),
            rows,
            "eda_readiness",
        )
    numeric_summary = payload.get("numeric_summary")
    if isinstance(numeric_summary, Mapping):
        rows = []
        for column, summary in numeric_summary.items():
            row = _object_mapping(summary)
            rows.append(
                {
                    "column": column,
                    "count": row.get("count"),
                    "mean": row.get("mean"),
                    "std": row.get("std"),
                    "min": row.get("min"),
                    "median": _coalesce(row.get("50%"), row.get("median")),
                    "max": row.get("max"),
                }
            )
        builder.table(
            "numeric_summary",
            "Numeric summaries",
            ("column", "count", "mean", "std", "min", "median", "max"),
            rows,
            "eda_readiness",
        )
    correlations = payload.get("correlations")
    if isinstance(correlations, Mapping):
        rows = []
        seen: set[tuple[str, str]] = set()
        for left, values in correlations.items():
            if not isinstance(values, Mapping):
                continue
            for right, score in values.items():
                pair = tuple(sorted((str(left), str(right))))
                if left == right or pair in seen:
                    continue
                seen.add(pair)
                rows.append({"a": left, "b": right, "correlation": score})
        rows.sort(
            key=lambda row: abs(float(row.get("correlation") or 0)),
            reverse=True,
        )
        builder.table(
            "correlations",
            "Pairwise correlations",
            ("a", "b", "correlation"),
            rows,
            "associations",
            footnote=ASSOCIATION_CAVEAT,
        )
        builder.caveats.append(ASSOCIATION_CAVEAT)
    for key, label in (
        ("warnings", "EDA warnings"),
        ("suggestions", "EDA suggestions"),
        ("leakage_hints", "Leakage hints"),
    ):
        builder.fact(
            key,
            label,
            payload.get(key),
            "eda_readiness",
            priority=90 if key == "leakage_hints" else 65,
        )


def _flatten_metric_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for container in ("estimate", "metrics", "data", "payload"):
        value = payload.get(container)
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(item, (str, int, float, bool)) or item is None:
                    result[str(key)] = item
    for key, item in payload.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            result.setdefault(str(key), item)
    return result


def _adapt_estimate(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    metrics = _flatten_metric_payload(payload)
    z = (
        metrics.get("z")
        or metrics.get("instrument")
        or payload.get("instrument")
    )
    instrument_context = dict(metrics)
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        instrument_context.update(
            {
                key: provenance.get(key)
                for key in (
                    "synthetic",
                    "auto_instrument",
                    "identification",
                    "instrument_provenance",
                )
                if key in provenance
            }
        )
    synthetic = _synthetic_instrument(z, instrument_context)
    category = "iv_evidence" if z or "2sls" in str(metrics.get("method") or "").lower() else "causal_estimates"
    evidence_eligible = not synthetic
    for key, label, priority in (
        ("ok", "Estimate completed", 85),
        ("method", "Estimator method", 75),
        ("backend", "Estimator backend", 75),
        ("estimand", "Causal estimand", 85),
        ("evidence_grade", "Evidence grade", 90),
        ("outcome", "Outcome", 65),
        ("y", "Outcome", 65),
        ("treatment", "Treatment", 65),
        ("d", "Treatment", 65),
        ("instrument", "Instrument", 85),
        ("z", "Instrument", 85),
        ("estimate", "Point estimate", 95),
        ("effect", "Effect estimate", 95),
        ("ate", "Average treatment effect", 95),
        ("coef", "Coefficient", 95),
        ("standard_error", "Standard error", 80),
        ("std_error", "Standard error", 80),
        ("stderr", "Standard error", 80),
        ("se", "Standard error", 80),
        ("ci_low", "Confidence interval lower", 85),
        ("ci_high", "Confidence interval upper", 85),
        ("ci_lower", "Confidence interval lower", 85),
        ("ci_upper", "Confidence interval upper", 85),
        ("pvalue", "P-value", 80),
        ("p_value", "P-value", 80),
        ("first_stage_f", "First-stage F statistic", 95),
        ("n", "Analysis observations", 70),
        ("n_obs", "Analysis observations", 70),
    ):
        if key in metrics:
            builder.fact(
                f"estimate_{key}",
                label,
                metrics.get(key),
                category,
                priority=priority,
                caveat=SYNTHETIC_IV_CAVEAT if synthetic else CAUSAL_CAVEAT,
                evidence_eligible=evidence_eligible,
                attributes={"synthetic_iv": synthetic, "metric": key},
            )
    for key, label, priority in (
        ("controls", "Adjustment controls", 80),
        ("confounders", "Specified confounders", 80),
        ("assumptions", "Identification assumptions", 100),
        ("diagnostics", "Estimator diagnostics", 90),
    ):
        builder.fact(
            f"estimate_{key}",
            label,
            payload.get(key),
            category,
            priority=priority,
            caveat=SYNTHETIC_IV_CAVEAT if synthetic else CAUSAL_CAVEAT,
            evidence_eligible=evidence_eligible,
            attributes={"synthetic_iv": synthetic},
        )
    sample_used = payload.get("sample_used")
    if isinstance(sample_used, Mapping):
        safe_counts = {
            key: value
            for key, value in sample_used.items()
            if (
                isinstance(value, (bool, int, float))
                and not _PII_KEY_RE.search(str(key))
                and not _SECRET_KEY_RE.search(str(key))
            )
        }
        builder.fact(
            "estimate_sample_summary",
            "Analysis sample summary",
            safe_counts,
            category,
            priority=70,
            caveat=CAUSAL_CAVEAT,
            evidence_eligible=evidence_eligible,
            attributes={"synthetic_iv": synthetic},
        )
    gates = payload.get("gates")
    if isinstance(gates, Mapping):
        _adapt_gate(builder, gates)
    elif gates:
        _adapt_gate(builder, {"results": gates})
    if synthetic:
        builder.caveats.append(SYNTHETIC_IV_CAVEAT)
    builder.caveats.append(CAUSAL_CAVEAT)


def _adapt_refute(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    metrics = _flatten_metric_payload(payload)
    for key, label, priority in (
        ("ok", "Refutation completed", 90),
        ("method", "Refutation method", 75),
        ("backend", "Refutation backend", 65),
        ("soft_skip", "Refutation skipped", 90),
        ("error", "Refutation error", 100),
        ("original_effect", "Original effect", 80),
        ("placebo_effect", "Placebo effect", 90),
        ("pvalue", "Refutation p-value", 80),
        ("p_value", "Refutation p-value", 80),
    ):
        if key in metrics and not _is_blank(metrics[key]):
            builder.fact(
                f"refute_{key}",
                label,
                metrics[key],
                "refutations_sensitivity",
                priority=priority,
                caveat=(
                    "Refutation checks probe robustness; they do not prove or "
                    "disprove causation by themselves."
                ),
            )
    edge = payload.get("edge")
    if isinstance(edge, Mapping):
        builder.fact(
            "refuted_edge",
            "Edge under refutation",
            {
                key: edge.get(key)
                for key in ("source", "target", "type", "score", "confidence")
            },
            "refutations_sensitivity",
            priority=80,
        )


def _adapt_sensitivity(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    builder.fact(
        "domain_hint",
        "Sensitivity domain hint",
        payload.get("domain_hint"),
        "refutations_sensitivity",
    )
    builder.fact(
        "recommended_source",
        "Recommended scientific source",
        payload.get("recommended_source"),
        "refutations_sensitivity",
    )
    metrics = _as_list(payload.get("metrics"))
    rows = []
    for metric in metrics:
        row = _object_mapping(metric)
        name = row.get("name")
        value = row.get("value")
        fact = builder.fact(
            f"sensitivity_{name}",
            f"Sensitivity: {name}",
            value,
            "refutations_sensitivity",
            priority=80,
            attributes={"metric": name},
        )
        rows.append(
            {
                "fact_id": fact.id if fact else "",
                "metric": name,
                "value": value,
                "detail": row.get("detail"),
            }
        )
    if rows:
        builder.table(
            "sensitivity_metrics",
            "Sensitivity metrics",
            ("fact_id", "metric", "value", "detail"),
            rows,
            "refutations_sensitivity",
        )


def _gate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in _as_list(payload.get("results") or payload.get("gates")):
        row = _object_mapping(result)
        status = row.get("status") or ("pass" if row.get("ok") else "fail")
        rows.append(
            {
                "id": row.get("id") or row.get("code"),
                "stage": row.get("stage"),
                "status": status,
                "detail": row.get("detail") or row.get("message"),
                "metric": row.get("metric"),
                "threshold": row.get("threshold"),
                "remediation": row.get("remediation") or row.get("recommendation"),
                "overridden": row.get("overridden"),
            }
        )
    return rows


def _adapt_gate(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    rows = _gate_rows(payload)
    failed = [
        row for row in rows if str(row.get("status")).lower() in {"fail", "escalate"}
        and not row.get("overridden")
    ]
    builder.fact("gate_profile", "Gate profile", payload.get("profile"), "scope_provenance")
    builder.fact(
        "gate_ok",
        "Production gates passed",
        payload.get("ok") if "ok" in payload else not failed,
        "refutations_sensitivity",
        priority=100,
    )
    builder.fact(
        "failed_gate_count",
        "Failed/escalated gates",
        len(failed),
        "refutations_sensitivity",
        priority=100 if failed else 80,
    )
    if rows:
        builder.table(
            "production_gates",
            "Production/evidence gates",
            (
                "id",
                "stage",
                "status",
                "detail",
                "metric",
                "threshold",
                "remediation",
                "overridden",
            ),
            rows,
            "refutations_sensitivity",
        )


def _adapt_manifest(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    for key, label, priority in (
        ("run_id", "Run id", 90),
        ("package_version", "AutoCausal version", 80),
        ("mode", "Run mode", 90),
        ("created_at", "Run created", 60),
        ("completed_at", "Run completed", 60),
        ("status", "Run status", 90),
        ("random_state", "Random state", 65),
    ):
        builder.fact(key, label, payload.get(key), "scope_provenance", priority=priority)
    fingerprint = payload.get("data_fingerprint")
    if isinstance(fingerprint, Mapping):
        builder.fact(
            "data_fingerprint",
            "Data fingerprint",
            {
                key: fingerprint.get(key)
                for key in (
                    "sha256",
                    "n_rows",
                    "n_columns",
                    "hashed_rows",
                    "contains_raw_values",
                )
                if key in fingerprint
            },
            "scope_provenance",
            priority=95,
        )
    policy = payload.get("policy")
    if isinstance(policy, Mapping):
        builder.fact(
            "manifest_policy",
            "Run policy thresholds",
            {
                key: policy.get(key)
                for key in (
                    "profile",
                    "qc",
                    "stability",
                    "bootstrap_n",
                    "ensemble",
                    "min_first_stage_f",
                    "min_stability",
                    "min_methods",
                    "required_evidence",
                    "allow_slm",
                )
                if key in policy
            },
            "technical_appendix",
            priority=75,
        )
    versions = payload.get("engine_versions")
    if isinstance(versions, Mapping):
        builder.table(
            "engine_versions",
            "Engine versions",
            ("engine", "version"),
            [
                {"engine": engine, "version": version}
                for engine, version in sorted(versions.items())
            ],
            "technical_appendix",
        )
    privacy = payload.get("privacy")
    if isinstance(privacy, Mapping):
        builder.fact(
            "pii_column_count",
            "Potential PII columns detected",
            len(_as_list(privacy.get("pii_columns"))),
            "scope_provenance",
            priority=95,
        )
        builder.fact(
            "high_cardinality_count",
            "High-cardinality columns detected",
            len(_as_list(privacy.get("high_cardinality_columns"))),
            "scope_provenance",
            priority=80,
        )
    events = _as_list(payload.get("events"))
    event_rows = []
    for event in events:
        row = _object_mapping(event)
        event_rows.append(
            {
                "stage": row.get("stage"),
                "status": row.get("status"),
                "duration_ms": row.get("duration_ms"),
                "started_at": row.get("started_at"),
                "ended_at": row.get("ended_at"),
            }
        )
    if event_rows:
        builder.table(
            "trace_events",
            "Trace/span summary",
            ("stage", "status", "duration_ms", "started_at", "ended_at"),
            event_rows,
            "technical_appendix",
            footnote="Event metadata is shape/scalar-only; raw payloads are omitted.",
        )
    if payload.get("gates"):
        _adapt_gate(builder, {"gates": payload.get("gates")})


def _adapt_insight(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    summary = payload.get("summary")
    builder.fact(
        "insight_summary",
        "Insight summary",
        summary,
        "insight_actions",
        priority=80,
        caveat=SLM_CAVEAT if payload.get("slm_used") else CAUSAL_CAVEAT,
    )
    builder.fact(
        "guide_backend",
        "Guide backend",
        payload.get("guide_backend") or payload.get("backend"),
        "insight_actions",
    )
    builder.fact(
        "slm_used",
        "SLM used",
        payload.get("slm_used"),
        "insight_actions",
        priority=75,
    )
    key_edges = _as_list(payload.get("key_edges"))
    if key_edges:
        _edge_rows_and_facts(builder, key_edges)
    experiments = _as_list(
        payload.get("experiments_recommended") or payload.get("experiments")
    )
    rows = []
    for item in experiments:
        row = _object_mapping(item)
        edge = _object_mapping(row.get("hypothesized_edge"))
        rows.append(
            {
                "priority": row.get("priority"),
                "kind": row.get("kind"),
                "title": row.get("title") or row.get("statement"),
                "rationale": row.get("rationale"),
                "source": edge.get("source"),
                "target": edge.get("target"),
            }
        )
    if rows:
        builder.table(
            "experiments",
            "Experiment recommendations",
            ("priority", "kind", "title", "rationale", "source", "target"),
            rows,
            "insight_actions",
            footnote="Recommendations require policy approval and human design review.",
        )
    if payload.get("slm_narrative") or payload.get("narrative"):
        narrative = payload.get("slm_narrative") or payload.get("narrative")
        builder.fact(
            "generative_narrative",
            "Generative narrative excerpt",
            narrative,
            "insight_actions",
            caveat=SLM_CAVEAT,
            attributes={"slm_generated": True},
        )
        builder.caveats.append(SLM_CAVEAT)
    builder.caveats.append(CAUSAL_CAVEAT)


def _adapt_agentic(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    _adapt_insight(builder, payload)
    for key, label, priority in (
        ("runtime_backend", "Agentic runtime backend", 70),
        ("chain_backend", "Chain backend", 70),
        ("slm_backend", "SLM backend", 80),
        ("model_name", "SLM model", 70),
        ("n_rounds", "Agentic rounds", 70),
        ("stop_reason", "Agentic stop reason", 85),
    ):
        builder.fact(key, label, payload.get(key), "insight_actions", priority=priority)
    validation = payload.get("validation")
    if isinstance(validation, Mapping):
        builder.fact(
            "agentic_validation",
            "Agentic validation",
            validation,
            "refutations_sensitivity",
            priority=85,
        )
    history = _as_list(payload.get("round_history"))
    if history:
        rows = []
        for item in history:
            row = _object_mapping(item)
            rows.append(
                {
                    "round": row.get("round"),
                    "edges": row.get("n_edges"),
                    "new_edges": row.get("n_new_edges"),
                    "dropped_edges": row.get("n_dropped_edges"),
                    "stop": row.get("stop"),
                }
            )
        builder.table(
            "agentic_rounds",
            "Agentic round history",
            ("round", "edges", "new_edges", "dropped_edges", "stop"),
            rows,
            "insight_actions",
        )


def _adapt_grail(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    for key, label, priority in (
        ("goal", "GRAIL goal", 70),
        ("domain", "GRAIL domain", 65),
        ("backend", "GRAIL backend", 65),
        ("live_kineteq", "Live Kineteq backend", 70),
        ("final_answer", "GRAIL final answer", 75),
        ("next_questions", "GRAIL next questions", 80),
        ("focus_columns", "GRAIL focus columns", 60),
    ):
        builder.fact(key, label, payload.get(key), "insight_actions", priority=priority)
    fold = payload.get("fold")
    if isinstance(fold, Mapping):
        builder.fact(
            "grail_fold",
            "GRAIL fold diagnosis",
            {
                key: fold.get(key)
                for key in ("action_s", "kinetic_t", "potential_v", "directive")
                if key in fold
            },
            "insight_actions",
        )
    builder.caveats.append(CAUSAL_CAVEAT)


def _adapt_automl(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    inner = payload.get("payload")
    merged = dict(payload)
    if isinstance(inner, Mapping):
        merged.update(inner)
    task = merged.get("task")
    if isinstance(task, Mapping):
        builder.fact(
            "automl_task",
            "Predictive task",
            {
                key: task.get(key)
                for key in ("task_type", "target", "metric", "positive_label")
                if key in task
            },
            "automl",
            priority=80,
        )
    else:
        builder.fact("automl_task", "Predictive task", task, "automl", priority=80)
    for key, label, priority in (
        ("imputer", "ML imputer", 60),
        ("predictor", "Predictive model", 75),
        ("outcome", "Predictive outcome", 75),
        ("target", "Predictive target", 75),
        ("selected_name", "Selected predictive model", 90),
        ("selected_model", "Selected predictive model", 90),
        ("metric", "Primary predictive metric", 80),
        ("mode", "AutoML mode", 70),
        ("torch_used", "PyTorch used", 50),
        ("sklearn_used", "scikit-learn used", 50),
        ("kpi_focus", "KPI focus", 65),
        ("ok", "AutoML completed", 80),
    ):
        builder.fact(key, label, merged.get(key), "automl", priority=priority)
    metrics = merged.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {
            key: value
            for key, value in merged.items()
            if _numeric(value)
            and key.lower()
            in {
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "roc_auc",
                "pr_auc",
                "rmse",
                "mae",
                "r2",
                "brier",
                "log_loss",
                "cv_mean",
                "cv_std",
            }
        }
    rows = []
    for key, value in (metrics or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            fact = builder.fact(
                f"automl_metric_{key}",
                f"Predictive metric: {key}",
                value,
                "automl",
                priority=80,
                caveat="Predictive performance is separate from causal estimation.",
                attributes={"predictive_only": True, "metric": key},
            )
            rows.append(
                {
                    "fact_id": fact.id if fact else "",
                    "metric": key,
                    "value": value,
                }
            )
    if rows:
        builder.table(
            "automl_metrics",
            "AutoML predictive metrics",
            ("fact_id", "metric", "value"),
            rows,
            "automl",
            footnote="Predictive metrics must not be presented as causal estimates.",
        )
    candidate_rows: list[dict[str, Any]] = []
    for candidate_index, candidate_value in enumerate(
        _as_list(merged.get("candidates"))
    ):
        candidate = _object_mapping(candidate_value)
        model = candidate.get("name") or candidate.get("model")
        candidate_metrics = candidate.get("metrics")
        if isinstance(candidate_metrics, Mapping):
            for metric_name, summary_value in candidate_metrics.items():
                summary = _object_mapping(summary_value)
                mean = (
                    summary.get("mean")
                    if summary
                    else (
                        summary_value
                        if isinstance(summary_value, (int, float))
                        else None
                    )
                )
                fact = builder.fact(
                    f"automl_candidate_{candidate_index + 1}_{metric_name}",
                    f"{model} predictive {metric_name}",
                    {
                        "mean": mean,
                        "std": summary.get("std"),
                        "ci95_low": summary.get("ci95_low"),
                        "ci95_high": summary.get("ci95_high"),
                    },
                    "automl",
                    priority=85 if candidate.get("selected") else 65,
                    caveat="Predictive performance is separate from causal estimation.",
                    attributes={
                        "predictive_only": True,
                        "metric": metric_name,
                        "model": model,
                    },
                )
                candidate_rows.append(
                    {
                        "fact_id": fact.id if fact else "",
                        "model": model,
                        "metric": metric_name,
                        "mean": mean,
                        "std": summary.get("std"),
                        "ci_low": summary.get("ci95_low"),
                        "ci_high": summary.get("ci95_high"),
                        "rank": candidate.get("rank"),
                        "selected": candidate.get("selected"),
                    }
                )
        elif candidate.get("mean_score") is not None:
            metric_name = candidate.get("metric") or merged.get("metric")
            fact = builder.fact(
                f"automl_candidate_{candidate_index + 1}_{metric_name}",
                f"{model} predictive {metric_name}",
                {
                    "mean": candidate.get("mean_score"),
                    "std": candidate.get("std_score"),
                    "ci_low": candidate.get("ci_low"),
                    "ci_high": candidate.get("ci_high"),
                },
                "automl",
                priority=(
                    85
                    if model
                    in {
                        merged.get("selected_model"),
                        merged.get("selected_name"),
                    }
                    else 65
                ),
                caveat="Predictive performance is separate from causal estimation.",
                attributes={
                    "predictive_only": True,
                    "metric": metric_name,
                    "model": model,
                },
            )
            candidate_rows.append(
                {
                    "fact_id": fact.id if fact else "",
                    "model": model,
                    "metric": metric_name,
                    "mean": candidate.get("mean_score"),
                    "std": candidate.get("std_score"),
                    "ci_low": candidate.get("ci_low"),
                    "ci_high": candidate.get("ci_high"),
                    "rank": candidate.get("rank"),
                    "selected": model
                    in {
                        merged.get("selected_model"),
                        merged.get("selected_name"),
                    },
                }
            )
    if candidate_rows:
        builder.table(
            "automl_candidates",
            "AutoML model-selection ledger",
            (
                "fact_id",
                "model",
                "metric",
                "mean",
                "std",
                "ci_low",
                "ci_high",
                "rank",
                "selected",
            ),
            candidate_rows,
            "automl",
            footnote="Cross-validation scores are predictive, not causal evidence.",
        )
    importance_rows = []
    for item in _as_list(merged.get("feature_importance")):
        row = _object_mapping(item)
        importance_rows.append(
            {
                "feature": row.get("feature") or row.get("name"),
                "importance_mean": (
                    row.get("importance_mean")
                    if row.get("importance_mean") is not None
                    else row.get("importance")
                ),
                "importance_std": row.get("importance_std"),
            }
        )
    if importance_rows:
        builder.table(
            "feature_importance",
            "Predictive feature importance",
            ("feature", "importance_mean", "importance_std"),
            importance_rows,
            "automl",
            footnote=(
                "Feature importance reflects predictive behavior, not intervention effects."
            ),
        )
    gates = merged.get("gates")
    if isinstance(gates, Mapping):
        _adapt_gate(builder, gates)
    elif gates:
        _adapt_gate(builder, {"results": gates})
    if merged.get("raw_predictions_included") or merged.get(
        "contains_raw_predictions"
    ):
        builder.warnings.append(
            "Raw predictions were present in the source report and omitted."
        )
    builder.caveats.append(
        "AutoML performance is predictive evidence and is separated from causal estimates."
    )


def _adapt_nlp(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    for raw_key in ("raw_text", "text", "document", "documents", "corpus", "tokens"):
        if payload.get(raw_key):
            builder.warnings.append(
                f"`{raw_key}` was excluded; raw NLP text is prohibited by default."
            )
    safe_keys = (
        "sentiment",
        "label",
        "score",
        "confidence",
        "valence",
        "arousal",
        "intent",
        "emotion",
        "language",
        "roles",
        "keywords",
        "caveat",
        "privacy",
    )
    for key in safe_keys:
        if key in payload and not _is_blank(payload[key]):
            builder.fact(
                f"nlp_{key}",
                f"NLP {key.replace('_', ' ')}",
                payload[key],
                "nlp_behavioral",
                priority=70,
                caveat=NLP_PRIVACY_CAVEAT,
            )
    profile = payload.get("profile")
    if isinstance(profile, Mapping):
        builder.fact(
            "nlp_profile_rows",
            "NLP-profiled rows",
            profile.get("n_rows"),
            "nlp_behavioral",
            priority=60,
        )
        builder.fact(
            "nlp_privacy_risk",
            "NLP privacy risk",
            profile.get("privacy_risk"),
            "nlp_behavioral",
            priority=95,
        )
        text_rows = []
        for item in _as_list(profile.get("text_columns")):
            row = _object_mapping(item)
            text_rows.append(
                {
                    "column": row.get("column"),
                    "rows": row.get("n_rows"),
                    "non_missing": row.get("n_non_missing"),
                    "missing_fraction": row.get("missing_fraction"),
                    "unique": row.get("unique_count"),
                    "duplication_fraction": row.get("duplication_fraction"),
                    "mean_length": row.get("mean_length"),
                    "pii_hits": sum(
                        int(value or 0)
                        for value in (
                            row.get("pii_risk")
                            if isinstance(row.get("pii_risk"), Mapping)
                            else {}
                        ).values()
                    ),
                    "secret_hits": sum(
                        int(value or 0)
                        for value in (
                            row.get("secret_risk")
                            if isinstance(row.get("secret_risk"), Mapping)
                            else {}
                        ).values()
                    ),
                }
            )
        if text_rows:
            builder.table(
                "nlp_text_profile",
                "NLP text-column profile (no sample text)",
                (
                    "column",
                    "rows",
                    "non_missing",
                    "missing_fraction",
                    "unique",
                    "duplication_fraction",
                    "mean_length",
                    "pii_hits",
                    "secret_hits",
                ),
                text_rows,
                "nlp_behavioral",
                footnote="No documents, snippets, or sample values are retained.",
            )
    claim_rows = []
    for index, item in enumerate(_as_list(payload.get("claims"))):
        claim = _object_mapping(item)
        fact = builder.fact(
            f"nlp_claim_{index + 1}",
            f"Linguistic hypothesis {claim.get('treatment')} → {claim.get('outcome')}",
            {
                "treatment": claim.get("treatment"),
                "outcome": claim.get("outcome"),
                "connector": claim.get("connector"),
                "negated": claim.get("negated"),
                "uncertain": claim.get("uncertain"),
                "confidence": claim.get("confidence"),
                "instruments": claim.get("instruments"),
                "confounders": claim.get("confounders"),
            },
            "nlp_behavioral",
            priority=75,
            caveat=NLP_PRIVACY_CAVEAT,
            evidence_eligible=False,
            attributes={
                "edge_pair": [claim.get("treatment"), claim.get("outcome")],
                "linguistic_hypothesis": True,
            },
        )
        if claim.get("evidence_span"):
            builder.warnings.append(
                "NLP evidence spans were omitted; only structured hypotheses remain."
            )
        claim_rows.append(
            {
                "fact_id": fact.id if fact else "",
                "treatment": claim.get("treatment"),
                "outcome": claim.get("outcome"),
                "negated": claim.get("negated"),
                "uncertain": claim.get("uncertain"),
                "confidence": claim.get("confidence"),
                "hypothesis": True,
            }
        )
    if claim_rows:
        builder.table(
            "nlp_claims",
            "Linguistic causal hypotheses",
            (
                "fact_id",
                "treatment",
                "outcome",
                "negated",
                "uncertain",
                "confidence",
                "hypothesis",
            ),
            claim_rows,
            "nlp_behavioral",
            footnote=(
                "Linguistic causal wording is a hypothesis, not evidence of "
                "causal identification."
            ),
        )
    role_rows = []
    for item in _as_list(payload.get("role_hypotheses")):
        role = _object_mapping(item)
        role_rows.append(
            {
                "role": role.get("role"),
                "text": role.get("text"),
                "confidence": role.get("confidence"),
                "basis": role.get("basis"),
                "hypothesis": role.get("hypothesis", True),
            }
        )
    if role_rows:
        builder.table(
            "nlp_roles",
            "NLP-derived role hypotheses",
            ("role", "text", "confidence", "basis", "hypothesis"),
            role_rows,
            "nlp_behavioral",
        )
    builder.caveats.append(NLP_PRIVACY_CAVEAT)


def _adapt_behavioral(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    builder.fact(
        "trace_name",
        "Behavioral trace collection",
        payload.get("trace_name"),
        "nlp_behavioral",
    )
    edges = _as_list(payload.get("edges") or payload.get("discovery_edges"))
    rows = []
    for index, item in enumerate(edges):
        row = _object_mapping(item)
        fact = builder.fact(
            f"behavioral_edge_{index + 1}",
            f"Behavioral hypothesis {row.get('source')} → {row.get('target')}",
            {
                "source": row.get("source"),
                "target": row.get("target"),
                "kind": row.get("kind") or row.get("type"),
                "score": row.get("score"),
                "evidence": row.get("evidence"),
            },
            "nlp_behavioral",
            caveat=NLP_PRIVACY_CAVEAT,
            evidence_eligible=False,
            attributes={"edge_pair": [row.get("source"), row.get("target")]},
        )
        rows.append(
            {
                "fact_id": fact.id if fact else "",
                "source": row.get("source"),
                "target": row.get("target"),
                "kind": row.get("kind") or row.get("type"),
                "score": row.get("score"),
                "evidence": row.get("evidence"),
            }
        )
    if rows:
        builder.table(
            "behavioral_edges",
            "Behavioral hypotheses",
            ("fact_id", "source", "target", "kind", "score", "evidence"),
            rows,
            "nlp_behavioral",
            footnote=(
                "Behavioral edges are exploratory hypotheses and may encode "
                "sensitive attributes."
            ),
        )
    panel_summary = payload.get("panel_summary")
    if isinstance(panel_summary, Mapping):
        builder.fact(
            "behavioral_panel_summary",
            "Behavioral panel summary",
            panel_summary,
            "nlp_behavioral",
        )
    builder.caveats.extend([NLP_PRIVACY_CAVEAT, ASSOCIATION_CAVEAT])


def _adapt_autoviz(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    plan = payload.get("plan")
    if isinstance(plan, Mapping):
        plan_payload = dict(plan)
    else:
        plan_payload = dict(payload)
    builder.fact(
        "viz_planner",
        "Visualization planner",
        plan_payload.get("planner"),
        "visualizations",
    )
    builder.fact(
        "viz_mode",
        "Visualization mode",
        plan_payload.get("mode"),
        "visualizations",
    )
    recommendations = _as_list(plan_payload.get("recommendations"))
    builder.fact(
        "viz_count",
        "Visualization recommendations",
        len(recommendations),
        "visualizations",
        priority=70,
    )
    for index, item in enumerate(recommendations):
        row = _object_mapping(item)
        rec_id = str(row.get("id") or f"viz-{index + 1}")
        builder.chart(
            rec_id,
            chart_type=str(row.get("chart_type") or row.get("type") or "chart"),
            title=str(row.get("title") or rec_id),
            alt_text=str(
                row.get("alt_text")
                or row.get("rationale")
                or f"Recommended {row.get('chart_type') or 'chart'} visualization."
            ),
            spec={
                "required_columns": row.get("required_columns") or [],
                "data_requirements": row.get("data_requirements") or {},
                "spec_hints": row.get("spec_hints") or row.get("spec") or {},
                "rationale": row.get("rationale"),
            },
            priority=int(row.get("priority") or 50),
            caption=ASSOCIATION_CAVEAT,
        )
    builder.caveats.append(ASSOCIATION_CAVEAT)


def _adapt_autochart(
    builder: SourceBuilder,
    payload: Mapping[str, Any],
    original: Any = None,
) -> None:
    runtime_items: list[Any] = []
    if original is not None and hasattr(original, "charts"):
        runtime_items = list(getattr(original, "charts") or [])
    elif original is not None and hasattr(original, "spec"):
        runtime_items = [original]
    if "artifacts" in payload or "charts" in payload:
        raw_items = (
            payload.get("artifacts")
            if payload.get("artifacts") is not None
            else payload.get("charts")
        )
        payload_items = _as_list(raw_items)
    else:
        payload_items = [payload]
    item_count = max(len(runtime_items), len(payload_items))
    for index in range(item_count):
        runtime_item = runtime_items[index] if index < len(runtime_items) else None
        item = payload_items[index] if index < len(payload_items) else runtime_item
        row = _object_mapping(item)
        spec = _object_mapping(row.get("spec"))
        if runtime_item is not None and hasattr(runtime_item, "spec"):
            runtime_spec = getattr(runtime_item, "spec")
            if hasattr(runtime_spec, "to_dict"):
                try:
                    spec = runtime_spec.to_dict(
                        redact_filter_values=builder.context.policy.production_mode,
                        redact_annotations=builder.context.policy.production_mode,
                    )
                except TypeError:
                    spec = _object_mapping(runtime_spec)
        accessibility = _object_mapping(spec.get("accessibility"))
        provenance = _object_mapping(row.get("provenance")) or _object_mapping(
            spec.get("provenance")
        )
        if row.get("contains_raw_values") or provenance.get("contains_raw_values"):
            builder.contains_raw_data = True
            builder.warnings.append(
                "AutoChart provenance indicates raw values; production reporting "
                "will fail closed."
            )
        runtime_artifact = (
            getattr(runtime_item, "artifact", None)
            if runtime_item is not None
            else None
        )
        path = (
            row.get("path")
            or row.get("image_path")
            or row.get("png_path")
            or row.get("svg_path")
            or (
                str(runtime_artifact)
                if isinstance(runtime_artifact, (str, Path))
                else ""
            )
            or ""
        )
        builder.chart(
            str(spec.get("id") or row.get("id") or f"chart-{index + 1}"),
            chart_type=str(
                spec.get("type")
                or row.get("chart_type")
                or row.get("type")
                or "chart"
            ),
            title=str(
                spec.get("title") or row.get("title") or f"Chart {index + 1}"
            ),
            alt_text=str(
                accessibility.get("alt_text") or row.get("alt_text")
                or row.get("description")
                or f"AutoChart artifact {index + 1}"
            ),
            image_path=str(path),
            spec={
                key: spec.get(key)
                for key in (
                    "id",
                    "type",
                    "title",
                    "x",
                    "y",
                    "color",
                    "facet",
                    "aggregation",
                    "filters",
                    "annotations",
                    "max_rows",
                    "max_cardinality",
                )
                if key in spec
            },
            provenance_ids=[
                str(item)
                for item in _as_list(
                    provenance.get("provenance_ids")
                    or provenance.get("source_ids")
                    or provenance.get("source_id")
                )
                if item
            ],
            caption=str(row.get("caption") or ASSOCIATION_CAVEAT),
            runtime_artifact=(
                None
                if isinstance(runtime_artifact, (str, Path))
                else runtime_artifact
            ),
        )


def _citation_ids_from_claim(claim: Mapping[str, Any]) -> list[str]:
    raw = (
        claim.get("citation_ids")
        or claim.get("source_ids")
        or claim.get("citations")
        or claim.get("sources")
        or []
    )
    ids = []
    for item in _as_list(raw):
        if isinstance(item, Mapping):
            item = (
                item.get("id")
                or item.get("source_id")
                or item.get("record_id")
                or ""
            )
        if item:
            ids.append(str(item))
    for span in _as_list(claim.get("evidence_spans")):
        span_payload = _object_mapping(span)
        source_id = span_payload.get("source_id")
        if source_id:
            ids.append(str(source_id))
    for key in ("supporting_sources", "contradicting_sources"):
        ids.extend(str(item) for item in _as_list(claim.get(key)) if item)
    return list(dict.fromkeys(ids))


def _adapt_deep_research(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    records = _as_list(
        payload.get("source_records")
        or payload.get("sources")
        or payload.get("references")
        or payload.get("citations")
    )
    for record in records:
        builder.citation(record)
        record_payload = _object_mapping(record)
        if record_payload.get("abstract") or record_payload.get("snippet"):
            builder.warnings.append(
                "Source abstracts/snippets were not retained in the report bundle."
            )
    citation_ids = {citation.id for citation in builder.citations}
    builder.fact(
        "research_run_id",
        "Research handoff run id",
        payload.get("handoff_run_id") or payload.get("run_id"),
        "scope_provenance",
        priority=80,
    )
    builder.fact(
        "research_status",
        "Research status",
        payload.get("status"),
        "deep_research",
        priority=80,
    )
    builder.fact(
        "research_stop_reason",
        "Research stop reason",
        payload.get("stop_reason"),
        "deep_research",
        priority=90,
    )
    builder.fact(
        "research_agenda_count",
        "Research questions",
        len(_as_list(payload.get("agenda"))),
        "deep_research",
        priority=65,
    )
    builder.fact(
        "research_question",
        "Research question",
        payload.get("question") or payload.get("query") or payload.get("purpose"),
        "deep_research",
        priority=70,
    )
    builder.fact(
        "verified_sources",
        "Verified literature sources",
        sum(1 for citation in builder.citations if citation.verified),
        "deep_research",
        priority=90,
    )
    claims = _as_list(
        payload.get("claims")
        or payload.get("findings")
        or payload.get("evidence")
    )
    rows = []
    for index, item in enumerate(claims):
        claim = _object_mapping(item)
        text = (
            claim.get("normalized_claim")
            or claim.get("claim")
            or claim.get("statement")
            or claim.get("finding")
            or claim.get("summary")
        )
        ids = _citation_ids_from_claim(claim)
        unsupported = [citation_id for citation_id in ids if citation_id not in citation_ids]
        if unsupported:
            builder.metadata["unsupported_citation_ids"] = list(
                dict.fromkeys(
                    _as_list(builder.metadata.get("unsupported_citation_ids"))
                    + unsupported
                )
            )
            builder.warnings.append(
                "Literature claim referenced unsupported citation ids: "
                + ", ".join(unsupported)
            )
        contradiction_status = str(
            claim.get("contradiction_status")
            or claim.get("literature_label")
            or ""
        ).lower()
        contradiction = bool(claim.get("contradiction")) or contradiction_status in {
            "contradicted",
            "contradicts",
            "mixed",
            "conflicted",
        }
        fact = builder.fact(
            f"literature_claim_{index + 1}",
            f"Literature finding {index + 1}",
            text,
            "deep_research",
            priority=85,
            citation_ids=ids,
            caveat="Literature evidence is reported with supplied source-record ids.",
            attributes={
                "contradiction": contradiction,
                "claim_id": claim.get("claim_id"),
                "literature_label": claim.get("literature_label"),
                "independent_source_count": claim.get("independent_source_count"),
            },
        )
        rows.append(
            {
                "fact_id": fact.id if fact else "",
                "finding": text,
                "citation_ids": ids,
                "confidence": claim.get("confidence"),
                "contradiction": contradiction,
            }
        )
    if rows:
        builder.table(
            "literature_findings",
            "Literature findings",
            ("fact_id", "finding", "citation_ids", "confidence", "contradiction"),
            rows,
            "deep_research",
            footnote="Only supplied/fetched SourceRecord ids may appear as citations.",
        )
    contradictions = _as_list(payload.get("contradictions"))
    contradiction_rows = []
    for index, item in enumerate(contradictions):
        row = _object_mapping(item)
        text = (
            row.get("statement")
            or row.get("contradiction")
            or row.get("detail")
            or (str(item) if not row else "")
        )
        ids = _citation_ids_from_claim(row)
        fact = builder.fact(
            f"contradiction_{index + 1}",
            f"Contradictory evidence {index + 1}",
            text,
            "deep_research",
            priority=100,
            citation_ids=ids,
            attributes={"contradiction": True},
        )
        contradiction_rows.append(
            {
                "fact_id": fact.id if fact else "",
                "contradiction": text,
                "citation_ids": ids,
            }
        )
    if contradiction_rows:
        builder.table(
            "contradictions",
            "Contradictory literature evidence",
            ("fact_id", "contradiction", "citation_ids"),
            contradiction_rows,
            "deep_research",
        )
        builder.caveats.append(
            "Contradictory evidence is retained and must not be suppressed by narrative."
        )
    unresolved = _as_list(
        payload.get("unresolved_questions")
        or payload.get("unresolved_evidence_gaps")
    )
    if unresolved:
        builder.fact(
            "unresolved_research_questions",
            "Unresolved research questions/evidence gaps",
            unresolved,
            "limitations",
            priority=95,
        )
    recommendations = _as_list(
        payload.get("experiment_recommendations")
        or payload.get("handback_recommendations")
    )
    if recommendations:
        builder.table(
            "research_recommendations",
            "Literature-informed follow-up recommendations",
            ("recommendation", "rationale", "priority"),
            [
                {
                    "recommendation": (
                        _object_mapping(item).get("recommendation")
                        or _object_mapping(item).get("title")
                        or _object_mapping(item).get("action")
                        or str(item)
                    ),
                    "rationale": _object_mapping(item).get("rationale"),
                    "priority": _object_mapping(item).get("priority"),
                }
                for item in recommendations
            ],
            "insight_actions",
        )


def _adapt_source_record(builder: SourceBuilder, value: Any) -> None:
    citation = builder.citation(value)
    if citation:
        builder.fact(
            "source_record",
            "Reference source record",
            {"id": citation.id, "title": citation.title, "verified": citation.verified},
            "deep_research",
            citation_ids=[citation.id],
        )


def _adapt_validation(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    builder.fact(
        "validation_ok",
        "Validation passed",
        payload.get("ok"),
        "refutations_sensitivity",
        priority=95,
    )
    builder.fact(
        "validation_score",
        "Validation score",
        payload.get("score"),
        "refutations_sensitivity",
        priority=85,
    )
    checks = _as_list(payload.get("checks"))
    rows = []
    for check in checks:
        row = _object_mapping(check)
        rows.append(
            {
                "id": row.get("id"),
                "ok": row.get("ok"),
                "detail": row.get("detail"),
            }
        )
    if rows:
        builder.table(
            "validation_checks",
            "Validation checks",
            ("id", "ok", "detail"),
            rows,
            "refutations_sensitivity",
        )


def _adapt_public_causal(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    _adapt_mining(builder, payload)
    if payload.get("edges"):
        _adapt_discovery(
            builder,
            {
                "edges": payload.get("edges"),
                "method": payload.get("method") or "public_causal",
            },
        )
    builder.fact(
        "public_sources",
        "Public data sources",
        payload.get("sources") or payload.get("data_sources"),
        "scope_provenance",
    )


def _adapt_generic(builder: SourceBuilder, payload: Mapping[str, Any]) -> None:
    retained = 0
    for key, value in sorted(payload.items()):
        if key in {
            "notes",
            "warnings",
            "caveats",
            "raw_text",
            "text",
            "rows",
            "records",
            "data",
            "frame",
        }:
            continue
        if isinstance(value, (str, int, float, bool)) and not _is_blank(value):
            builder.fact(
                f"generic_{key}",
                str(key).replace("_", " ").title(),
                value,
                "technical_appendix",
            )
            retained += 1
        if retained >= 30:
            break
    if not retained:
        builder.warnings.append(
            "The generic adapter found no safe scalar facts; raw/nested payloads were omitted."
        )


class AutoCausalSourceAdapter:
    def matches(self, value: Any, family_hint: str | None = None) -> bool:
        return family_hint == "autocausal" or (
            _class_name(value) == "AutoCausal"
            and _module_name(value).startswith("autocausal")
        )

    def adapt(
        self,
        value: Any,
        context: AdapterContext,
        family_hint: str | None = None,
    ) -> list[ReportSource]:
        return [_adapt_autocausal(value, context)]


class StructuredReportSourceAdapter:
    def matches(self, value: Any, family_hint: str | None = None) -> bool:
        return (
            family_hint is not None
            or isinstance(value, Mapping)
            or is_dataclass(value)
            or (hasattr(value, "to_dict") and callable(value.to_dict))
        )

    def adapt(
        self,
        value: Any,
        context: AdapterContext,
        family_hint: str | None = None,
    ) -> list[ReportSource]:
        payload = _object_mapping(value)
        family = family_hint or _family_for(value, payload)
        builder = SourceBuilder(
            context,
            family=family,
            title=_source_title(family),
            payload=payload,
        )
        dispatch: dict[str, Callable[[SourceBuilder, Mapping[str, Any]], None]] = {
            "auto_result": _adapt_auto_result,
            "discovery": _adapt_discovery,
            "mining": _adapt_mining,
            "mine": _adapt_mining,
            "qc": _adapt_qc,
            "cleanse": _adapt_cleanse,
            "eda": _adapt_eda,
            "estimate": _adapt_estimate,
            "refute": _adapt_refute,
            "sensitivity": _adapt_sensitivity,
            "gate": _adapt_gate,
            "manifest": _adapt_manifest,
            "insight": _adapt_insight,
            "agentic": _adapt_agentic,
            "grail": _adapt_grail,
            "automl": _adapt_automl,
            "nlp": _adapt_nlp,
            "nlp_behavioral": _adapt_behavioral,
            "autoviz": _adapt_autoviz,
            "autochart": _adapt_autochart,
            "deep_research": _adapt_deep_research,
            "validation": _adapt_validation,
            "public_causal": _adapt_public_causal,
            "generic": _adapt_generic,
        }
        if family == "source_record":
            _adapt_source_record(builder, value)
        elif family == "autochart":
            _adapt_autochart(builder, payload, value)
        else:
            dispatch.get(family, _adapt_generic)(builder, payload)
        _common_notes(builder, payload)
        return [builder.finish()]


def default_adapter_registry() -> AdapterRegistry:
    return AdapterRegistry(
        [
            AutoCausalSourceAdapter(),
            StructuredReportSourceAdapter(),
        ]
    )


def _nested_sources(value: Any, family: str) -> list[tuple[Any, str | None]]:
    """Return safe child artifacts without importing their defining modules."""
    children: list[tuple[Any, str | None]] = []
    if family == "autocausal":
        for attr, hint in (
            ("run_manifest", "manifest"),
            ("qc_report", "qc"),
            ("cleanse_report", "cleanse"),
            ("eda_report", "eda"),
            ("mining", "mining"),
            ("mine_report", "mine"),
            ("result", "discovery"),
            ("sensitivity_report", "sensitivity"),
            ("insight_report", "insight"),
            ("agentic_report", "agentic"),
            ("grail_report", "grail"),
            ("automl_report", "automl"),
            ("nlp_hints", "nlp"),
            ("behavioral_result", "nlp_behavioral"),
            ("autoviz_report", "autoviz"),
            ("autochart_report", "autochart"),
            ("deep_research_report", "deep_research"),
        ):
            try:
                item = getattr(value, attr, None)
            except Exception:
                item = None
            if item is not None:
                children.append((item, hint))
        for attr, hint in (
            ("estimate_results", "estimate"),
            ("causal_inference_results", "estimate"),
            ("refute_results", "refute"),
        ):
            try:
                values = getattr(value, attr, None)
            except Exception:
                values = None
            for item in _as_list(values):
                if item is not None:
                    children.append((item, hint))
    elif family == "auto_result":
        payload = _object_mapping(value)
        for attr, payload_key, hint in (
            ("discovery", "discovery", "discovery"),
            ("mining", "mining", "mining"),
            ("qc", "qc", "qc"),
            ("nlp_hints", "nlp_hints", "nlp"),
            ("sensitivity_report", "sensitivity", "sensitivity"),
            ("grounding", "grounding", "insight"),
        ):
            item = getattr(value, attr, None)
            if item is None:
                item = payload.get(payload_key)
            if item is not None:
                children.append((item, hint))
    elif family == "discovery":
        for attr, hint in (
            ("estimate_results", "estimate"),
            ("refute_results", "refute"),
            ("sensitivity_report", "sensitivity"),
            ("manifest", "manifest"),
        ):
            try:
                item = getattr(value, attr, None)
            except Exception:
                item = None
            for nested in _as_list(item):
                if nested is not None:
                    children.append((nested, hint))
    return children


def normalize_report_sources(
    source: Any,
    *,
    policy: ReportPolicy | None = None,
    registry: AdapterRegistry | None = None,
) -> tuple[list[ReportSource], list[str]]:
    """Normalize arbitrary supported report objects without retaining raw data."""
    resolved_policy = policy or ReportPolicy.production()
    context = AdapterContext(resolved_policy)
    adapters = registry or default_adapter_registry()
    output: list[ReportSource] = []

    def visit(value: Any, family_hint: str | None = None) -> None:
        if value is None:
            return
        if isinstance(value, ReportSource):
            output.append(value)
            return
        if _is_dataframe_like(value):
            raise ReportSafetyError(
                "Raw DataFrame/Series sources are not report artifacts. Run an "
                "AutoCausal analysis first or wrap only approved aggregate facts."
            )
        if (
            family_hint is None
            and isinstance(value, (list, tuple, set, frozenset))
        ):
            for item in value:
                visit(item)
            return
        object_id = id(value)
        if not isinstance(value, (str, int, float, bool, bytes)):
            if object_id in context.seen_objects:
                return
            context.seen_objects.add(object_id)
        payload = _object_mapping(value)
        family = family_hint or (
            "autocausal"
            if _class_name(value) == "AutoCausal"
            and _module_name(value).startswith("autocausal")
            else _family_for(value, payload)
        )
        sources = adapters.adapt(value, context, family)
        output.extend(sources)
        for child, hint in _nested_sources(value, family):
            visit(child, hint)

    visit(source)
    if not output:
        raise TypeError("No reportable artifacts were supplied")
    ensure_unique_ids(output, kind="source")
    for item in output:
        ensure_unique_ids(item.facts, kind=f"fact for source {item.id}")
        ensure_unique_ids(item.tables, kind=f"table for source {item.id}")
        ensure_unique_ids(item.charts, kind=f"chart for source {item.id}")
    warnings = list(context.warnings)
    for item in output:
        warnings.extend(item.warnings)
    if context.unsafe_findings:
        warnings.extend(context.unsafe_findings)
    return output, list(dict.fromkeys(warnings))


__all__ = [
    "ASSOCIATION_CAVEAT",
    "AdapterRegistry",
    "AutoCausalSourceAdapter",
    "CAUSAL_CAVEAT",
    "NLP_PRIVACY_CAVEAT",
    "ReportSourceAdapter",
    "SLM_CAVEAT",
    "SYNTHETIC_IV_CAVEAT",
    "StructuredReportSourceAdapter",
    "default_adapter_registry",
    "normalize_report_sources",
]
