"""Production policy, evidence, provenance, reproducibility, and safety gates.

The core library remains alpha research software.  ``mode="production"`` means
*fail closed and report honestly*; it does not turn heuristic discovery into
causal identification.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
from importlib.util import find_spec
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence

import pandas as pd

from autocausal.__version__ import __version__
from autocausal.iv import AUTO_INSTRUMENT_COL

__all__ = [
    "AnalysisMode",
    "AutoMLRiskPolicy",
    "CausalEvidencePolicy",
    "DataQualityPolicy",
    "EPISTEMIC",
    "EPISTEMIC_BANNER",
    "EvidenceGateError",
    "EvidenceGrade",
    "GateReport",
    "GateResult",
    "OperationalPolicy",
    "PolicyOverride",
    "PrivacyPolicy",
    "MATURITY",
    "ProductionGateError",
    "ProductionPolicy",
    "ProductionSettings",
    "ResourceLimitError",
    "RunEvent",
    "RunManifest",
    "RunPolicy",
    "RunRecorder",
    "StatisticalValidityPolicy",
    "SYNTHETIC_CONFIDENCE_CAP",
    "UnsafePayloadError",
    "annotate_and_gate_edges",
    "apply_mode_defaults",
    "build_data_fingerprint",
    "build_manifest",
    "check_required_engines",
    "engine_versions",
    "grade_edge",
    "is_production",
    "is_synthetic_instrument",
    "privacy_scan",
    "production_checklist",
    "ProductionRun",
    "run_production_pipeline",
    "refuse_synthetic_iv",
    "resolve_mode",
    "resolve_policy",
    "tag_synthetic_iv_edge",
]

AnalysisMode = Literal["exploratory", "review", "production"]
FallbackBehavior = Literal["fail", "warn"]
GateStatus = Literal["pass", "warn", "fail", "escalate", "skip"]

EPISTEMIC = (
    "AutoCausal outputs are exploratory assistance — not causal identification. "
    "Heuristic discovery, synthetic instruments, and soft estimate/refute paths "
    "must not be presented as identified effects."
)

EPISTEMIC_BANNER = (
    "> **EPISTEMIC:** Exploratory assistance only — not causal identification. "
    "Associations ≠ effects; synthetic IV is demo plumbing; require a real design "
    "and observed instruments before making IV claims."
)

SYNTHETIC_CONFIDENCE_CAP = 0.25

MATURITY: dict[str, str] = {
    "core_impute_discover": "stable-alpha",
    "iv_numpy_2sls": "alpha (observed Z required for production claims)",
    "auto_instrument": "demo-only",
    "heuristic_discovery": "alpha (PC-lite / ensemble heuristics)",
    "estimate_builtin": "alpha",
    "estimate_doubleml_econml": "soft-optional beta",
    "refute_dowhy": "soft-optional beta",
    "grail_stub": "stub / offline scaffold",
    "physics_demo": "demo",
    "agentic_loop": "alpha",
    "insight_suite": "alpha",
}


class EvidenceGrade(str, Enum):
    """Evidence state; deliberately excludes an ``identified`` label."""

    EXPLORATORY = "exploratory"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INSUFFICIENT = "insufficient"


def _meets_required_evidence(
    grade: EvidenceGrade,
    required: EvidenceGrade,
) -> bool:
    if required == EvidenceGrade.SUPPORTED:
        return grade == EvidenceGrade.SUPPORTED
    if required == EvidenceGrade.EXPLORATORY:
        return grade in (EvidenceGrade.EXPLORATORY, EvidenceGrade.SUPPORTED)
    if required == EvidenceGrade.REFUTED:
        return grade == EvidenceGrade.REFUTED
    return grade == EvidenceGrade.INSUFFICIENT


@dataclass
class GateResult:
    """Structured production/evidence gate result."""

    id: str
    ok: Optional[bool]
    detail: str
    status: GateStatus | str = ""
    severity: str = "error"
    metric: Optional[Any] = None
    threshold: Optional[Any] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    remediation: Optional[str] = None
    stage: str = "unspecified"
    policy_version: str = "0.13.0"
    edge: Optional[dict[str, Any]] = None
    recommendation: Optional[str] = None
    overridden: bool = False
    override_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.status:
            self.status = "pass" if self.ok else "fail"
        if self.status not in ("pass", "warn", "fail", "escalate", "skip"):
            raise ValueError(f"unknown gate status {self.status!r}")
        if self.ok is None:
            self.ok = self.status in ("pass", "warn", "skip")
        if self.remediation is None and self.recommendation:
            self.remediation = self.recommendation
        if self.recommendation is None and self.remediation:
            self.recommendation = self.remediation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateResult":
        return cls(**dict(value))


@dataclass
class GateReport:
    """Collection of gate decisions with machine and human-readable summaries."""

    schema: str = "AutoCausalGateReport.v1"
    profile: str = "production"
    policy_version: str = "0.13.0"
    results: list[GateResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, *results: GateResult) -> None:
        self.results.extend(results)

    def extend(self, results: Sequence[GateResult]) -> None:
        self.results.extend(results)

    @property
    def failed(self) -> list[GateResult]:
        return [
            result
            for result in self.results
            if result.status in ("fail", "escalate") and not result.overridden
        ]

    @property
    def warnings(self) -> list[GateResult]:
        return [result for result in self.results if result.status == "warn"]

    @property
    def ok(self) -> bool:
        return not self.failed

    def by_stage(self) -> dict[str, list[GateResult]]:
        out: dict[str, list[GateResult]] = {}
        for result in self.results:
            out.setdefault(result.stage, []).append(result)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile": self.profile,
            "policy_version": self.policy_version,
            "ok": self.ok,
            "n_fail": len(self.failed),
            "n_warn": len(self.warnings),
            "results": [result.to_dict() for result in self.results],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateReport":
        payload = dict(value)
        payload.pop("ok", None)
        payload.pop("n_fail", None)
        payload.pop("n_warn", None)
        payload["results"] = [
            result
            if isinstance(result, GateResult)
            else GateResult.from_dict(result)
            for result in payload.get("results") or []
        ]
        return cls(**payload)

    def report(self) -> str:
        lines = [
            "# AutoCausal production gates",
            "",
            f"**Profile:** `{self.profile}` · **Overall:** "
            f"{'PASS' if self.ok else 'FAIL / ESCALATE'}",
            "",
            EPISTEMIC_BANNER,
            "",
        ]
        for stage, results in self.by_stage().items():
            lines.extend([f"## {stage}", ""])
            for result in results:
                marker = str(result.status).upper()
                override = " (OVERRIDDEN)" if result.overridden else ""
                lines.append(
                    f"- [{marker}]{override} `{result.id}` — {result.detail}"
                )
                if result.metric is not None or result.threshold is not None:
                    lines.append(
                        f"  - metric={result.metric!r}; threshold={result.threshold!r}"
                    )
                if result.remediation:
                    lines.append(f"  - Remediation: {result.remediation}")
            lines.append("")
        if self.notes:
            lines.extend(["## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
            lines.append("")
        return "\n".join(lines)


@dataclass
class PolicyOverride:
    """Auditable explicit policy override."""

    field: str
    old_value: Any
    new_value: Any
    reason: str
    actor: str = "user"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualityPolicy:
    min_rows: int = 30
    max_missing_fraction: float = 0.40
    near_zero_variance: float = 1e-12
    allow_destructive_column_drop: bool = False
    allow_row_drop: bool = False
    allow_type_coercion: bool = False
    allow_winsorization: bool = False
    allow_imputation: bool = True
    duplicate_action: str = "flag"
    range_constraints: dict[str, tuple[Optional[float], Optional[float]]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StatisticalValidityPolicy:
    min_sample_size: int = 50
    min_events_per_variable: float = 10.0
    max_vif: float = 10.0
    max_condition_number: float = 1000.0
    heteroskedasticity_alpha: float = 0.05
    durbin_watson_min: float = 1.0
    durbin_watson_max: float = 3.0
    min_overlap_fraction: float = 0.90
    propensity_epsilon: float = 0.05
    fdr_alpha: float = 0.05
    min_stability: float = 0.60
    max_engine_disagreement: float = 0.50
    target_power: float = 0.80
    alpha: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CausalEvidencePolicy:
    require_explicit_roles: bool = True
    require_observed_instrument: bool = True
    min_first_stage_f: float = 10.0
    require_confounders: bool = False
    require_refutation: bool = False
    require_sensitivity: bool = False
    required_evidence: str = EvidenceGrade.SUPPORTED.value
    propensity_trim_quantile: float = 0.01
    max_propensity_weight: float = 20.0
    max_abs_standardized_mean_difference: float = 0.10
    min_design_group_size: int = 20
    min_pre_periods: int = 3
    min_rdd_side: int = 20
    crossfit_folds: int = 5
    hac_max_lags: int = 4
    require_parallel_trends_review: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutoMLRiskPolicy:
    enabled: bool = True
    cv_folds: int = 5
    test_fraction: float = 0.20
    max_cv_coefficient_variation: float = 0.50
    max_brier_score: float = 0.25
    min_class_fraction: float = 0.05
    require_group_split_when_grouped: bool = True
    require_time_split_when_temporal: bool = True
    allow_feature_importance: bool = True
    include_raw_predictions: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrivacyPolicy:
    detect_pii: bool = True
    fail_on_pii: bool = False
    redact_sample_values: bool = True
    allow_raw_data_external: bool = False
    high_cardinality_ratio: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OperationalPolicy:
    required_engines: tuple[str, ...] = ()
    fallback_behavior: FallbackBehavior = "fail"
    allow_slm: bool = False
    max_rows: int = 100_000
    max_columns: int = 200
    max_rounds: int = 3
    max_seconds: float = 300.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["required_engines"] = list(self.required_engines)
        return out


class ProductionGateError(ValueError):
    """Base fail-closed exception carrying safe, structured diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "production_gate_failed",
        gates: Optional[Sequence[GateResult | Mapping[str, Any]]] = None,
        recommendations: Optional[Sequence[str]] = None,
        partial_result: Any = None,
        manifest: Optional["RunManifest"] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.gates = [
            g if isinstance(g, GateResult) else GateResult.from_dict(g)
            for g in (gates or [])
        ]
        self.recommendations = list(recommendations or [])
        self.partial_result = partial_result
        self.manifest = manifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "code": self.code,
            "message": str(self),
            "gates": [g.to_dict() for g in self.gates],
            "recommendations": list(self.recommendations),
            "partial_result": (
                self.partial_result.to_dict()
                if self.partial_result is not None
                and hasattr(self.partial_result, "to_dict")
                else None
            ),
            "manifest": self.manifest.to_dict() if self.manifest else None,
        }


class EvidenceGateError(ProductionGateError):
    """Evidence did not meet the configured production threshold."""


class ResourceLimitError(ProductionGateError):
    """Configured row/column/round/time limit was exceeded."""


class UnsafePayloadError(ProductionGateError):
    """Raw data or unsafe SLM/MCP payload was refused."""


@dataclass
class ProductionPolicy:
    """Serializable run policy used by both production and exploratory sessions.

    Defaults are production-safe.  Use :meth:`for_mode` for mode-specific
    defaults or :meth:`exploratory` for the permissive research profile.
    """

    schema: str = "AutoCausalRunPolicy.v1"
    policy_version: str = "0.13.0"
    profile: str = "production"
    qc: str = "block"
    stability: bool = True
    bootstrap_n: int = 20
    ensemble: bool = True
    use_iv: bool = True
    require_observed_instrument: bool = True
    min_first_stage_f: float = 10.0
    min_stability: float = 0.60
    min_methods: int = 2
    required_evidence: str = EvidenceGrade.SUPPORTED.value
    required_engines: tuple[str, ...] = ()
    fallback_behavior: FallbackBehavior = "fail"
    allow_iv_fallback: bool = False
    allow_synthetic_iv: bool = False
    allow_slm: bool = False
    allow_raw_data_external: bool = False
    redact_sample_values: bool = True
    fail_on_pii: bool = False
    max_rows: int = 100_000
    max_columns: int = 200
    max_rounds: int = 3
    max_seconds: float = 300.0
    random_state: int = 0
    data_quality: Optional[DataQualityPolicy] = None
    statistical_validity: Optional[StatisticalValidityPolicy] = None
    causal_evidence: Optional[CausalEvidencePolicy] = None
    automl_risk: Optional[AutoMLRiskPolicy] = None
    privacy: Optional[PrivacyPolicy] = None
    operational: Optional[OperationalPolicy] = None
    overrides: list[PolicyOverride] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.profile = str(self.profile).strip().lower()
        if self.profile not in ("exploratory", "review", "production"):
            raise ValueError(
                "ProductionPolicy.profile must be exploratory, review, or production"
            )
        if isinstance(self.data_quality, Mapping):
            self.data_quality = DataQualityPolicy(**dict(self.data_quality))
        if isinstance(self.statistical_validity, Mapping):
            self.statistical_validity = StatisticalValidityPolicy(
                **dict(self.statistical_validity)
            )
        if isinstance(self.causal_evidence, Mapping):
            self.causal_evidence = CausalEvidencePolicy(
                **dict(self.causal_evidence)
            )
        if isinstance(self.automl_risk, Mapping):
            self.automl_risk = AutoMLRiskPolicy(**dict(self.automl_risk))
        if isinstance(self.privacy, Mapping):
            self.privacy = PrivacyPolicy(**dict(self.privacy))
        if isinstance(self.operational, Mapping):
            operational_payload = dict(self.operational)
            operational_payload["required_engines"] = tuple(
                operational_payload.get("required_engines") or ()
            )
            self.operational = OperationalPolicy(**operational_payload)
        self.overrides = [
            override
            if isinstance(override, PolicyOverride)
            else PolicyOverride(**dict(override))
            for override in self.overrides
        ]

        # Nested policies are authoritative when explicitly supplied; otherwise
        # derive them from backward-compatible flat fields.
        if self.statistical_validity is not None:
            self.min_stability = self.statistical_validity.min_stability
        else:
            self.statistical_validity = StatisticalValidityPolicy(
                min_stability=self.min_stability
            )
        if self.causal_evidence is not None:
            self.require_observed_instrument = (
                self.causal_evidence.require_observed_instrument
            )
            self.min_first_stage_f = self.causal_evidence.min_first_stage_f
            self.required_evidence = self.causal_evidence.required_evidence
        else:
            self.causal_evidence = CausalEvidencePolicy(
                require_observed_instrument=self.require_observed_instrument,
                min_first_stage_f=self.min_first_stage_f,
                required_evidence=str(self.required_evidence),
            )
        if self.privacy is not None:
            self.allow_raw_data_external = self.privacy.allow_raw_data_external
            self.redact_sample_values = self.privacy.redact_sample_values
            self.fail_on_pii = self.privacy.fail_on_pii
        else:
            self.privacy = PrivacyPolicy(
                fail_on_pii=self.fail_on_pii,
                redact_sample_values=self.redact_sample_values,
                allow_raw_data_external=self.allow_raw_data_external,
            )
        if self.operational is not None:
            self.required_engines = self.operational.required_engines
            self.fallback_behavior = self.operational.fallback_behavior
            self.allow_slm = self.operational.allow_slm
            self.max_rows = self.operational.max_rows
            self.max_columns = self.operational.max_columns
            self.max_rounds = self.operational.max_rounds
            self.max_seconds = self.operational.max_seconds
        else:
            self.operational = OperationalPolicy(
                required_engines=self.required_engines,
                fallback_behavior=self.fallback_behavior,
                allow_slm=self.allow_slm,
                max_rows=self.max_rows,
                max_columns=self.max_columns,
                max_rounds=self.max_rounds,
                max_seconds=self.max_seconds,
            )
        if self.data_quality is None:
            self.data_quality = DataQualityPolicy()
        if self.automl_risk is None:
            self.automl_risk = AutoMLRiskPolicy()

        self.qc = str(self.qc).lower()
        if self.qc not in ("off", "warn", "block"):
            raise ValueError("ProductionPolicy.qc must be off, warn, or block")
        self.fallback_behavior = str(self.fallback_behavior).lower()  # type: ignore[assignment]
        if self.fallback_behavior not in ("fail", "warn"):
            raise ValueError("fallback_behavior must be 'fail' or 'warn'")
        evidence = (
            self.required_evidence.value
            if isinstance(self.required_evidence, EvidenceGrade)
            else str(self.required_evidence).lower()
        )
        if evidence not in {grade.value for grade in EvidenceGrade}:
            raise ValueError(f"unknown required_evidence {self.required_evidence!r}")
        self.required_evidence = evidence
        self.required_engines = tuple(str(x) for x in self.required_engines)
        self.bootstrap_n = max(1, int(self.bootstrap_n))
        self.min_methods = max(1, int(self.min_methods))
        self.max_rows = max(1, int(self.max_rows))
        self.max_columns = max(1, int(self.max_columns))
        self.max_rounds = max(1, int(self.max_rounds))
        self.max_seconds = max(0.001, float(self.max_seconds))
        self.random_state = int(self.random_state)

    @classmethod
    def production(cls, **overrides: Any) -> "ProductionPolicy":
        values: dict[str, Any] = {"profile": "production"}
        values.update(overrides)
        return cls(**values)

    @classmethod
    def strict(cls, **overrides: Any) -> "ProductionPolicy":
        """Strict fail-closed profile used by the production pipeline."""
        return cls.production(**overrides)

    @classmethod
    def review(cls, **overrides: Any) -> "ProductionPolicy":
        """Review profile: strong checks, escalation instead of hard fallback."""
        base: dict[str, Any] = {
            "profile": "review",
            "qc": "warn",
            "stability": True,
            "ensemble": True,
            "required_evidence": EvidenceGrade.EXPLORATORY.value,
            "fallback_behavior": "warn",
            "allow_slm": False,
            "causal_evidence": CausalEvidencePolicy(
                require_explicit_roles=True,
                require_observed_instrument=True,
                min_first_stage_f=10.0,
                required_evidence=EvidenceGrade.EXPLORATORY.value,
            ),
            "operational": OperationalPolicy(
                fallback_behavior="warn",
                allow_slm=False,
            ),
        }
        base.update(overrides)
        return cls(**base)

    @classmethod
    def exploratory(cls, **overrides: Any) -> "ProductionPolicy":
        base: dict[str, Any] = {
            "profile": "exploratory",
            "qc": "warn",
            "stability": False,
            "bootstrap_n": 20,
            "ensemble": False,
            "use_iv": True,
            "require_observed_instrument": False,
            "min_first_stage_f": 0.0,
            "min_stability": 0.0,
            "min_methods": 1,
            "required_evidence": EvidenceGrade.EXPLORATORY.value,
            "fallback_behavior": "warn",
            "allow_iv_fallback": False,
            "allow_synthetic_iv": True,
            "allow_slm": True,
            "allow_raw_data_external": False,
            "redact_sample_values": True,
            "max_rows": 250_000,
            "max_columns": 500,
            "max_rounds": 5,
            "max_seconds": 600.0,
            "random_state": 0,
            "data_quality": DataQualityPolicy(
                min_rows=10,
                max_missing_fraction=0.80,
                allow_destructive_column_drop=True,
                allow_row_drop=True,
                allow_type_coercion=True,
                allow_winsorization=True,
                duplicate_action="drop",
            ),
            "statistical_validity": StatisticalValidityPolicy(
                min_sample_size=10,
                min_events_per_variable=2.0,
                max_vif=50.0,
                max_condition_number=100_000.0,
                min_overlap_fraction=0.50,
                min_stability=0.0,
                max_engine_disagreement=1.0,
            ),
            "causal_evidence": CausalEvidencePolicy(
                require_explicit_roles=False,
                require_observed_instrument=False,
                min_first_stage_f=0.0,
                required_evidence=EvidenceGrade.EXPLORATORY.value,
            ),
            "automl_risk": AutoMLRiskPolicy(
                cv_folds=3,
                max_cv_coefficient_variation=2.0,
                max_brier_score=1.0,
                min_class_fraction=0.01,
            ),
            "privacy": PrivacyPolicy(
                fail_on_pii=False,
                redact_sample_values=True,
                allow_raw_data_external=False,
            ),
            "operational": OperationalPolicy(
                fallback_behavior="warn",
                allow_slm=True,
                max_rows=250_000,
                max_columns=500,
                max_rounds=5,
                max_seconds=600.0,
            ),
        }
        base.update(overrides)
        return cls(**base)

    @classmethod
    def for_mode(cls, mode: AnalysisMode | str, **overrides: Any) -> "ProductionPolicy":
        resolved = resolve_mode(mode)
        if resolved == "production":
            return cls.production(**overrides)
        if resolved == "review":
            return cls.review(**overrides)
        return cls.exploratory(**overrides)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["required_engines"] = list(self.required_engines)
        if self.operational is not None:
            out["operational"]["required_engines"] = list(
                self.operational.required_engines
            )
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductionPolicy":
        payload = dict(value)
        payload.pop("schema", None)
        for key, policy_type in (
            ("data_quality", DataQualityPolicy),
            ("statistical_validity", StatisticalValidityPolicy),
            ("causal_evidence", CausalEvidencePolicy),
            ("automl_risk", AutoMLRiskPolicy),
            ("privacy", PrivacyPolicy),
        ):
            if isinstance(payload.get(key), Mapping):
                payload[key] = policy_type(**dict(payload[key]))
        if isinstance(payload.get("operational"), Mapping):
            op = dict(payload["operational"])
            op["required_engines"] = tuple(op.get("required_engines") or ())
            payload["operational"] = OperationalPolicy(**op)
        payload["overrides"] = [
            override
            if isinstance(override, PolicyOverride)
            else PolicyOverride(**dict(override))
            for override in payload.get("overrides") or []
        ]
        return cls(**payload)

    def with_overrides(
        self,
        *,
        reason: str,
        actor: str = "user",
        **changes: Any,
    ) -> "ProductionPolicy":
        """Return a cloned policy with an auditable explicit override trail."""
        if not reason.strip():
            raise ValueError("Policy overrides require a non-empty reason")
        payload = self.to_dict()
        audit = list(payload.pop("overrides", []))
        for name, new_value in changes.items():
            if name not in payload:
                raise KeyError(f"Unknown policy field {name!r}")
            old_value = payload[name]
            payload[name] = new_value
            audit.append(
                PolicyOverride(
                    field=name,
                    old_value=old_value,
                    new_value=new_value,
                    reason=reason,
                    actor=actor,
                ).to_dict()
            )
            nested_map = {
                "min_stability": ("statistical_validity", "min_stability"),
                "require_observed_instrument": (
                    "causal_evidence",
                    "require_observed_instrument",
                ),
                "min_first_stage_f": ("causal_evidence", "min_first_stage_f"),
                "required_evidence": ("causal_evidence", "required_evidence"),
                "fail_on_pii": ("privacy", "fail_on_pii"),
                "redact_sample_values": ("privacy", "redact_sample_values"),
                "allow_raw_data_external": (
                    "privacy",
                    "allow_raw_data_external",
                ),
                "required_engines": ("operational", "required_engines"),
                "fallback_behavior": ("operational", "fallback_behavior"),
                "allow_slm": ("operational", "allow_slm"),
                "max_rows": ("operational", "max_rows"),
                "max_columns": ("operational", "max_columns"),
                "max_rounds": ("operational", "max_rounds"),
                "max_seconds": ("operational", "max_seconds"),
            }
            if name in nested_map:
                section, nested_name = nested_map[name]
                nested = dict(payload.get(section) or {})
                nested[nested_name] = new_value
                payload[section] = nested
        payload["overrides"] = audit
        return ProductionPolicy.from_dict(payload)

    @classmethod
    def from_json(cls, value: str) -> "ProductionPolicy":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise TypeError("policy JSON must contain an object")
        return cls.from_dict(payload)


# Public synonym for users who prefer mode-neutral naming.
RunPolicy = ProductionPolicy


@dataclass
class ProductionSettings:
    """Resolved discover knobs after combining mode, policy, and call overrides."""

    mode: AnalysisMode = "exploratory"
    auto_instrument: bool = False
    allow_iv_fallback: bool = False
    qc: str = "warn"
    stability: bool = False
    bootstrap_n: int = 20
    ensemble: bool = False
    use_iv: bool = True
    min_methods: int = 1
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunEvent:
    """One redacted observability event/span."""

    stage: str
    status: str
    started_at: str
    ended_at: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunEvent":
        return cls(**dict(value))


@dataclass
class RunManifest:
    """Replayable, privacy-safe run manifest (no raw rows or sample values)."""

    schema: str = "AutoCausalRunManifest.v1"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    package_version: str = __version__
    mode: str = "exploratory"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None
    status: str = "running"
    random_state: int = 0
    policy: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    data_fingerprint: dict[str, Any] = field(default_factory=dict)
    engine_versions: dict[str, Optional[str]] = field(default_factory=dict)
    privacy: dict[str, Any] = field(default_factory=dict)
    events: list[RunEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gates: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "package_version": self.package_version,
            "mode": self.mode,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "random_state": self.random_state,
            "policy": dict(self.policy),
            "config": dict(self.config),
            "data_fingerprint": dict(self.data_fingerprint),
            "engine_versions": dict(self.engine_versions),
            "privacy": dict(self.privacy),
            "events": [event.to_dict() for event in self.events],
            "warnings": list(self.warnings),
            "gates": [gate.to_dict() for gate in self.gates],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunManifest":
        payload = dict(value)
        payload["events"] = [
            event if isinstance(event, RunEvent) else RunEvent.from_dict(event)
            for event in payload.get("events") or []
        ]
        payload["gates"] = [
            gate if isinstance(gate, GateResult) else GateResult.from_dict(gate)
            for gate in payload.get("gates") or []
        ]
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> "RunManifest":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise TypeError("manifest JSON must contain an object")
        return cls.from_dict(payload)

    def finish(self, status: str = "ok") -> None:
        self.status = status
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def replay_config(self) -> dict[str, Any]:
        """Return safe config needed to replay against a separately supplied frame."""
        return {
            "schema": "AutoCausalReplayConfig.v1",
            "package_version": self.package_version,
            "mode": self.mode,
            "random_state": self.random_state,
            "policy": dict(self.policy),
            "discover": dict(self.config.get("discover") or {}),
            "expected_data_fingerprint": dict(self.data_fingerprint),
        }


class RunRecorder:
    """Small no-op-friendly span recorder; metadata must never contain raw data."""

    def __init__(self, manifest: RunManifest) -> None:
        self.manifest = manifest
        self._monotonic_start = time.perf_counter()

    @contextmanager
    def span(self, stage: str, **metadata_values: Any) -> Iterator[None]:
        start_wall = datetime.now(timezone.utc)
        start = time.perf_counter()
        safe_metadata = _redact_metadata(metadata_values)
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            end = time.perf_counter()
            self.manifest.events.append(
                RunEvent(
                    stage=str(stage),
                    status=status,
                    started_at=start_wall.isoformat(),
                    ended_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=round((end - start) * 1000.0, 3),
                    metadata=safe_metadata,
                )
            )

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._monotonic_start

    def check_deadline(self, max_seconds: float, *, partial_result: Any = None) -> None:
        elapsed = self.elapsed_seconds
        if elapsed <= max_seconds:
            return
        gate = GateResult(
            id="max_seconds",
            ok=False,
            detail=f"elapsed={elapsed:.3f}s exceeds max_seconds={max_seconds:.3f}",
            recommendation="Increase policy.max_seconds or reduce data/method scope.",
        )
        self.manifest.gates.append(gate)
        self.manifest.finish("aborted")
        raise ResourceLimitError(
            gate.detail,
            code="time_limit_exceeded",
            gates=[gate],
            recommendations=[gate.recommendation or ""],
            partial_result=partial_result,
            manifest=self.manifest,
        )


def _redact_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep observability metadata scalar/shape-only; never rows or frames."""
    out: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, pd.DataFrame):
            out[str(key)] = {"n_rows": len(item), "n_cols": len(item.columns)}
        elif isinstance(item, pd.Series):
            out[str(key)] = {"length": len(item), "dtype": str(item.dtype)}
        elif isinstance(item, (str, int, float, bool)) or item is None:
            out[str(key)] = item
        elif isinstance(item, (list, tuple, set)):
            out[str(key)] = [str(x) for x in list(item)[:50]]
        elif isinstance(item, Mapping):
            out[str(key)] = {
                str(k): v
                for k, v in item.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }
        else:
            out[str(key)] = type(item).__name__
    return out


def resolve_mode(
    mode: Optional[AnalysisMode | str] = None,
    *,
    strict: Optional[bool] = None,
    default: AnalysisMode = "exploratory",
) -> AnalysisMode:
    if strict is True:
        return "production"
    if mode is None:
        return default
    normalized = str(mode).strip().lower()
    if normalized in ("production", "prod", "strict"):
        return "production"
    if normalized in ("review", "reviewed", "audit"):
        return "review"
    if normalized in ("exploratory", "explore", "demo", "flexible"):
        return "exploratory"
    raise ValueError(
        f"Unknown analysis mode {mode!r}; use 'exploratory', 'review', or "
        "'production' (or strict=True)."
    )


def resolve_policy(
    mode: AnalysisMode | str,
    policy: Optional[ProductionPolicy | Mapping[str, Any]] = None,
    *,
    random_state: Optional[int] = None,
) -> ProductionPolicy:
    resolved_mode = resolve_mode(mode)
    if policy is None:
        resolved = ProductionPolicy.for_mode(resolved_mode)
    elif isinstance(policy, ProductionPolicy):
        resolved = ProductionPolicy.from_dict(policy.to_dict())
    elif isinstance(policy, Mapping):
        base = ProductionPolicy.for_mode(resolved_mode).to_dict()
        base.update(dict(policy))
        resolved = ProductionPolicy.from_dict(base)
    else:
        raise TypeError("policy must be ProductionPolicy, mapping, or None")
    if resolved.profile != resolved_mode:
        raise ProductionGateError(
            f"analysis mode {resolved_mode!r} conflicts with policy profile "
            f"{resolved.profile!r}",
            code="policy_profile_mismatch",
            recommendations=[
                f"Use ProductionPolicy.{resolved_mode}() or omit the explicit policy."
            ],
        )
    if random_state is not None:
        resolved.random_state = int(random_state)
    return resolved


def is_production(
    mode: Optional[AnalysisMode | str] = None,
    *,
    strict: Optional[bool] = None,
) -> bool:
    return resolve_mode(mode, strict=strict) == "production"


def is_synthetic_instrument(z: Optional[str]) -> bool:
    if z is None:
        return False
    text = str(z)
    return text == AUTO_INSTRUMENT_COL or text.startswith("auto_instrument")


def refuse_synthetic_iv(z: Optional[str], *, mode: AnalysisMode) -> None:
    if mode == "production" and is_synthetic_instrument(z):
        gate = GateResult(
            id="synthetic_iv",
            ok=False,
            detail=f"synthetic instrument `{z}` is forbidden in production",
            recommendation="Provide an observed Z and document the IV design.",
        )
        raise EvidenceGateError(
            gate.detail,
            code="synthetic_iv_forbidden",
            gates=[gate],
            recommendations=[gate.recommendation or ""],
        )


def tag_synthetic_iv_edge(edge: dict[str, Any]) -> dict[str, Any]:
    out = dict(edge)
    out["auto_instrument"] = True
    out["synthetic"] = True
    out["identification"] = "none"
    out["evidence_grade"] = EvidenceGrade.INSUFFICIENT.value
    confidence = float(out.get("confidence") or 0.0)
    out["confidence"] = round(min(confidence, SYNTHETIC_CONFIDENCE_CAP), 4)
    notes = list(out.get("notes") or [])
    note = "SYNTHETIC IV (demo only) — identification=none; not an identified effect."
    if note not in notes:
        notes.append(note)
    out["notes"] = notes
    return out


def apply_mode_defaults(
    *,
    mode: Optional[AnalysisMode | str] = None,
    strict: Optional[bool] = None,
    policy: Optional[ProductionPolicy | Mapping[str, Any]] = None,
    auto_instrument: Optional[bool] = None,
    allow_iv_fallback: Optional[bool] = None,
    qc: Optional[str] = None,
    stability: Optional[bool] = None,
    bootstrap_n: Optional[int] = None,
    ensemble: Optional[bool] = None,
    use_iv: Optional[bool] = None,
    min_methods: Optional[int] = None,
) -> ProductionSettings:
    """Resolve mode/policy/call overrides without silently weakening production."""
    resolved_mode = resolve_mode(mode, strict=strict)
    resolved_policy = resolve_policy(resolved_mode, policy)
    notes: list[str] = [f"Analysis mode: {resolved_mode}."]

    if resolved_mode == "production":
        resolved_qc = str(qc if qc is not None else resolved_policy.qc)
        if resolved_qc == "off":
            raise ProductionGateError(
                "production mode refuses qc='off'",
                code="qc_disabled",
                recommendations=["Use qc='warn' or the production default qc='block'."],
            )
        if auto_instrument is True or resolved_policy.allow_synthetic_iv:
            gate = GateResult(
                id="synthetic_iv",
                ok=False,
                detail="production mode refuses auto_instrument=True",
                recommendation="Provide an observed instrument or use exploratory mode.",
            )
            raise EvidenceGateError(
                gate.detail,
                code="synthetic_iv_forbidden",
                gates=[gate],
                recommendations=[gate.recommendation or ""],
            )
        if allow_iv_fallback is True or resolved_policy.allow_iv_fallback:
            gate = GateResult(
                id="iv_fallback",
                ok=False,
                detail="production mode refuses weak-correlate instrument fallback",
                recommendation="Pin an observed instrument with set_iv_roles/candidates.",
            )
            raise EvidenceGateError(
                gate.detail,
                code="iv_fallback_forbidden",
                gates=[gate],
                recommendations=[gate.recommendation or ""],
            )
        if stability is False and resolved_policy.stability:
            raise ProductionGateError(
                "production policy requires bootstrap stability",
                code="stability_disabled",
                recommendations=["Set stability=True or provide an explicit reviewed policy."],
            )
        if ensemble is False and resolved_policy.ensemble:
            raise ProductionGateError(
                "production policy requires ensemble discovery",
                code="ensemble_disabled",
                recommendations=["Set ensemble=True or provide an explicit reviewed policy."],
            )
        notes.append(
            "PRODUCTION: observed instruments only; fail-closed gates; explicit "
            "Y/D(/Z) for estimate/refute."
        )
    elif resolved_mode == "review":
        if auto_instrument:
            notes.append(
                "REVIEW: auto_instrument=True was not run; synthetic IV requires "
                "an explicit exploratory session."
            )
        if allow_iv_fallback:
            notes.append(
                "REVIEW: weak-correlate IV fallback was not run; provide an "
                "observed instrument for review."
            )
        notes.append(
            "REVIEW: strong checks run with escalation records; unresolved "
            "gates require human/domain review."
        )
    elif auto_instrument:
        notes.append(
            "EXPLORATORY: auto_instrument=True synthesizes demo Z "
            f"`{AUTO_INSTRUMENT_COL}` with identification=none."
        )

    return ProductionSettings(
        mode=resolved_mode,
        auto_instrument=bool(auto_instrument) if resolved_mode == "exploratory" else False,
        allow_iv_fallback=(
            bool(allow_iv_fallback)
            if resolved_mode == "exploratory"
            else False
        ),
        qc=str(qc if qc is not None else resolved_policy.qc),
        stability=bool(
            resolved_policy.stability if stability is None else stability
        ),
        bootstrap_n=int(
            resolved_policy.bootstrap_n if bootstrap_n is None else bootstrap_n
        ),
        ensemble=bool(resolved_policy.ensemble if ensemble is None else ensemble),
        use_iv=bool(resolved_policy.use_iv if use_iv is None else use_iv),
        min_methods=int(
            resolved_policy.min_methods if min_methods is None else min_methods
        ),
        notes=notes,
    )


def build_data_fingerprint(df: pd.DataFrame) -> dict[str, Any]:
    """Create a deterministic schema/value digest without exposing raw values."""
    schema = [
        {"name": str(column), "dtype": str(df[column].dtype)}
        for column in df.columns
    ]
    schema_blob = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256()
    digest.update(schema_blob)
    digest.update(f"{len(df)}:{len(df.columns)}".encode())
    try:
        # A bounded deterministic sample keeps manifests cheap for large frames.
        sample = df.iloc[: min(len(df), 10_000)]
        hashes = pd.util.hash_pandas_object(sample, index=True).to_numpy()
        digest.update(hashes.tobytes())
        value_scope = len(sample)
    except Exception:
        value_scope = 0
    return {
        "schema": schema,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "sha256": digest.hexdigest(),
        "hashed_rows": int(value_scope),
        "contains_raw_values": False,
    }


_PII_NAME_RE = re.compile(
    r"(^|_)(ssn|social_security|email|e_mail|phone|mobile|address|"
    r"first_name|last_name|full_name|dob|date_of_birth|passport|"
    r"credit_card|account_number|ip_address)($|_)",
    re.I,
)


def privacy_scan(
    df: pd.DataFrame,
    *,
    high_cardinality_ratio: float = 0.95,
) -> dict[str, Any]:
    """Column-name/cardinality-only privacy scan; no sample values returned."""
    n_rows = max(int(len(df)), 1)
    pii_columns: list[str] = []
    high_cardinality: list[str] = []
    warnings: list[str] = []
    for column in df.columns:
        name = str(column)
        if _PII_NAME_RE.search(name):
            pii_columns.append(name)
        try:
            ratio = float(df[column].nunique(dropna=True)) / n_rows
        except Exception:
            ratio = 0.0
        if n_rows >= 5 and ratio >= high_cardinality_ratio:
            high_cardinality.append(name)
    if pii_columns:
        warnings.append(
            "Potential PII column names detected: " + ", ".join(pii_columns[:12])
        )
    if high_cardinality:
        warnings.append(
            "High-cardinality columns detected: "
            + ", ".join(high_cardinality[:12])
        )
    return {
        "pii_columns": pii_columns,
        "high_cardinality_columns": high_cardinality,
        "warnings": warnings,
        "sample_values_included": False,
    }


def engine_versions(names: Optional[Sequence[str]] = None) -> dict[str, Optional[str]]:
    """Best-effort package versions without importing heavy engines."""
    package_map = {
        "autocausal": "auto-causal-lib",
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "causallearn": "causal-learn",
        "dowhy": "dowhy",
        "doubleml": "DoubleML",
        "econml": "econml",
        "lingam": "lingam",
        "castle": "gcastle",
    }
    selected = list(names or package_map)
    versions: dict[str, Optional[str]] = {}
    for name in selected:
        package = package_map.get(name, name)
        try:
            versions[name] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[name] = None
        except Exception:
            versions[name] = None
    versions["autocausal"] = __version__
    return versions


def build_manifest(
    df: pd.DataFrame,
    *,
    mode: AnalysisMode | str,
    policy: ProductionPolicy,
    run_id: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> RunManifest:
    privacy = privacy_scan(df)
    return RunManifest(
        run_id=run_id or str(uuid.uuid4()),
        mode=resolve_mode(mode),
        random_state=policy.random_state,
        policy=policy.to_dict(),
        config=dict(config or {}),
        data_fingerprint=build_data_fingerprint(df),
        engine_versions=engine_versions(),
        privacy=privacy,
        warnings=list(privacy.get("warnings") or []),
    )


def check_required_engines(
    policy: ProductionPolicy,
    *,
    manifest: Optional[RunManifest] = None,
) -> list[GateResult]:
    """Fail closed for explicitly required engines; builtins remain available."""
    from autocausal.engines import engine_status

    gates: list[GateResult] = []
    missing: list[str] = []
    for name in policy.required_engines:
        status = engine_status(name)
        ok = bool(status.get("available"))
        gate = GateResult(
            id=f"required_engine:{name}",
            ok=ok,
            detail=f"required engine `{name}` is {'available' if ok else 'missing'}",
            recommendation=(
                None
                if ok
                else "Install auto-causal-lib[causal-extra] or change reviewed policy."
            ),
        )
        gates.append(gate)
        if not ok:
            missing.append(name)
    if manifest is not None:
        manifest.gates.extend(gates)
    if missing and policy.fallback_behavior == "fail":
        raise ProductionGateError(
            "Missing required production engine(s): " + ", ".join(missing),
            code="required_engine_missing",
            gates=[gate for gate in gates if not gate.ok],
            recommendations=[
                "Install auto-causal-lib[causal-extra] and rerun doctor --production."
            ],
            manifest=manifest,
        )
    return gates


def _edge_methods(edge: Mapping[str, Any], fallback_method: str) -> list[str]:
    methods = edge.get("methods") or []
    if isinstance(methods, str):
        methods = [methods]
    out = [str(method) for method in methods]
    single = edge.get("method") or edge.get("orientation")
    if single and str(single) not in out:
        out.append(str(single))
    if not out:
        out.append(str(fallback_method))
    return out


def grade_edge(edge: Mapping[str, Any], policy: ProductionPolicy) -> EvidenceGrade:
    """Grade support without ever claiming causal identification."""
    if edge.get("refuted") is True:
        return EvidenceGrade.REFUTED
    instrument = edge.get("instrument")
    is_iv = str(edge.get("type") or "") == "iv_2sls" or instrument is not None
    if is_iv:
        if (
            edge.get("synthetic")
            or edge.get("auto_instrument")
            or is_synthetic_instrument(str(instrument) if instrument is not None else None)
        ):
            return EvidenceGrade.INSUFFICIENT
        first_stage = float(edge.get("first_stage_f") or 0.0)
        return (
            EvidenceGrade.SUPPORTED
            if first_stage >= policy.min_first_stage_f
            else EvidenceGrade.INSUFFICIENT
        )
    stability = edge.get("stability")
    methods = edge.get("methods") or []
    n_methods = int(edge.get("n_methods") or len(methods) or 1)
    if (
        stability is not None
        and float(stability) >= policy.min_stability
        and n_methods >= policy.min_methods
    ):
        return EvidenceGrade.SUPPORTED
    return EvidenceGrade.EXPLORATORY


def annotate_and_gate_edges(
    edges: Sequence[Mapping[str, Any]],
    *,
    source: str,
    run_id: str,
    method: str,
    policy: ProductionPolicy,
    mode: AnalysisMode | str,
    source_columns: Optional[Sequence[str]] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[GateResult]]:
    """Attach provenance/evidence and filter production-ineligible edges."""
    resolved_mode = resolve_mode(mode)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    gates: list[GateResult] = []
    allowed_columns = set(str(column) for column in (source_columns or []))
    required = EvidenceGrade(policy.required_evidence)

    for raw_edge in edges:
        edge = dict(raw_edge)
        instrument = edge.get("instrument")
        synthetic = bool(
            edge.get("synthetic")
            or edge.get("auto_instrument")
            or is_synthetic_instrument(
                str(instrument) if instrument is not None else None
            )
        )
        origin = "synthetic" if synthetic else ("observed" if instrument else "none")
        columns = [
            str(value)
            for value in (
                edge.get("source"),
                edge.get("target"),
                instrument,
            )
            if value is not None
            and (not allowed_columns or str(value) in allowed_columns)
        ]
        grade = grade_edge(edge, policy)
        edge["evidence_grade"] = grade.value
        edge["identification"] = "none" if synthetic else "unverified"
        edge["provenance"] = {
            "source_columns": list(dict.fromkeys(columns)),
            "datasets": [source],
            "discovery_methods": _edge_methods(edge, method),
            "bootstrap_stability": edge.get("stability"),
            "estimator": (
                edge.get("orientation")
                if str(edge.get("type") or "") == "iv_2sls"
                else None
            ),
            "refuters": [],
            "instrument_origin": origin,
            "run_id": run_id,
            "package_version": __version__,
        }

        failure: Optional[GateResult] = None
        if resolved_mode == "production":
            if synthetic:
                failure = GateResult(
                    id="observed_instrument",
                    ok=False,
                    detail="synthetic IV edge rejected (identification=none)",
                    edge={
                        "source": edge.get("source"),
                        "target": edge.get("target"),
                        "instrument": instrument,
                    },
                    recommendation="Provide an observed instrument and design review.",
                )
            elif (
                instrument is not None
                and policy.require_observed_instrument
                and float(edge.get("first_stage_f") or 0.0)
                < policy.min_first_stage_f
            ):
                failure = GateResult(
                    id="instrument_strength",
                    ok=False,
                    detail=(
                        f"first_stage_f={float(edge.get('first_stage_f') or 0.0):.3f} "
                        f"< {policy.min_first_stage_f:.3f}"
                    ),
                    edge={
                        "source": edge.get("source"),
                        "target": edge.get("target"),
                        "instrument": instrument,
                    },
                    recommendation="Use a stronger observed instrument or do not report IV.",
                )
            elif not _meets_required_evidence(grade, required):
                failure = GateResult(
                    id="required_evidence",
                    ok=False,
                    detail=f"edge grade={grade.value}, required={required.value}",
                    edge={
                        "source": edge.get("source"),
                        "target": edge.get("target"),
                        "type": edge.get("type"),
                    },
                    recommendation=(
                        "Increase bootstrap stability/method agreement or collect "
                        "stronger design evidence."
                    ),
                )

        if failure is not None:
            edge["production_eligible"] = False
            edge["failed_gates"] = [failure.id]
            rejected.append(edge)
            gates.append(failure)
        else:
            edge["production_eligible"] = resolved_mode == "production"
            edge["failed_gates"] = []
            accepted.append(edge)

    return accepted, rejected, gates


def production_checklist(
    *,
    production: bool = True,
    policy: Optional[ProductionPolicy | Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Health + safety checklist for ``python -m autocausal doctor --production``."""
    from autocausal.api import AutoCausal
    from autocausal.discovery import discover_relationships
    from autocausal.engines import engine_status, list_engines

    resolved_policy = resolve_policy(
        "production" if production else "exploratory", policy
    )
    checks: list[dict[str, Any]] = []

    signature = inspect.signature(AutoCausal.discover)
    auto_default = signature.parameters["auto_instrument"].default
    checks.append(
        {
            "id": "default_auto_instrument_false",
            "ok": auto_default is False,
            "detail": f"AutoCausal.discover auto_instrument default={auto_default!r}",
        }
    )
    relationship_signature = inspect.signature(discover_relationships)
    relationship_default = relationship_signature.parameters[
        "auto_instrument"
    ].default
    checks.append(
        {
            "id": "discover_relationships_auto_instrument_false",
            "ok": relationship_default is False,
            "detail": (
                "discover_relationships auto_instrument "
                f"default={relationship_default!r}"
            ),
        }
    )
    checks.append(
        {
            "id": "policy_serialization",
            "ok": (
                ProductionPolicy.from_json(resolved_policy.to_json()).to_dict()
                == resolved_policy.to_dict()
            ),
            "detail": f"policy schema={resolved_policy.schema}",
        }
    )
    checks.append(
        {
            "id": "policy_qc",
            "ok": resolved_policy.qc == "block" if production else True,
            "detail": f"qc={resolved_policy.qc}",
        }
    )
    checks.append(
        {
            "id": "policy_resource_limits",
            "ok": all(
                (
                    resolved_policy.max_rows > 0,
                    resolved_policy.max_columns > 0,
                    resolved_policy.max_rounds > 0,
                    resolved_policy.max_seconds > 0,
                )
            ),
            "detail": (
                f"rows={resolved_policy.max_rows}, cols={resolved_policy.max_columns}, "
                f"rounds={resolved_policy.max_rounds}, seconds={resolved_policy.max_seconds}"
            ),
        }
    )

    engines = list_engines()
    status = engine_status()
    available = sum(1 for engine in engines if engine.available)
    checks.append(
        {
            "id": "engines_available",
            "ok": available >= 1,
            "detail": f"n_available={available} / n={status.get('n')}",
        }
    )
    required_status: dict[str, bool] = {}
    for engine_name in resolved_policy.required_engines:
        ok = bool(engine_status(engine_name).get("available"))
        required_status[engine_name] = ok
        checks.append(
            {
                "id": f"required_engine_{engine_name}",
                "ok": ok,
                "detail": f"{engine_name}={'available' if ok else 'missing'}",
            }
        )

    for name in ("dowhy", "doubleml", "econml", "causallearn"):
        ok = find_spec(name) is not None
        checks.append(
            {
                "id": f"optional_{name}",
                "ok": ok,
                "warn_only": not ok and name not in resolved_policy.required_engines,
                "detail": (
                    f"{name}={'installed' if ok else 'MISSING — unavailable if requested'}"
                ),
            }
        )
    checks.append(
        {
            "id": "version",
            "ok": bool(__version__),
            "detail": f"auto-causal-lib=={__version__}",
        }
    )

    failed = [
        check
        for check in checks
        if not check.get("ok") and not check.get("warn_only")
    ]
    warnings = [
        check
        for check in checks
        if check.get("warn_only") and not check.get("ok")
    ]
    return {
        "schema": "AutoCausalProductionChecklist.v1",
        "version": __version__,
        "production": production,
        "epistemic": EPISTEMIC,
        "policy": resolved_policy.to_dict(),
        "required_engine_status": required_status,
        "maturity": dict(MATURITY),
        "checks": checks,
        "ok": not failed,
        "n_failed": len(failed),
        "n_warnings": len(warnings),
        "notes": [
            EPISTEMIC,
            "Production mode refuses synthetic IV and raw external payloads.",
            "Heuristic discovery remains alpha even when evidence gates pass.",
        ],
    }


@dataclass
class ProductionRun:
    """Unified, privacy-safe production pipeline result.

    ``frame`` remains in memory for downstream work but is deliberately omitted
    from serialization.
    """

    gates: GateReport
    manifest: RunManifest
    policy: ProductionPolicy
    status: str
    frame: Optional[pd.DataFrame] = field(default=None, repr=False)
    cleanse_report: Any = None
    eda_report: Any = None
    automl_report: Any = None
    inference_result: Any = None
    discovery_result: Any = None
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)
    schema: str = "AutoCausalProductionRun.v1"

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.gates.ok

    def to_dict(self) -> dict[str, Any]:
        def serialize(value: Any) -> Any:
            if value is None:
                return None
            if hasattr(value, "to_dict"):
                return value.to_dict()
            return value

        return {
            "schema": self.schema,
            "status": self.status,
            "ok": self.ok,
            "policy": self.policy.to_dict(),
            "gates": self.gates.to_dict(),
            "manifest": self.manifest.to_dict(),
            "cleanse_report": serialize(self.cleanse_report),
            "eda_report": serialize(self.eda_report),
            "automl_report": serialize(self.automl_report),
            "inference_result": serialize(self.inference_result),
            "discovery_result": serialize(self.discovery_result),
            "recommendations": list(self.recommendations),
            "escalations": list(self.escalations),
            "frame_included": False,
        }

    def report(self) -> str:
        lines = [
            "# AutoCausal production-oriented run",
            "",
            EPISTEMIC_BANNER,
            "",
            f"- Status: `{self.status}`",
            f"- Run id: `{self.manifest.run_id}`",
            f"- Policy: `{self.policy.profile}` / `{self.policy.policy_version}`",
            "",
            self.gates.report(),
        ]
        if self.escalations:
            lines.extend(["", "## Escalations", ""])
            lines.extend(f"- {value}" for value in self.escalations)
        return "\n".join(lines)


def run_production_pipeline(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    instrument: Optional[str | Sequence[str]] = None,
    confounders: Optional[Sequence[str]] = None,
    unit: Optional[str] = None,
    time: Optional[str] = None,
    post: Optional[str] = None,
    running: Optional[str] = None,
    cutoff: Optional[float] = None,
    bandwidth: Optional[float] = None,
    method: Optional[str] = None,
    automl_target: Optional[str] = None,
    automl_features: Optional[Sequence[str]] = None,
    run_discovery: bool = False,
    dry_run_cleanse: bool = False,
    policy: Optional[ProductionPolicy | Mapping[str, Any]] = None,
    random_state: int = 42,
) -> ProductionRun:
    """Run aligned cleanse, EDA, statistics, optional ML and inference gates.

    When ``method`` is omitted, the planner records reviewed candidates and the
    pipeline stops at ``review_required`` rather than blindly choosing a causal
    estimator.  Supplying a method runs it fail-closed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("run_production_pipeline expects a pandas DataFrame.")
    resolved_policy = resolve_policy(
        "production",
        policy or ProductionPolicy.strict(random_state=random_state),
        random_state=random_state,
    )
    # A passed-in review/exploratory policy remains auditable, but this entry
    # point still does not enable synthetic IV.
    if resolved_policy.allow_synthetic_iv:
        raise ProductionGateError(
            "Production pipeline refuses policies that allow synthetic IV.",
            code="unsafe_production_policy",
            recommendations=[
                "Use ProductionPolicy.strict() or override allow_synthetic_iv=False."
            ],
        )
    manifest = build_manifest(
        df,
        mode="production",
        policy=resolved_policy,
        config={
            "production_pipeline": {
                "treatment": treatment,
                "outcome": outcome,
                "instrument": (
                    [instrument]
                    if isinstance(instrument, str)
                    else list(instrument or [])
                ),
                "confounders": list(confounders or []),
                "unit": unit,
                "time": time,
                "post": post,
                "running": running,
                "cutoff": cutoff,
                "bandwidth": bandwidth,
                "method": method,
                "automl_target": automl_target,
                "run_discovery": run_discovery,
                "dry_run_cleanse": dry_run_cleanse,
            }
        },
    )
    recorder = RunRecorder(manifest)
    gates = GateReport(
        profile=resolved_policy.profile,
        policy_version=resolved_policy.policy_version,
        notes=[
            "Passing gates provides production-oriented safeguards, not certification."
        ],
    )

    with recorder.span("production_cleanse", dry_run=dry_run_cleanse):
        from autocausal.suites.autocleanse import AutoCleanseSuite

        cleanse_suite = AutoCleanseSuite(
            df,
            mode="production",
            policy=resolved_policy,
            use_slm=False,
            dry_run=dry_run_cleanse,
        ).run()
    clean_frame = cleanse_suite.frame
    cleanse_report = cleanse_suite.report
    assert clean_frame is not None and cleanse_report is not None
    if cleanse_report.gate_report:
        cleanse_gates = GateReport.from_dict(cleanse_report.gate_report)
        gates.extend(cleanse_gates.results)
        manifest.gates.extend(cleanse_gates.results)

    with recorder.span("production_eda"):
        from autocausal.suites.autoeda import AutoEDASuite

        eda_suite = AutoEDASuite(
            clean_frame,
            mode="production",
            policy=resolved_policy,
            use_slm=False,
            treatment=treatment,
            outcome=outcome,
            instrument=instrument,
            confounders=confounders,
            unit=unit,
            time=time,
        ).run()
    eda_report = eda_suite.report
    assert eda_report is not None
    if eda_report.gate_report:
        eda_gates = GateReport.from_dict(eda_report.gate_report)
        gates.extend(eda_gates.results)
        manifest.gates.extend(eda_gates.results)

    automl_report = None
    if automl_target:
        with recorder.span("production_automl", target=automl_target):
            from autocausal.automl import AutoTabularML

            try:
                automl_report = AutoTabularML(
                    clean_frame,
                    policy=resolved_policy,
                    mode="production",
                    random_state=random_state,
                ).run(
                    target=automl_target,
                    feature_columns=automl_features,
                    group_column=unit,
                    time_column=time,
                    enforce_gates=True,
                )
                for raw_gate in list(getattr(automl_report, "gates", []) or []):
                    gate = (
                        GateResult.from_dict(raw_gate)
                        if isinstance(raw_gate, dict)
                        else raw_gate
                    )
                    gates.append(gate)
                    manifest.gates.append(gate)
            except Exception as exc:
                # Compatibility fallback to the compact ml.AutoML portfolio.
                from autocausal.ml import AutoML

                automl_report = AutoML(
                    policy=resolved_policy,
                    mode="production",
                    random_state=random_state,
                ).fit(
                    clean_frame,
                    target=automl_target,
                    features=automl_features,
                    group=unit,
                    time=time,
                )
                gates.extend(automl_report.gates.results)
                manifest.gates.extend(automl_report.gates.results)
                manifest.warnings.append(
                    f"AutoTabularML unavailable ({exc}); used autocausal.ml.AutoML fallback."
                )

    from autocausal.inference import (
        AutoInference,
        AutoInferencePlanner,
        CausalSpec,
    )

    spec = CausalSpec(
        treatment=treatment,
        outcome=outcome,
        confounders=list(confounders or []),
        instrument=(
            instrument
            if isinstance(instrument, str) or instrument is None
            else list(instrument)
        ),
        unit=unit,
        time=time,
        post=post,
        running=running,
        cutoff=cutoff,
        bandwidth=bandwidth,
        instrument_provenance="observed" if instrument else "unknown",
    )
    planner = AutoInferencePlanner(
        spec,
        policy=resolved_policy,
        mode="production",
    )
    recommendations = [
        value.to_dict() for value in planner.recommend(clean_frame)
    ]
    manifest.config["production_pipeline"]["planner_recommendations"] = (
        recommendations
    )
    inference_result = None
    escalations: list[str] = []
    if method is None:
        review_gate = GateResult(
            id="inference_method_review",
            ok=False,
            status="escalate",
            detail=(
                "No causal method was selected; candidates were recommended but "
                "not run automatically."
            ),
            evidence={
                "candidate_methods": [
                    value["method"]
                    for value in recommendations
                    if value["status"] == "candidate"
                ]
            },
            remediation="Have a causal/domain reviewer select and justify a method.",
            stage="causal_inference",
            policy_version=resolved_policy.policy_version,
        )
        gates.add(review_gate)
        manifest.gates.append(review_gate)
        escalations.append(
            "Human/domain review: select an estimand and defensible method."
        )
        status = "review_required"
    else:
        with recorder.span("production_inference", method=method):
            inference_result = AutoInference(
                spec,
                policy=resolved_policy,
                mode="production",
                random_state=random_state,
                run_id=manifest.run_id,
            ).fit(clean_frame, method=method)
        gates.extend(inference_result.gates.results)
        manifest.gates.extend(inference_result.gates.results)
        status = "ok"

    discovery_result = None
    if run_discovery:
        with recorder.span("production_discovery"):
            from autocausal.api import AutoCausal

            discovery_session = AutoCausal.from_dataframe(
                clean_frame,
                source="production_pipeline",
                mode="production",
                policy=resolved_policy,
                random_state=random_state,
                run_id=manifest.run_id,
            )
            discovery_result = discovery_session.discover(
                auto_instrument=False,
                mode="production",
                policy=resolved_policy,
            )
        if getattr(discovery_result, "gates", None):
            for value in discovery_result.gates:
                gate = (
                    value
                    if isinstance(value, GateResult)
                    else GateResult.from_dict(value)
                )
                gates.add(gate)
                manifest.gates.append(gate)

    manifest.finish(status)
    run = ProductionRun(
        gates=gates,
        manifest=manifest,
        policy=resolved_policy,
        status=status,
        frame=clean_frame,
        cleanse_report=cleanse_report,
        eda_report=eda_report,
        automl_report=automl_report,
        inference_result=inference_result,
        discovery_result=discovery_result,
        recommendations=recommendations,
        escalations=escalations,
    )
    non_review_failures = [
        gate for gate in gates.failed if gate.id != "inference_method_review"
    ]
    if non_review_failures:
        raise ProductionGateError(
            "Production pipeline failed one or more gates.",
            code="production_pipeline_gate_failed",
            gates=non_review_failures,
            recommendations=[
                gate.remediation or "Review failed stage."
                for gate in non_review_failures
            ],
            partial_result=run,
            manifest=manifest,
        )
    return run
