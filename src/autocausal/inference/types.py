"""Unified causal design and result contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from autocausal.production import EPISTEMIC, GateReport, RunManifest


@dataclass
class CausalSpec:
    """Explicit design metadata for effect estimation.

    A spec records the proposed design; it does not itself establish
    identification. Domain assumptions should be documented in
    ``assumptions``.
    """

    treatment: str
    outcome: str
    confounders: list[str] = field(default_factory=list)
    instrument: Optional[str | list[str]] = None
    unit: Optional[str] = None
    time: Optional[str] = None
    post: Optional[str] = None
    running: Optional[str] = None
    cutoff: Optional[float] = None
    bandwidth: Optional[float] = None
    cluster: Optional[str] = None
    weights: Optional[str] = None
    estimand: str = "ATE"
    treatment_value: Any = 1
    control_value: Any = 0
    assumptions: dict[str, str | bool] = field(default_factory=dict)
    instrument_provenance: str = "observed"
    source: Optional[str] = None
    schema: str = "AutoCausalCausalSpec.v1"

    def __post_init__(self) -> None:
        self.treatment = str(self.treatment)
        self.outcome = str(self.outcome)
        self.confounders = [str(value) for value in self.confounders]
        if isinstance(self.instrument, tuple):
            self.instrument = [str(value) for value in self.instrument]
        elif isinstance(self.instrument, list):
            self.instrument = [str(value) for value in self.instrument]
        elif self.instrument is not None:
            self.instrument = str(self.instrument)
        for attribute in ("unit", "time", "post", "running", "cluster", "weights"):
            value = getattr(self, attribute)
            if value is not None:
                setattr(self, attribute, str(value))
        self.instrument_provenance = str(self.instrument_provenance).lower()
        if self.instrument_provenance not in ("observed", "synthetic", "unknown"):
            raise ValueError(
                "instrument_provenance must be observed, synthetic, or unknown"
            )

    @property
    def instruments(self) -> list[str]:
        if self.instrument is None:
            return []
        if isinstance(self.instrument, str):
            return [self.instrument]
        return list(self.instrument)

    def required_columns(self, method: Optional[str] = None) -> list[str]:
        columns = [self.treatment, self.outcome, *self.confounders]
        normalized = str(method or "").lower()
        if normalized in ("iv", "2sls", "iv_2sls"):
            columns.extend(self.instruments)
        if normalized in ("did", "difference_in_differences"):
            columns.extend(
                value for value in (self.unit, self.time, self.post) if value
            )
        if normalized in ("panel_fe", "fixed_effects", "within"):
            columns.extend(value for value in (self.unit, self.time) if value)
        if normalized in ("rdd", "regression_discontinuity"):
            if self.running:
                columns.append(self.running)
        if normalized in ("its", "interrupted_time_series"):
            columns.extend(value for value in (self.time, self.post) if value)
        if self.cluster:
            columns.append(self.cluster)
        if self.weights:
            columns.append(self.weights)
        return list(dict.fromkeys(str(value) for value in columns if value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CausalSpec":
        return cls(**dict(value))


@dataclass
class MethodRecommendation:
    method: str
    rank: int
    rationale: str
    required_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    status: str = "candidate"
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CausalInferenceResult:
    """One causal estimate plus diagnostics, gates, and provenance."""

    method: str
    estimand: str
    estimate: Optional[float]
    standard_error: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    p_value: Optional[float]
    n: int
    treatment: str
    outcome: str
    controls: list[str] = field(default_factory=list)
    sample_used: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    evidence_grade: str = "insufficient"
    gates: GateReport = field(default_factory=GateReport)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=lambda: [EPISTEMIC])
    manifest: Optional[RunManifest] = None
    ok: bool = True
    soft_skip: bool = False
    error: Optional[str] = None
    schema: str = "AutoCausalInferenceResult.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ok": self.ok,
            "soft_skip": self.soft_skip,
            "error": self.error,
            "method": self.method,
            "estimand": self.estimand,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n": self.n,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "controls": list(self.controls),
            "sample_used": dict(self.sample_used),
            "assumptions": list(self.assumptions),
            "diagnostics": dict(self.diagnostics),
            "provenance": dict(self.provenance),
            "evidence_grade": self.evidence_grade,
            "gates": self.gates.to_dict(),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "manifest": self.manifest.to_dict() if self.manifest else None,
        }

    def report(self) -> str:
        estimate = (
            "not estimated" if self.estimate is None else f"{self.estimate:.6g}"
        )
        uncertainty = (
            "not available"
            if self.standard_error is None
            else f"SE={self.standard_error:.6g}"
        )
        interval = (
            "not available"
            if self.ci_low is None or self.ci_high is None
            else f"[{self.ci_low:.6g}, {self.ci_high:.6g}]"
        )
        pvalue = (
            "not available" if self.p_value is None else f"{self.p_value:.6g}"
        )
        assumptions = "\n".join(f"- {value}" for value in self.assumptions)
        warnings = "\n".join(f"- {value}" for value in self.warnings)
        return "\n".join(
            [
                "# Causal inference result",
                "",
                f"> **{EPISTEMIC}**",
                "",
                f"- Method: `{self.method}`",
                f"- Estimand: `{self.estimand}`",
                f"- Estimate: {estimate}",
                f"- Uncertainty: {uncertainty}; 95% CI {interval}",
                f"- p-value: {pvalue}",
                f"- Complete sample: {self.n}",
                f"- Evidence grade: `{self.evidence_grade}`",
                "",
                "## Assumptions",
                assumptions or "- Not documented.",
                "",
                "## Gate decisions",
                self.gates.report(),
                "",
                "## Warnings",
                warnings or "- None beyond the epistemic banner.",
            ]
        )

    def to_fabric_metadata(self) -> dict[str, Any]:
        """Privacy-safe fields for a FabricBundle ``meta`` payload."""
        return {
            "inference": {
                "schema": self.schema,
                "method": self.method,
                "estimand": self.estimand,
                "estimate": self.estimate,
                "standard_error": self.standard_error,
                "ci": [self.ci_low, self.ci_high],
                "p_value": self.p_value,
                "n": self.n,
                "evidence_grade": self.evidence_grade,
                "diagnostics": dict(self.diagnostics),
                "provenance": dict(self.provenance),
                "gates": self.gates.to_dict(),
            }
        }


InferenceResult = CausalInferenceResult
