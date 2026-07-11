"""Typed contracts for optional AutoCausal integrations.

This module intentionally depends only on the Python standard library.  It is
safe to import in minimal environments and never imports an integration
package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable


class LicensePolicy(str, Enum):
    PERMISSIVE = "permissive"
    COPYLEFT = "copyleft"
    UNKNOWN = "unknown"


class IntegrationMaturity(str, Enum):
    NATIVE = "native adapter"
    EXTERNAL = "external adapter"
    AWARENESS = "awareness-only"
    BLOCKED = "deprecated/blocked"


class RuntimeRequirement(str, Enum):
    CPU = "CPU"
    GPU = "GPU"
    JAVA = "Java"
    R = "R"
    NATIVE = "native"


class HealthState(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"
    UNHEALTHY = "unhealthy"
    BLOCKED = "blocked"
    UNPROBED = "unprobed"


@dataclass(frozen=True)
class CapabilitySpec:
    """One callable capability, not merely a package feature claim."""

    id: str
    description: str
    input_kind: str = "python"
    output_kind: str = "python"
    deterministic: bool = True
    production_ready: bool = False
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["caveats"] = list(self.caveats)
        return value


@dataclass(frozen=True)
class SecurityProfile:
    """Security properties used by policy routing."""

    network_required: bool = False
    may_access_network: bool = False
    data_egress: bool = False
    executes_external_code: bool = False
    telemetry_supported: bool = False
    telemetry_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntegrationSpec:
    """Static catalog metadata plus lazily resolved installation state."""

    id: str
    category: str
    package: str
    import_name: str
    description: str
    install_extra: Optional[str] = None
    install_hint: str = ""
    homepage: str = ""
    license: str = "unknown"
    license_policy: LicensePolicy = LicensePolicy.UNKNOWN
    maturity: IntegrationMaturity = IntegrationMaturity.AWARENESS
    capabilities: tuple[CapabilitySpec, ...] = ()
    required_runtime: tuple[RuntimeRequirement, ...] = (RuntimeRequirement.CPU,)
    compatibility: tuple[str, ...] = ()
    requires_python: str = ">=3.10"
    version_constraint: Optional[str] = None
    health_probe: str = "module_spec"
    deterministic_fallback: Optional[str] = None
    security: SecurityProfile = field(default_factory=SecurityProfile)
    profiles: tuple[str, ...] = ()
    blocked_reason: Optional[str] = None
    version_detected: Optional[str] = None
    health_state: HealthState = HealthState.UNPROBED
    failure_reason: Optional[str] = None

    @property
    def capability_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "package": self.package,
            "import_name": self.import_name,
            "description": self.description,
            "version_detected": self.version_detected,
            "install_extra": self.install_extra,
            "install_hint": self.install_hint,
            "homepage": self.homepage,
            "license": self.license,
            "license_policy": self.license_policy.value,
            "maturity": self.maturity.value,
            "capabilities": [item.to_dict() for item in self.capabilities],
            "required_runtime": [item.value for item in self.required_runtime],
            "compatibility": list(self.compatibility),
            "requires_python": self.requires_python,
            "version_constraint": self.version_constraint,
            "health_probe": self.health_probe,
            "health_state": self.health_state.value,
            "failure_reason": self.failure_reason,
            "deterministic_fallback": self.deterministic_fallback,
            "security": self.security.to_dict(),
            "profiles": list(self.profiles),
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True)
class ProbeResult:
    healthy: bool
    detail: str = ""
    version: Optional[str] = None


@runtime_checkable
class IntegrationAdapter(Protocol):
    """Stable adapter protocol for built-ins and explicitly trusted plugins."""

    id: str
    integration_id: str
    capabilities: Sequence[str]

    def probe(self) -> ProbeResult:
        """Perform an explicit, potentially importing health check."""

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        """Invoke a declared capability."""


@dataclass(frozen=True)
class IntegrationStatus:
    spec: IntegrationSpec
    installed: bool
    healthy: Optional[bool]
    adapter_registered: bool
    probe_performed: bool = False

    @property
    def state(self) -> HealthState:
        return self.spec.health_state

    def to_dict(self) -> dict[str, Any]:
        value = self.spec.to_dict()
        value.update(
            {
                "installed": self.installed,
                "healthy": self.healthy,
                "adapter_registered": self.adapter_registered,
                "probe_performed": self.probe_performed,
            }
        )
        return value


@dataclass(frozen=True)
class PluginDescriptor:
    """Entry-point metadata discovered without loading plugin code."""

    name: str
    value: str
    group: str
    distribution: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntegrationPlugin:
    """Object shape that a trusted entry point must expose."""

    spec: IntegrationSpec
    adapter: IntegrationAdapter


@dataclass(frozen=True)
class PluginLoadPolicy:
    """Explicit approval required before any entry point is loaded."""

    allow_entry_point_loading: bool = False
    trusted_distributions: frozenset[str] = frozenset()
    trusted_names: frozenset[str] = frozenset()

    def approves(self, descriptor: PluginDescriptor) -> bool:
        if not self.allow_entry_point_loading:
            return False
        if descriptor.name in self.trusted_names:
            return True
        return bool(
            descriptor.distribution
            and descriptor.distribution in self.trusted_distributions
        )


@dataclass(frozen=True)
class ResourceBudget:
    hardware: str = "cpu"
    max_memory_mb: int = 2_048
    max_rows: int = 100_000
    max_seconds: float = 300.0


@dataclass(frozen=True)
class RoutingPolicy:
    """Capability selection policy.  Secure/offline defaults are deliberate."""

    allowed_licenses: frozenset[LicensePolicy] = frozenset(
        {LicensePolicy.PERMISSIVE}
    )
    allow_unknown_license: bool = False
    allow_network: bool = False
    allow_data_egress: bool = False
    allow_external_code: bool = False
    allow_gpu: bool = False
    allow_native_runtime: bool = True
    allow_java: bool = False
    allow_r: bool = False
    production: bool = False
    require_deterministic: bool = False
    require_production_ready: bool = False
    explicit_integration: Optional[str] = None


@dataclass(frozen=True)
class RoutingCandidate:
    integration_id: str
    adapter_id: Optional[str]
    installed: bool
    healthy: Optional[bool]
    eligible: bool
    score: int
    reasons: tuple[str, ...] = ()
    version: Optional[str] = None
    fallback: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True)
class RoutingDecision:
    capability: str
    candidates: tuple[RoutingCandidate, ...]
    selected_integration: Optional[str]
    selected_adapter: Optional[str]
    fallback: Optional[str]
    escalation: Optional[str]
    policy_reasons: tuple[str, ...] = ()
    versions: Mapping[str, Optional[str]] = field(default_factory=dict)
    caveats: tuple[str, ...] = ()
    schema: str = "AutoCausalRoutingDecision.v1"

    @property
    def selected(self) -> bool:
        return self.selected_adapter is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "capability": self.capability,
            "selected": self.selected,
            "selected_integration": self.selected_integration,
            "selected_adapter": self.selected_adapter,
            "fallback": self.fallback,
            "escalation": self.escalation,
            "policy_reasons": list(self.policy_reasons),
            "versions": dict(self.versions),
            "caveats": list(self.caveats),
            "candidates": [item.to_dict() for item in self.candidates],
        }


@dataclass(frozen=True)
class InstallPlan:
    profile: str
    hardware: str
    packages: tuple[str, ...]
    constraints: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    command: Optional[str] = None
    schema: str = "AutoCausalInstallPlan.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("packages", "constraints", "excluded", "warnings"):
            value[key] = list(value[key])
        return value


def coerce_routing_policy(
    value: Optional[RoutingPolicy | Mapping[str, Any]],
) -> RoutingPolicy:
    if value is None:
        return RoutingPolicy()
    if isinstance(value, RoutingPolicy):
        return value
    payload = dict(value)
    licenses = payload.get("allowed_licenses")
    if licenses is not None:
        payload["allowed_licenses"] = frozenset(
            item if isinstance(item, LicensePolicy) else LicensePolicy(str(item))
            for item in licenses
        )
    return RoutingPolicy(**payload)


__all__ = [
    "CapabilitySpec",
    "HealthState",
    "InstallPlan",
    "IntegrationAdapter",
    "IntegrationMaturity",
    "IntegrationPlugin",
    "IntegrationSpec",
    "IntegrationStatus",
    "LicensePolicy",
    "PluginDescriptor",
    "PluginLoadPolicy",
    "ProbeResult",
    "ResourceBudget",
    "RoutingCandidate",
    "RoutingDecision",
    "RoutingPolicy",
    "RuntimeRequirement",
    "SecurityProfile",
    "coerce_routing_policy",
]
