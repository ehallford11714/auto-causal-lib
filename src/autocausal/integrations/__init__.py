"""Production-safe optional integration and capability layer.

Normal imports perform no third-party imports, downloads, network calls,
entry-point loading, or telemetry.
"""

from __future__ import annotations

from dataclasses import replace
from threading import RLock
from typing import Any, Mapping, Optional

from autocausal.integrations.catalog import catalog_specs
from autocausal.integrations.manifest import record_routing_decision
from autocausal.integrations.profiles import (
    build_install_plan,
    list_install_profiles,
)
from autocausal.integrations.registry import ENTRY_POINT_GROUP, IntegrationRegistry
from autocausal.integrations.router import CapabilityRouter
from autocausal.integrations.types import (
    CapabilitySpec,
    HealthState,
    InstallPlan,
    IntegrationAdapter,
    IntegrationMaturity,
    IntegrationPlugin,
    IntegrationSpec,
    IntegrationStatus,
    LicensePolicy,
    PluginDescriptor,
    PluginLoadPolicy,
    ProbeResult,
    ResourceBudget,
    RoutingCandidate,
    RoutingDecision,
    RoutingPolicy,
    RuntimeRequirement,
    SecurityProfile,
)


_DEFAULT_REGISTRY: Optional[IntegrationRegistry] = None
_DEFAULT_LOCK = RLock()


def build_default_registry() -> IntegrationRegistry:
    """Create an isolated registry with the maintained catalog and adapters."""

    from autocausal.integrations.adapters import builtin_adapters

    registry = IntegrationRegistry(catalog_specs())
    for adapter in builtin_adapters():
        registry.register_adapter(adapter)
    return registry


def get_default_registry() -> IntegrationRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_REGISTRY is None:
                _DEFAULT_REGISTRY = build_default_registry()
    return _DEFAULT_REGISTRY


def list_integrations(
    *,
    category: Optional[str] = None,
    deep: bool = False,
    registry: Optional[IntegrationRegistry] = None,
) -> list[IntegrationSpec]:
    """List resolved catalog specs without importing optional packages."""

    active = registry or get_default_registry()
    return active.list_specs(category=category, resolved=True, deep=deep)


def integration_status(
    integration_id: str,
    *,
    deep: bool = False,
    registry: Optional[IntegrationRegistry] = None,
) -> IntegrationStatus:
    active = registry or get_default_registry()
    return active.status(integration_id, deep=deep)


def _with_explicit_integration(
    policy: Optional[RoutingPolicy | Mapping[str, Any]],
    integration_id: Optional[str],
) -> Optional[RoutingPolicy | Mapping[str, Any]]:
    if integration_id is None:
        return policy
    if policy is None:
        return RoutingPolicy(explicit_integration=integration_id)
    if isinstance(policy, RoutingPolicy):
        return replace(policy, explicit_integration=integration_id)
    payload = dict(policy)
    payload["explicit_integration"] = integration_id
    return payload


def get_capability(
    capability: str,
    *,
    integration_id: Optional[str] = None,
    policy: Optional[RoutingPolicy | Mapping[str, Any]] = None,
    budget: Optional[ResourceBudget] = None,
    registry: Optional[IntegrationRegistry] = None,
) -> IntegrationAdapter:
    """Return the selected adapter; use ``CapabilityRouter.route`` for evidence."""

    active = registry or get_default_registry()
    router = CapabilityRouter(active)
    decision = router.route(
        capability,
        policy=_with_explicit_integration(policy, integration_id),
        budget=budget,
    )
    if not decision.selected_integration:
        raise RuntimeError(
            decision.escalation or f"no eligible adapter for {capability!r}"
        )
    return active.get_capability(
        capability,
        integration_id=decision.selected_integration,
    )


def invoke_capability(
    capability: str,
    *,
    integration_id: Optional[str] = None,
    policy: Optional[RoutingPolicy | Mapping[str, Any]] = None,
    budget: Optional[ResourceBudget] = None,
    registry: Optional[IntegrationRegistry] = None,
    manifest: Any = None,
    return_decision: bool = False,
    routing_context: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Route and invoke a capability with safe offline policy defaults."""

    active = registry or get_default_registry()
    return CapabilityRouter(active).invoke(
        capability,
        policy=_with_explicit_integration(policy, integration_id),
        budget=budget,
        manifest=manifest,
        return_decision=return_decision,
        routing_context=routing_context,
        **kwargs,
    )


__all__ = [
    "ENTRY_POINT_GROUP",
    "CapabilityRouter",
    "CapabilitySpec",
    "HealthState",
    "InstallPlan",
    "IntegrationAdapter",
    "IntegrationMaturity",
    "IntegrationPlugin",
    "IntegrationRegistry",
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
    "build_default_registry",
    "build_install_plan",
    "get_capability",
    "get_default_registry",
    "integration_status",
    "invoke_capability",
    "list_install_profiles",
    "list_integrations",
    "record_routing_decision",
]
