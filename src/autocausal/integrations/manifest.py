"""Privacy-safe routing provenance helpers for production manifests."""

from __future__ import annotations

from typing import Any, MutableMapping

from autocausal.integrations.types import RoutingDecision


def record_routing_decision(manifest: Any, decision: RoutingDecision) -> None:
    """Attach a routing decision and selected versions without raw data."""

    if not isinstance(decision, RoutingDecision):
        raise TypeError("decision must be a RoutingDecision")
    if isinstance(manifest, MutableMapping):
        config = manifest.setdefault("config", {})
        versions = manifest.setdefault("engine_versions", {})
    else:
        config = getattr(manifest, "config", None)
        versions = getattr(manifest, "engine_versions", None)
    if not isinstance(config, MutableMapping) or not isinstance(
        versions, MutableMapping
    ):
        raise TypeError(
            "manifest must expose mutable config and engine_versions mappings"
        )
    integrations = config.setdefault(
        "integrations",
        {
            "schema": "AutoCausalIntegrationManifest.v1",
            "telemetry_enabled": False,
            "routing_decisions": [],
        },
    )
    if not isinstance(integrations, MutableMapping):
        raise TypeError("manifest.config['integrations'] must be a mapping")
    decisions = integrations.setdefault("routing_decisions", [])
    if not isinstance(decisions, list):
        raise TypeError("integration routing_decisions must be a list")
    if len(decisions) >= 100:
        raise ValueError("manifest routing decision cap (100) exceeded")
    decisions.append(decision.to_dict())
    for integration_id, version in decision.versions.items():
        if version is not None:
            versions[f"integration:{integration_id}"] = version
    integrations["selected_packages"] = {
        key.removeprefix("integration:"): value
        for key, value in versions.items()
        if str(key).startswith("integration:")
    }


__all__ = ["record_routing_decision"]
