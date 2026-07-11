"""Lazy integration registry and opt-in third-party plugin discovery."""

from __future__ import annotations

import sys
from dataclasses import replace
from importlib import metadata
from importlib.util import find_spec
from threading import RLock
from typing import Any, Iterable, Optional

from autocausal.integrations.types import (
    HealthState,
    IntegrationAdapter,
    IntegrationMaturity,
    IntegrationPlugin,
    IntegrationSpec,
    IntegrationStatus,
    PluginDescriptor,
    PluginLoadPolicy,
)


ENTRY_POINT_GROUP = "autocausal.integrations"


class IntegrationRegistry:
    """Thread-safe catalog and adapter registry.

    Construction and normal status checks use package metadata and module
    specs only.  Deep probes and entry-point loading are explicit operations.
    """

    def __init__(self, specs: Optional[Iterable[IntegrationSpec]] = None) -> None:
        self._specs: dict[str, IntegrationSpec] = {}
        self._adapters: dict[str, IntegrationAdapter] = {}
        self._capability_adapters: dict[str, list[str]] = {}
        self._entry_points: dict[str, Any] = {}
        self._plugin_descriptors: dict[str, PluginDescriptor] = {}
        self._lock = RLock()
        for spec in specs or ():
            self.register(spec)

    def register(
        self,
        spec: IntegrationSpec,
        adapter: Optional[IntegrationAdapter] = None,
        *,
        replace_existing: bool = False,
    ) -> None:
        if not isinstance(spec, IntegrationSpec):
            raise TypeError("spec must be an IntegrationSpec")
        with self._lock:
            if spec.id in self._specs and not replace_existing:
                raise ValueError(f"integration {spec.id!r} is already registered")
            self._specs[spec.id] = spec
        if adapter is not None:
            self.register_adapter(adapter, replace_existing=replace_existing)

    def register_adapter(
        self,
        adapter: IntegrationAdapter,
        *,
        replace_existing: bool = False,
    ) -> None:
        if not isinstance(adapter, IntegrationAdapter):
            raise TypeError(
                "adapter must implement id, integration_id, capabilities, probe(), and invoke()"
            )
        integration_id = str(adapter.integration_id)
        adapter_id = str(adapter.id)
        with self._lock:
            if integration_id not in self._specs:
                raise KeyError(
                    f"adapter {adapter_id!r} references unknown integration {integration_id!r}"
                )
            spec = self._specs[integration_id]
            declared = set(spec.capability_ids)
            provided = {str(item) for item in adapter.capabilities}
            undeclared = provided - declared
            if undeclared:
                raise ValueError(
                    f"adapter {adapter_id!r} exposes undeclared capabilities: "
                    f"{sorted(undeclared)}"
                )
            if adapter_id in self._adapters and not replace_existing:
                raise ValueError(f"adapter {adapter_id!r} is already registered")
            if replace_existing and adapter_id in self._adapters:
                self._remove_adapter_bindings(adapter_id)
            self._adapters[adapter_id] = adapter
            for capability in sorted(provided):
                bindings = self._capability_adapters.setdefault(capability, [])
                if adapter_id not in bindings:
                    bindings.append(adapter_id)

    def _remove_adapter_bindings(self, adapter_id: str) -> None:
        for capability, bindings in list(self._capability_adapters.items()):
            self._capability_adapters[capability] = [
                item for item in bindings if item != adapter_id
            ]
            if not self._capability_adapters[capability]:
                del self._capability_adapters[capability]

    def get_spec(self, integration_id: str, *, resolved: bool = False) -> IntegrationSpec:
        try:
            spec = self._specs[str(integration_id)]
        except KeyError as exc:
            raise KeyError(
                f"unknown integration {integration_id!r}; known={sorted(self._specs)}"
            ) from exc
        return self.status(spec.id).spec if resolved else spec

    def list_specs(
        self,
        *,
        category: Optional[str] = None,
        resolved: bool = True,
        deep: bool = False,
    ) -> list[IntegrationSpec]:
        specs = [
            item
            for item in self._specs.values()
            if category is None or item.category == category
        ]
        specs.sort(key=lambda item: (item.category, item.id))
        if not resolved:
            return specs
        return [self.status(item.id, deep=deep).spec for item in specs]

    def list_adapters(self, capability: Optional[str] = None) -> list[IntegrationAdapter]:
        if capability is None:
            return [self._adapters[key] for key in sorted(self._adapters)]
        return [
            self._adapters[adapter_id]
            for adapter_id in self._capability_adapters.get(str(capability), [])
        ]

    def adapter_for_integration(
        self,
        integration_id: str,
        *,
        capability: Optional[str] = None,
    ) -> Optional[IntegrationAdapter]:
        for adapter in self._adapters.values():
            if adapter.integration_id != integration_id:
                continue
            if capability is None or capability in adapter.capabilities:
                return adapter
        return None

    def get_capability(
        self,
        capability: str,
        *,
        integration_id: Optional[str] = None,
        require_available: bool = True,
    ) -> IntegrationAdapter:
        bindings = list(self._capability_adapters.get(str(capability), []))
        if integration_id is not None:
            bindings = [
                adapter_id
                for adapter_id in bindings
                if self._adapters[adapter_id].integration_id == integration_id
            ]
        if not bindings:
            suffix = f" for {integration_id!r}" if integration_id else ""
            raise KeyError(f"no adapter for capability {capability!r}{suffix}")
        for adapter_id in bindings:
            adapter = self._adapters[adapter_id]
            if not require_available:
                return adapter
            status = self.status(adapter.integration_id)
            if status.state == HealthState.AVAILABLE:
                return adapter
        raise RuntimeError(
            f"capability {capability!r} has adapters, but none are installed and policy-eligible"
        )

    def invoke_capability(
        self,
        capability: str,
        *,
        integration_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        adapter = self.get_capability(
            capability,
            integration_id=integration_id,
            require_available=True,
        )
        return adapter.invoke(capability, **kwargs)

    def status(self, integration_id: str, *, deep: bool = False) -> IntegrationStatus:
        spec = self.get_spec(integration_id, resolved=False)
        adapter = self.adapter_for_integration(spec.id)
        if (
            spec.maturity == IntegrationMaturity.BLOCKED
            or spec.blocked_reason is not None
        ):
            resolved = replace(
                spec,
                health_state=HealthState.BLOCKED,
                failure_reason=spec.blocked_reason or "blocked by catalog policy",
            )
            return IntegrationStatus(
                spec=resolved,
                installed=self._module_installed(spec.import_name),
                healthy=False,
                adapter_registered=adapter is not None,
                probe_performed=False,
            )

        installed = self._module_installed(spec.import_name)
        version = self._distribution_version(spec.package) if installed else None
        if not installed:
            resolved = replace(
                spec,
                version_detected=None,
                health_state=HealthState.MISSING,
                failure_reason=f"module {spec.import_name!r} is not installed",
            )
            return IntegrationStatus(
                spec=resolved,
                installed=False,
                healthy=False,
                adapter_registered=adapter is not None,
                probe_performed=False,
            )

        compatibility_failure = self._compatibility_failure(spec, version)
        if compatibility_failure is not None:
            resolved = replace(
                spec,
                version_detected=version,
                health_state=HealthState.INCOMPATIBLE,
                failure_reason=compatibility_failure,
            )
            return IntegrationStatus(
                spec=resolved,
                installed=True,
                healthy=False,
                adapter_registered=adapter is not None,
                probe_performed=False,
            )

        healthy: Optional[bool] = None
        detail: Optional[str] = None
        state = HealthState.AVAILABLE
        probe_performed = False
        if deep and adapter is not None:
            probe_performed = True
            try:
                result = adapter.probe()
                healthy = bool(result.healthy)
                detail = result.detail or None
                version = result.version or version
                if not healthy:
                    state = HealthState.UNHEALTHY
            except Exception as exc:
                healthy = False
                state = HealthState.UNHEALTHY
                detail = f"{type(exc).__name__}: {exc}"

        resolved = replace(
            spec,
            version_detected=version,
            health_state=state,
            failure_reason=detail if state != HealthState.AVAILABLE else None,
        )
        return IntegrationStatus(
            spec=resolved,
            installed=True,
            healthy=healthy,
            adapter_registered=adapter is not None,
            probe_performed=probe_performed,
        )

    def statuses(
        self,
        *,
        category: Optional[str] = None,
        deep: bool = False,
    ) -> list[IntegrationStatus]:
        specs = self.list_specs(category=category, resolved=False)
        return [self.status(item.id, deep=deep) for item in specs]

    @staticmethod
    def _module_installed(import_name: str) -> bool:
        try:
            return find_spec(import_name) is not None
        except (ImportError, AttributeError, ModuleNotFoundError, ValueError):
            return False
        except Exception:
            return False

    @staticmethod
    def _distribution_version(package: str) -> Optional[str]:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None
        except Exception:
            return None

    @staticmethod
    def _version_matches(version: str, constraint: str) -> Optional[bool]:
        if not constraint:
            return True
        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version

            return Version(str(version)) in SpecifierSet(str(constraint))
        except ImportError:
            return None
        except Exception:
            return False

    @classmethod
    def _compatibility_failure(
        cls,
        spec: IntegrationSpec,
        version: Optional[str],
    ) -> Optional[str]:
        python_version = ".".join(str(item) for item in sys.version_info[:3])
        python_match = cls._version_matches(
            python_version,
            spec.requires_python,
        )
        if python_match is False:
            return (
                f"Python {python_version} does not satisfy "
                f"{spec.requires_python}"
            )
        if spec.version_constraint and version is not None:
            package_match = cls._version_matches(
                version,
                spec.version_constraint,
            )
            if package_match is False:
                return (
                    f"{spec.package}=={version} does not satisfy "
                    f"{spec.version_constraint}"
                )
        return None

    def discover_plugins(self) -> list[PluginDescriptor]:
        """Inspect entry-point metadata without importing or executing plugins."""

        try:
            entry_points = metadata.entry_points()
            if hasattr(entry_points, "select"):
                selected = list(entry_points.select(group=ENTRY_POINT_GROUP))
            else:  # pragma: no cover - compatibility with old importlib.metadata
                selected = list(entry_points.get(ENTRY_POINT_GROUP, ()))
        except Exception:
            selected = []

        descriptors: list[PluginDescriptor] = []
        with self._lock:
            self._entry_points.clear()
            self._plugin_descriptors.clear()
            for entry_point in selected:
                distribution = None
                dist = getattr(entry_point, "dist", None)
                if dist is not None:
                    distribution = getattr(dist, "name", None)
                    if distribution is None:
                        try:
                            distribution = dist.metadata["Name"]
                        except Exception:
                            distribution = None
                descriptor = PluginDescriptor(
                    name=str(entry_point.name),
                    value=str(entry_point.value),
                    group=ENTRY_POINT_GROUP,
                    distribution=str(distribution) if distribution else None,
                )
                descriptors.append(descriptor)
                self._entry_points[descriptor.name] = entry_point
                self._plugin_descriptors[descriptor.name] = descriptor
        return sorted(descriptors, key=lambda item: item.name)

    def load_plugin(
        self,
        name: str,
        *,
        policy: PluginLoadPolicy,
        replace_existing: bool = False,
    ) -> IntegrationPlugin:
        """Load one explicitly trusted entry point.

        Merely discovering entry points never calls ``EntryPoint.load``.
        """

        if name not in self._entry_points:
            self.discover_plugins()
        descriptor = self._plugin_descriptors.get(name)
        entry_point = self._entry_points.get(name)
        if descriptor is None or entry_point is None:
            raise KeyError(f"unknown integration plugin entry point {name!r}")
        if not policy.approves(descriptor):
            raise PermissionError(
                f"plugin {name!r} was discovered but not approved for code loading"
            )
        loaded = entry_point.load()
        if not isinstance(loaded, IntegrationPlugin):
            raise TypeError(
                "trusted integration entry point must expose an IntegrationPlugin "
                "instance; factories are not invoked automatically"
            )
        self.register(
            loaded.spec,
            loaded.adapter,
            replace_existing=replace_existing,
        )
        return loaded

    def doctor(self, *, deep: bool = False) -> dict[str, Any]:
        statuses = self.statuses(deep=deep)
        counts = {state.value: 0 for state in HealthState}
        license_blocked = sum(
            1
            for item in statuses
            if item.spec.license_policy.value == "copyleft"
        )
        incompatible = 0
        for item in statuses:
            counts[item.state.value] += 1
            if item.state == HealthState.BLOCKED:
                if item.spec.license_policy.value != "copyleft":
                    incompatible += 1
        return {
            "schema": "AutoCausalIntegrationDoctor.v1",
            "python": sys.version.split()[0],
            "total": len(statuses),
            "installed": sum(1 for item in statuses if item.installed),
            "missing": counts[HealthState.MISSING.value],
            "available": counts[HealthState.AVAILABLE.value],
            "unhealthy": counts[HealthState.UNHEALTHY.value],
            "incompatible": counts[HealthState.INCOMPATIBLE.value] + incompatible,
            "blocked": counts[HealthState.BLOCKED.value],
            "license_blocked": license_blocked,
            "deep_probes": bool(deep),
            "telemetry_enabled": False,
            "notes": [
                "Status probes are lazy; deep probes may import only registered adapters.",
                "No integration telemetry or data egress is enabled by default.",
                "Package availability does not establish causal validity.",
            ],
        }


__all__ = ["ENTRY_POINT_GROUP", "IntegrationRegistry"]
