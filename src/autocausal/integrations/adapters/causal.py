"""Bridges from integration capabilities to existing causal engines."""

from __future__ import annotations

from typing import Any

from autocausal.integrations.adapters.base import CAUSAL_CAVEAT, LazyAdapter


class ExistingCausalAdapter(LazyAdapter):
    def __init__(
        self,
        *,
        integration_id: str,
        module_name: str,
        package_name: str,
        capabilities: tuple[str, ...],
        default_method: str,
    ) -> None:
        self.integration_id = integration_id
        self.module_name = module_name
        self.package_name = package_name
        self.capabilities = capabilities
        self.default_method = default_method
        self.id = f"{integration_id}.autocausal-engine"

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        if capability not in self.capabilities:
            raise KeyError(capability)
        if capability in ("causal.discovery.pc", "causal.discovery"):
            return self.discover(**kwargs)
        if capability == "causal.estimate.ate":
            return self.estimate(**kwargs)
        if capability == "causal.refute":
            return self.refute(**kwargs)
        raise KeyError(capability)

    def discover(self, *, frame: Any, **kwargs: Any) -> dict[str, Any]:
        from autocausal.engines import discover_with

        allowed: dict[str, set[str]] = {
            "causal-learn": {
                "causal_learn_pc",
                "causal_learn_ges",
                "causal_learn_fci",
            },
            "lingam": {"lingam", "direct_lingam"},
            "gcastle": {"gcastle_notears", "notears", "gcastle"},
        }
        method = str(kwargs.pop("method", self.default_method))
        if method not in allowed.get(self.integration_id, {self.default_method}):
            raise ValueError(
                f"method {method!r} is not provided by {self.integration_id}"
            )
        result = discover_with(frame, method=method, **kwargs)
        result.setdefault("notes", []).append(CAUSAL_CAVEAT)
        return result

    def estimate(
        self,
        *,
        frame: Any,
        outcome: str | None = None,
        treatment: str | None = None,
        controls: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from autocausal.engines import estimate

        requested = str(kwargs.pop("method", self.default_method))
        if self.integration_id == "doubleml":
            backend = "doubleml"
        elif self.integration_id == "econml":
            if requested in ("causal_forest", "econml_causal_forest"):
                backend = "econml_causal_forest"
            else:
                backend = "econml"
        else:
            raise RuntimeError(f"{self.integration_id} is not an estimate engine")
        result = estimate(
            frame,
            backend=backend,
            y=outcome or kwargs.pop("y", None),
            d=treatment or kwargs.pop("d", None),
            x=controls or kwargs.pop("x", None),
            **kwargs,
        )
        output = result.to_dict()
        output.setdefault("notes", []).append(CAUSAL_CAVEAT)
        return output

    def refute(self, *, edge: Any = None, frame: Any = None, **kwargs: Any) -> Any:
        from autocausal.engines import refute

        method = str(kwargs.pop("method", self.default_method))
        if not method.startswith("dowhy"):
            method = "dowhy"
        result = refute(edge, method=method, df=frame, **kwargs)
        if hasattr(result, "to_dict"):
            output = result.to_dict()
            output.setdefault("notes", []).append(CAUSAL_CAVEAT)
            return output
        return result


def causal_adapters() -> tuple[LazyAdapter, ...]:
    return (
        ExistingCausalAdapter(
            integration_id="causal-learn",
            module_name="causallearn",
            package_name="causal-learn",
            capabilities=("causal.discovery.pc", "causal.discovery"),
            default_method="causal_learn_pc",
        ),
        ExistingCausalAdapter(
            integration_id="lingam",
            module_name="lingam",
            package_name="lingam",
            capabilities=("causal.discovery",),
            default_method="lingam",
        ),
        ExistingCausalAdapter(
            integration_id="gcastle",
            module_name="castle",
            package_name="gcastle",
            capabilities=("causal.discovery",),
            default_method="gcastle_notears",
        ),
        ExistingCausalAdapter(
            integration_id="doubleml",
            module_name="doubleml",
            package_name="DoubleML",
            capabilities=("causal.estimate.ate",),
            default_method="doubleml",
        ),
        ExistingCausalAdapter(
            integration_id="econml",
            module_name="econml",
            package_name="econml",
            capabilities=("causal.estimate.ate",),
            default_method="econml",
        ),
        ExistingCausalAdapter(
            integration_id="dowhy",
            module_name="dowhy",
            package_name="dowhy",
            capabilities=("causal.refute",),
            default_method="dowhy",
        ),
    )


__all__ = ["ExistingCausalAdapter", "causal_adapters"]
