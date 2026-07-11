"""Policy- and resource-aware capability routing."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from autocausal.integrations.registry import IntegrationRegistry
from autocausal.integrations.types import (
    HealthState,
    LicensePolicy,
    ResourceBudget,
    RoutingCandidate,
    RoutingDecision,
    RoutingPolicy,
    RuntimeRequirement,
    coerce_routing_policy,
)


DEFAULT_ROUTES: dict[str, tuple[str, ...]] = {
    "stats.test": ("scipy",),
    "stats.partial_correlation": (
        "statsmodels",
        "scipy",
        "autocausal-native",
    ),
    "stats.robust_covariance": ("statsmodels",),
    "ml.preprocessing": ("scikit-learn",),
    "ml.tabular_classifier": (
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "autocausal-native",
    ),
    "ml.tabular_regressor": (
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "autocausal-native",
    ),
    "ml.estimator_factory": ("xgboost", "lightgbm", "catboost"),
    "ml.cross_validation": ("scikit-learn",),
    "ml.metrics": ("scikit-learn",),
    "ml.tune": ("optuna",),
    "ml.resampling_pipeline": ("imbalanced-learn",),
    "ml.explain": ("shap",),
    "nlp.entities": ("spacy",),
    "nlp.embeddings": (
        "sentence-transformers",
        "scikit-learn",
        "autocausal-native",
    ),
    "nlp.embeddings.tfidf": ("scikit-learn", "autocausal-native"),
    "nlp.vector_search": ("faiss", "chromadb", "autocausal-native"),
    "causal.discovery.pc": ("causal-learn", "autocausal-native"),
    "causal.discovery": ("causal-learn", "lingam", "gcastle"),
    "causal.estimate.ate": ("doubleml", "econml", "autocausal-native"),
    "causal.refute": ("dowhy",),
    "viz.chart": ("plotly", "matplotlib", "autocausal-native"),
    "viz.dag": ("networkx", "plotly", "autocausal-native"),
    "viz.export": ("kaleido",),
    "data.validate": ("pandera",),
    "data.convert": ("polars", "pyarrow"),
    "ops.mlflow.log_manifest": ("mlflow",),
}


class CapabilityRouter:
    """Select adapters without importing optional packages during routing."""

    def __init__(
        self,
        registry: IntegrationRegistry,
        *,
        routes: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.registry = registry
        self.routes = {
            key: tuple(value) for key, value in (routes or DEFAULT_ROUTES).items()
        }

    def _ordered_integrations(
        self,
        capability: str,
        *,
        explicit: Optional[str],
        context: Mapping[str, Any],
    ) -> tuple[str, ...]:
        if explicit:
            return (explicit,)
        configured = self.routes.get(capability)
        if configured is None:
            configured = tuple(
                spec.id
                for spec in self.registry.list_specs(resolved=False)
                if capability in spec.capability_ids
            )
        order = list(configured)
        if capability == "causal.estimate.ate":
            design = str(context.get("design") or "").lower()
            if design in ("cate", "heterogeneous", "causal_forest"):
                order = ["econml", "doubleml", "autocausal-native"]
            elif design in ("binary_aipw", "aipw"):
                order = ["autocausal-native", "doubleml", "econml"]
        return tuple(dict.fromkeys(order))

    def route(
        self,
        capability: str,
        *,
        policy: Optional[RoutingPolicy | Mapping[str, Any]] = None,
        budget: Optional[ResourceBudget] = None,
        data_type: Optional[str] = None,
        n_rows: Optional[int] = None,
        estimated_memory_mb: Optional[int] = None,
        deep_health: bool = False,
        context: Optional[Mapping[str, Any]] = None,
    ) -> RoutingDecision:
        resolved_policy = coerce_routing_policy(policy)
        resolved_budget = budget or ResourceBudget()
        route_context = dict(context or {})
        explicit = resolved_policy.explicit_integration
        order = self._ordered_integrations(
            capability,
            explicit=explicit,
            context=route_context,
        )
        candidates: list[RoutingCandidate] = []
        versions: dict[str, Optional[str]] = {}
        global_reasons: list[str] = []

        if n_rows is not None and int(n_rows) > resolved_budget.max_rows:
            global_reasons.append(
                f"n_rows={int(n_rows)} exceeds budget.max_rows={resolved_budget.max_rows}"
            )
        if (
            estimated_memory_mb is not None
            and int(estimated_memory_mb) > resolved_budget.max_memory_mb
        ):
            global_reasons.append(
                "estimated memory exceeds max_memory_mb="
                f"{resolved_budget.max_memory_mb}"
            )
        estimated_seconds = route_context.get("estimated_seconds")
        if (
            estimated_seconds is not None
            and float(estimated_seconds) > resolved_budget.max_seconds
        ):
            global_reasons.append(
                "estimated runtime exceeds max_seconds="
                f"{resolved_budget.max_seconds}"
            )

        for priority, integration_id in enumerate(order):
            reasons = list(global_reasons)
            try:
                spec = self.registry.get_spec(integration_id, resolved=False)
            except KeyError:
                candidates.append(
                    RoutingCandidate(
                        integration_id=integration_id,
                        adapter_id=None,
                        installed=False,
                        healthy=False,
                        eligible=False,
                        score=0,
                        reasons=("not registered",),
                    )
                )
                continue
            adapter = self.registry.adapter_for_integration(
                integration_id,
                capability=capability,
            )
            status = self.registry.status(integration_id, deep=deep_health)
            versions[integration_id] = status.spec.version_detected

            if adapter is None:
                reasons.append("no callable adapter for requested capability")
            if status.state == HealthState.MISSING:
                reasons.append("package is not installed")
            elif status.state == HealthState.BLOCKED:
                reasons.append(
                    status.spec.failure_reason or "integration is policy-blocked"
                )
            elif status.state in (HealthState.UNHEALTHY, HealthState.INCOMPATIBLE):
                reasons.append(
                    status.spec.failure_reason or f"health state={status.state.value}"
                )

            license_policy = spec.license_policy
            if (
                license_policy == LicensePolicy.UNKNOWN
                and not resolved_policy.allow_unknown_license
                and license_policy not in resolved_policy.allowed_licenses
            ):
                reasons.append("unknown license is not allowed")
            elif license_policy not in resolved_policy.allowed_licenses:
                reasons.append(f"{license_policy.value} license is not allowed")

            runtime = set(spec.required_runtime)
            hardware = str(resolved_budget.hardware).lower()
            if RuntimeRequirement.GPU in runtime and hardware != "gpu":
                reasons.append(
                    f"GPU runtime is incompatible with hardware={hardware!r}"
                )
            if RuntimeRequirement.GPU in runtime and not resolved_policy.allow_gpu:
                reasons.append("GPU runtime is not allowed")
            if (
                RuntimeRequirement.NATIVE in runtime
                and not resolved_policy.allow_native_runtime
            ):
                reasons.append("native runtime is not allowed")
            if RuntimeRequirement.JAVA in runtime and not resolved_policy.allow_java:
                reasons.append("Java runtime is not allowed")
            if RuntimeRequirement.R in runtime and not resolved_policy.allow_r:
                reasons.append("R runtime is not allowed")

            security = spec.security
            if security.network_required and not resolved_policy.allow_network:
                reasons.append("network access is required but not allowed")
            if security.data_egress and not resolved_policy.allow_data_egress:
                reasons.append("integration can egress data and egress is not allowed")
            if (
                security.executes_external_code
                and not resolved_policy.allow_external_code
            ):
                reasons.append("external code execution is not allowed")

            capability_spec = next(
                (
                    item
                    for item in spec.capabilities
                    if item.id == capability
                ),
                None,
            )
            if (
                resolved_policy.require_deterministic
                and capability_spec is not None
                and not capability_spec.deterministic
            ):
                reasons.append("capability is not declared deterministic")
            if (
                resolved_policy.require_production_ready
                and capability_spec is not None
                and not capability_spec.production_ready
            ):
                reasons.append("capability is not declared production-ready")
            if data_type and capability.startswith("ml.") and data_type == "text":
                if integration_id in ("xgboost", "lightgbm", "catboost"):
                    reasons.append("boosted tabular adapter does not accept raw text")

            eligible = (
                adapter is not None
                and status.state == HealthState.AVAILABLE
                and not reasons
            )
            score = (len(order) - priority) * 100 if eligible else 0
            if integration_id == "autocausal-native" and eligible:
                score -= 10
            candidates.append(
                RoutingCandidate(
                    integration_id=integration_id,
                    adapter_id=adapter.id if adapter is not None else None,
                    installed=status.installed,
                    healthy=status.healthy,
                    eligible=eligible,
                    score=score,
                    reasons=tuple(reasons or ("eligible",)),
                    version=status.spec.version_detected,
                    fallback=spec.deterministic_fallback,
                )
            )

        ranked = sorted(
            (item for item in candidates if item.eligible),
            key=lambda item: item.score,
            reverse=True,
        )
        selected = ranked[0] if ranked else None
        fallback = None
        if len(ranked) > 1:
            fallback = ranked[1].integration_id
        elif selected is not None:
            fallback = selected.fallback

        caveats: list[str] = []
        if selected is not None:
            selected_spec = self.registry.get_spec(selected.integration_id)
            selected_capability = next(
                (
                    item
                    for item in selected_spec.capabilities
                    if item.id == capability
                ),
                None,
            )
            if selected_capability is not None:
                caveats.extend(selected_capability.caveats)
        if capability.startswith("causal."):
            caveats.append(
                "Model/package availability does not establish causal validity or identification."
            )
        if resolved_policy.production:
            caveats.append(
                "Production routing records availability and policy fit; evidence gates still apply."
            )

        escalation = None
        if selected is None:
            escalation = (
                "No installed adapter satisfies policy and resource constraints; "
                "review the install plan or explicitly revise policy."
            )
        elif (
            capability.startswith("causal.")
            and resolved_policy.production
            and not bool(route_context.get("design_validated"))
        ):
            escalation = (
                "A causal adapter is available, but production use requires an "
                "independently validated design and evidence review."
            )

        return RoutingDecision(
            capability=capability,
            candidates=tuple(candidates),
            selected_integration=(
                selected.integration_id if selected is not None else None
            ),
            selected_adapter=selected.adapter_id if selected is not None else None,
            fallback=fallback,
            escalation=escalation,
            policy_reasons=tuple(global_reasons),
            versions=versions,
            caveats=tuple(dict.fromkeys(caveats)),
        )

    def invoke(
        self,
        capability: str,
        *,
        policy: Optional[RoutingPolicy | Mapping[str, Any]] = None,
        budget: Optional[ResourceBudget] = None,
        manifest: Any = None,
        return_decision: bool = False,
        routing_context: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        frame = kwargs.get("frame")
        inferred_rows = None
        if frame is not None:
            try:
                inferred_rows = len(frame)
            except Exception:
                inferred_rows = None
        decision = self.route(
            capability,
            policy=policy,
            budget=budget,
            n_rows=inferred_rows,
            context=routing_context,
        )
        if manifest is not None:
            from autocausal.integrations.manifest import record_routing_decision

            record_routing_decision(manifest, decision)
        if not decision.selected_integration:
            raise RuntimeError(
                decision.escalation
                or f"no eligible adapter for capability {capability!r}"
            )
        adapter = self.registry.get_capability(
            capability,
            integration_id=decision.selected_integration,
        )
        result = adapter.invoke(capability, **kwargs)
        return (result, decision) if return_decision else result


__all__ = ["CapabilityRouter", "DEFAULT_ROUTES"]
