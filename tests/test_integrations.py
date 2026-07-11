from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autocausal.integrations import (
    CapabilityRouter,
    CapabilitySpec,
    IntegrationMaturity,
    IntegrationPlugin,
    IntegrationRegistry,
    IntegrationSpec,
    LicensePolicy,
    PluginLoadPolicy,
    ProbeResult,
    ResourceBudget,
    RoutingPolicy,
    build_default_registry,
    build_install_plan,
    integration_status,
    list_integrations,
    record_routing_decision,
)
from autocausal.integrations.adapters.ml import (
    BoostedEstimatorAdapter,
    SklearnAdapter,
)
from autocausal.integrations.adapters.native import NativeAdapter
from autocausal.integrations.registry import ENTRY_POINT_GROUP
from autocausal.integrations.types import HealthState


CAP = CapabilitySpec("test.capability", "test", production_ready=True)


def _subprocess_env() -> dict[str, str]:
    root = Path(__file__).parents[1]
    existing = os.environ.get("PYTHONPATH", "")
    value = str(root / "src")
    if existing:
        value = value + os.pathsep + existing
    return {**os.environ, "PYTHONPATH": value}


class FakeAdapter:
    def __init__(
        self,
        integration_id: str,
        *,
        adapter_id: str | None = None,
        healthy: bool = True,
        value: Any = "ok",
        capabilities: tuple[str, ...] = ("test.capability",),
    ) -> None:
        self.integration_id = integration_id
        self.id = adapter_id or f"{integration_id}.adapter"
        self.capabilities = capabilities
        self.healthy = healthy
        self.value = value
        self.invocations = 0

    def probe(self) -> ProbeResult:
        return ProbeResult(self.healthy, "fake probe", "1.2.3")

    def invoke(self, capability: str, **kwargs: Any) -> Any:
        self.invocations += 1
        return {"capability": capability, "kwargs": kwargs, "value": self.value}


def _spec(
    integration_id: str,
    *,
    import_name: str = "json",
    license_policy: LicensePolicy = LicensePolicy.PERMISSIVE,
    deterministic: bool = True,
) -> IntegrationSpec:
    capability = CapabilitySpec(
        "test.capability",
        "test",
        deterministic=deterministic,
        production_ready=True,
    )
    return IntegrationSpec(
        id=integration_id,
        category="test",
        package=integration_id,
        import_name=import_name,
        description="test",
        license="MIT" if license_policy == LicensePolicy.PERMISSIVE else "GPL",
        license_policy=license_policy,
        maturity=IntegrationMaturity.EXTERNAL,
        capabilities=(capability,),
    )


def test_catalog_is_broad_and_honest() -> None:
    specs = list_integrations()
    ids = {item.id for item in specs}
    required = {
        "numpy",
        "scipy",
        "statsmodels",
        "pingouin",
        "linearmodels",
        "arch",
        "pymc",
        "arviz",
        "lifelines",
        "scikit-posthocs",
        "patsy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "optuna",
        "imbalanced-learn",
        "shap",
        "joblib",
        "torch",
        "tensorflow",
        "mlflow",
        "nltk",
        "spacy",
        "transformers",
        "sentence-transformers",
        "gensim",
        "stanza",
        "textblob",
        "keybert",
        "faiss",
        "chromadb",
        "causal-learn",
        "dowhy",
        "econml",
        "doubleml",
        "lingam",
        "gcastle",
        "tigramite",
        "pgmpy",
        "causalml",
        "pysensemakr",
        "py-tetrad",
        "causalnex",
        "cdt",
        "pandera",
        "great-expectations",
        "ydata-profiling",
        "matplotlib",
        "seaborn",
        "plotly",
        "kaleido",
        "networkx",
        "graphviz",
        "polars",
        "pyarrow",
    }
    assert required <= ids
    assert len(specs) >= 55
    awareness = [item for item in specs if item.maturity == IntegrationMaturity.AWARENESS]
    assert awareness
    assert all(not item.capabilities for item in awareness)
    assert integration_status("causalnex").state == HealthState.BLOCKED
    assert integration_status("cdt").state == HealthState.BLOCKED


def test_default_registry_has_contract_adapter_for_every_claim() -> None:
    registry = build_default_registry()
    for spec in registry.list_specs(resolved=False):
        for capability in spec.capability_ids:
            assert registry.adapter_for_integration(
                spec.id,
                capability=capability,
            ), (spec.id, capability)


def test_integration_import_does_not_import_heavy_optional_packages() -> None:
    code = """
import sys
import autocausal.integrations
for name in (
    "xgboost", "lightgbm", "catboost", "optuna", "shap", "spacy",
    "sentence_transformers", "chromadb", "dowhy", "econml", "doubleml"
):
    assert name not in sys.modules, name
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        env=_subprocess_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_lazy_status_detection_and_deep_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    import autocausal.integrations.registry as registry_module

    monkeypatch.setattr(registry_module, "find_spec", lambda name: object())
    monkeypatch.setattr(registry_module.metadata, "version", lambda name: "9.9.9")
    registry = IntegrationRegistry()
    adapter = FakeAdapter("demo")
    registry.register(_spec("demo"), adapter)

    shallow = registry.status("demo")
    assert shallow.installed
    assert shallow.healthy is None
    assert shallow.spec.version_detected == "9.9.9"

    deep = registry.status("demo", deep=True)
    assert deep.healthy is True
    assert deep.probe_performed
    assert deep.spec.version_detected == "1.2.3"


def test_missing_package_status(monkeypatch: pytest.MonkeyPatch) -> None:
    import autocausal.integrations.registry as registry_module

    monkeypatch.setattr(registry_module, "find_spec", lambda name: None)
    registry = IntegrationRegistry([_spec("missing", import_name="not_real")])
    status = registry.status("missing")
    assert status.state == HealthState.MISSING
    assert not status.installed
    assert "not installed" in (status.spec.failure_reason or "")


def test_incompatible_python_or_package_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autocausal.integrations.registry as registry_module

    monkeypatch.setattr(registry_module, "find_spec", lambda name: object())
    monkeypatch.setattr(registry_module.metadata, "version", lambda name: "9.0")
    incompatible = replace(
        _spec("incompatible"),
        requires_python="<2",
        version_constraint="<1",
    )
    status = IntegrationRegistry([incompatible]).status("incompatible")
    assert status.state == HealthState.INCOMPATIBLE
    assert status.installed
    assert status.healthy is False
    assert "does not satisfy" in (status.spec.failure_reason or "")


def test_explicit_registration_and_invocation() -> None:
    registry = IntegrationRegistry()
    adapter = FakeAdapter("explicit", value=7)
    registry.register(_spec("explicit"), adapter)
    result = registry.invoke_capability(
        "test.capability",
        integration_id="explicit",
        answer=42,
    )
    assert result["value"] == 7
    assert result["kwargs"]["answer"] == 42
    assert adapter.invocations == 1


def test_entry_point_discovery_never_loads_without_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autocausal.integrations.registry as registry_module

    plugin_spec = _spec("plugin")
    plugin_adapter = FakeAdapter("plugin")
    loaded = {"count": 0}

    class Dist:
        name = "trusted-plugin"

    class EntryPoint:
        name = "demo"
        value = "demo:plugin"
        dist = Dist()

        def load(self) -> IntegrationPlugin:
            loaded["count"] += 1
            return IntegrationPlugin(plugin_spec, plugin_adapter)

    class EntryPoints(list):
        def select(self, **kwargs: Any) -> list[Any]:
            assert kwargs == {"group": ENTRY_POINT_GROUP}
            return list(self)

    monkeypatch.setattr(
        registry_module.metadata,
        "entry_points",
        lambda: EntryPoints([EntryPoint()]),
    )
    registry = IntegrationRegistry()
    descriptors = registry.discover_plugins()
    assert descriptors[0].distribution == "trusted-plugin"
    assert loaded["count"] == 0
    with pytest.raises(PermissionError):
        registry.load_plugin("demo", policy=PluginLoadPolicy())
    assert loaded["count"] == 0

    plugin = registry.load_plugin(
        "demo",
        policy=PluginLoadPolicy(
            allow_entry_point_loading=True,
            trusted_distributions=frozenset({"trusted-plugin"}),
        ),
    )
    assert plugin.spec.id == "plugin"
    assert loaded["count"] == 1
    assert registry.get_spec("plugin").id == "plugin"


def test_router_skips_missing_and_license_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autocausal.integrations.registry as registry_module

    monkeypatch.setattr(
        registry_module,
        "find_spec",
        lambda name: object() if name == "json" else None,
    )
    registry = IntegrationRegistry()
    registry.register(
        _spec(
            "copyleft",
            license_policy=LicensePolicy.COPYLEFT,
        ),
        FakeAdapter("copyleft"),
    )
    registry.register(
        _spec("missing", import_name="definitely_missing"),
        FakeAdapter("missing"),
    )
    registry.register(_spec("native"), FakeAdapter("native"))
    router = CapabilityRouter(
        registry,
        routes={"test.capability": ("copyleft", "missing", "native")},
    )
    decision = router.route("test.capability")
    assert decision.selected_integration == "native"
    assert "copyleft license is not allowed" in decision.candidates[0].reasons
    assert "package is not installed" in decision.candidates[1].reasons


def test_router_honors_determinism_and_resource_budget() -> None:
    registry = IntegrationRegistry()
    registry.register(
        _spec("stochastic", deterministic=False),
        FakeAdapter("stochastic"),
    )
    registry.register(
        _spec("deterministic", deterministic=True),
        FakeAdapter("deterministic"),
    )
    router = CapabilityRouter(
        registry,
        routes={"test.capability": ("stochastic", "deterministic")},
    )
    decision = router.route(
        "test.capability",
        policy=RoutingPolicy(require_deterministic=True),
    )
    assert decision.selected_integration == "deterministic"
    assert any(
        "not declared deterministic" in reason
        for reason in decision.candidates[0].reasons
    )

    constrained = router.route(
        "test.capability",
        budget=ResourceBudget(max_rows=10),
        n_rows=11,
    )
    assert constrained.selected_integration is None
    assert constrained.escalation


def test_causal_decision_never_claims_validity() -> None:
    registry = build_default_registry()
    decision = CapabilityRouter(registry).route(
        "causal.estimate.ate",
        policy=RoutingPolicy(production=True),
        context={"design": "dml", "design_validated": False},
    )
    assert any("does not establish causal validity" in item for item in decision.caveats)
    if decision.selected:
        assert decision.escalation


def test_all_safe_plan_excludes_gpl_cuda_java_r_and_blocked() -> None:
    plan = build_install_plan("all-safe", hardware="cpu")
    packages = {item.lower() for item in plan.packages}
    assert "tigramite" not in packages
    assert "pingouin" not in packages
    assert "gensim" not in packages
    assert "py-tetrad" not in packages
    assert "cdt" not in packages
    assert "causalnex" not in packages
    assert "tensorflow" not in packages
    assert any("CPU plan" in item for item in plan.excluded)
    assert plan.command and plan.command.startswith("python -m pip install")


def test_research_copyleft_requires_explicit_policy() -> None:
    default = build_install_plan("research")
    assert "tigramite" not in default.packages
    approved = build_install_plan(
        "research",
        policy={"allow_copyleft": True},
    )
    assert {"pingouin", "gensim", "tigramite"} <= set(approved.packages)


def test_native_fallbacks_are_deterministic() -> None:
    adapter = NativeAdapter()
    control = np.arange(50, dtype=float)
    x = control + np.linspace(0.0, 1.0, 50)
    y = 2.0 * control + np.linspace(1.0, 0.0, 50)
    first = adapter.invoke(
        "stats.partial_correlation",
        x=x,
        y=y,
        controls=control,
    )
    second = adapter.invoke(
        "stats.partial_correlation",
        x=x,
        y=y,
        controls=control,
    )
    assert first == second

    texts = ["alpha beta", "beta gamma"]
    embeddings_a = adapter.invoke("nlp.embeddings", texts=texts)["embeddings"]
    embeddings_b = adapter.invoke("nlp.embeddings", texts=texts)["embeddings"]
    np.testing.assert_allclose(embeddings_a, embeddings_b)


def test_sklearn_adapter_contract() -> None:
    pytest.importorskip("sklearn")
    adapter = SklearnAdapter()
    classifier = adapter.invoke(
        "ml.tabular_classifier",
        model="logistic",
        random_state=3,
    )
    assert classifier.random_state == 3
    result = adapter.invoke(
        "nlp.embeddings",
        texts=["causal graph", "predictive model"],
        max_features=32,
    )
    assert result["shape"][0] == 2
    assert result["data_egress"] is False


def test_boosted_adapter_can_be_contract_tested_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Estimator:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    class Module:
        XGBClassifier = Estimator
        XGBRegressor = Estimator

    adapter = BoostedEstimatorAdapter("xgboost", "xgboost", "xgboost")
    monkeypatch.setattr(adapter, "_module", lambda: Module())
    estimator = adapter.invoke(
        "ml.tabular_classifier",
        n_estimators=50,
        n_jobs=2,
    )
    assert isinstance(estimator, Estimator)
    assert captured["device"] == "cpu"
    assert captured["n_estimators"] == 50
    with pytest.raises(ValueError):
        adapter.invoke("ml.tabular_classifier", n_estimators=50_000)


def test_manifest_records_versions_and_decision() -> None:
    registry = IntegrationRegistry()
    registry.register(_spec("selected"), FakeAdapter("selected"))
    decision = CapabilityRouter(
        registry,
        routes={"test.capability": ("selected",)},
    ).route("test.capability")
    manifest: dict[str, Any] = {"config": {}, "engine_versions": {}}
    record_routing_decision(manifest, decision)
    section = manifest["config"]["integrations"]
    assert section["telemetry_enabled"] is False
    assert section["routing_decisions"][0]["selected_integration"] == "selected"
    assert "raw" not in json.dumps(section).lower()


def test_cli_and_mcp_surface(capsys: pytest.CaptureFixture[str]) -> None:
    from autocausal.cli import main
    from autocausal.mcp.registry import build_default_registry as build_mcp_registry

    assert main(["integrations", "plan", "--profile", "all-safe", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["profile"] == "all-safe"
    tools = build_mcp_registry()
    for name in (
        "autocausal_list_integrations",
        "autocausal_integration_status",
        "autocausal_route_capability",
    ):
        assert name in tools.list_names()
    routed = tools.invoke(
        "autocausal_route_capability",
        {"capability": "viz.dag"},
    )
    assert routed["ok"]
    assert routed["invoked"] is False


def test_core_import_regression() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import autocausal; print(autocausal.__version__)"],
        cwd=Path(__file__).parents[1],
        env=_subprocess_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
