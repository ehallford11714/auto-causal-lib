"""Contract/integration coverage for the 0.14 production-readiness release."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from autocausal import (
    AutoCausal,
    DiscoveryResult,
    EvidenceGateError,
    EvidenceGrade,
    ProductionGateError,
    ProductionPolicy,
    ResourceLimitError,
    RunManifest,
    RunPolicy,
    UnsafePayloadError,
    load_dataset,
)
from autocausal.production import build_data_fingerprint


def test_public_policy_contract_and_roundtrip():
    policy = ProductionPolicy(
        required_engines=("builtin_ols",),
        max_rows=123,
        max_columns=17,
        max_rounds=2,
        random_state=42,
    )
    assert RunPolicy is ProductionPolicy
    restored = ProductionPolicy.from_json(policy.to_json())
    assert restored.to_dict() == policy.to_dict()
    assert restored.qc == "block"
    assert restored.stability is True
    assert restored.ensemble is True
    assert restored.allow_synthetic_iv is False
    assert restored.allow_slm is True


def test_manifest_roundtrip_and_fingerprint_are_private_deterministic():
    df = pd.DataFrame(
        {
            "email": ["a@example.test", "b@example.test"],
            "value": [123.456, 987.654],
        }
    )
    first = build_data_fingerprint(df)
    second = build_data_fingerprint(df.copy())
    assert first == second
    assert first["contains_raw_values"] is False
    assert "a@example.test" not in str(first)

    ac = AutoCausal.from_dataframe(df, random_state=11)
    manifest = RunManifest.from_json(ac.run_manifest.to_json())
    assert manifest.to_dict() == ac.run_manifest.to_dict()
    blob = manifest.to_json()
    assert "a@example.test" not in blob
    assert "123.456" not in blob


def test_production_iv_demo_integration_provenance_and_estimate_refute():
    df = load_dataset("iv_demo", allow_network=False)
    policy = ProductionPolicy(
        bootstrap_n=5,
        min_stability=0.30,
        random_state=7,
    )
    ac = AutoCausal.from_dataframe(
        df,
        source="dataset:iv_demo",
        mode="production",
        policy=policy,
    )
    result = ac.discover(qc="block")
    assert result.mode == "production"
    assert result.run_id
    assert result.manifest.status == "ok"
    assert result.manifest.random_state == 7
    assert result.manifest.events

    iv_edges = [edge for edge in result.edges if edge.get("type") == "iv_2sls"]
    assert iv_edges
    iv_edge = iv_edges[0]
    assert iv_edge["instrument"] == "z"
    assert iv_edge["evidence_grade"] == EvidenceGrade.SUPPORTED.value
    assert iv_edge["identification"] == "unverified"
    provenance = iv_edge["provenance"]
    assert provenance["instrument_origin"] == "observed"
    assert provenance["run_id"] == result.run_id
    assert provenance["package_version"] == "0.14.2"
    assert set(("z", "treatment", "outcome")) <= set(provenance["source_columns"])

    estimate = ac.estimate(
        backend="builtin_2sls",
        y="outcome",
        d="treatment",
        z="z",
    )
    assert estimate.ok and not estimate.soft_skip
    refutation = ac.refute(
        edge=iv_edge,
        method="placebo",
        y="outcome",
        d="treatment",
    )
    assert refutation.ok and not refutation.soft_skip
    assert iv_edge["provenance"]["estimator"]
    assert iv_edge["provenance"]["refuters"]

    report = result.report()
    assert "EPISTEMIC" in report
    assert "Evidence gates" in report
    assert "IV (only if real Z)" in report
    assert "Reproducibility and privacy" in report


def test_exploratory_iris_replay_is_deterministic_and_serializable():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df, random_state=19)
    result = ac.discover(
        use_iv=False,
        qc="warn",
        ensemble=True,
        stability=True,
        bootstrap_n=4,
    )
    restored = DiscoveryResult.from_json(result.to_json())
    assert restored.run_id == result.run_id
    assert restored.policy == result.policy
    assert restored.manifest.to_dict() == result.manifest.to_dict()

    replay = result.reproduce(df)
    edge_key = lambda edge: (
        edge.get("source"),
        edge.get("target"),
        edge.get("type"),
        edge.get("score"),
        edge.get("stability"),
    )
    assert [edge_key(edge) for edge in replay.edges] == [
        edge_key(edge) for edge in result.edges
    ]
    assert replay.run_id != result.run_id
    assert replay.manifest.random_state == 19


def test_required_engine_and_resource_limits_fail_closed():
    df = load_dataset("iris", allow_network=False)
    missing_policy = ProductionPolicy(required_engines=("not_a_real_engine",))
    ac = AutoCausal.from_dataframe(df, mode="production", policy=missing_policy)
    with pytest.raises(ProductionGateError) as exc:
        ac.discover(use_iv=False)
    assert exc.value.code == "required_engine_missing"
    assert exc.value.to_dict()["recommendations"]

    limited = AutoCausal.from_dataframe(
        df,
        mode="production",
        policy=ProductionPolicy(max_rows=10),
    )
    with pytest.raises(ResourceLimitError) as limit_exc:
        limited.discover(use_iv=False)
    assert limit_exc.value.code == "shape_limit_exceeded"
    assert limit_exc.value.manifest is not None
    assert limit_exc.value.manifest.status == "aborted"


def test_privacy_external_payload_and_slm_gates():
    n = 30
    df = pd.DataFrame(
        {
            "email": [f"person-{idx}@example.test" for idx in range(n)],
            "treatment": np.tile([0, 1], n // 2),
            "outcome": np.linspace(0.0, 1.0, n),
        }
    )
    ac = AutoCausal.from_dataframe(df, mode="production")
    summary = ac.external_payload()
    assert summary["contains_raw_values"] is False
    assert "person-0@example.test" not in str(summary)
    with pytest.raises(UnsafePayloadError):
        ac.external_payload(include_frame=True)
    # SLM guides by default; unstructured model output soft-falls to rules.
    guided = ac.guide(use_slm=True, text="what drives outcome?")
    assert guided is not None
    assert ac.policy.allow_slm is True

    strict_privacy = AutoCausal.from_dataframe(
        df,
        mode="production",
        policy=ProductionPolicy(fail_on_pii=True),
    )
    with pytest.raises(ProductionGateError) as exc:
        strict_privacy.discover(use_iv=False)
    assert exc.value.code in ("qc_blocked", "pii_gate_failed")


def test_production_rejects_unsafe_overrides_and_synthetic_iv():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal.from_dataframe(df, mode="production")
    with pytest.raises(ProductionGateError, match="qc='off'"):
        ac.discover(qc="off")
    with pytest.raises(EvidenceGateError):
        ac.discover(auto_instrument=True)


def test_weak_observed_iv_is_filtered_and_fill_values_are_redacted():
    rng = np.random.default_rng(101)
    n = 180
    treatment = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "z": rng.normal(size=n),  # intentionally irrelevant instrument
            "treatment": treatment,
            "outcome": treatment + rng.normal(scale=0.2, size=n),
            "control": rng.normal(size=n),
        }
    )
    df.loc[0, "control"] = np.nan
    ac = AutoCausal.from_dataframe(
        df,
        mode="production",
        policy=ProductionPolicy(bootstrap_n=4, min_stability=0.25),
    )
    result = ac.discover(
        candidates={
            "instrument": ["z"],
            "treatment": ["treatment"],
            "outcome": ["outcome"],
        }
    )
    assert not [
        edge for edge in result.edges if edge.get("type") == "iv_2sls"
    ]
    rejected_iv = [
        edge for edge in result.rejected_edges if edge.get("type") == "iv_2sls"
    ]
    assert rejected_iv
    assert rejected_iv[0]["evidence_grade"] == EvidenceGrade.INSUFFICIENT.value
    assert "instrument_strength" in rejected_iv[0]["failed_gates"]

    serialized = result.to_json()
    assert '"fill_value": "<redacted>"' in serialized


def test_default_api_contract_has_safe_auto_instrument():
    signature = inspect.signature(AutoCausal.discover)
    assert signature.parameters["auto_instrument"].default is False
    exploratory = ProductionPolicy.exploratory()
    assert exploratory.allow_synthetic_iv is True
    assert exploratory.fallback_behavior == "warn"


def test_doctor_production_cli_contract(capsys):
    from autocausal.cli import main
    from autocausal.doctor import doctor_report

    report = doctor_report(
        production=True,
        policy=ProductionPolicy(required_engines=("builtin_ols",)),
    )
    assert report["production_ok"] is True
    checks = {
        item["id"]: item
        for item in report["production_checklist"]["checks"]
    }
    assert checks["policy_serialization"]["ok"] is True
    assert checks["policy_resource_limits"]["ok"] is True
    assert checks["required_engine_builtin_ols"]["ok"] is True

    exit_code = main(["doctor", "--production", "--json"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"production_checklist"' in output
    assert '"default_auto_instrument_false"' in output
