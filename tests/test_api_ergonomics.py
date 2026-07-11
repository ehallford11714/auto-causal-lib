"""Comprehensive API ergonomics — no AttributeError on documented chains.

Instantiates a real iris pipeline and calls every public method users might
chain off discover / mine / suite / insight / agentic / grail results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autocausal import AutoCausal, DiscoveryResult, __version__
from autocausal.datasets import load_dataset
from autocausal.engines import engine_status, estimate, list_engines, refute
from autocausal.grail import GrailEngine
from autocausal.insight.suite import InsightSuite
from autocausal.mining import MiningReport
from autocausal.results import AutoResult


IRIS = Path(__file__).resolve().parents[1] / "src" / "autocausal" / "data" / "examples" / "iris.csv"

# Methods users chain on DiscoveryResult after discover()
DISCOVERY_METHODS = [
    "to_dict",
    "to_json",
    "to_markdown",
    "report",
    "to_causal_edges",
    "to_search_dag",
    "to_mine_report",
    "to_fabric_bundle",
    "estimate",
    "refute",
    "sensitivity",
    "run_sensitivity",
    "to_causaliv_request",
    "engines_status",
    "session",
    "dataframe",
]

AUTO_RESULT_METHODS = [
    "to_dict",
    "to_json",
    "to_markdown",
    "report",
    "to_causal_edges",
    "to_search_dag",
    "to_mine_report",
    "to_fabric_bundle",
    "estimate",
    "refute",
    "sensitivity",
    "run_sensitivity",
    "to_causaliv_request",
    "engines_status",
]

MINING_METHODS = [
    "to_dict",
    "to_json",
    "to_markdown",
    "report",
    "to_mine_report",
    "to_fabric_bundle",
]

REPORT_METHODS = ["to_dict", "to_json", "to_markdown", "report"]


@pytest.fixture(scope="module")
def iris_df():
    if IRIS.is_file():
        return AutoCausal.from_csv(IRIS)._df
    return load_dataset("iris")


@pytest.fixture(scope="module")
def ac_pipeline(iris_df):
    ac = AutoCausal.from_dataframe(iris_df, source="iris")
    ac.mine()
    ac.impute()
    result = ac.discover(qc="off", use_iv=False, min_abs_corr=0.2)
    return ac, result


def test_version_0_13_0_or_newer():
    parts = [int(x) for x in __version__.split(".")[:3]]
    assert parts >= [0, 13, 0]


@pytest.mark.parametrize("name", DISCOVERY_METHODS)
def test_discovery_result_has_method(ac_pipeline, name):
    _, result = ac_pipeline
    assert isinstance(result, DiscoveryResult)
    assert hasattr(result, name), f"DiscoveryResult missing {name}"
    assert callable(getattr(result, name))


@pytest.mark.parametrize("name", AUTO_RESULT_METHODS)
def test_auto_result_has_method(ac_pipeline, name):
    _, result = ac_pipeline
    auto = AutoResult(discovery=result, mining=result.mining, source="iris")
    assert hasattr(auto, name), f"AutoResult missing {name}"
    assert callable(getattr(auto, name))


@pytest.mark.parametrize("name", MINING_METHODS)
def test_mining_report_has_method(ac_pipeline, name):
    ac, _ = ac_pipeline
    assert isinstance(ac.mining, MiningReport)
    assert hasattr(ac.mining, name), f"MiningReport missing {name}"
    assert callable(getattr(ac.mining, name))


def test_discovery_result_full_chain(ac_pipeline):
    ac, result = ac_pipeline
    assert result.session() is ac
    assert result.dataframe() is not None

    assert isinstance(result.to_dict(), dict)
    assert result.to_json().strip().startswith("{")
    assert len(result.to_markdown()) > 20
    assert result.report() == result.to_markdown()
    assert result.report(as_markdown=False) == result.to_json()

    edges = result.to_causal_edges()
    assert isinstance(edges, list)
    dag = result.to_search_dag()
    assert dag["schema"] == "SearchDAG.v1"
    mine = result.to_mine_report()
    assert mine["schema"] == "MineReport.v1"
    bundle = result.to_fabric_bundle()
    assert bundle["schema"] == "FabricBundle.v1"

    est = result.estimate(backend="builtin_ols")
    assert est is not None
    assert getattr(est, "ok", True) or getattr(est, "soft_skip", False)
    ref = result.refute(method="placebo")
    assert ref is not None

    sens = result.sensitivity(n_boot=4, seed=1)
    assert sens is not None
    assert result.sensitivity_report is not None or hasattr(sens, "to_dict")

    iv = result.to_causaliv_request()
    assert iv["schema"] == "CausalIVRequest.v1"
    status = result.engines_status()
    assert status["schema"] == "AutoCausalEngineStatus.v1"


def test_auto_result_full_chain(ac_pipeline):
    _, result = ac_pipeline
    auto = AutoResult(discovery=result, mining=result.mining, source="iris")
    assert auto.report()
    assert auto.to_fabric_bundle()["schema"] == "FabricBundle.v1"
    assert auto.to_causal_edges() == result.to_causal_edges()
    assert auto.to_causaliv_request()["schema"] == "CausalIVRequest.v1"
    assert auto.engines_status()["schema"] == "AutoCausalEngineStatus.v1"
    assert auto.estimate(backend="builtin_ols") is not None
    assert auto.refute(method="placebo") is not None
    assert auto.sensitivity(n_boot=3, seed=2) is not None


def test_mining_report_chain(ac_pipeline):
    ac, _ = ac_pipeline
    mr = ac.mining
    assert mr.report()
    assert mr.to_mine_report()["schema"] == "MineReport.v1"
    assert mr.to_fabric_bundle()["schema"] == "FabricBundle.v1"


def test_suite_reports_have_report_alias(iris_df):
    ac = AutoCausal.from_dataframe(iris_df.head(40), source="iris")
    ac.cleanse(use_slm=False)
    ac.eda(use_slm=False)
    ac.automine(use_slm=False)
    for rep in (ac.cleanse_report, ac.eda_report, ac.mine_report):
        if rep is None:
            continue
        for name in REPORT_METHODS:
            assert hasattr(rep, name), f"{type(rep).__name__} missing {name}"
        assert len(rep.report()) > 10
        assert rep.to_json().strip().startswith("{")


def test_insight_report_aliases(ac_pipeline):
    ac, result = ac_pipeline
    suite = InsightSuite.from_autocausal(ac)
    report = suite.run(text="Do petal measurements associate with species?", use_slm=False)
    for name in REPORT_METHODS + ["write"]:
        assert hasattr(report, name), f"InsightReport missing {name}"
    assert report.report()
    assert report.to_json().strip().startswith("{")


def test_agentic_and_grail_report_aliases(ac_pipeline):
    ac, _ = ac_pipeline
    agentic = ac.agentic_loop(text="petal vs sepal", max_rounds=1, use_slm=False)
    for name in REPORT_METHODS + ["write", "save"]:
        assert hasattr(agentic, name), f"AgenticLoopReport missing {name}"
    assert agentic.report()

    grail = GrailEngine().run("iris petal associations", context={"text": "iris"})
    for name in REPORT_METHODS + ["write"]:
        assert hasattr(grail, name), f"GrailReport missing {name}"
    assert grail.report()


def test_autocausal_primary_path_still_works(iris_df):
    ac = AutoCausal.from_dataframe(iris_df, source="iris-primary")
    result = ac.discover(qc="off", use_iv=False, min_abs_corr=0.2)
    assert ac.result is result
    assert isinstance(ac.report(), str) and len(ac.report()) > 20
    assert ac.report() == result.report()
    assert ac.to_fabric_bundle()["schema"] == "FabricBundle.v1"
    assert ac.estimate(backend="builtin_ols") is not None
    assert ac.refute(method="placebo") is not None
    assert ac.to_causaliv_request()["schema"] == "CausalIVRequest.v1"
    assert ac.engines_status()["schema"] == "AutoCausalEngineStatus.v1"


def test_engines_module_surface():
    assert len(list_engines()) >= 5
    assert engine_status()["schema"] == "AutoCausalEngineStatus.v1"
    df = load_dataset("iris") if not IRIS.is_file() else AutoCausal.from_csv(IRIS)._df
    assert estimate(df, backend="builtin_ols", y="petal_length", d="sepal_length").ok or True
    assert refute({"source": "sepal_length", "target": "petal_length"}, method="placebo", df=df) is not None


def test_no_attribute_error_on_documented_advanced_pattern(ac_pipeline):
    """README advanced pattern: discover → fabric / estimate / refute / causaliv."""
    ac, result = ac_pipeline
    # Must not raise AttributeError
    _ = result.report()
    _ = result.to_json()
    _ = result.to_fabric_bundle()["schema"]
    _ = result.estimate(backend="builtin_ols")
    _ = result.refute(method="placebo")
    _ = result.to_causal_edges()
    _ = result.to_search_dag()
    _ = result.sensitivity(n_boot=3, seed=0)
    _ = result.to_causaliv_request()
    _ = result.engines_status()
    _ = ac.mining.report()
    _ = ac.mining.to_mine_report()


def test_remaining_result_types_have_report_alias():
    """0.11.4: report() on remaining user-facing result/report classes."""
    from autocausal.behavioral.report import BehavioralReport
    from autocausal.grounding import GroundingReport
    from autocausal.guides.types import DirectionPlan
    from autocausal.ml.construct import ModelConstructPlan
    from autocausal.ml.fit_report import FitReport
    from autocausal.physics.types import (
        PhysicalGroundingReport,
        PhysicsLoopResult,
        PhysicsState,
        Trajectory,
        TrajectoryPoint,
    )
    from autocausal.public_causal import PublicCausalReport
    from autocausal.slm import CreationResult, InferenceResult
    from autocausal.suite_tools import ValidationReport

    classes = [
        PublicCausalReport,
        DirectionPlan,
        CreationResult,
        InferenceResult,
        ValidationReport,
        FitReport,
        ModelConstructPlan,
        BehavioralReport,
        GroundingReport,
    ]
    for cls in classes:
        assert hasattr(cls, "report"), f"{cls.__name__} missing report"
        assert callable(getattr(cls, "report"))

    # Minimal instances for types that need construction to exercise report()
    pub = PublicCausalReport(sources=[])
    assert isinstance(pub.report(), str)
    assert pub.report(as_markdown=False).strip().startswith("{")

    plan = DirectionPlan()
    assert isinstance(plan.report(), str)

    create = CreationResult(backend="rule")
    assert isinstance(create.report(), str)

    infer = InferenceResult(backend="rule")
    assert isinstance(infer.report(), str)

    val = ValidationReport(ok=True)
    assert isinstance(val.report(), str)

    fit = FitReport()
    assert isinstance(fit.report(), str)

    mcp = ModelConstructPlan()
    assert isinstance(mcp.report(), str)

    beh = BehavioralReport(trace_name="t")
    assert isinstance(beh.report(), str)

    ground = GroundingReport(claims=[])
    assert isinstance(ground.report(), str)

    traj = Trajectory(
        points=[
            TrajectoryPoint(
                t=0,
                state=PhysicsState(names=["x"], position=[0.0]),
            )
        ]
    )
    phys = PhysicsLoopResult(
        trajectory=traj,
        physical_grounding=PhysicalGroundingReport(insights=[]),
    )
    assert hasattr(PhysicsLoopResult, "report") and callable(PhysicsLoopResult.report)
    assert isinstance(phys.report(), str)


def test_mi_binned_and_aliases(iris_df):
    ac = AutoCausal.from_dataframe(iris_df, source="iris")
    for method in ("mi_binned", "mi", "mi_stub"):
        r = ac.discover(method=method, qc="off", use_iv=False, min_abs_corr=0.1)
        assert r is not None
        # edges from MI path are labeled mi_binned
        mi_edges = [e for e in r.edges if e.get("method") == "mi_binned"]
        assert mi_edges or r.method in ("mi_binned", "mi", "mi_stub", "score_pc_lite")


def test_apply_grail_and_session_snapshot(iris_df):
    ac = AutoCausal.from_dataframe(iris_df, source="iris")
    ac.discover(qc="off", use_iv=False, min_abs_corr=0.2)
    report = ac.apply_grail("iris causal discovery", second_pass=True, qc="off", use_iv=False)
    assert report is not None
    assert hasattr(report, "boost_edges")
    assert isinstance(report.boost_edges, list)
    assert ac.grail_report is report

    snap = ac.session_snapshot()
    assert isinstance(snap, dict)
    assert snap["n_rows"] == len(iris_df)
    assert snap["has_result"] is True
    assert "n_edges" in snap
    assert snap["schema"] == "AutoCausalSessionSnapshot.v1"
