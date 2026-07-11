"""Offline tests for modular suites + action registries + skilling surface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autocausal import (
    AutoCausal,
    AutoCleanseSuite,
    AutoEDASuite,
    AutoMineSuite,
    CleanseActions,
    EDAActions,
    MineActions,
    SLMAutoDirector,
    SLMToolBroker,
    suite_tool_surface,
)
from autocausal.skilling import SkillDrill, skill_catalog
from autocausal.suites.autocleanse import CLEANSE_REGISTRY
from autocausal.suites.autoeda import EDA_REGISTRY
from autocausal.suites.automine import MINE_REGISTRY


def _messy_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "unit_id": np.arange(n),
            "treat_x": rng.integers(0, 2, size=n),
            "instrument_z": rng.normal(0, 1, size=n),
            "confound_age": rng.normal(40, 10, size=n),
            "outcome_y": rng.normal(0, 1, size=n),
            "spend": rng.normal(100, 20, size=n),
            "revenue": rng.normal(200, 50, size=n),
            "const_col": 1,
            "almost_missing": [np.nan] * (n - 2) + [1.0, 2.0],
            "obj_num": [str(x) for x in rng.normal(0, 1, size=n)],
        }
    )
    df.loc[0:5, "spend"] = np.nan
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


def test_cleanse_action_registry():
    names = CleanseActions.list()
    assert "impute" in names
    assert "profile_missingness" in names
    assert set(names) == set(CLEANSE_REGISTRY.list())
    df = _messy_df(20)
    r = CleanseActions.profile_missingness(df)
    assert r.name == "profile_missingness"
    assert "missingness" in r.payload
    r2 = CleanseActions.impute(df, method="auto")
    assert r2.frame is not None
    assert r2.frame.isna().sum().sum() <= df.isna().sum().sum()


def test_eda_action_registry():
    assert "suggest_roles" in EDAActions.list()
    assert set(EDAActions.list()) == set(EDA_REGISTRY.list())
    df = _messy_df(30)
    r = EDAActions.suggest_roles(df)
    assert r.payload.get("roles", {}).get("outcome") or r.payload.get("roles", {}).get("treatment")
    r2 = EDAActions.correlation_matrix(df, max_cols=6)
    assert isinstance(r2.payload.get("correlations"), dict)


def test_mine_action_registry():
    assert "mine_associations" in MineActions.list()
    assert set(MineActions.list()) == set(MINE_REGISTRY.list())
    df = _messy_df(40)
    r = MineActions.mine_associations(df, min_score=0.1)
    assert "associations" in r.payload
    r2 = MineActions.to_mine_report(
        df,
        columns=r.payload.get("columns"),
        associations=r.payload.get("associations"),
        kpis=r.payload.get("kpis"),
    )
    env = r2.payload["fabric_envelope"]
    assert "MineReport" in str(env.get("schema", env))


def test_suites_run_with_actions():
    df = _messy_df()
    clean = AutoCleanseSuite(df, use_slm=False).run()
    assert clean.report is not None
    assert clean.report.actions_run
    assert "impute" in clean.report.actions_run or "profile_missingness" in clean.report.actions_run
    eda = AutoEDASuite(clean.frame, use_slm=False).run()
    assert eda.report is not None
    assert eda.report.actions_run
    mine = AutoMineSuite(clean.frame, use_slm=False, join_public=None).run()
    assert mine.report is not None
    assert "mine_associations" in mine.report.actions_run
    assert mine.report.slm_directives is not None


def test_fluent_chain():
    df = _messy_df(50)
    ac = AutoCausal.from_dataframe(df).cleanse(use_slm=False).eda(use_slm=False).automine(
        use_slm=False, join_public=None
    )
    assert ac.cleanse_report is not None
    assert ac.eda_report is not None
    assert ac.mine_report is not None


def test_tool_surface_and_broker():
    surface = suite_tool_surface()
    names = surface.list_names()
    assert any(n.startswith("autocleanse.") for n in names)
    assert any(n.startswith("autoeda.") for n in names)
    assert any(n.startswith("automine.") for n in names)
    assert "autocausal.discover" in names

    broker = SLMToolBroker(surface, use_slm=False)
    tools = broker.list_tools(skill="skill:autocleanse")
    assert tools
    assert all(t["name"].startswith("autocleanse.") for t in tools)

    df = _messy_df(25)
    tr = broker.invoke("autocleanse.impute", {"method": "auto"}, df=df)
    assert tr.ok
    assert EPISTEMIC_IN(tr)

    frame2, results, trace = broker.run_skill("skill:autoeda", df)
    assert len(results) >= 3
    assert trace.skill == "skill:autoeda"
    assert trace.outcomes


def EPISTEMIC_IN(tr) -> bool:
    blob = " ".join(tr.notes + tr.warnings).lower()
    return "generative" in blob or "exploratory" in blob or "identification" in blob


def test_skill_catalog_and_drill():
    cat = skill_catalog()
    assert cat["n_skills"] >= 4
    assert cat["n_tools"] >= 15
    ids = {s["id"] for s in cat["skills"]}
    assert "skill:autocleanse" in ids
    assert "skill:autocausal_loop" in ids

    drill = SkillDrill(skill="skill:automine", use_slm=False)
    trace = drill.run()
    assert trace.tool_calls
    md = drill.to_markdown()
    assert "Skill" in md


def test_director_tools_invoked():
    df = _messy_df(20)
    d = SLMAutoDirector(use_slm=False).direct("cleanse", df)
    assert d.actions
    dd = d.to_dict()
    assert "tools_invoked" in dd or d.actions


def test_skilling_cli(tmp_path: Path):
    from autocausal.cli import main

    out = tmp_path / "skills.md"
    rc = main(["skilling", "list", "-o", str(out)])
    assert rc == 0
    assert out.exists()
    out2 = tmp_path / "drill.json"
    rc2 = main(["skilling", "drill", "--skill", "skill:autocleanse", "--format", "json", "-o", str(out2)])
    assert rc2 == 0
    payload = json.loads(out2.read_text(encoding="utf-8"))
    assert "tool_calls" in payload


def test_suite_cli(tmp_path: Path):
    from autocausal.cli import main

    df = _messy_df(20)
    csv = tmp_path / "x.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "c.md"
    rc = main(["suite", "cleanse", "--csv", str(csv), "--no-slm", "-o", str(out)])
    assert rc == 0
    assert "AutoCleanse" in out.read_text(encoding="utf-8")
