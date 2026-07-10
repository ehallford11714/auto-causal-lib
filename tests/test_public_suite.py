"""Public suite list/load/join offline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autocausal import AutoCausal
from autocausal.cli import main
from autocausal.public_suite import (
    ensure_bundled_public_data,
    get_public,
    join_public_frames,
    list_public,
    load_public,
    suggest_join_keys,
)


def test_list_and_load_bundled():
    ensure_bundled_public_data()
    sources = list_public(offline_only=True)
    ids = {s.id for s in sources}
    assert "finance_demo" in ids
    assert "marketing_demo" in ids
    assert "policy_demo" in ids
    assert "demographics_demo" in ids
    assert "vision_stub" in ids
    assert "instruments_demo" in ids
    assert "climate_demo" in ids
    assert "health_demo" in ids
    df = load_public("demographics_demo", allow_network=False)
    assert "region" in df.columns
    assert len(df) >= 3
    info = get_public("finance_demo")
    assert info.access == "bundled"
    assert info.suggested_join_keys


def test_join_public_into_user_and_discover(tmp_path: Path):
    rng = np.random.default_rng(3)
    n = 80
    user = pd.DataFrame(
        {
            "region": rng.choice(["US", "EU", "APAC"], size=n),
            "treatment": rng.integers(0, 2, size=n),
            "outcome": rng.normal(size=n),
        }
    )
    user["outcome"] = user["outcome"] + 1.2 * user["treatment"]
    path = tmp_path / "user.csv"
    user.to_csv(path, index=False)

    ac = AutoCausal.from_csv(path)
    ac.join_public("demographics_demo")
    assert any(j.get("ok") for j in ac.join_log)
    assert "median_age" in ac.df.columns or "median_income" in ac.df.columns
    ac.mine(min_score=0.05)
    assert ac.mining is not None and ac.mining.columns
    result = ac.impute().discover(use_iv=False, min_abs_corr=0.05)
    assert result.edges


def test_suggest_join_keys():
    left = pd.DataFrame({"region": ["US"], "x": [1]})
    right = pd.DataFrame({"region": ["US"], "y": [2]})
    assert suggest_join_keys(left, right) == ["region"]


def test_join_public_frames_multi():
    user = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(20)],
            "region": ["US"] * 10 + ["EU"] * 10,
            "y": list(range(20)),
        }
    )
    joined, log = join_public_frames(user, ["instruments_demo"], allow_network=False)
    assert any(x.get("ok") for x in log)
    assert len(joined) >= 1


def test_cli_public_list():
    assert main(["public", "list", "--offline"]) == 0


@pytest.mark.skipif(
    True,  # network downloads skipped by default
    reason="Network download tests require AUTOCAUSAL_PUBLIC_NET=1",
)
def test_iris_download_skipped_placeholder():
    load_public("iris_open", allow_network=True)


def test_auto_with_join(tmp_path: Path):
    rng = np.random.default_rng(4)
    n = 60
    df = pd.DataFrame(
        {
            "region": rng.choice(["US", "EU", "APAC"], size=n),
            "treatment": rng.integers(0, 2, size=n),
            "outcome": rng.normal(size=n),
            "age": rng.normal(40, 8, size=n),
        }
    )
    df["outcome"] = df["outcome"] + 1.5 * df["treatment"]
    path = tmp_path / "a.csv"
    df.to_csv(path, index=False)
    result = AutoCausal.auto(
        str(path),
        join="demographics_demo",
        text="what causes outcome?",
        use_slm=False,
        second_pass=True,
        use_iv=False,
        min_abs_corr=0.05,
    )
    assert result.discovery.edges
    assert result.mining
    assert result.guide
    assert result.grounding
    assert result.to_markdown()
