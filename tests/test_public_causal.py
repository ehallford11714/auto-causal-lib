"""Public multi-source causal mining (offline bundled fixtures)."""

from __future__ import annotations

from pathlib import Path

from autocausal import AutoCausal, PublicCausalMiner, mine_public
from autocausal.cli import main
from autocausal.public_suite import (
    BUNDLED_IDS,
    ensure_bundled_public_data,
    join_public_corpus,
    list_public,
)


def test_bundled_ids_present():
    ensure_bundled_public_data()
    ids = {s.id for s in list_public(offline_only=True)}
    for bid in BUNDLED_IDS:
        assert bid in ids


def test_join_public_corpus_multi():
    df, log = join_public_corpus(
        ["finance_demo", "demographics_demo", "health_demo"],
        allow_network=False,
    )
    assert any(x.get("ok") for x in log)
    assert "region" in df.columns
    assert len(df) >= 1
    # demographics / health columns should appear after join
    assert any(c in df.columns for c in ("median_age", "median_income", "life_expectancy"))


def test_mine_public_discover_edges():
    report = mine_public(
        ["finance_demo", "demographics_demo", "health_demo"],
        join_on="region",
        discover=True,
        use_iv=False,
        min_abs_corr=0.05,
        min_score=0.05,
        validate=True,
    )
    assert report.n_rows >= 1
    assert report.n_cols >= 3
    assert report.mining is not None
    assert report.edges  # synthetic joins should yield some associations/edges
    assert report.caveats
    assert "finance_demo" in report.sources
    md = report.to_markdown()
    assert "Public causal mining" in md or "causal" in md.lower()
    assert report.to_json()


def test_public_causal_miner_class():
    miner = PublicCausalMiner(
        ["marketing_demo", "demographics_demo"],
        join_on="region",
    )
    report = miner.run(discover=True, use_iv=True, min_abs_corr=0.05, min_score=0.05)
    assert report.discovery is not None
    assert miner.df is not None


def test_autocausal_mine_public():
    report = AutoCausal.mine_public(
        "climate_demo,demographics_demo,health_demo",
        discover=True,
        use_iv=False,
        min_abs_corr=0.05,
    )
    assert report.edges or report.mining
    assert any(j.get("ok") for j in report.join_log)


def test_cli_public_mine_and_causal(tmp_path: Path):
    out = tmp_path / "pub.md"
    assert (
        main(
            [
                "public",
                "mine",
                "--sources",
                "finance_demo,demographics_demo",
                "--discover",
                "-o",
                str(out),
            ]
        )
        == 0
    )
    assert out.exists() and out.stat().st_size > 50

    out2 = tmp_path / "causal.md"
    assert (
        main(
            [
                "public",
                "causal",
                "--sources",
                "finance_demo,demographics_demo,health_demo",
                "--validate",
                "-o",
                str(out2),
            ]
        )
        == 0
    )
    text = out2.read_text(encoding="utf-8")
    assert "Caveats" in text or "caveat" in text.lower()
