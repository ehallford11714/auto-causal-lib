"""Smoke tests: bundled real example CSVs load and discover offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from autocausal import AutoCausal, load_dataset, list_datasets
from autocausal.cli import main
from autocausal.datasets import DATASET_IDS, dataset_path, ensure_example_datasets, get_dataset
from autocausal.insight import demo_insight
from autocausal.public_suite import REAL_EXAMPLE_IDS, ensure_bundled_public_data, load_public


def test_example_csvs_exist():
    root = ensure_example_datasets()
    for did in DATASET_IDS:
        path = dataset_path(did)
        assert path.is_file(), path
        assert path.parent == root or path.parent.resolve() == root.resolve()


def test_list_and_load_iris_offline():
    ids = {d.id for d in list_datasets()}
    assert "iris" in ids
    assert "wine" in ids
    assert "titanic" in ids
    df = load_dataset("iris", allow_network=False)
    assert len(df) == 150
    assert {"sepal_length", "petal_length", "species"} <= set(df.columns)
    meta = get_dataset("iris")
    assert "illustrative" in meta.epistemic_note.lower() or "not" in meta.epistemic_note.lower()


@pytest.mark.parametrize("dataset_id", list(DATASET_IDS))
def test_load_each_dataset_offline(dataset_id: str):
    df = load_dataset(dataset_id, allow_network=False)
    assert len(df) >= 10
    assert len(df.columns) >= 2


def test_iris_discover_offline():
    df = load_dataset("iris", allow_network=False)
    ac = AutoCausal(df)
    ac.mine(min_score=0.05)
    result = ac.impute().discover(use_iv=False, min_abs_corr=0.15)
    assert result.edges
    assert ac.report()


def test_public_load_iris_offline():
    ensure_bundled_public_data()
    df = load_public("iris", allow_network=False)
    assert len(df) == 150
    # alias soft-falls back offline
    df2 = load_public("iris_open", allow_network=False)
    assert len(df2) >= 100


def test_public_suite_lists_real_examples():
    ensure_bundled_public_data()
    from autocausal.public_suite import list_public

    ids = {s.id for s in list_public(offline_only=True)}
    for rid in REAL_EXAMPLE_IDS:
        assert rid in ids


def test_cli_public_load_iris():
    assert main(["public", "load", "iris"]) == 0


def test_cli_insight_demo_iris():
    assert main(["insight", "demo", "--dataset", "iris", "--no-slm", "--rounds", "1", "--format", "json"]) == 0


def test_demo_insight_iris_offline():
    report = demo_insight(dataset="iris", use_slm=False, max_rounds=1, research_loop=False)
    assert report.key_edges is not None
    assert report.to_markdown()


def test_examples_data_mirror_optional():
    """Repo checkout may mirror CSVs under examples/data/."""
    mirror = Path(__file__).resolve().parents[1] / "examples" / "data" / "iris.csv"
    if mirror.is_file():
        import pandas as pd

        assert len(pd.read_csv(mirror)) == 150
