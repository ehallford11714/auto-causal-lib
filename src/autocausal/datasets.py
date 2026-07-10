"""Bundled real / public-style example datasets (offline-first).

Prefer::

    from autocausal.datasets import load_dataset, list_datasets
    df = load_dataset("iris")

CSVs ship under ``autocausal/data/examples/`` and are mirrored in
``examples/data/`` for discoverability. Optional network refresh soft-fails
back to the bundled file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen

import pandas as pd

__all__ = [
    "ExampleDataset",
    "EXAMPLES_DIR",
    "DATASET_IDS",
    "list_datasets",
    "get_dataset",
    "load_dataset",
    "dataset_path",
    "ensure_example_datasets",
]


def _resolve_examples_dir() -> Path:
    try:
        root = resources.files("autocausal.data.examples")
        return Path(str(root))
    except Exception:
        return Path(__file__).resolve().parent / "data" / "examples"


EXAMPLES_DIR = _resolve_examples_dir()


@dataclass(frozen=True)
class ExampleDataset:
    """Metadata for a bundled example dataset."""

    id: str
    name: str
    filename: str
    domain: str
    description: str
    license_note: str
    rows_approx: int
    suggested_outcome: Optional[str] = None
    columns: list[str] = field(default_factory=list)
    url: Optional[str] = None  # optional soft-refresh mirror
    epistemic_note: str = (
        "Exploratory edges are illustrative library demos, not scientific "
        "causal claims about the domain."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_REGISTRY: dict[str, ExampleDataset] = {
    "iris": ExampleDataset(
        id="iris",
        name="Iris (Fisher)",
        filename="iris.csv",
        domain="demo",
        description="Classic 150-row flower measurements (sepal/petal × species).",
        license_note="UCI / Fisher Iris — public domain / open educational use.",
        rows_approx=150,
        suggested_outcome="species",
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
        ],
        url="https://raw.githubusercontent.com/plotly/datasets/master/iris.csv",
        epistemic_note=(
            "Iris edges are exploratory associations among morphological "
            "measurements — not claims that one flower trait causes another."
        ),
    ),
    "wine": ExampleDataset(
        id="wine",
        name="Wine recognition (UCI)",
        filename="wine.csv",
        domain="chemistry",
        description="178 Italian wines × 13 chemical features + cultivar label.",
        license_note="UCI Wine — open educational / redistributable research use.",
        rows_approx=178,
        suggested_outcome="cultivar",
        columns=[
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280/od315_of_diluted_wines",
            "proline",
            "cultivar",
        ],
    ),
    "diabetes": ExampleDataset(
        id="diabetes",
        name="Diabetes (sklearn / Efron)",
        filename="diabetes.csv",
        domain="health",
        description="442 patients × physiologic features → disease progression.",
        license_note="sklearn diabetes — redistributable educational dataset.",
        rows_approx=442,
        suggested_outcome="disease_progression",
        columns=[
            "age",
            "sex",
            "bmi",
            "bp",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6",
            "disease_progression",
        ],
    ),
    "titanic": ExampleDataset(
        id="titanic",
        name="Titanic passengers (educational)",
        filename="titanic.csv",
        domain="demographics",
        description="Passenger class/sex/age/fare → survival (tabular classic).",
        license_note=(
            "Public-domain passenger records via open educational CSV mirrors; "
            "cite upstream when publishing."
        ),
        rows_approx=891,
        suggested_outcome="Survived",
        columns=[
            "PassengerId",
            "Survived",
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ],
        url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    ),
    "gapminder_subset": ExampleDataset(
        id="gapminder_subset",
        name="Gapminder country indicators (subset)",
        filename="gapminder_subset.csv",
        domain="demographics",
        description="Small country-year panel: lifeExp, pop, gdpPercap.",
        license_note="Gapminder open data subset — cite Gapminder when publishing.",
        rows_approx=36,
        suggested_outcome="lifeExp",
        columns=["country", "continent", "year", "lifeExp", "pop", "gdpPercap"],
        url="https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv",
    ),
    "california_housing_sample": ExampleDataset(
        id="california_housing_sample",
        name="California housing (250-row sample)",
        filename="california_housing_sample.csv",
        domain="housing",
        description="Sklearn California housing sample → median house value.",
        license_note="sklearn fetch_california_housing sample — educational use.",
        rows_approx=250,
        suggested_outcome="median_house_value",
        columns=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
            "median_house_value",
        ],
    ),
}

# Public aliases used by older manifest / CLI docs
_ALIASES: dict[str, str] = {
    "iris_open": "iris",
    "gapminder_open": "gapminder_subset",
    "gapminder": "gapminder_subset",
    "housing": "california_housing_sample",
    "california_housing": "california_housing_sample",
}

DATASET_IDS: tuple[str, ...] = tuple(_REGISTRY.keys())


def list_datasets() -> list[ExampleDataset]:
    """Return metadata for all bundled example datasets."""
    return [_REGISTRY[i] for i in DATASET_IDS]


def get_dataset(dataset_id: str) -> ExampleDataset:
    """Resolve dataset id (including aliases) to metadata."""
    key = _ALIASES.get(dataset_id, dataset_id)
    if key not in _REGISTRY:
        known = ", ".join(DATASET_IDS)
        raise KeyError(f"Unknown dataset {dataset_id!r}. Known: {known}")
    return _REGISTRY[key]


def dataset_path(dataset_id: str) -> Path:
    """Filesystem path to the bundled CSV (may not exist until ensure)."""
    meta = get_dataset(dataset_id)
    return EXAMPLES_DIR / meta.filename


def ensure_example_datasets() -> Path:
    """Ensure the examples data directory exists; return its path.

    CSVs are shipped in the package. This only validates presence and raises
    a clear error if the install is incomplete.
    """
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    missing = [m.filename for m in list_datasets() if not (EXAMPLES_DIR / m.filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing bundled example CSVs under {EXAMPLES_DIR}: {missing}. "
            "Reinstall autocausal or run scripts/generate_example_datasets.py."
        )
    return EXAMPLES_DIR


def _download_csv(url: str, timeout: float = 8.0) -> pd.DataFrame:
    req = Request(url, headers={"User-Agent": "autocausal/0.8"})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — curated open URLs
        return pd.read_csv(resp)


def _normalize_iris_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common Iris mirrors onto the bundled column names."""
    out = df.copy()
    rename = {
        "sepal.length": "sepal_length",
        "sepal.width": "sepal_width",
        "petal.length": "petal_length",
        "petal.width": "petal_width",
        "SepalLength": "sepal_length",
        "SepalWidth": "sepal_width",
        "PetalLength": "petal_length",
        "PetalWidth": "petal_width",
        "Species": "species",
        "Name": "species",
        "variety": "species",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    return out


def load_dataset(
    dataset_id: str,
    *,
    allow_network: bool = False,
    timeout: float = 8.0,
    prefer_network: bool = False,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load a bundled example dataset (offline by default).

    Parameters
    ----------
    dataset_id:
        One of :data:`DATASET_IDS` or an alias (``iris_open``, ``gapminder``, …).
    allow_network:
        If True and a mirror URL is configured, attempt a soft download.
        On failure (or when False), load the bundled CSV.
    prefer_network:
        If True with ``allow_network``, try the network first; otherwise
        bundled-first (offline-first).
    use_cache:
        When a network fetch succeeds, write a local cache CSV under
        ``cache_dir`` (default: ``EXAMPLES_DIR / .cache``).
    cache_dir:
        Optional override for the network cache directory.
    """
    meta = get_dataset(dataset_id)
    path = EXAMPLES_DIR / meta.filename
    cache_root = Path(cache_dir) if cache_dir else (EXAMPLES_DIR / ".cache")
    cache_path = cache_root / meta.filename

    def _from_bundle() -> pd.DataFrame:
        if not path.is_file():
            ensure_example_datasets()
        if not path.is_file():
            raise FileNotFoundError(f"Bundled dataset missing: {path}")
        return pd.read_csv(path)

    def _from_cache() -> Optional[pd.DataFrame]:
        if use_cache and cache_path.is_file():
            try:
                return pd.read_csv(cache_path)
            except Exception:
                return None
        return None

    def _write_cache(df: pd.DataFrame) -> None:
        if not use_cache:
            return
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
        except Exception:
            pass

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if meta.id == "iris":
            return _normalize_iris_columns(df)
        if meta.id == "titanic":
            keep = [c for c in meta.columns if c in df.columns]
            return df[keep] if keep else df
        if meta.id == "gapminder_subset":
            cols = [c for c in meta.columns if c in df.columns]
            if cols:
                df = df[cols]
                if len(df) > 200 and "country" in df.columns:
                    countries = {
                        "Afghanistan",
                        "Brazil",
                        "China",
                        "Egypt",
                        "France",
                        "Germany",
                        "India",
                        "Japan",
                        "Nigeria",
                        "United States",
                        "United Kingdom",
                        "Mexico",
                    }
                    df = df[df["country"].isin(countries)]
            return df
        return df

    if allow_network and meta.url and prefer_network:
        try:
            df = _normalize(_download_csv(meta.url, timeout=timeout))
            _write_cache(df)
            return df
        except Exception:
            cached = _from_cache()
            if cached is not None:
                return cached
            return _from_bundle()

    if path.is_file():
        return _from_bundle()

    cached = _from_cache()
    if cached is not None:
        return cached

    if allow_network and meta.url:
        try:
            df = _normalize(_download_csv(meta.url, timeout=timeout))
            _write_cache(df)
            return df
        except Exception as e:
            raise RuntimeError(
                f"{meta.id}: bundled CSV missing and download soft-fail "
                f"({type(e).__name__}: {e})"
            ) from e

    return _from_bundle()
