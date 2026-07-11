"""Generate bundled real/public-style example CSVs (offline fixtures).

Run from repo root:
  python scripts/generate_example_datasets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, load_wine

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "autocausal" / "data" / "examples"
EXAMPLES = ROOT / "examples" / "data"


def main() -> None:
    PKG.mkdir(parents=True, exist_ok=True)
    EXAMPLES.mkdir(parents=True, exist_ok=True)

    # --- Iris (UCI / Fisher; public domain educational classic) ---
    iris = load_iris(as_frame=True)
    idf = iris.frame.copy()
    idf.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    idf["species"] = idf["species"].map(dict(enumerate(iris.target_names)))
    idf.to_csv(PKG / "iris.csv", index=False)
    print("iris", len(idf), list(idf.columns))

    # --- Wine (UCI Wine recognition) ---
    wine = load_wine(as_frame=True)
    wdf = wine.frame.copy()
    last = list(wdf.columns)[-1]
    if last in ("target", "class"):
        wdf = wdf.rename(columns={last: "cultivar"})
    wdf["cultivar"] = wdf["cultivar"].astype(int)
    wdf.to_csv(PKG / "wine.csv", index=False)
    print("wine", len(wdf), list(wdf.columns))

    # --- Diabetes (sklearn / Efron et al. educational) ---
    diab = load_diabetes(as_frame=True)
    ddf = diab.frame.copy()
    ddf = ddf.rename(columns={"target": "disease_progression"})
    ddf.to_csv(PKG / "diabetes.csv", index=False)
    print("diabetes", len(ddf), list(ddf.columns))

    # --- Gapminder subset (classic open indicators; cite Gapminder) ---
    gap_rows = [
        ("Afghanistan", "Asia", 1952, 28.801, 8425333, 779.4453145),
        ("Afghanistan", "Asia", 1977, 30.982, 14880372, 786.11336),
        ("Afghanistan", "Asia", 2007, 43.828, 31889923, 974.5803384),
        ("Brazil", "Americas", 1952, 50.917, 56602560, 2108.944355),
        ("Brazil", "Americas", 1977, 61.489, 114259877, 6660.118654),
        ("Brazil", "Americas", 2007, 72.390, 190010647, 9065.800825),
        ("China", "Asia", 1952, 44.000, 556263527, 400.4486107),
        ("China", "Asia", 1977, 63.967, 943455000, 741.7337246),
        ("China", "Asia", 2007, 72.961, 1318683096, 4959.114854),
        ("Egypt", "Africa", 1952, 41.895, 22223309, 1418.822445),
        ("Egypt", "Africa", 1977, 53.319, 40840741, 3195.112777),
        ("Egypt", "Africa", 2007, 71.338, 80264543, 5581.180998),
        ("France", "Europe", 1952, 67.410, 42459667, 7029.809327),
        ("France", "Europe", 1977, 73.830, 53165019, 17496.31322),
        ("France", "Europe", 2007, 80.657, 61083916, 30470.0167),
        ("Germany", "Europe", 1952, 67.500, 69145952, 7144.114393),
        ("Germany", "Europe", 1977, 72.510, 78160773, 17740.86369),
        ("Germany", "Europe", 2007, 79.406, 82400996, 32170.37434),
        ("India", "Asia", 1952, 37.373, 372000000, 546.5657493),
        ("India", "Asia", 1977, 51.805, 634000000, 813.3143461),
        ("India", "Asia", 2007, 64.698, 1110396331, 2452.210407),
        ("Japan", "Asia", 1952, 63.166, 86480977, 3216.956347),
        ("Japan", "Asia", 1977, 75.380, 113872473, 15758.51658),
        ("Japan", "Asia", 2007, 82.603, 127467972, 31656.06806),
        ("Nigeria", "Africa", 1952, 36.698, 33119096, 1077.281856),
        ("Nigeria", "Africa", 1977, 44.514, 73040776, 1981.951806),
        ("Nigeria", "Africa", 2007, 46.859, 135031164, 2013.977305),
        ("United States", "Americas", 1952, 68.440, 157553000, 13990.48208),
        ("United States", "Americas", 1977, 73.380, 220239000, 24072.63213),
        ("United States", "Americas", 2007, 78.242, 301139947, 42951.65309),
        ("United Kingdom", "Europe", 1952, 69.180, 50430000, 9979.508487),
        ("United Kingdom", "Europe", 1977, 72.760, 56179000, 16332.3497),
        ("United Kingdom", "Europe", 2007, 79.425, 60776238, 33207.1528),
        ("Mexico", "Americas", 1952, 50.789, 30112313, 3478.125529),
        ("Mexico", "Americas", 1977, 62.442, 62286143, 7674.929108),
        ("Mexico", "Americas", 2007, 76.195, 108700891, 11977.57496),
    ]
    gdf = pd.DataFrame(
        gap_rows,
        columns=["country", "continent", "year", "lifeExp", "pop", "gdpPercap"],
    )
    gdf.to_csv(PKG / "gapminder_subset.csv", index=False)
    print("gapminder_subset", len(gdf))

    # --- Titanic (open educational passenger table; soft-download preferred) ---
    titanic_url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    try:
        tdf = pd.read_csv(titanic_url)
        keep = [
            c
            for c in [
                "PassengerId",
                "Survived",
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
            ]
            if c in tdf.columns
        ]
        tdf = tdf[keep].copy()
        tdf.to_csv(PKG / "titanic.csv", index=False)
        print("titanic downloaded", len(tdf), list(tdf.columns))
    except Exception as e:  # noqa: BLE001 — offline generator fallback
        print("titanic download failed:", type(e).__name__, e)
        rng = np.random.default_rng(42)
        n = 200
        pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
        sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
        age = np.clip(rng.normal(30, 14, size=n), 0.5, 80)
        fare = np.clip(rng.lognormal(2.5, 1.0, size=n) * (4 - pclass), 0, 500)
        logit = -1.0 + 1.5 * (sex == "female") - 0.7 * (pclass - 1) + 0.01 * fare
        survived = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        tdf = pd.DataFrame(
            {
                "PassengerId": np.arange(1, n + 1),
                "Survived": survived,
                "Pclass": pclass,
                "Sex": sex,
                "Age": np.round(age, 1),
                "SibSp": rng.integers(0, 4, size=n),
                "Parch": rng.integers(0, 3, size=n),
                "Fare": np.round(fare, 2),
                "Embarked": rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09]),
            }
        )
        tdf.to_csv(PKG / "titanic.csv", index=False)
        print("titanic synthetic fallback", len(tdf))

    # --- California housing sample ---
    try:
        from sklearn.datasets import fetch_california_housing

        cal = fetch_california_housing(as_frame=True)
        cdf = cal.frame.copy()
        cdf = cdf.rename(columns={"MedHouseVal": "median_house_value"})
        cdf = cdf.sample(n=250, random_state=7).reset_index(drop=True)
        cdf.to_csv(PKG / "california_housing_sample.csv", index=False)
        print("california_housing_sample", len(cdf), list(cdf.columns))
    except Exception as e:  # noqa: BLE001
        print("california housing failed:", type(e).__name__, e)

    # --- IV demo (synthetic z → treatment → outcome) ---
    rng = np.random.default_rng(11)
    n = 200
    z = rng.normal(size=n)
    confounder = rng.normal(size=n)
    noise = rng.normal(size=n)
    treatment = ((0.9 * z + 0.35 * confounder + 0.25 * rng.normal(size=n)) > 0).astype(int)
    outcome = 1.6 * treatment + 0.5 * confounder + 0.3 * noise
    ivdf = pd.DataFrame(
        {
            "z": np.round(z, 6),
            "treatment": treatment,
            "outcome": np.round(outcome, 6),
            "confounder": np.round(confounder, 6),
            "noise": np.round(noise, 6),
        }
    )
    ivdf.to_csv(PKG / "iv_demo.csv", index=False)
    print("iv_demo", len(ivdf), list(ivdf.columns))

    for p in sorted(PKG.glob("*.csv")):
        shutil.copy2(p, EXAMPLES / p.name)
        print("mirrored", p.name)

    print("DONE", sorted(x.name for x in PKG.glob("*.csv")))


if __name__ == "__main__":
    main()
