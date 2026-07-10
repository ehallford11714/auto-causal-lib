"""Public / demo dataset suite for joins into mining & discovery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd


__all__ = [
    "PublicSource",
    "list_public",
    "get_public",
    "load_public",
    "suggest_join_keys",
    "join_public_frames",
    "join_public_corpus",
    "ensure_bundled_public_data",
    "manifest_path",
    "PUBLIC_DIR",
    "BUNDLED_IDS",
]

# Expected offline fixtures (regenerate manifest if any missing).
BUNDLED_IDS = (
    "finance_demo",
    "marketing_demo",
    "policy_demo",
    "demographics_demo",
    "vision_stub",
    "instruments_demo",
    "climate_demo",
    "health_demo",
)


PUBLIC_DIR = Path(__file__).resolve().parent / "data" / "public"


@dataclass
class PublicSource:
    id: str
    name: str
    domain: str
    access: str  # bundled | download | sql_demo
    license_note: str
    description: str
    path: Optional[str] = None  # relative to PUBLIC_DIR or absolute
    url: Optional[str] = None
    sql_url_env: Optional[str] = None
    table: Optional[str] = None
    schema_summary: list[dict[str, str]] = field(default_factory=list)
    suggested_join_keys: list[str] = field(default_factory=list)
    rows_approx: int = 0
    offline: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manifest_path() -> Path:
    return PUBLIC_DIR / "manifest.json"


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def ensure_bundled_public_data(*, force: bool = False) -> Path:
    """Create bundled public CSVs + manifest if missing (idempotent)."""
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    man = manifest_path()
    if man.exists() and not force:
        # ensure files listed exist and expected bundled ids are present
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
            ids = {item.get("id") for item in data.get("sources", [])}
            ok = all(bid in ids for bid in BUNDLED_IDS)
            for item in data.get("sources", []):
                if item.get("access") == "bundled" and item.get("path"):
                    if not (PUBLIC_DIR / item["path"]).exists():
                        ok = False
                        break
            if ok and int(data.get("version", 0)) >= 2:
                return PUBLIC_DIR
        except Exception:
            pass

    rng = np.random.default_rng(7)

    # 1) finance_demo — daily-ish returns / macro stub
    n = 120
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    region = rng.choice(["US", "EU", "APAC"], size=n)
    finance = pd.DataFrame(
        {
            "date": dates.astype(str),
            "region": region,
            "ticker": rng.choice(["AAA", "BBB", "CCC"], size=n),
            "return": rng.normal(0.001, 0.02, size=n),
            "interest_rate": 0.03 + 0.01 * np.sin(np.arange(n) / 20) + rng.normal(0, 0.001, n),
            "volume": rng.lognormal(10, 0.4, size=n),
            "default_flag": (rng.random(n) < 0.05).astype(int),
        }
    )
    _write_csv(PUBLIC_DIR / "finance_demo.csv", finance)

    # 2) marketing_demo — campaign panel
    n = 100
    marketing = pd.DataFrame(
        {
            "user_id": [f"u{i%40}" for i in range(n)],
            "region": rng.choice(["US", "EU", "APAC"], size=n),
            "campaign": rng.choice(["spring", "summer", "retarget"], size=n),
            "treatment": rng.integers(0, 2, size=n),
            "ctr": np.clip(rng.normal(0.04, 0.02, size=n), 0, 1),
            "conversion": rng.integers(0, 2, size=n),
            "spend": rng.lognormal(3, 0.5, size=n),
            "revenue": rng.lognormal(4, 0.7, size=n),
        }
    )
    # make treatment affect conversion/revenue a bit
    marketing["conversion"] = (
        (0.3 * marketing["treatment"] + 0.2 * marketing["ctr"] + rng.normal(0, 0.2, n)) > 0.35
    ).astype(int)
    marketing["revenue"] = (
        marketing["revenue"] * (1 + 0.4 * marketing["treatment"] + 0.2 * marketing["conversion"])
    )
    _write_csv(PUBLIC_DIR / "marketing_demo.csv", marketing)

    # 3) policy_demo — DiD-style panel
    n = 96
    policy = pd.DataFrame(
        {
            "unit_id": [f"s{i%16}" for i in range(n)],
            "region": rng.choice(["US", "EU", "APAC"], size=n),
            "year": np.tile(np.arange(2018, 2024), 16)[:n],
            "post": (np.tile(np.arange(2018, 2024), 16)[:n] >= 2021).astype(int),
            "treated_unit": ([1] * 48 + [0] * 48)[:n],
            "eligibility": rng.integers(0, 2, size=n),
            "outcome": rng.normal(10, 2, size=n),
            "population": rng.integers(50_000, 500_000, size=n),
        }
    )
    policy["treatment"] = (policy["treated_unit"] * policy["post"]).astype(int)
    policy["outcome"] = policy["outcome"] + 1.5 * policy["treatment"] + 0.00001 * policy["population"]
    _write_csv(PUBLIC_DIR / "policy_demo.csv", policy)

    # 4) demographics_demo — region-level stub
    demo = pd.DataFrame(
        {
            "region": ["US", "EU", "APAC", "LATAM", "MEA"],
            "median_age": [38.5, 43.0, 32.0, 29.5, 26.0],
            "median_income": [65000, 42000, 28000, 18000, 12000],
            "urban_pct": [82.0, 75.0, 51.0, 80.0, 48.0],
            "population_m": [330.0, 450.0, 2800.0, 650.0, 400.0],
        }
    )
    _write_csv(PUBLIC_DIR / "demographics_demo.csv", demo)

    # 5) vision_stub — frame/motion features (join on region or clip_id)
    n = 60
    vision = pd.DataFrame(
        {
            "clip_id": [f"c{i%20}" for i in range(n)],
            "region": rng.choice(["US", "EU", "APAC"], size=n),
            "frame_idx": rng.integers(0, 30, size=n),
            "motion_score": rng.random(n),
            "object_count": rng.integers(0, 8, size=n),
            "next_frame_err": rng.exponential(0.2, size=n),
        }
    )
    _write_csv(PUBLIC_DIR / "vision_stub.csv", vision)

    # 6) instruments_demo — IV-friendly assignment table
    n = 80
    z = rng.normal(size=n)
    instruments = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],
            "region": rng.choice(["US", "EU", "APAC"], size=n),
            "instrument_z": z,
            "assignment": (z > 0).astype(int),
            "eligibility": (rng.random(n) < 0.7).astype(int),
        }
    )
    _write_csv(PUBLIC_DIR / "instruments_demo.csv", instruments)

    # 7) climate_demo — region-year weather / emissions stub (join on region, year)
    climate = pd.DataFrame(
        {
            "region": ["US", "EU", "APAC", "LATAM", "MEA"] * 3,
            "year": [2019] * 5 + [2020] * 5 + [2021] * 5,
            "temp_anomaly": rng.normal(0.8, 0.3, size=15),
            "precip_mm": rng.lognormal(6.5, 0.25, size=15),
            "co2_index": 400 + np.arange(15) * 0.4 + rng.normal(0, 1, 15),
            "extreme_events": rng.integers(0, 12, size=15),
        }
    )
    _write_csv(PUBLIC_DIR / "climate_demo.csv", climate)

    # 8) health_demo — region-level health / access stub
    health = pd.DataFrame(
        {
            "region": ["US", "EU", "APAC", "LATAM", "MEA"],
            "life_expectancy": [78.5, 81.0, 74.0, 75.5, 70.0],
            "hospital_beds_per_1k": [2.9, 5.1, 2.2, 2.0, 1.4],
            "vaccination_pct": [68.0, 72.0, 55.0, 60.0, 48.0],
            "smoking_pct": [14.0, 22.0, 18.0, 16.0, 12.0],
        }
    )
    _write_csv(PUBLIC_DIR / "health_demo.csv", health)

    sources = [
        PublicSource(
            id="finance_demo",
            name="Finance returns stub",
            domain="finance",
            access="bundled",
            license_note="Synthetic MIT-licensed fixture (not real market data).",
            description="Synthetic daily returns, rates, volume, default flag by region/ticker.",
            path="finance_demo.csv",
            schema_summary=[
                {"column": "date", "dtype": "str"},
                {"column": "region", "dtype": "str"},
                {"column": "ticker", "dtype": "str"},
                {"column": "return", "dtype": "float"},
                {"column": "interest_rate", "dtype": "float"},
                {"column": "volume", "dtype": "float"},
                {"column": "default_flag", "dtype": "int"},
            ],
            suggested_join_keys=["region", "date"],
            rows_approx=len(finance),
            offline=True,
        ),
        PublicSource(
            id="marketing_demo",
            name="Marketing campaign panel",
            domain="marketing",
            access="bundled",
            license_note="Synthetic MIT-licensed fixture.",
            description="User-level campaign exposure, CTR, conversion, spend, revenue.",
            path="marketing_demo.csv",
            schema_summary=[
                {"column": "user_id", "dtype": "str"},
                {"column": "region", "dtype": "str"},
                {"column": "campaign", "dtype": "str"},
                {"column": "treatment", "dtype": "int"},
                {"column": "ctr", "dtype": "float"},
                {"column": "conversion", "dtype": "int"},
                {"column": "spend", "dtype": "float"},
                {"column": "revenue", "dtype": "float"},
            ],
            suggested_join_keys=["user_id", "region"],
            rows_approx=len(marketing),
            offline=True,
        ),
        PublicSource(
            id="policy_demo",
            name="Policy DiD panel",
            domain="policy",
            access="bundled",
            license_note="Synthetic MIT-licensed fixture.",
            description="Unit-year panel with post × treated_unit design and outcomes.",
            path="policy_demo.csv",
            schema_summary=[
                {"column": "unit_id", "dtype": "str"},
                {"column": "region", "dtype": "str"},
                {"column": "year", "dtype": "int"},
                {"column": "treatment", "dtype": "int"},
                {"column": "outcome", "dtype": "float"},
            ],
            suggested_join_keys=["region", "year", "unit_id"],
            rows_approx=len(policy),
            offline=True,
        ),
        PublicSource(
            id="demographics_demo",
            name="Demographics by region",
            domain="demographics",
            access="bundled",
            license_note="Synthetic MIT-licensed fixture (illustrative numbers).",
            description="Region-level age, income, urbanization, population.",
            path="demographics_demo.csv",
            schema_summary=[
                {"column": "region", "dtype": "str"},
                {"column": "median_age", "dtype": "float"},
                {"column": "median_income", "dtype": "float"},
            ],
            suggested_join_keys=["region"],
            rows_approx=len(demo),
            offline=True,
        ),
        PublicSource(
            id="vision_stub",
            name="Vision / motion stub",
            domain="vision",
            access="bundled",
            license_note="Synthetic MIT-licensed fixture.",
            description="Clip/frame motion and object counts for NextFrameSeq-style joins.",
            path="vision_stub.csv",
            schema_summary=[
                {"column": "clip_id", "dtype": "str"},
                {"column": "region", "dtype": "str"},
                {"column": "motion_score", "dtype": "float"},
                {"column": "object_count", "dtype": "int"},
            ],
            suggested_join_keys=["region", "clip_id"],
            rows_approx=len(vision),
            offline=True,
        ),
        PublicSource(
            id="instruments_demo",
            name="IV assignment table",
            domain="policy",
            access="bundled",
            license_note="Synthetic MIT-licensed fixture.",
            description="Instrument Z, assignment, eligibility by user/region for IV joins.",
            path="instruments_demo.csv",
            schema_summary=[
                {"column": "user_id", "dtype": "str"},
                {"column": "region", "dtype": "str"},
                {"column": "instrument_z", "dtype": "float"},
                {"column": "assignment", "dtype": "int"},
            ],
            suggested_join_keys=["user_id", "region"],
            rows_approx=len(instruments),
            offline=True,
        ),
        PublicSource(
            id="climate_demo",
            name="Climate / weather stub",
            domain="climate",
            access="bundled",
            license_note="Synthetic MIT-licensed fixture (not real climate data).",
            description="Region-year temperature anomaly, precip, CO2 index, extreme events.",
            path="climate_demo.csv",
            schema_summary=[
                {"column": "region", "dtype": "str"},
                {"column": "year", "dtype": "int"},
                {"column": "temp_anomaly", "dtype": "float"},
                {"column": "co2_index", "dtype": "float"},
            ],
            suggested_join_keys=["region", "year"],
            rows_approx=len(climate),
            offline=True,
        ),
        PublicSource(
            id="health_demo",
            name="Health indicators by region",
            domain="health",
            access="bundled",
            license_note="Synthetic MIT-licensed fixture (illustrative numbers).",
            description="Region-level life expectancy, beds, vaccination, smoking.",
            path="health_demo.csv",
            schema_summary=[
                {"column": "region", "dtype": "str"},
                {"column": "life_expectancy", "dtype": "float"},
                {"column": "vaccination_pct", "dtype": "float"},
            ],
            suggested_join_keys=["region"],
            rows_approx=len(health),
            offline=True,
        ),
        PublicSource(
            id="iris_open",
            name="Iris (open CSV mirror)",
            domain="demo",
            access="download",
            license_note="Classic UCI Iris; public domain / open educational use.",
            description="Optional network download of Iris CSV for soft-fail demos.",
            url="https://raw.githubusercontent.com/plotly/datasets/master/iris.csv",
            schema_summary=[
                {"column": "sepal_length", "dtype": "float"},
                {"column": "species", "dtype": "str"},
            ],
            suggested_join_keys=[],
            rows_approx=150,
            offline=False,
        ),
        PublicSource(
            id="gapminder_open",
            name="Gapminder life expectancy (open CSV)",
            domain="demographics",
            access="download",
            license_note="Gapminder open data; cite Gapminder when publishing.",
            description="Optional network download — soft-fails offline.",
            url="https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv",
            schema_summary=[
                {"column": "country", "dtype": "str"},
                {"column": "year", "dtype": "int"},
                {"column": "lifeExp", "dtype": "float"},
            ],
            suggested_join_keys=["year"],
            rows_approx=1704,
            offline=False,
        ),
        PublicSource(
            id="public_pg_env",
            name="Optional public Postgres (env URL)",
            domain="sql_demo",
            access="sql_demo",
            license_note="User-supplied read-only demo URL only; nothing hardcoded.",
            description="Ping/load via AUTOCAUSAL_PUBLIC_PG_URL if set. Prefer bundled fixtures.",
            sql_url_env="AUTOCAUSAL_PUBLIC_PG_URL",
            table=None,
            schema_summary=[],
            suggested_join_keys=[],
            rows_approx=0,
            offline=False,
        ),
    ]

    payload = {
        "version": 2,
        "description": "AutoCausalLib public / demo dataset suite",
        "sources": [s.to_dict() for s in sources],
    }
    man.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return PUBLIC_DIR


def _load_manifest() -> list[PublicSource]:
    ensure_bundled_public_data()
    data = json.loads(manifest_path().read_text(encoding="utf-8"))
    return [PublicSource(**item) for item in data.get("sources", [])]


def list_public(*, offline_only: bool = False) -> list[PublicSource]:
    sources = _load_manifest()
    if offline_only:
        return [s for s in sources if s.offline or s.access == "bundled"]
    return sources


def get_public(public_id: str) -> PublicSource:
    for s in list_public():
        if s.id == public_id:
            return s
    known = ", ".join(x.id for x in list_public())
    raise KeyError(f"Unknown public source {public_id!r}. Known: {known}")


def _download_csv(url: str, timeout: float = 8.0) -> pd.DataFrame:
    req = Request(url, headers={"User-Agent": "autocausal/0.2"})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — curated open URLs only
        return pd.read_csv(resp)


def load_public(
    public_id: str,
    *,
    allow_network: bool = True,
    timeout: float = 8.0,
) -> pd.DataFrame:
    """Load a suite member. Network sources soft-fail with a clear error."""
    src = get_public(public_id)
    if src.access == "bundled":
        assert src.path
        path = PUBLIC_DIR / src.path
        if not path.exists():
            ensure_bundled_public_data(force=True)
        return pd.read_csv(path)

    if src.access == "download":
        if not allow_network:
            raise RuntimeError(f"{public_id}: network disabled (offline mode)")
        if not src.url:
            raise RuntimeError(f"{public_id}: no download URL configured")
        try:
            return _download_csv(src.url, timeout=timeout)
        except Exception as e:
            raise RuntimeError(
                f"{public_id}: download soft-fail ({type(e).__name__}: {e})"
            ) from e

    if src.access == "sql_demo":
        import os

        from autocausal.ingest import load_sqlalchemy

        if not allow_network:
            raise RuntimeError(f"{public_id}: network disabled")
        env = src.sql_url_env or "AUTOCAUSAL_PUBLIC_PG_URL"
        url = os.environ.get(env, "")
        if not url:
            raise RuntimeError(
                f"{public_id}: set {env} to a read-only demo SQLAlchemy URL "
                "(no secrets are shipped)."
            )
        if not src.table:
            raise RuntimeError(
                f"{public_id}: set table via load_sqlalchemy manually; "
                "env URL alone is for ping/docs."
            )
        return load_sqlalchemy(url, table=src.table)

    raise RuntimeError(f"Unsupported access method: {src.access}")


def suggest_join_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    hints: Optional[list[str]] = None,
) -> list[str]:
    """Suggest join keys by exact name overlap, then fuzzy token overlap."""
    left_cols = {str(c): str(c).lower() for c in left.columns}
    right_cols = {str(c): str(c).lower() for c in right.columns}
    exact = sorted(set(left_cols) & set(right_cols))
    if hints:
        hinted = [h for h in hints if h in left.columns and h in right.columns]
        if hinted:
            return hinted

    if exact:
        # prefer id-like / region / date keys
        priority = ("id", "key", "region", "date", "year", "user", "unit", "ticker")
        exact_sorted = sorted(
            exact,
            key=lambda c: (
                0 if any(p in c.lower() for p in priority) else 1,
                c.lower(),
            ),
        )
        return exact_sorted[:3]

    # fuzzy: shared tokens
    fuzzy: list[tuple[str, str, float]] = []
    for lc, ll in left_cols.items():
        for rc, rl in right_cols.items():
            if ll == rl:
                continue
            if ll in rl or rl in ll:
                score = min(len(ll), len(rl)) / max(len(ll), len(rl))
                if score >= 0.6:
                    fuzzy.append((lc, rc, score))
    fuzzy.sort(key=lambda x: -x[2])
    # return left names that fuzzy-match (user can rename); prefer empty over wrong
    return []


def join_public_frames(
    user_df: pd.DataFrame,
    public_ids: str | list[str],
    *,
    on: Optional[str | list[str]] = None,
    how: str = "left",
    allow_network: bool = False,
    suffixes: tuple[str, str] = ("", "_pub"),
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Left-join one or more public suite tables into user_df.

    Returns (joined_df, join_log).
    """
    if isinstance(public_ids, str):
        ids = [x.strip() for x in public_ids.split(",") if x.strip()]
    else:
        ids = list(public_ids)

    out = user_df.copy()
    log: list[dict[str, Any]] = []
    for i, pid in enumerate(ids):
        src = get_public(pid)
        try:
            right = load_public(pid, allow_network=allow_network)
        except Exception as e:
            log.append({"id": pid, "ok": False, "error": str(e)})
            continue

        keys: Optional[list[str]]
        if on is None:
            keys = suggest_join_keys(out, right, hints=src.suggested_join_keys)
        elif isinstance(on, str):
            keys = [on]
        else:
            keys = list(on)

        if not keys:
            log.append(
                {
                    "id": pid,
                    "ok": False,
                    "error": "no join keys found; pass on=...",
                    "right_columns": list(map(str, right.columns)),
                }
            )
            continue

        missing_l = [k for k in keys if k not in out.columns]
        missing_r = [k for k in keys if k not in right.columns]
        if missing_l or missing_r:
            log.append(
                {
                    "id": pid,
                    "ok": False,
                    "error": f"join keys missing left={missing_l} right={missing_r}",
                    "keys": keys,
                }
            )
            continue

        before = len(out.columns)
        # avoid duplicate non-key columns colliding badly
        rsuf = suffixes[1] if suffixes[1] else f"_{pid}"
        out = out.merge(right, on=keys, how=how, suffixes=(suffixes[0], rsuf))
        log.append(
            {
                "id": pid,
                "ok": True,
                "keys": keys,
                "how": how,
                "rows": len(out),
                "cols_added": len(out.columns) - before,
            }
        )
    return out, log


def join_public_corpus(
    public_ids: str | list[str],
    *,
    on: Optional[str | list[str]] = None,
    how: str = "outer",
    allow_network: bool = False,
    base: Optional[pd.DataFrame] = None,
    base_label: Optional[str] = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Join multiple public suite tables into one mining corpus.

    If ``base`` is provided, left-join public tables into it (same as
    :func:`join_public_frames`). Otherwise the first successfully loaded
    public id becomes the base frame and subsequent ids are joined on
    suggested keys (default ``how='outer'`` for multi-source demos).
    """
    if isinstance(public_ids, str):
        ids = [x.strip() for x in public_ids.split(",") if x.strip()]
    else:
        ids = list(public_ids)

    if not ids and base is None:
        return pd.DataFrame(), [{"id": None, "ok": False, "error": "no sources"}]

    if base is not None:
        joined, log = join_public_frames(
            base,
            ids,
            on=on,
            how=how if how != "outer" else "left",
            allow_network=allow_network,
        )
        if base_label:
            log.insert(0, {"id": base_label, "ok": True, "keys": [], "how": "base", "rows": len(base)})
        return joined, log

    log: list[dict[str, Any]] = []
    out: Optional[pd.DataFrame] = None
    for pid in ids:
        try:
            right = load_public(pid, allow_network=allow_network)
        except Exception as e:
            log.append({"id": pid, "ok": False, "error": str(e)})
            continue

        if out is None:
            out = right.copy()
            log.append(
                {
                    "id": pid,
                    "ok": True,
                    "keys": [],
                    "how": "base",
                    "rows": len(out),
                    "cols_added": len(out.columns),
                }
            )
            continue

        src = get_public(pid)
        if on is None:
            keys = suggest_join_keys(out, right, hints=src.suggested_join_keys)
            # prefer region (+ year when both have it) for multi-domain corpus demos
            if "region" in out.columns and "region" in right.columns:
                if "year" in out.columns and "year" in right.columns:
                    keys = ["region", "year"]
                else:
                    keys = ["region"]
        elif isinstance(on, str):
            keys = [on]
        else:
            keys = list(on)

        if not keys:
            log.append(
                {
                    "id": pid,
                    "ok": False,
                    "error": "no join keys found; pass on=...",
                    "right_columns": list(map(str, right.columns)),
                }
            )
            continue

        missing_l = [k for k in keys if k not in out.columns]
        missing_r = [k for k in keys if k not in right.columns]
        if missing_l or missing_r:
            # fall back to region-only if possible
            if "region" in out.columns and "region" in right.columns:
                keys = ["region"]
            else:
                log.append(
                    {
                        "id": pid,
                        "ok": False,
                        "error": f"join keys missing left={missing_l} right={missing_r}",
                        "keys": keys,
                    }
                )
                continue

        before = len(out.columns)
        # drop duplicate non-key columns from right to reduce suffix noise
        drop_cols = [c for c in right.columns if c in out.columns and c not in keys]
        right_clean = right.drop(columns=drop_cols, errors="ignore")
        out = out.merge(right_clean, on=keys, how=how, suffixes=("", f"_{pid}"))
        log.append(
            {
                "id": pid,
                "ok": True,
                "keys": keys,
                "how": how,
                "rows": len(out),
                "cols_added": len(out.columns) - before,
            }
        )

    if out is None:
        return pd.DataFrame(), log
    return out, log
