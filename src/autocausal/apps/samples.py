"""Bundled / synthetic frames for the physics Streamlit demo (no Streamlit import)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

SampleKind = Literal["oscillator", "kpi_panel", "markets", "affect"]

__all__ = [
    "SampleKind",
    "synthetic_oscillator",
    "synthetic_kpi_panel",
    "synthetic_markets",
    "synthetic_affect",
    "load_demo_frame",
    "bundled_sample_path",
]


def synthetic_oscillator(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Damped harmonic oscillator proxies — mechanics-lite demo."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * np.pi, n)
    damp = np.exp(-0.08 * t)
    position = damp * np.cos(t) + 0.04 * rng.normal(size=n)
    velocity = -damp * np.sin(t) + 0.04 * rng.normal(size=n)
    force = -0.25 * position - 0.1 * velocity + 0.05 * rng.normal(size=n)
    energy = 0.5 * velocity**2 + 0.12 * position**2
    momentum = velocity  # unit mass
    treatment = (force > np.median(force)).astype(float)
    outcome = 0.7 * treatment + 0.25 * energy + 0.08 * rng.normal(size=n)
    return pd.DataFrame(
        {
            "t": t,
            "position": position,
            "velocity": velocity,
            "force": force,
            "momentum": momentum,
            "energy": energy,
            "damping": damp,
            "treatment": treatment,
            "outcome": outcome,
        }
    )


def synthetic_kpi_panel(n: int = 100, seed: int = 1) -> pd.DataFrame:
    """Generic KPI panel with mild causal structure for linear-ODE / drift demos."""
    rng = np.random.default_rng(seed)
    spend = rng.normal(50, 12, size=n).clip(5)
    conf = rng.normal(0, 1, size=n)
    treatment = (spend + 3 * conf + rng.normal(0, 2, size=n) > 48).astype(float)
    revenue = 0.6 * spend + 8 * treatment + 2 * conf + rng.normal(0, 3, size=n)
    conversion = 0.02 * spend + 0.15 * treatment + 0.05 * conf + rng.normal(0, 0.05, size=n)
    retention = 0.4 + 0.01 * revenue / 10 + 0.08 * treatment + rng.normal(0, 0.03, size=n)
    return pd.DataFrame(
        {
            "spend": spend,
            "confounder": conf,
            "treatment": treatment,
            "revenue": revenue,
            "conversion": conversion.clip(0, 1),
            "retention": retention.clip(0, 1),
            "noise": rng.normal(size=n),
        }
    )


def synthetic_markets(n: int = 100, seed: int = 2) -> pd.DataFrame:
    """Markets-as-dynamics analogy panel (price / return / volatility)."""
    rng = np.random.default_rng(seed)
    log_price = np.cumsum(0.001 + 0.02 * rng.normal(size=n))
    price = 100 * np.exp(log_price - log_price[0])
    ret = np.concatenate([[0.0], np.diff(np.log(price))])
    vol = pd.Series(ret).rolling(5, min_periods=1).std().fillna(0.01).to_numpy()
    volume = np.exp(10 + 0.5 * rng.normal(size=n) + 2 * vol)
    revenue = price * volume * 1e-4 + rng.normal(0, 1, size=n)
    leverage = 1.0 + 0.5 * (vol / (vol.mean() + 1e-9))
    return pd.DataFrame(
        {
            "price": price,
            "return": ret,
            "volatility": vol,
            "volume": volume,
            "revenue": revenue,
            "leverage": leverage,
            "interest": 0.03 + 0.01 * rng.normal(size=n).cumsum() / n,
        }
    )


def synthetic_affect(n: int = 80, seed: int = 3) -> pd.DataFrame:
    """Affect-as-dynamics analogy panel (valence / arousal)."""
    rng = np.random.default_rng(seed)
    valence = np.tanh(np.cumsum(0.05 * rng.normal(size=n)))
    arousal = np.abs(np.sin(np.linspace(0, 4 * np.pi, n)) + 0.2 * rng.normal(size=n))
    intent = (valence > 0).astype(float)
    treatment = ((arousal > np.median(arousal)) & (intent > 0)).astype(float)
    outcome = 0.5 * treatment + 0.3 * valence + 0.2 * arousal + 0.1 * rng.normal(size=n)
    emotion = np.where(valence > 0.3, 1.0, np.where(valence < -0.3, -1.0, 0.0))
    return pd.DataFrame(
        {
            "valence": valence,
            "arousal": arousal,
            "emotion": emotion,
            "intent": intent,
            "treatment": treatment,
            "outcome": outcome,
        }
    )


def bundled_sample_path(kind: SampleKind = "oscillator") -> Path:
    """Path under package data for a written CSV (created on demand by load_demo_frame)."""
    root = Path(__file__).resolve().parent.parent / "data" / "physics"
    return root / f"{kind}_demo.csv"


def load_demo_frame(
    kind: SampleKind = "oscillator",
    *,
    n: Optional[int] = None,
    seed: Optional[int] = None,
    write_bundled: bool = False,
) -> pd.DataFrame:
    """
    Load a demo frame: prefer bundled CSV if present, else synthesize.

    ``write_bundled=True`` writes the synthetic frame under ``autocausal/data/physics/``.
    """
    factories = {
        "oscillator": synthetic_oscillator,
        "kpi_panel": synthetic_kpi_panel,
        "markets": synthetic_markets,
        "affect": synthetic_affect,
    }
    factory = factories[kind]
    kwargs: dict = {}
    if n is not None:
        kwargs["n"] = n
    if seed is not None:
        kwargs["seed"] = seed

    path = bundled_sample_path(kind)
    if path.is_file() and n is None and seed is None:
        return pd.read_csv(path)

    df = factory(**kwargs)
    if write_bundled:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df
