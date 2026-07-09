"""Sensitivity metrics for scientific-database selection and physical grounding.

Offline-first: bootstrap / finite-diff stubs never require network or HF.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "SensitivityMetric",
    "SensitivityReport",
    "compute_sensitivity",
    "select_scientific_source",
    "DOMAIN_SOURCE_PRIOR",
]


# Soft prior: domain keyword → preferred scientific registry ids (ordered).
DOMAIN_SOURCE_PRIOR: dict[str, list[str]] = {
    "physics": ["physics_constants", "materials_stub"],
    "mechanics": ["physics_constants", "materials_stub"],
    "materials": ["materials_stub", "physics_constants"],
    "climate": ["climate_energy_stub", "noaa_open"],
    "energy": ["climate_energy_stub", "nasa_open"],
    "epidemiology": ["epi_panel_stub", "world_bank_open"],
    "health": ["epi_panel_stub", "pubmed_meta"],
    "genomics": ["genomics_lite_stub", "pubmed_meta"],
    "biology": ["genomics_lite_stub", "pubmed_meta", "openalex_meta"],
    "finance": ["world_bank_open", "climate_energy_stub"],
    "policy": ["world_bank_open", "epi_panel_stub"],
    "markets": ["world_bank_open", "climate_energy_stub"],
    "affect": ["epi_panel_stub", "pubmed_meta"],
    "vision": ["physics_constants", "materials_stub"],
    "general": ["physics_constants", "epi_panel_stub", "climate_energy_stub"],
}


@dataclass
class SensitivityMetric:
    name: str
    value: float
    detail: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SensitivityReport:
    """Cross-talk artifact for CausalBridge / MineReport notes."""

    metrics: list[SensitivityMetric] = field(default_factory=list)
    domain_hint: str = "general"
    domain_match_scores: dict[str, float] = field(default_factory=dict)
    recommended_source: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    schema: str = "SensitivityReport.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "domain_hint": self.domain_hint,
            "domain_match_scores": self.domain_match_scores,
            "recommended_source": self.recommended_source,
            "metrics": [m.to_dict() for m in self.metrics],
            "notes": self.notes,
        }

    def metric_map(self) -> dict[str, float]:
        return {m.name: float(m.value) for m in self.metrics}

    def to_mine_notes(self) -> list[str]:
        notes = [
            f"sensitivity domain_hint={self.domain_hint}",
            f"recommended_scientific_source={self.recommended_source or 'none'}",
        ]
        for m in self.metrics:
            notes.append(f"sensitivity.{m.name}={m.value:.4f} ({m.detail})")
        return notes + list(self.notes)


def _tokens(*parts: str) -> set[str]:
    blob = " ".join(p for p in parts if p)
    return {t for t in re.split(r"[^a-zA-Z0-9]+", blob.lower()) if len(t) > 2}


def _domain_match_scores(
    columns: Sequence[str],
    text: str = "",
    domain: Optional[str] = None,
) -> dict[str, float]:
    """Score how well column/text tokens match each DOMAIN_SOURCE_PRIOR key."""
    toks = _tokens(*(list(columns) + [text, domain or ""]))
    scores: dict[str, float] = {}
    for dom, _srcs in DOMAIN_SOURCE_PRIOR.items():
        if dom == "general":
            scores[dom] = 0.15
            continue
        # token overlap with domain name + a few seed keywords
        seeds = {dom} | set(dom.split("-"))
        seed_extra = {
            "physics": {"force", "mass", "velocity", "energy", "damping", "momentum"},
            "mechanics": {"force", "acceleration", "position", "oscillator"},
            "materials": {"alloy", "crystal", "modulus", "hardness", "density"},
            "climate": {"temp", "temperature", "precip", "co2", "climate"},
            "energy": {"watt", "power", "kwh", "renewable", "grid"},
            "epidemiology": {"incidence", "prevalence", "cases", "infection", "mortality"},
            "health": {"patient", "dose", "treatment", "outcome", "hospital"},
            "genomics": {"gene", "snp", "expression", "sequence", "allele"},
            "biology": {"cell", "protein", "organism", "species"},
            "finance": {"return", "price", "ticker", "interest", "volume"},
            "policy": {"treatment", "eligibility", "post", "unit", "policy"},
            "markets": {"revenue", "campaign", "conversion", "spend"},
            "affect": {"valence", "arousal", "emotion", "intent"},
            "vision": {"frame", "motion", "clip", "object", "pixel"},
        }.get(dom, set())
        hits = len(toks & (seeds | seed_extra))
        scores[dom] = min(1.0, hits / max(3.0, len(seed_extra) * 0.35 + 1.0))
    if domain and domain in scores:
        scores[domain] = max(scores[domain], 0.85)
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def _missingness_sensitivity(df: pd.DataFrame) -> SensitivityMetric:
    if df.empty or len(df.columns) == 0:
        return SensitivityMetric("feature_missingness", 0.0, "empty frame")
    rates = df.isna().mean()
    mean_miss = float(rates.mean())
    # high variance across columns → imputation choice matters more
    var = float(rates.var()) if len(rates) > 1 else 0.0
    value = float(min(1.0, mean_miss + 0.5 * math.sqrt(var)))
    top = rates.sort_values(ascending=False).head(5)
    detail = ", ".join(f"{c}={v:.2f}" for c, v in top.items() if v > 0) or "no missing"
    return SensitivityMetric(
        "feature_missingness",
        round(value, 4),
        detail=detail,
        meta={"mean_missing": mean_miss, "per_column": {str(k): float(v) for k, v in rates.items()}},
    )


def _edge_stability(
    edges: Optional[Sequence[dict[str, Any]]],
    df: Optional[pd.DataFrame] = None,
    *,
    n_boot: int = 8,
    seed: int = 0,
) -> SensitivityMetric:
    """Bootstrap / sign-flip stub for causal edge stability.

    Without edges: estimate pairwise corr sign stability on numeric cols.
    With edges: re-sample rows and check whether |corr(source,target)| keeps sign.
    """
    rng = np.random.default_rng(seed)
    if df is None or df.empty:
        return SensitivityMetric("causal_edge_stability", 0.5, "no frame; neutral stub")

    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2 or len(num) < 4:
        return SensitivityMetric("causal_edge_stability", 0.5, "insufficient numeric data")

    pairs: list[tuple[str, str]] = []
    if edges:
        for e in edges:
            s, t = str(e.get("source", "")), str(e.get("target", ""))
            if s in num.columns and t in num.columns:
                pairs.append((s, t))
    if not pairs:
        cols = list(num.columns)[:6]
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                pairs.append((a, b))

    if not pairs:
        return SensitivityMetric("causal_edge_stability", 0.5, "no pairs")

    stables: list[float] = []
    n = len(num)
    for a, b in pairs[:12]:
        base = float(num[a].corr(num[b]))
        if not math.isfinite(base) or abs(base) < 1e-9:
            continue
        agree = 0
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sample = num.iloc[idx]
            c = float(sample[a].corr(sample[b]))
            if math.isfinite(c) and (c == 0 or (c > 0) == (base > 0)):
                agree += 1
        stables.append(agree / n_boot)

    if not stables:
        return SensitivityMetric("causal_edge_stability", 0.5, "no finite correlations")
    value = float(np.mean(stables))
    return SensitivityMetric(
        "causal_edge_stability",
        round(value, 4),
        detail=f"mean sign-agree over {len(stables)} pair(s), n_boot={n_boot}",
        meta={"n_pairs": len(stables), "n_boot": n_boot},
    )


def _physics_rollout_sensitivity(
    trajectory: Any = None,
    *,
    param_eps: float = 0.05,
) -> SensitivityMetric:
    """Finite-diff / energy-spread stub for ∂predict/∂param sensitivity.

    Uses trajectory point energy spread when available; else neutral mid value.
    """
    if trajectory is None:
        return SensitivityMetric(
            "physics_rollout_sensitivity",
            0.4,
            "no trajectory; neutral stub",
            meta={"param_eps": param_eps},
        )
    points = getattr(trajectory, "points", None) or []
    if not points:
        td = trajectory.to_dict() if hasattr(trajectory, "to_dict") else {}
        points = td.get("points") or []
    energies: list[float] = []
    for p in points:
        if hasattr(p, "kinetic_energy"):
            energies.append(float(p.kinetic_energy) + float(getattr(p, "potential_energy", 0.0)))
        elif isinstance(p, dict):
            energies.append(
                float(p.get("kinetic_energy") or 0.0) + float(p.get("potential_energy") or 0.0)
            )
    if len(energies) < 2:
        return SensitivityMetric(
            "physics_rollout_sensitivity",
            0.35,
            "short trajectory",
            meta={"param_eps": param_eps},
        )
    spread = float(np.std(energies))
    mean_e = float(np.mean(np.abs(energies))) + 1e-6
    # normalize: larger relative energy swing → higher sensitivity
    value = float(min(1.0, spread / mean_e))
    # finite-diff stub: perturb last energy by eps and report relative change
    last = energies[-1]
    pert = abs(last) * param_eps + param_eps
    rel = abs(pert) / (abs(last) + 1e-6)
    value = float(min(1.0, 0.6 * value + 0.4 * min(1.0, rel * 5)))
    return SensitivityMetric(
        "physics_rollout_sensitivity",
        round(value, 4),
        detail=f"energy_std={spread:.4f} finite_diff_rel≈{rel:.4f}",
        meta={"param_eps": param_eps, "n_points": len(energies)},
    )


def compute_sensitivity(
    df: Optional[pd.DataFrame] = None,
    *,
    edges: Optional[Sequence[dict[str, Any]]] = None,
    trajectory: Any = None,
    text: str = "",
    domain: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    n_boot: int = 8,
    seed: int = 0,
    auto_select: bool = True,
) -> SensitivityReport:
    """Compute sensitivity metrics and optionally pick a scientific source."""
    cols = list(columns) if columns is not None else (list(df.columns) if df is not None else [])
    domain_scores = _domain_match_scores(cols, text=text, domain=domain)
    best_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "general"
    if domain and domain in domain_scores:
        best_domain = domain

    metrics: list[SensitivityMetric] = []
    if df is not None:
        metrics.append(_missingness_sensitivity(df))
        metrics.append(_edge_stability(edges, df, n_boot=n_boot, seed=seed))
    else:
        metrics.append(SensitivityMetric("feature_missingness", 0.0, "no frame"))
        metrics.append(SensitivityMetric("causal_edge_stability", 0.5, "no frame"))
    metrics.append(_physics_rollout_sensitivity(trajectory))

    # domain match as a first-class metric
    top_score = float(domain_scores.get(best_domain, 0.0))
    metrics.append(
        SensitivityMetric(
            "domain_match",
            round(top_score, 4),
            detail=f"best_domain={best_domain}",
            meta={"scores": domain_scores},
        )
    )

    report = SensitivityReport(
        metrics=metrics,
        domain_hint=best_domain,
        domain_match_scores=domain_scores,
        notes=[
            "Sensitivity metrics are exploratory stubs for source selection — not formal ID tests.",
        ],
    )
    if auto_select:
        report.recommended_source = select_scientific_source(report, domain=best_domain)
        report.notes.append(f"auto-selected scientific source: {report.recommended_source}")
    return report


def select_scientific_source(
    sensitivity_report: Union[SensitivityReport, dict[str, Any]],
    domain: Optional[str] = None,
) -> str:
    """Map sensitivity + domain → scientific registry ``database_id``.

    Selection logic (offline, deterministic):
    1. Prefer ``DOMAIN_SOURCE_PRIOR[domain]`` ordered list.
    2. If missingness high → prefer panel/stub sources with rich covariates
       (epi / climate) over sparse constants.
    3. If physics rollout sensitivity high → prefer ``physics_constants`` /
       ``materials_stub``.
    4. If edge stability low → prefer metadata search stubs (pubmed/openalex)
       for literature grounding rather than numeric joins.
    5. Fall back to ``physics_constants`` (always bundled).
    """
    if isinstance(sensitivity_report, dict):
        domain_hint = str(sensitivity_report.get("domain_hint") or domain or "general")
        mmap = {
            m.get("name"): float(m.get("value") or 0.0)
            for m in (sensitivity_report.get("metrics") or [])
            if isinstance(m, dict)
        }
        recommended = sensitivity_report.get("recommended_source")
        if recommended and domain is None:
            return str(recommended)
    else:
        domain_hint = domain or sensitivity_report.domain_hint or "general"
        mmap = sensitivity_report.metric_map()

    priors = list(DOMAIN_SOURCE_PRIOR.get(domain_hint) or DOMAIN_SOURCE_PRIOR["general"])
    miss = mmap.get("feature_missingness", 0.0)
    edge_stab = mmap.get("causal_edge_stability", 0.5)
    phys = mmap.get("physics_rollout_sensitivity", 0.4)
    dom_match = mmap.get("domain_match", 0.0)

    # re-rank priors with soft bonuses
    scored: list[tuple[float, str]] = []
    for i, sid in enumerate(priors):
        score = 10.0 - i  # base rank
        if miss >= 0.25 and sid in ("epi_panel_stub", "climate_energy_stub", "world_bank_open"):
            score += 2.0
        if phys >= 0.55 and sid in ("physics_constants", "materials_stub"):
            score += 2.5
        if edge_stab < 0.45 and sid in ("pubmed_meta", "openalex_meta"):
            score += 1.5
        if dom_match >= 0.5:
            score += 1.0
        # prefer bundled offline sources slightly
        if sid.endswith("_stub") or sid == "physics_constants":
            score += 0.5
        scored.append((score, sid))
    scored.sort(key=lambda x: -x[0])
    return scored[0][1] if scored else "physics_constants"
