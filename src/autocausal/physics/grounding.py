"""Physical insight grounding — link causal edges to dynamical mechanisms."""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence, Union

from autocausal.physics.types import (
    PhysicalGroundingReport,
    PhysicalInsight,
    Trajectory,
)

__all__ = [
    "PHYSICS_DOMAIN_GLOSSARIES",
    "ground_physical",
    "merge_with_domain_grounding",
]

# Domain glossaries carefully labeled: literal mechanics vs analogy domains.
PHYSICS_DOMAIN_GLOSSARIES: dict[str, dict[str, dict[str, Any]]] = {
    "mechanics-lite": {
        "force": {
            "label": "Force / drive",
            "mechanism": "force_balance",
            "relations": ["acceleration", "momentum", "velocity", "mass"],
            "evidence": "Newton: net force relates to acceleration (F ≈ ma).",
            "analogy_label": "literal",
        },
        "velocity": {
            "label": "Velocity",
            "mechanism": "kinematics",
            "relations": ["position", "acceleration", "momentum", "motion"],
            "evidence": "Kinematic state: position integrates velocity.",
            "analogy_label": "literal",
        },
        "momentum": {
            "label": "Momentum",
            "mechanism": "momentum_transfer",
            "relations": ["force", "velocity", "mass", "impulse"],
            "evidence": "Momentum change tracks impulse / force over time.",
            "analogy_label": "literal",
        },
        "energy": {
            "label": "Energy",
            "mechanism": "energy_exchange",
            "relations": ["kinetic", "potential", "damping", "power"],
            "evidence": "Energy exchange / dissipation under damping.",
            "analogy_label": "literal",
        },
        "damping": {
            "label": "Damping / friction",
            "mechanism": "dissipation",
            "relations": ["velocity", "energy", "stability"],
            "evidence": "Damping dissipates kinetic energy toward equilibrium.",
            "analogy_label": "literal",
        },
        "stability": {
            "label": "Stability",
            "mechanism": "stability",
            "relations": ["damping", "stiffness", "equilibrium"],
            "evidence": "Stable fixed points when restoring + damping dominate.",
            "analogy_label": "literal",
        },
        "acceleration": {
            "label": "Acceleration",
            "mechanism": "force_balance",
            "relations": ["force", "velocity", "position"],
            "evidence": "Second derivative of position; responds to net force.",
            "analogy_label": "literal",
        },
        "mass": {
            "label": "Mass / inertia",
            "mechanism": "inertia",
            "relations": ["force", "acceleration", "momentum"],
            "evidence": "Inertia resists acceleration for a given force.",
            "analogy_label": "literal",
        },
        "position": {
            "label": "Position / state",
            "mechanism": "kinematics",
            "relations": ["velocity", "displacement"],
            "evidence": "Configuration variable evolved by the dynamical system.",
            "analogy_label": "literal",
        },
        "motion": {
            "label": "Motion",
            "mechanism": "kinematics",
            "relations": ["velocity", "frame", "object"],
            "evidence": "Motion features mediate state change across time.",
            "analogy_label": "literal",
        },
    },
    "markets-as-dynamics": {
        "price": {
            "label": "Price (state analogy)",
            "mechanism": "mean_reversion_analogy",
            "relations": ["return", "volatility", "volume", "revenue"],
            "evidence": (
                "ANALOGY: price as position; returns as velocity; "
                "mean-reversion ≈ restoring force — not literal physics."
            ),
            "analogy_label": "analogy",
        },
        "return": {
            "label": "Return (velocity analogy)",
            "mechanism": "momentum_analogy",
            "relations": ["price", "volatility", "momentum"],
            "evidence": "ANALOGY: returns ≈ velocity of log-price state.",
            "analogy_label": "analogy",
        },
        "volatility": {
            "label": "Volatility (energy/noise analogy)",
            "mechanism": "diffusion_analogy",
            "relations": ["return", "price", "risk"],
            "evidence": "ANALOGY: volatility ≈ diffusion / kinetic energy scale.",
            "analogy_label": "analogy",
        },
        "volume": {
            "label": "Volume (mass/flow analogy)",
            "mechanism": "inertia_analogy",
            "relations": ["price", "liquidity", "return"],
            "evidence": "ANALOGY: volume as inertial / flow mass proxy.",
            "analogy_label": "analogy",
        },
        "revenue": {
            "label": "Revenue (output analogy)",
            "mechanism": "driven_output_analogy",
            "relations": ["price", "volume", "treatment", "campaign"],
            "evidence": "ANALOGY: revenue as driven output of price×quantity dynamics.",
            "analogy_label": "analogy",
        },
        "interest": {
            "label": "Interest rate (forcing analogy)",
            "mechanism": "exogenous_force_analogy",
            "relations": ["investment", "default", "price"],
            "evidence": "ANALOGY: rates as exogenous forcing on investment/default.",
            "analogy_label": "analogy",
        },
        "leverage": {
            "label": "Leverage (gain analogy)",
            "mechanism": "gain_amplification_analogy",
            "relations": ["default", "volatility", "return"],
            "evidence": "ANALOGY: leverage amplifies shocks like gain in a linear system.",
            "analogy_label": "analogy",
        },
    },
    "affect-as-dynamics": {
        "valence": {
            "label": "Affect valence (state analogy)",
            "mechanism": "affect_state_analogy",
            "relations": ["arousal", "emotion", "stability"],
            "evidence": (
                "ANALOGY: valence as slow affective state — "
                "not a physical conserved quantity."
            ),
            "analogy_label": "analogy",
        },
        "arousal": {
            "label": "Affect arousal (energy analogy)",
            "mechanism": "affect_energy_analogy",
            "relations": ["valence", "motion", "intensity"],
            "evidence": "ANALOGY: arousal ≈ kinetic / activation energy of affect.",
            "analogy_label": "analogy",
        },
        "emotion": {
            "label": "Emotion label (regime analogy)",
            "mechanism": "regime_analogy",
            "relations": ["valence", "arousal", "intent"],
            "evidence": "ANALOGY: discrete emotion as regime of continuous affect dynamics.",
            "analogy_label": "analogy",
        },
        "intent": {
            "label": "Intent (control analogy)",
            "mechanism": "control_input_analogy",
            "relations": ["emotion", "action", "treatment"],
            "evidence": "ANALOGY: intent as control input / intervention on affect state.",
            "analogy_label": "analogy",
        },
        "treatment": {
            "label": "Treatment (impulse analogy)",
            "mechanism": "impulse_analogy",
            "relations": ["outcome", "revenue", "conversion"],
            "evidence": "ANALOGY: treatment as impulse / force applied to outcome state.",
            "analogy_label": "analogy",
        },
        "outcome": {
            "label": "Outcome (response analogy)",
            "mechanism": "response_analogy",
            "relations": ["treatment", "confounder"],
            "evidence": "ANALOGY: outcome as driven response of the dynamical system.",
            "analogy_label": "analogy",
        },
    },
}


def _tokens(name: str) -> list[str]:
    parts = re.split(r"[^a-zA-Z0-9]+", str(name).lower())
    return [p for p in parts if p]


def _hits_for(
    col: str,
    domains: Sequence[str],
) -> list[tuple[str, str, dict[str, Any]]]:
    toks = _tokens(col)
    hits: list[tuple[str, str, dict[str, Any]]] = []
    for domain in domains:
        glossary = PHYSICS_DOMAIN_GLOSSARIES.get(domain) or {}
        for key, meta in glossary.items():
            if key in toks or any(key in t or t in key for t in toks if len(t) > 2):
                hits.append((domain, key, meta))
    return hits


def _trajectory_signal(
    trajectory: Optional[Trajectory],
    source: str,
    target: str,
) -> str:
    if trajectory is None or not trajectory.points:
        return ""
    names = trajectory.points[0].state.names
    name_l = {n.lower(): n for n in names}
    src_n = name_l.get(source.lower())
    tgt_n = name_l.get(target.lower())
    parts: list[str] = []
    last = trajectory.points[-1]
    ke = last.kinetic_energy
    pe = last.potential_energy
    parts.append(f"At horizon t+{last.t}: KE={ke:.3f}, PE={pe:.3f}.")
    if src_n and src_n in names:
        i = names.index(src_n)
        parts.append(f"`{src_n}`→{last.state.position[i]:.3f} (±{last.uncertainty[i]:.3f}).")
    if tgt_n and tgt_n in names:
        j = names.index(tgt_n)
        parts.append(f"`{tgt_n}`→{last.state.position[j]:.3f} (±{last.uncertainty[j]:.3f}).")
    # stability heuristic
    if ke + pe < 0.05:
        parts.append("Trajectory near equilibrium (low energy) → stability-like regime.")
    elif ke > pe * 2:
        parts.append("Kinetic-dominated trajectory → momentum / transient drive.")
    return " ".join(parts)


def ground_physical(
    edges: list[dict[str, Any]],
    trajectory: Optional[Trajectory] = None,
    *,
    domain: Union[str, Sequence[str]] = "auto",
) -> PhysicalGroundingReport:
    """
    Link causal edges to physical (or carefully labeled analogy) mechanisms.

    ``domain``: ``mechanics-lite`` | ``markets-as-dynamics`` | ``affect-as-dynamics``
    | ``auto`` (all) | list of domains.
    """
    if domain == "auto":
        domains: list[str] = list(PHYSICS_DOMAIN_GLOSSARIES.keys())
    elif isinstance(domain, str):
        domains = [domain]
    else:
        domains = list(domain)

    insights: list[PhysicalInsight] = []
    glossary_hits: list[dict[str, Any]] = []
    notes = [
        "Physical grounding is interpretive: literal for mechanics-lite; "
        "analogy-labeled for markets/affect/policy proxies.",
        "Does not claim conservation laws hold for non-physical domains.",
    ]

    for e in edges:
        src, tgt = str(e.get("source", "")), str(e.get("target", ""))
        if not src or not tgt:
            continue
        src_hits = _hits_for(src, domains)
        tgt_hits = _hits_for(tgt, domains)
        for d, k, _meta in src_hits:
            glossary_hits.append({"column": src, "domain": d, "key": k})
        for d, k, _meta in tgt_hits:
            glossary_hits.append({"column": tgt, "domain": d, "key": k})

        mechanism = "coupled_dynamics"
        analogy_label = "analogy"
        evidence_parts: list[str] = []
        domain_used = domains[0] if domains else "mechanics-lite"
        related = False

        for d, key, meta in src_hits:
            domain_used = d
            mechanism = str(meta.get("mechanism") or mechanism)
            analogy_label = str(meta.get("analogy_label") or analogy_label)
            evidence_parts.append(str(meta.get("evidence") or ""))
            rels = [str(r).lower() for r in meta.get("relations", [])]
            if any(r in _tokens(tgt) or any(r in t for t in _tokens(tgt)) for r in rels):
                related = True

        for d, key, meta in tgt_hits:
            domain_used = d
            if not src_hits:
                mechanism = str(meta.get("mechanism") or mechanism)
                analogy_label = str(meta.get("analogy_label") or analogy_label)
            evidence_parts.append(str(meta.get("evidence") or ""))
            rels = [str(r).lower() for r in meta.get("relations", [])]
            if any(r in _tokens(src) or any(r in t for t in _tokens(src)) for r in rels):
                related = True

        if not src_hits and not tgt_hits:
            # generic dynamical reading from edge score
            score = float(e.get("score") or e.get("confidence") or 0.0)
            mechanism = "edge_coupled_force"
            analogy_label = "analogy"
            evidence_parts.append(
                f"No glossary hit; treat `{src}`→`{tgt}` as soft force coupling "
                f"(score={score:.3f}) in the rollout graph."
            )
            conf = 0.25
        elif related:
            conf = 0.7
        else:
            conf = 0.45
            if not evidence_parts:
                evidence_parts.append("Partial glossary match on edge endpoints.")

        edge_conf = float(e.get("confidence") or e.get("score") or 0.0)
        conf = float(min(1.0, 0.55 * conf + 0.45 * min(1.0, abs(edge_conf))))

        insights.append(
            PhysicalInsight(
                source=src,
                target=tgt,
                mechanism=mechanism,
                domain=domain_used,
                analogy_label=analogy_label,
                confidence=round(conf, 3),
                evidence=" ".join(evidence_parts)[:500],
                trajectory_signal=_trajectory_signal(trajectory, src, tgt),
            )
        )

    primary = domains[0] if len(domains) == 1 else "auto"
    return PhysicalGroundingReport(
        insights=insights,
        domain=primary,
        method="physics_glossary+trajectory",
        glossary_hits=glossary_hits[:40],
        notes=notes,
    )


def merge_with_domain_grounding(
    physical: PhysicalGroundingReport,
    domain_report: Any,
) -> PhysicalGroundingReport:
    """Attach existing ``autocausal.grounding.GroundingReport`` into physical report."""
    merged = dict(physical.to_dict())
    if domain_report is None:
        return physical
    if hasattr(domain_report, "to_dict"):
        merged["merged_grounding"] = domain_report.to_dict()
    elif isinstance(domain_report, dict):
        merged["merged_grounding"] = domain_report
    else:
        return physical
    physical.merged_grounding = merged["merged_grounding"]
    physical.method = "physics_glossary+trajectory+domain_grounding"
    physical.notes = list(physical.notes) + [
        "Merged with autocausal.grounding domain glossary claims."
    ]
    return physical
