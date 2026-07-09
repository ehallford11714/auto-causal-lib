"""Real-world grounding for proposed causal edges."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


__all__ = [
    "GroundedClaim",
    "GroundingReport",
    "ground_edges",
    "DOMAIN_GLOSSARIES",
]


# Curated offline domain glossaries (stubs). Keys are lowercase tokens.
DOMAIN_GLOSSARIES: dict[str, dict[str, dict[str, Any]]] = {
    "finance": {
        "revenue": {
            "label": "Revenue / sales income",
            "relations": ["price", "volume", "conversion", "churn", "marketing"],
            "evidence": "Standard accounting identity: revenue ≈ price × quantity.",
            "source": "offline:finance_glossary",
        },
        "price": {
            "label": "Unit price",
            "relations": ["demand", "revenue", "elasticity"],
            "evidence": "Price often affects demand; reverse causality with revenue possible.",
            "source": "offline:finance_glossary",
        },
        "interest": {
            "label": "Interest rate",
            "relations": ["investment", "loan", "default"],
            "evidence": "Macro/finance: rates influence borrowing and investment.",
            "source": "offline:finance_glossary",
        },
        "default": {
            "label": "Credit default",
            "relations": ["leverage", "income", "interest"],
            "evidence": "Credit risk literature links leverage and ability-to-pay to default.",
            "source": "offline:finance_glossary",
        },
    },
    "marketing": {
        "treatment": {
            "label": "Treatment / campaign exposure",
            "relations": ["outcome", "conversion", "revenue", "ctr"],
            "evidence": "A/B and uplift literature: exposure may affect conversion/revenue.",
            "source": "offline:marketing_glossary",
        },
        "campaign": {
            "label": "Marketing campaign",
            "relations": ["ctr", "conversion", "spend", "revenue"],
            "evidence": "Campaign spend/exposure commonly hypothesized to drive conversions.",
            "source": "offline:marketing_glossary",
        },
        "conversion": {
            "label": "Conversion rate / event",
            "relations": ["campaign", "price", "ctr", "revenue"],
            "evidence": "Funnel metrics: CTR → conversion → revenue is a common causal story.",
            "source": "offline:marketing_glossary",
        },
        "ctr": {
            "label": "Click-through rate",
            "relations": ["campaign", "creative", "conversion"],
            "evidence": "Ad response models treat CTR as intermediate outcome of creatives.",
            "source": "offline:marketing_glossary",
        },
        "churn": {
            "label": "Customer churn",
            "relations": ["satisfaction", "price", "engagement", "retention"],
            "evidence": "Retention research: dissatisfaction and price shocks raise churn risk.",
            "source": "offline:marketing_glossary",
        },
    },
    "policy": {
        "instrument": {
            "label": "Instrument / assignment",
            "relations": ["treatment", "outcome"],
            "evidence": "IV design: instrument affects treatment, not outcome except via treatment.",
            "source": "offline:policy_glossary",
        },
        "assignment": {
            "label": "Policy assignment / lottery",
            "relations": ["treatment", "eligibility"],
            "evidence": "Randomized assignment is a classic instrument for treatment uptake.",
            "source": "offline:policy_glossary",
        },
        "eligibility": {
            "label": "Program eligibility",
            "relations": ["treatment", "outcome"],
            "evidence": "RDD/IV settings often use eligibility cutoffs as exogenous variation.",
            "source": "offline:policy_glossary",
        },
        "outcome": {
            "label": "Policy outcome",
            "relations": ["treatment", "confounder"],
            "evidence": "Outcomes are downstream of treatments conditional on confounders.",
            "source": "offline:policy_glossary",
        },
    },
    "vision": {
        "frame": {
            "label": "Video / image frame feature",
            "relations": ["motion", "object", "next_frame"],
            "evidence": "Vision pipelines: past frames inform next-frame / motion prediction.",
            "source": "offline:vision_glossary",
        },
        "motion": {
            "label": "Motion / optical flow",
            "relations": ["frame", "object"],
            "evidence": "Motion features mediate appearance change across frames.",
            "source": "offline:vision_glossary",
        },
        "object": {
            "label": "Detected object",
            "relations": ["frame", "label", "bbox"],
            "evidence": "Detection models map image features to object presence/labels.",
            "source": "offline:vision_glossary",
        },
    },
}


@dataclass
class GroundedClaim:
    source: str
    target: str
    label: str  # unsupported | plausible | documented
    confidence: float
    domains: list[str] = field(default_factory=list)
    citations: list[dict[str, str]] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundingReport:
    claims: list[GroundedClaim]
    method: str = "glossary+optional_web"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "claims": [c.to_dict() for c in self.claims],
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        lines = ["# Real-world grounding", "", f"**Method:** `{self.method}`", ""]
        if not self.claims:
            lines.append("_No edges to ground._")
            lines.append("")
            return "\n".join(lines)
        lines.append("| edge | label | confidence | domains | sources |")
        lines.append("|---|---|---:|---|---|")
        for c in self.claims:
            edge = f"`{c.source}` → `{c.target}`"
            domains = ", ".join(c.domains) or "—"
            srcs = "; ".join(x.get("source", "") for x in c.citations[:3]) or "—"
            lines.append(
                f"| {edge} | {c.label} | {c.confidence:.2f} | {domains} | {srcs} |"
            )
        lines.append("")
        for c in self.claims:
            if c.rationale:
                lines.append(f"- **{c.source}→{c.target}:** {c.rationale}")
        lines.append("")
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")
        return "\n".join(lines)


def _tokens(name: str) -> list[str]:
    parts = re.split(r"[^a-zA-Z0-9]+", str(name).lower())
    return [p for p in parts if p]


def _glossary_hits(col: str) -> list[tuple[str, str, dict[str, Any]]]:
    toks = _tokens(col)
    hits: list[tuple[str, str, dict[str, Any]]] = []
    for domain, glossary in DOMAIN_GLOSSARIES.items():
        for key, meta in glossary.items():
            if key in toks or any(key in t or t in key for t in toks if len(t) > 2):
                hits.append((domain, key, meta))
    return hits


def _try_causalsearch(query: str, timeout: float = 3.0) -> list[dict[str, str]]:
    """Soft reuse of CausalSearch if installed."""
    try:
        import causalsearch  # type: ignore
    except ImportError:
        return []
    try:
        if hasattr(causalsearch, "search"):
            hits = causalsearch.search(query, limit=3)  # type: ignore[attr-defined]
            out: list[dict[str, str]] = []
            for h in hits or []:
                if isinstance(h, dict):
                    out.append(
                        {
                            "source": str(h.get("source") or h.get("url") or "causalsearch"),
                            "title": str(h.get("title") or query),
                            "snippet": str(h.get("snippet") or h.get("text") or ""),
                        }
                    )
            return out
    except Exception:
        return []
    return []


def _try_web_ground(query: str, timeout: float = 3.0) -> list[dict[str, str]]:
    """Optional duckduckgo/httpx grounding; soft-fail with timeout."""
    if os.environ.get("AUTOCAUSAL_NO_WEB", "").strip() in ("1", "true", "yes"):
        return []
    # Prefer duckduckgo_search if present
    try:
        from duckduckgo_search import DDGS  # type: ignore

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return [
            {
                "source": str(r.get("href") or r.get("link") or "duckduckgo"),
                "title": str(r.get("title") or query),
                "snippet": str(r.get("body") or r.get("snippet") or ""),
            }
            for r in results
        ]
    except Exception:
        pass
    # httpx Instant Answer API (no key)
    try:
        import httpx  # type: ignore

        url = "https://api.duckduckgo.com/"
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url, params={"q": query, "format": "json", "no_html": 1})
            if r.status_code != 200:
                return []
            data = r.json()
        cites: list[dict[str, str]] = []
        if data.get("AbstractText"):
            cites.append(
                {
                    "source": str(data.get("AbstractURL") or "duckduckgo"),
                    "title": str(data.get("Heading") or query),
                    "snippet": str(data["AbstractText"])[:280],
                }
            )
        for t in (data.get("RelatedTopics") or [])[:2]:
            if isinstance(t, dict) and t.get("Text"):
                cites.append(
                    {
                        "source": str(t.get("FirstURL") or "duckduckgo"),
                        "title": query,
                        "snippet": str(t["Text"])[:280],
                    }
                )
        return cites
    except Exception:
        return []


def ground_edges(
    edges: list[dict[str, Any]],
    *,
    use_web: bool = False,
    timeout: float = 3.0,
) -> GroundingReport:
    """Map edge hypotheses to domain glossaries (+ optional soft web evidence)."""
    claims: list[GroundedClaim] = []
    notes = [
        "Grounding is heuristic: glossary matches and optional soft web evidence.",
        "Labels: documented (glossary+relation), plausible (partial), unsupported.",
    ]
    for e in edges:
        src, tgt = str(e.get("source", "")), str(e.get("target", ""))
        if not src or not tgt:
            continue
        src_hits = _glossary_hits(src)
        tgt_hits = _glossary_hits(tgt)
        domains = sorted({d for d, _, _ in src_hits + tgt_hits})
        citations: list[dict[str, str]] = []
        rationale_parts: list[str] = []
        related = False

        for domain, key, meta in src_hits:
            citations.append(
                {
                    "source": str(meta.get("source", "offline")),
                    "title": f"{domain}:{key}",
                    "snippet": str(meta.get("evidence", "")),
                }
            )
            rels = [str(r).lower() for r in meta.get("relations", [])]
            tgt_toks = _tokens(tgt)
            if any(r in tgt_toks or any(r in t for t in tgt_toks) for r in rels):
                related = True
                rationale_parts.append(
                    f"Glossary `{domain}/{key}` lists relation toward `{tgt}`."
                )

        for domain, key, meta in tgt_hits:
            citations.append(
                {
                    "source": str(meta.get("source", "offline")),
                    "title": f"{domain}:{key}",
                    "snippet": str(meta.get("evidence", "")),
                }
            )
            rels = [str(r).lower() for r in meta.get("relations", [])]
            src_toks = _tokens(src)
            if any(r in src_toks or any(r in t for t in src_toks) for r in rels):
                related = True
                rationale_parts.append(
                    f"Glossary `{domain}/{key}` is consistent with cause `{src}`."
                )

        if use_web:
            q = f"causal relationship between {src} and {tgt}"
            web = _try_causalsearch(q, timeout=timeout) or _try_web_ground(q, timeout=timeout)
            citations.extend(web)
            if web:
                rationale_parts.append(f"Optional web/evidence returned {len(web)} hit(s).")

        if related and (src_hits or tgt_hits):
            label = "documented"
            conf = 0.75
        elif src_hits or tgt_hits:
            label = "plausible"
            conf = 0.45
            if not rationale_parts:
                rationale_parts.append("Column names match domain glossary terms.")
        else:
            label = "unsupported"
            conf = 0.15
            rationale_parts.append("No glossary match; treat as data-driven hypothesis only.")

        # blend with discovery confidence if present
        edge_conf = float(e.get("confidence") or e.get("score") or 0.0)
        conf = float(min(1.0, 0.6 * conf + 0.4 * min(1.0, edge_conf)))

        claims.append(
            GroundedClaim(
                source=src,
                target=tgt,
                label=label,
                confidence=round(conf, 3),
                domains=domains,
                citations=citations[:8],
                rationale=" ".join(rationale_parts),
            )
        )

    method = "glossary"
    if use_web:
        method = "glossary+optional_web"
    return GroundingReport(claims=claims, method=method, notes=notes)
