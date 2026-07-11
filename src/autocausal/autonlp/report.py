"""Structured AutoNLP report contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from autocausal.autonlp.features import NLPFeaturePlan
from autocausal.autonlp.profile import NLPProfile
from autocausal.autonlp.roles import (
    CausalClaim,
    RoleHypothesis,
    role_hypotheses_to_guide_context,
)


NLP_CAVEAT = (
    "NLP-derived claims and causal roles are linguistic hypotheses. They do not "
    "identify causal effects or verify study-design assumptions."
)


@dataclass
class AutoNLPReport:
    profile: NLPProfile
    claims: list[CausalClaim] = field(default_factory=list)
    role_hypotheses: list[RoleHypothesis] = field(default_factory=list)
    feature_plans: list[NLPFeaturePlan] = field(default_factory=list)
    analysis_summary: dict[str, Any] = field(default_factory=dict)
    privacy: dict[str, Any] = field(default_factory=dict)
    external_enrichment: Optional[dict[str, Any]] = None
    mode: str = "exploratory"
    notes: list[str] = field(default_factory=list)
    schema: str = "AutoCausalAutoNLPReport.v1"

    def to_guide_context(self) -> dict[str, Any]:
        context = role_hypotheses_to_guide_context(self.role_hypotheses)
        claims = [
            claim.to_dict(production=self.mode == "production")
            for claim in self.claims[:100]
        ]
        context["nlp_claims"] = [
            {
                "treatment": claim["treatment"],
                "outcome": claim["outcome"],
                "negated": claim["negated"],
                "uncertain": claim["uncertain"],
                "confidence": claim["confidence"],
                "hypothesis": True,
            }
            for claim in claims
        ]
        context["notes"] = list(context.get("notes") or []) + [NLP_CAVEAT]
        return context

    def to_dict(self) -> dict[str, Any]:
        production = self.mode == "production"
        return {
            "schema": self.schema,
            "mode": self.mode,
            "profile": self.profile.to_dict(),
            "claims": [
                claim.to_dict(production=production) for claim in self.claims
            ],
            "role_hypotheses": [
                hypothesis.to_dict(production=production)
                for hypothesis in self.role_hypotheses
            ],
            "feature_plans": [plan.to_dict() for plan in self.feature_plans],
            "analysis_summary": dict(self.analysis_summary),
            "privacy": dict(self.privacy),
            "external_enrichment": self.external_enrichment,
            "guide_context": self.to_guide_context(),
            "notes": list(self.notes),
            "epistemic_caveat": NLP_CAVEAT,
            "contains_raw_documents": False,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# AutoNLP report",
            "",
            f"- mode: `{self.mode}`",
            f"- text columns: {len(self.profile.text_columns)}",
            f"- extracted claims: {len(self.claims)}",
            f"- role hypotheses: {len(self.role_hypotheses)}",
            f"- privacy risk: `{self.profile.privacy_risk}`",
            "",
            f"> {NLP_CAVEAT}",
            "",
            "## Text profile",
            "",
            "| column | missing | duplicate | mean length | PII hits | secret hits |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for profile in self.profile.text_columns:
            lines.append(
                f"| `{profile.column}` | {profile.missing_fraction:.1%} | "
                f"{profile.duplication_fraction:.1%} | {profile.mean_length:.1f} | "
                f"{sum(profile.pii_risk.values())} | "
                f"{sum(profile.secret_risk.values())} |"
            )
        if self.claims:
            lines.extend(["", "## Linguistic claims", ""])
            for claim in self.claims[:25]:
                safe_claim = claim.to_dict(production=self.mode == "production")
                flags = ", ".join(claim.modality) or "assertive wording"
                lines.append(
                    f"- `{safe_claim['treatment']}` → `{safe_claim['outcome']}` "
                    f"({flags}; confidence={claim.confidence:.2f}; hypothesis)"
                )
        if self.feature_plans:
            lines.extend(["", "## Feature plans", ""])
            for plan in self.feature_plans:
                lines.append(
                    f"- `{', '.join(plan.text_columns)}`: {plan.vectorizer}, "
                    f"ngrams={plan.ngram_range}, fit_scope={plan.fit_scope}"
                )
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
        lines.append("")
        return "\n".join(lines)

    def report(self, *, as_markdown: bool = True) -> str:
        return self.to_markdown() if as_markdown else self.to_json()

    def write(self, path: str | Path, *, fmt: str = "auto") -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        selected = fmt.lower()
        if selected == "auto":
            selected = "json" if output.suffix.lower() == ".json" else "markdown"
        if selected not in ("json", "markdown", "md"):
            raise ValueError("AutoNLPReport.write fmt must be json or markdown")
        output.write_text(
            self.to_json() if selected == "json" else self.to_markdown(),
            encoding="utf-8",
        )
        return output


__all__ = ["AutoNLPReport", "NLP_CAVEAT"]
