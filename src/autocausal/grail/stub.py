"""Rich offline GRAIL stub — same primitive names as Kineteq, local heuristics.

This is an **embellished AutoCausal adaptation**, not a claim of parity with
live Kineteq GRAIL (LM conductor, genome archive, MCP tool surfaces).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

from autocausal.grail.types import (
    Assumption,
    CycleTrace,
    ExpertChain,
    ExpertStep,
    FoldDiagnosis,
    GrailReport,
    GraphMemoryNode,
    ImputationAudit,
)

__all__ = ["GrailStub", "DEFAULT_ROLES"]

DEFAULT_ROLES = (
    "imputer",
    "causal_analyst",
    "confounder_critic",
    "instrument_scout",
    "evidence_retriever",
    "synthesizer",
)

_ROLE_HINTS = {
    "treat": "treatment",
    "spend": "treatment",
    "campaign": "treatment",
    "policy": "treatment",
    "intervention": "treatment",
    "revenue": "outcome",
    "sales": "outcome",
    "churn": "outcome",
    "outcome": "outcome",
    "y_": "outcome",
    "z_": "instrument",
    "iv": "instrument",
    "instrument": "instrument",
    "assign": "instrument",
    "lottery": "instrument",
    "age": "confounder",
    "region": "confounder",
    "demo": "confounder",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", (text or "").lower()) if len(t) > 1]


def _col_names(context: Optional[dict[str, Any]]) -> list[str]:
    if not context:
        return []
    cols = context.get("columns") or []
    out: list[str] = []
    for c in cols:
        if isinstance(c, dict):
            out.append(str(c.get("name", c)))
        else:
            out.append(str(c))
    return out


def _guess_roles(columns: list[str]) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {
        "treatment": [],
        "outcome": [],
        "instrument": [],
        "confounder": [],
    }
    for c in columns:
        cl = c.lower()
        for hint, role in _ROLE_HINTS.items():
            if hint in cl:
                if c not in roles[role]:
                    roles[role].append(c)
                break
    return roles


def _genome_id(goal: str, domain: str, chain_length: int) -> str:
    h = hashlib.sha256(f"{goal}|{domain}|{chain_length}".encode("utf-8")).hexdigest()
    return f"grail_stub_{h[:12]}"


class GrailStub:
    """Offline GRAIL primitives: impute → compose → fold → run + memory/graph."""

    name = "grail_stub"

    def __init__(self, *, domain: str = "causal") -> None:
        self.domain = domain
        self._memory: list[GraphMemoryNode] = []

    def available(self) -> bool:
        return True

    def impute(
        self,
        goal: str,
        *,
        context: Optional[dict[str, Any]] = None,
        domain: Optional[str] = None,
    ) -> ImputationAudit:
        """Detect underspecified goal slots and declare ASSUMPTIONS."""
        domain = domain or self.domain
        columns = _col_names(context)
        roles = _guess_roles(columns)
        text = (context or {}).get("text") or goal
        tokens = set(_tokenize(text))

        underspecified: list[str] = []
        assumptions: list[Assumption] = []

        if not roles["treatment"]:
            underspecified.append("treatment")
            # pick column overlapping goal tokens
            pick = next(
                (c for c in columns if any(t in c.lower() for t in tokens)),
                columns[0] if columns else "treatment",
            )
            assumptions.append(
                Assumption(
                    parameter="treatment",
                    value=pick,
                    confidence=0.45 if columns else 0.25,
                    rationale="Imputed from column/token overlap (stub).",
                )
            )
        else:
            assumptions.append(
                Assumption(
                    parameter="treatment",
                    value=roles["treatment"][0],
                    confidence=0.7,
                    rationale="Inferred from column name heuristics.",
                )
            )

        if not roles["outcome"]:
            underspecified.append("outcome")
            pick = next(
                (
                    c
                    for c in columns
                    if c != assumptions[0].value
                    and any(t in c.lower() for t in ("y", "out", "rev", "sale"))
                ),
                columns[1] if len(columns) > 1 else "outcome",
            )
            assumptions.append(
                Assumption(
                    parameter="outcome",
                    value=pick,
                    confidence=0.4 if columns else 0.25,
                    rationale="Imputed outcome slot (stub).",
                )
            )
        else:
            assumptions.append(
                Assumption(
                    parameter="outcome",
                    value=roles["outcome"][0],
                    confidence=0.7,
                    rationale="Inferred from column name heuristics.",
                )
            )

        if roles["instrument"]:
            assumptions.append(
                Assumption(
                    parameter="instrument",
                    value=roles["instrument"][0],
                    confidence=0.65,
                    rationale="Instrument-like column name.",
                )
            )
        else:
            underspecified.append("instrument")
            assumptions.append(
                Assumption(
                    parameter="instrument",
                    value=None,
                    confidence=0.2,
                    rationale="No Z-like column; leave open for design review.",
                )
            )

        if not any(k in tokens for k in ("cause", "effect", "impact", "does", "lead")):
            underspecified.append("causal_question_framing")
            assumptions.append(
                Assumption(
                    parameter="framing",
                    value="exploratory_association",
                    confidence=0.55,
                    rationale="Goal lacks explicit causal verbs; treat as exploratory.",
                )
            )

        treat = next((a.value for a in assumptions if a.parameter == "treatment"), "?")
        outc = next((a.value for a in assumptions if a.parameter == "outcome"), "?")
        enriched = (
            f"{goal.strip()} "
            f"[ASSUMPTIONS: treatment=`{treat}`, outcome=`{outc}`, "
            f"domain=`{domain}`, exploratory≠identification]"
        ).strip()

        audit = ImputationAudit(
            original_goal=goal,
            enriched_goal=enriched,
            assumptions=assumptions,
            underspecified=underspecified,
            domain=domain,
            backend=self.name,
            notes=[
                "Offline grail_impute stub — declares assumptions; not live Kineteq.",
            ],
        )
        self._remember(
            GraphMemoryNode(
                key="impute:goal",
                kind="assumption",
                content=enriched[:400],
                score=0.9,
            )
        )
        for a in assumptions:
            self._remember(
                GraphMemoryNode(
                    key=f"assume:{a.parameter}",
                    kind="assumption",
                    content=f"{a.parameter}={a.value} ({a.confidence:.2f})",
                    score=a.confidence,
                    meta={"parameter": a.parameter, "value": a.value},
                )
            )
        return audit

    def compose(
        self,
        goal: str,
        *,
        context: Optional[dict[str, Any]] = None,
        chain_length: int = 3,
        imputation: Optional[ImputationAudit] = None,
        domain: Optional[str] = None,
    ) -> ExpertChain:
        """Compose a dense expert prompt chain (no LM execution)."""
        domain = domain or self.domain
        chain_length = max(1, min(int(chain_length), len(DEFAULT_ROLES)))
        audit = imputation or self.impute(goal, context=context, domain=domain)
        columns = _col_names(context)
        roles = _guess_roles(columns)
        edges = list((context or {}).get("edges") or [])

        treat = next(
            (a.value for a in audit.assumptions if a.parameter == "treatment"),
            (roles["treatment"] or [None])[0],
        )
        outc = next(
            (a.value for a in audit.assumptions if a.parameter == "outcome"),
            (roles["outcome"] or [None])[0],
        )

        steps: list[ExpertStep] = []
        for i, role in enumerate(DEFAULT_ROLES[:chain_length], start=1):
            if role == "imputer":
                prompt = (
                    f"Audit underspecified slots in: {audit.enriched_goal}. "
                    f"List ASSUMPTIONS with confidence."
                )
                charges = {"epistemic": 0.8, "certainty": 0.4, "constraint": 0.6}
            elif role == "causal_analyst":
                prompt = (
                    f"Propose exploratory edges for treatment=`{treat}` → "
                    f"outcome=`{outc}` given columns {columns[:12]}."
                )
                charges = {"epistemic": 0.7, "certainty": 0.35, "constraint": 0.5}
            elif role == "confounder_critic":
                conf = roles["confounder"] or columns[:3]
                prompt = (
                    f"Critique confounders {conf} and reverse-causality risks "
                    f"for `{treat}` ↔ `{outc}`."
                )
                charges = {"epistemic": 0.75, "certainty": 0.3, "constraint": 0.7}
            elif role == "instrument_scout":
                zs = roles["instrument"] or []
                prompt = (
                    f"Scout instruments {zs or '(none)'}; check relevance/exclusion "
                    f"as design hypotheses only."
                )
                charges = {"epistemic": 0.65, "certainty": 0.25, "constraint": 0.8}
            elif role == "evidence_retriever":
                prompt = (
                    f"Retrieve related associations/edges ({len(edges)} known) "
                    f"and form search queries for `{goal[:80]}`."
                )
                charges = {"epistemic": 0.6, "certainty": 0.4, "constraint": 0.4}
            else:
                prompt = (
                    f"Synthesize a DirectionPlan-style answer for: {goal[:120]}. "
                    "Emphasize exploratory caveats."
                )
                charges = {"epistemic": 0.85, "certainty": 0.45, "constraint": 0.55}
            steps.append(ExpertStep(step=i, role=role, prompt=prompt, charges=charges))

        mutation = (
            f"Mutate chain toward clearer treatment/outcome framing and "
            f"stronger confounder critique for domain={domain}."
        )
        chain = ExpertChain(
            goal=audit.enriched_goal,
            steps=steps,
            mutation_prompt=mutation,
            chain_length=len(steps),
            backend=self.name,
            notes=["Offline grail_compose stub — prompts only, no LM conductor."],
        )
        self._remember(
            GraphMemoryNode(
                key="compose:chain",
                kind="query",
                content=f"chain_length={len(steps)} roles={[s.role for s in steps]}",
                score=0.8,
            )
        )
        return chain

    def fold(self, chain: ExpertChain) -> FoldDiagnosis:
        """Lightweight T/V proxy over chain charges (not full Kineteq fold)."""
        per_step: list[dict[str, Any]] = []
        kinetic = 0.0
        potential = 0.0
        prev_ep: Optional[float] = None
        for s in chain.steps:
            ep = float(s.charges.get("epistemic", 0.5))
            cert = float(s.charges.get("certainty", 0.5))
            cons = float(s.charges.get("constraint", 0.5))
            t = abs(ep - (prev_ep if prev_ep is not None else ep)) + 0.1 * (1.0 - cert)
            v = (1.0 - cert) + 0.5 * cons
            kinetic += t
            potential += v
            per_step.append(
                {
                    "step": s.step,
                    "role": s.role,
                    "T": round(t, 4),
                    "V": round(v, 4),
                    "L": round(t - v, 4),
                }
            )
            prev_ep = ep
        action = kinetic - potential
        directive = (
            "Increase confounder critique / reduce certainty inflation"
            if potential > kinetic
            else "Chain kinetic OK — proceed to reflective run"
        )
        return FoldDiagnosis(
            action_s=round(action, 4),
            kinetic_t=round(kinetic, 4),
            potential_v=round(potential, 4),
            per_step=per_step,
            directive=directive,
            backend=self.name,
            notes=[
                "Stub fold uses charge proxies only — not Fisher-weighted Kineteq fold."
            ],
        )

    def memory_step(
        self,
        query: str,
        *,
        context: Optional[dict[str, Any]] = None,
        top_k: int = 8,
    ) -> list[GraphMemoryNode]:
        """Retrieve / upsert graph-memory nodes relevant to ``query``."""
        columns = _col_names(context)
        edges = list((context or {}).get("edges") or [])
        qtoks = set(_tokenize(query))

        # Seed from context edges/columns if memory empty-ish
        for c in columns:
            score = 0.3 + (0.4 if any(t in c.lower() for t in qtoks) else 0.0)
            self._remember(
                GraphMemoryNode(
                    key=f"col:{c}",
                    kind="column",
                    content=c,
                    score=score,
                )
            )
        for i, e in enumerate(edges[:20]):
            src, tgt = e.get("source"), e.get("target")
            if not src or not tgt:
                continue
            conf = float(e.get("confidence") or e.get("score") or 0)
            hit = any(t in str(src).lower() or t in str(tgt).lower() for t in qtoks)
            self._remember(
                GraphMemoryNode(
                    key=f"edge:{src}->{tgt}",
                    kind="edge",
                    content=f"{src} → {tgt} (score={conf})",
                    score=conf + (0.2 if hit else 0.0),
                    meta={"source": src, "target": tgt, "confidence": conf},
                )
            )

        ranked = sorted(self._memory, key=lambda m: m.score, reverse=True)
        # de-dupe by key keeping highest score
        seen: set[str] = set()
        out: list[GraphMemoryNode] = []
        for m in ranked:
            if m.key in seen:
                continue
            seen.add(m.key)
            out.append(m)
            if len(out) >= top_k:
                break
        return out

    def graph_retrieve(
        self,
        *,
        context: Optional[dict[str, Any]] = None,
        focus: Optional[list[str]] = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return boost-edge candidates from memory + context graph."""
        focus_set = set(focus or [])
        hits = self.memory_step(
            " ".join(focus or []) or ((context or {}).get("text") or "causal"),
            context=context,
            top_k=top_k * 2,
        )
        boost: list[dict[str, Any]] = []
        for m in hits:
            if m.kind != "edge":
                continue
            src = (m.meta or {}).get("source")
            tgt = (m.meta or {}).get("target")
            if not src or not tgt:
                continue
            reason = "grail_graph_retrieve"
            if focus_set and (src in focus_set or tgt in focus_set):
                reason = "grail_graph_focus"
            boost.append(
                {
                    "source": src,
                    "target": tgt,
                    "reason": reason,
                    "backend": self.name,
                    "score": m.score,
                }
            )
            if len(boost) >= top_k:
                break
        return boost

    def run(
        self,
        goal: str,
        *,
        context: Optional[dict[str, Any]] = None,
        max_cycles: int = 2,
        chain_length: int = 3,
        domain: Optional[str] = None,
    ) -> GrailReport:
        """Full offline reflective loop: impute → compose → fold → cycles."""
        domain = domain or self.domain
        max_cycles = max(1, int(max_cycles))
        audit = self.impute(goal, context=context, domain=domain)
        chain = self.compose(
            goal,
            context=context,
            chain_length=chain_length,
            imputation=audit,
            domain=domain,
        )
        fold = self.fold(chain)
        columns = _col_names(context)
        roles = _guess_roles(columns)
        treat = next(
            (a.value for a in audit.assumptions if a.parameter == "treatment"), None
        )
        outc = next(
            (a.value for a in audit.assumptions if a.parameter == "outcome"), None
        )
        focus = [c for c in (treat, outc, *(roles["instrument"][:2]), *(roles["confounder"][:2])) if c]
        focus = list(dict.fromkeys(str(c) for c in focus))

        mem = self.memory_step(goal, context=context, top_k=10)
        boost = self.graph_retrieve(context=context, focus=focus, top_k=8)

        cycles: list[CycleTrace] = []
        answer = (
            f"Exploratory GRAIL stub for `{goal[:100]}`: "
            f"focus on {', '.join(f'`{c}`' for c in focus[:4]) or 'available columns'}. "
            f"Fold directive: {fold.directive}. Not identification."
        )
        for cyc in range(1, max_cycles + 1):
            reflection = (
                f"Cycle {cyc}: revisit assumptions {audit.underspecified or ['none']}; "
                f"boost {len(boost)} graph edges; memory hits={len(mem)}."
            )
            verdict = "continue" if cyc < max_cycles else "accept_exploratory"
            if cyc == max_cycles:
                answer_delta = "Finalize DirectionPlan hints + caveats."
            else:
                answer_delta = "Tighten confounder critique; refresh graph retrieve."
                # refresh memory mid-loop
                mem = self.memory_step(f"{goal} cycle{cyc}", context=context, top_k=10)
                boost = self.graph_retrieve(context=context, focus=focus, top_k=8)
            cycles.append(
                CycleTrace(
                    cycle=cyc,
                    reflection=reflection,
                    verdict=verdict,
                    answer_delta=answer_delta,
                    memory_keys=[m.key for m in mem[:6]],
                    graph_hits=[f"{b['source']}→{b['target']}" for b in boost[:6]],
                )
            )
            self._remember(
                GraphMemoryNode(
                    key=f"cycle:{cyc}",
                    kind="cycle",
                    content=reflection,
                    score=0.5 + 0.1 * cyc,
                )
            )

        next_q = [
            f"Is `{treat}` a plausible treatment for `{outc}` under confounding?"
            if treat and outc
            else "Which columns are treatment vs outcome?",
            "What instruments (Z) satisfy relevance and exclusion as design hypotheses?",
        ]
        queries = [
            goal[:160],
            f"confounders of {treat} and {outc}" if treat and outc else "causal confounders",
        ]

        return GrailReport(
            goal=goal,
            domain=domain,
            backend=self.name,
            live_kineteq=False,
            imputation=audit,
            chain=chain,
            fold=fold,
            cycles=cycles,
            final_answer=answer,
            genome_id=_genome_id(goal, domain, chain_length),
            memory=mem,
            focus_columns=focus[:16],
            next_questions=next_q,
            search_queries=queries,
            boost_edges=boost,
            notes=[
                "Offline GRAIL stub path — rich structure, not live Kineteq LM/MCP.",
                fold.directive,
            ],
        )

    def _remember(self, node: GraphMemoryNode) -> None:
        # replace same key if present
        self._memory = [m for m in self._memory if m.key != node.key]
        self._memory.append(node)
