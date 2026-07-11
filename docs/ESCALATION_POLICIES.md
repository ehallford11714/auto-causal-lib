# Escalation policies for AutoCausal edge cases

**Status:** draft design (not fully implemented).  
**Scope:** when AutoCausal should *defer*, *cascade*, or *abstain* under uncertainty.  
**Epistemic honesty:** escalation improves triage and auditability; it does **not** create causal identification. Soft engines, SLMs, and GRAIL remain exploratory.

---

## 1. Ranked SOTA papers (arXiv 2023–2026)

| Rank | Paper | arXiv | Why relevant to AutoCausal |
|------|-------|-------|----------------------------|
| 1 | Cascaded LLMs for cost-effective human–AI decision-making | [2506.11887](https://arxiv.org/abs/2506.11887) | **Primary pattern:** deferral (base→large) + abstention (cascade→human) + online threshold updates. Maps to Qwen-small → Qwen-larger → human review. |
| 2 | FrugalGPT | [2305.05176](https://arxiv.org/abs/2305.05176) | Cascade with quality estimator / stop judge. Template for `escalate(decision)` cost tiers. |
| 3 | Unified routing + cascading (“cascade routing”) | [2410.10347](https://arxiv.org/abs/2410.10347) | Formal optimality: quality estimators are the bottleneck. Justifies scoring IV F-stat, stability, engine agreement—not verbal SLM confidence alone. |
| 4 | Language model cascades: token-level uncertainty | [2404.10136](https://arxiv.org/abs/2404.10136) | Prefer token/quantile uncertainty over sequence-level confidence for SLM routing (Qwen local). |
| 5 | Bootstrap aggregation & confidence for causal discovery | [2306.08946](https://arxiv.org/abs/2306.08946) (CLeaR 2024) | Edge-frequency / bagging as honest confidence. Aligns with existing `discovery._apply_stability`. |
| 6 | Semiparametric discovery with invalid instruments (PLACID) | [2504.12085](https://arxiv.org/abs/2504.12085) | Invalid/weak IVs distort graphs; escalate to CausalIV / human when instruments fail assumptions. |
| 7 | Dimension-agnostic bootstrap Anderson–Rubin for IV | [2412.01603](https://arxiv.org/abs/2412.01603) | Weak-ID-robust testing when first-stage is weak or #IVs ambiguous—policy trigger for AR-style handoff. |
| 8 | Multiagent debate (Du et al.) + conditional MAD | [2305.14325](https://arxiv.org/abs/2305.14325), [2505.22960](https://arxiv.org/abs/2505.22960) | Debate as *intra-tier* verification before costly upgrade. MAD helps most on hard / weak-model cases—not blanket. CascadeDebate-style deliberation at escalation boundaries (industry 2026). |

**Also useful (causal edge cases, not ranked as escalation machinery):**

- Selection / post-treatment selection: [2509.25800](https://arxiv.org/abs/2509.25800), [2512.11219](https://arxiv.org/abs/2512.11219)
- Latent confounders + selection in PAGs: [2603.26301](https://arxiv.org/abs/2603.26301)

**Policy patterns distilled from SOTA**

| Pattern | Meaning in AutoCausal |
|---------|----------------------|
| **Threshold escalate** | If score < τ → next tier |
| **Timeout escalate** | Soft engine / SLM / GRAIL exceeds budget → next tier or stub-labeled continue |
| **Disagreement escalate** | Engines / agents disagree → ensemble debate → human if unresolved |
| **Risk-tier gate** | Irreversible claims (publish IV effect, act on policy) require human even if confidence high |
| **Abstain vs defer** | Defer = larger model/engine; abstain = human / CausalIV handoff / stop with artifacts |

---

## 2. Edge-case catalog (AutoCausal-specific)

Grounded in current code: `iv.try_iv_edges`, `discovery` bootstrap stability + consensus, `qc.validate_frame`, `engines` soft-skip, `grail` stub vs live, `agentic` route/stop, `slm` confidence / weak-IV caveats.

| ID | Edge case | Signal today | Risk if ignored |
|----|-----------|--------------|-----------------|
| `E-IV-SKIP` | IV pass skipped (missing T/Y/Z) | note in `try_iv_edges` | Silent OLS-like graphs look “causal” |
| `E-IV-EMPTY` | Empty instruments after role merge | same + no IV edges | Auto-instrument temptation without labeling |
| `E-IV-AUTO` | Synthetic `auto_instrument_z` used | `auto_instrument=True`, conf×0.35 | Fake identification |
| `E-IV-WEAK` | First-stage F &lt; 10 (code warns &lt;10; lite path halves conf if F&lt;5) | `first_stage_f`, SLM caveats | Biased 2SLS, overconfident edges |
| `E-IV-SUITE` | CausalIV unavailable → numpy lite | notes | Weaker diagnostics than suite |
| `E-EDGE-UNSTABLE` | Bootstrap stability low | `stability`, `confidence = min(raw, stab)` | Spurious PC/score edges |
| `E-EDGE-LOW-N` | Few complete rows | discovery notes “unstable” | CI tests fail faithfulness in finite sample |
| `E-ENGINE-DISAGREE` | Soft backends disagree on skeleton | `consensus_edges` / `n_methods` | Method-specific artifacts |
| `E-ENGINE-SKIP` | Soft backend `soft_skip` | engines status | False sense of multi-engine rigor |
| `E-QC-BLOCK` | QC `severity=block` | `validate_qc` raises | Bad frames → nonsense graphs |
| `E-QC-WARN` | ID leakage / hygiene warns | `QCReport` | Leakage masquerades as causation |
| `E-SLM-LOWCONF` | SLM/Insight confidence low | `slm` report confidence | Hallucinated narratives |
| `E-SLM-TIMEOUT` | Local Qwen slow/OOM | soft-fail to rules | Silent quality drop |
| `E-AGENT-PLATEAU` | Agentic edge set unchanged | `_node_route` plateau stop | Spin without progress |
| `E-AGENT-EMPTY` | No edges after validate | route stop | Empty “success” reports |
| `E-GRAIL-STUB` | `grail_stub` not `kineteq_grail` | `grail_backend_status` | Stub memory treated as live GRAIL |
| `E-COLLIDER` | Conditioning / selection risk | not fully automated | Spurious associations (Berkson) |
| `E-SELECT` | Sampling / QC retention bias | post-treatment selection literature | Intervention responses look causal |

---

## 3. Draft named escalation policies

**Cascade ladder (default):**

```
rule / heuristics
  → Qwen-small (local SLM)
    → Qwen-larger / multi-engine ensemble (+ optional intra-tier debate)
      → CausalIVSuite handoff (IV path) OR soft FCI/DoWhy refute
        → human review (abstain)
```

Each policy: **trigger → escalate-to → required artifacts → stop/continue**.

### P1 — `ThresholdConfidenceEscalate`

| | |
|--|--|
| **Trigger** | Edge or SLM decision with `confidence < τ_conf` (suggested defaults: edges 0.35; SLM narrative 0.45). Prefer *calibrated* scores: `min(confidence, stability)`, first-stage F penalty, auto-IV penalty—not raw verbal confidence ([2410.10347](https://arxiv.org/abs/2410.10347), [2506.11887](https://arxiv.org/abs/2506.11887)). |
| **Escalate-to** | Next cascade tier (rule→Qwen-small→Qwen-larger). |
| **Artifacts** | Decision payload, score vector (`confidence`, `stability`, `first_stage_f`, `n_methods`), model id, prompt hash. |
| **Stop/continue** | Accept if score ≥ τ after tier; else continue ladder. Cap max tiers = 3 then abstain. |

### P2 — `WeakInstrumentEscalate`

| | |
|--|--|
| **Trigger** | IV edge with `first_stage_f < 10` (Stock–Yogo-ish rule of thumb) **or** `auto_instrument=True` **or** F missing. |
| **Escalate-to** | CausalIVSuite if available; else mark `identification=false` and escalate narrative to Qwen-larger + **human** for any publishable claim. Optionally AR-robust path ([2412.01603](https://arxiv.org/abs/2412.01603)). |
| **Artifacts** | T/Y/Z columns, n, F, coef/se/p, auto-instrument flag, exclusion notes. |
| **Stop/continue** | **Stop claiming IV ID** immediately. Continue exploratory graph with `type=iv_2sls` demoted / flagged. Human required for action. |

### P3 — `EmptyOrSkippedIVEscalate`

| | |
|--|--|
| **Trigger** | `E-IV-SKIP` / `E-IV-EMPTY`. |
| **Escalate-to** | Role repair: user `set_iv_roles` / `instruments_demo` / `iv_demo`; SLM instrument-scout prompt; **do not** auto-synthesize Z unless `auto_instrument=True` and labeled. |
| **Artifacts** | Missing roles list, candidate columns, prior notes. |
| **Stop/continue** | Continue discovery without IV; block IV estimate endpoints until Z present. |

### P4 — `UnstableEdgeEscalate`

| | |
|--|--|
| **Trigger** | `stability < τ_stab` (suggest 0.5) or low-N note ([2306.08946](https://arxiv.org/abs/2306.08946)). |
| **Escalate-to** | More bootstrap reps → multi-engine `consensus_edges` → Qwen critique of edge → human if still unstable. |
| **Artifacts** | Bootstrap frequency, method votes, n_rows, CI test params. |
| **Stop/continue** | Drop or gray-list edge in reports; continue loop only if ≥1 stable edge or user overrides. |

### P5 — `EngineDisagreementEscalate`

| | |
|--|--|
| **Trigger** | Same undirected pair: conflicting orientation **or** present in &lt; `min_methods` backends when ≥2 available. |
| **Escalate-to** | Intra-tier debate (analyst vs confounder-critic personas) *before* larger model ([2305.14325](https://arxiv.org/abs/2305.14325), CascadeDebate pattern); then ensemble; then human. |
| **Artifacts** | Per-engine adjacency snippets, agreement matrix, debate transcript. |
| **Stop/continue** | If debate consensus + stability ≥ τ → continue; else abstain with PAG/undirected suggestion. |

### P6 — `QCBlockEscalate`

| | |
|--|--|
| **Trigger** | `QCReport.blocked` or `mode=block` issues. |
| **Escalate-to** | AutoCleanse / role fix path; **no** discover until cleared (or explicit `qc="warn"|"off"` override logged). |
| **Artifacts** | Full `QCReport.to_dict()`, column roles. |
| **Stop/continue** | Hard stop discover; escalate to human for override. |

### P7 — `SLMUncertaintyEscalate`

| | |
|--|--|
| **Trigger** | Low SLM confidence, high token entropy / self-reported uncertainty, or rule-fallback after Qwen fail ([2404.10136](https://arxiv.org/abs/2404.10136)). |
| **Escalate-to** | Qwen-larger (or cloud if configured); optional 2-agent critique; human if abstention ξ exceeded ([2506.11887](https://arxiv.org/abs/2506.11887)). |
| **Artifacts** | Logprobs/entropy if available, hypothesis, tool traces, compacted memory handles. |
| **Stop/continue** | Defer regenerates answer; abstain freezes agentic `route=stop` with `stop_reason=escalation_abstain`. |

### P8 — `TimeoutEscalate`

| | |
|--|--|
| **Trigger** | Soft engine / SLM / live GRAIL exceeds `timeout_s` or OOM soft-fail. |
| **Escalate-to** | Next cheaper-complete path (builtin PC → skip heavy backend; SLM→rules; GRAIL live→stub **with label**). |
| **Artifacts** | Duration, backend id, error class, fallback chosen. |
| **Stop/continue** | Continue degraded; raise severity if fallback is stub/rules for high-risk claim. |

### P9 — `GrailStubHonestyEscalate`

| | |
|--|--|
| **Trigger** | `preferred == grail_stub` while user/docs imply live Kineteq. |
| **Escalate-to** | Label all GRAIL outputs `backend=grail_stub`; optional prompt to enable MCP; human if genomes/live memory expected. |
| **Artifacts** | `grail_backend_status()`, report backend field. |
| **Stop/continue** | Always continue with honest labeling; never silent upgrade. |

### P10 — `DisagreementDebateThenAbstain`

| | |
|--|--|
| **Trigger** | SLM vs engines vs IV notes conflict (e.g. narrative claims ID, IV F weak). |
| **Escalate-to** | Structured multi-role critique (insight GRAIL roles) → if unresolved, **human**. MAD is conditional—prefer on hard cases ([2505.22960](https://arxiv.org/abs/2505.22960)). |
| **Artifacts** | Conflict tuple, both sides’ evidence, final verdict enum: `accept` / `revise` / `abstain`. |
| **Stop/continue** | Abstain stops publishable language; exploratory JSON may continue flagged. |

### P11 — `RiskTierHumanGate` (cross-cutting)

| Action tier | Examples | Oversight |
|-------------|----------|-----------|
| Read-only | discover markdown, exploratory edges | auto OK |
| Reversible | re-run with new params | auto / log |
| External | MCP write, export “identified effect” | escalate |
| Irreversible / high-stakes | policy, clinical, spend decisions | **mandatory human** regardless of confidence |

---

## 4. How to build (`autocausal.escalation`)

### 4.1 Module sketch (do not fully implement yet)

```text
autocausal/escalation/
  __init__.py          # escalate, list_policies
  types.py             # Decision, EscalationVerdict, Tier
  policy.py            # EscalationPolicy protocol + registry
  policies.py          # P1–P11 callables
  cascade.py           # Frugal-style ladder runner
  spans.py             # AgentSpan recording
```

**Core types (sketch):**

```python
@dataclass
class Decision:
    kind: str                 # "edge" | "iv" | "slm" | "qc" | "engine" | "grail" | "route"
    payload: dict[str, Any]
    scores: dict[str, float]  # confidence, stability, first_stage_f, entropy, ...
    risk_tier: str            # read|reversible|external|irreversible
    source: str               # module:function
    timeout_s: float | None = None

@dataclass
class EscalationVerdict:
    action: Literal["accept", "defer", "abstain", "degrade"]
    to_tier: str | None       # "qwen_small" | "qwen_large" | "ensemble" | "causaliv" | "human"
    policy_ids: list[str]
    artifacts: dict[str, Any]
    stop_reason: str = ""

@dataclass
class AgentSpan:
    span_id: str
    parent_id: str | None
    name: str                 # "escalate.WeakInstrument"
    t0: float
    t1: float | None
    decision: dict
    verdict: dict
    notes: list[str]

class EscalationPolicy(Protocol):
    id: str
    def evaluate(self, decision: Decision) -> EscalationVerdict | None: ...

def escalate(decision: Decision, *, policies: Sequence[EscalationPolicy] | None = None) -> EscalationVerdict:
    """Run policies; first abstain wins; else highest defer tier; else accept."""
    ...
```

### 4.2 Hook points

| Hook | File / symbol | Call when |
|------|---------------|-----------|
| Discover post-stability | `discovery.py` after `_apply_stability` / `consensus_edges` | Per-edge `Decision(kind="edge")` |
| IV | `iv.try_iv_edges` before append | `E-IV-*` policies |
| QC | `api.AutoCausal.validate_qc` / `discover(qc=...)` | `P6` |
| Engines | `engines.discover_with` / `estimate` | soft_skip + disagreement |
| Insight / SLM | `slm.py` report build | `P7` |
| Agentic route | `agentic.loop._node_route` | inject `escalate` before `continue`; map abstain→`route=stop` |
| LangGraph | `langgraph_chain` validate/route nodes | same verdict into chain notes |
| GRAIL | `grail.adapter` / report | `P9` |
| Persist | `agentic.persist` | store `AgentSpan` in episode JSONL |

### 4.3 Span recording

- Emit `AgentSpan` on every `escalate()` call.
- Persist beside episodes: `{persist_dir}/escalations.jsonl`.
- Surface in `AgenticLoopReport` / Insight markdown: **Escalations** section (policy id, tier, stop/continue).
- MCP: optional tool `autocausal_list_escalations`.

### 4.4 Suggested defaults (tunable)

| Symbol | Default | Notes |
|--------|---------|-------|
| `τ_conf` | 0.35 | edges |
| `τ_slm` | 0.45 | narratives |
| `τ_stab` | 0.50 | bootstrap freq |
| `τ_F` | 10.0 | weak IV |
| `min_methods` | 2 | when ≥2 engines up |
| `max_cascade_tiers` | 3 | then abstain |
| `timeout_s` | 60 / 120 | engine / SLM |

Online threshold updates ([2506.11887](https://arxiv.org/abs/2506.11887)) are **phase 2**: log human accept/reject → adjust τ.

### 4.5 Implementation phases

1. **Types + `escalate()` no-ops** returning `accept` + span log.  
2. **P2, P3, P6, P9** (deterministic, no SLM).  
3. **P1, P4, P5** wired to discovery/IV/engines.  
4. **P7, P8, P10** + Qwen cascade.  
5. **P11** risk tiers on export/MCP write paths.  
6. Threshold learning from human feedback.

---

## 5. Limits (read this)

- Verbal / SLM confidence is often miscalibrated; prefer stability, F-stats, engine agreement, and risk tier.
- Escalation ≠ identification. Weak IV + human review still needs exclusion/relevance assumptions.
- MAD/debate is expensive and not always better than self-consistency; use at disagreement boundaries only.
- GRAIL stub and auto-instruments must stay labeled forever in artifacts.

---

## See also

- [SOTA.md](SOTA.md) — discovery caveats  
- [AGENTIC_LOOP.md](AGENTIC_LOOP.md) — route/stop  
- [GRAIL.md](GRAIL.md) — stub vs live  
- [LAYER_CAUSAL_IV.md](LAYER_CAUSAL_IV.md) — IV bridge  
