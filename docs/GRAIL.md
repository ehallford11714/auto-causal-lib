# GRAIL in AutoCausalLib

Library-first adaptation of **Kineteq GRAIL** for exploratory causal workflows.

## What GRAIL is (Kineteq origin)

**GRAIL** = **G**enerative **R**eflective **A**gentic **I**mputation **L**oop.

On the Kineteq MCP bus / Cognitive Receiver (`kineteq-mcp-app`, `kernel-os`), GRAIL is a strategy that:

| Tool | Role |
|------|------|
| `grail_impute` | Self-imputation audit — detect underspecified goal slots, declare **ASSUMPTIONS** |
| `grail_compose` | Meta-prompt composer — dense expert-chain + mutation prompt (no execution) |
| `grail_run` | Reflective cycles over the chain; traces, verdicts, final answer, `genome_id` |
| `grail_fold` | “Lagrangian of intelligence” T/V diagnosis over prompt charges |
| `grail_metaloop` / `grail_evolve` / `grail_genomes` | Outer evolutionary / genome archive (live platform) |
| `grail_surface` / `grail_science` | Tool-surface invoke + science-agent variant |

Kernel-os strategy `grail` is typically **impute → compose**; AutoLoop may call `grail_run`.

**Strengths for AutoCausal:** goal enrichment, structured expert roles (analyst / confounder critic / instrument scout), reflective memory, graph-aware focus — maps cleanly onto `DirectionPlan` and the insight research loop.

**Gaps:** live GRAIL needs Kineteq MCP/LM conductor, genome archive, and approved tool surfaces. Those are **not** vendored here. AutoCausal never hard-requires Kineteq.

## What AutoCausal adds (embellishment)

Package: `autocausal.grail`

- Clear Python API: `GrailEngine.impute / compose / fold / run / memory_step / graph_retrieve`
- Structured reports: `GrailReport.to_markdown()` / `.to_dict()`
- Soft live adapter (module or MCP) + **rich offline stub** with the same primitive names
- Guide backend `grail` / `kineteq_grail` → `DirectionPlan`
- Insight / SLM research loop soft stage `grail` (memory + graph)
- Skilling `ToolSurface` + MCP/AgentHook tools `autocausal_grail_*`

**Epistemic honesty:** the offline stub is an AutoCausal scaffold. It is **not** full live Kineteq GRAIL (no LM conductor, no genome archive parity). Backend labels distinguish `grail_stub` vs `kineteq_grail`.

## Install / live path

```bash
pip install -e ".[dev]"
# optional live MCP client:
pip install -e ".[web]"
# optional AutoCausal MCP server (exposes grail tools to agents):
pip install -e ".[mcp]"
```

| Env | Effect |
|-----|--------|
| `KINETEQ_MCP_URL` / `AUTOCAUSAL_KINETEQ_MCP_URL` | Kineteq JSON-RPC endpoint |
| `AUTOCAUSAL_KINETEQ_MCP=1` or `AUTOCAUSAL_GRAIL_MCP=1` | Enable live GRAIL MCP calls |
| `KINETEQ_AUTH_TOKEN` | `x-api-key` |

Without live config, everything runs on the stub.

## Library API

```python
from autocausal.grail import GrailEngine, run_grail, grail_backend_status

print(grail_backend_status())
# {'preferred': 'grail_stub', 'kineteq_mcp': False, ...}

eng = GrailEngine()
ctx = {
    "text": "Does campaign_spend cause revenue?",
    "columns": [{"name": "campaign_spend"}, {"name": "revenue"}, {"name": "instrument_z"}],
    "edges": [{"source": "campaign_spend", "target": "revenue", "score": 0.6}],
}
report = eng.run("Does campaign_spend cause revenue?", context=ctx, max_cycles=2)
print(report.to_markdown())
print(report.focus_columns, report.boost_edges[:3])
```

Wire GRAIL into discovery (focus rediscover + merge `boost_edges`):

```python
from autocausal import AutoCausal

ac = AutoCausal.from_csv("data.csv")
ac.discover(qc="off")
report = ac.apply_grail("Does spend cause revenue?")  # stores ac.grail_report
# boost edges tagged method=grail_boost when both endpoints exist
```

Primitives:

```python
audit = eng.impute(goal, context=ctx)
chain = eng.compose(goal, context=ctx, chain_length=4)
fold = eng.fold(chain)
mem = eng.memory_step(goal, context=ctx)
boost = eng.graph_retrieve(context=ctx, focus=["campaign_spend", "revenue"])
```

## Direction guide

```python
from autocausal import AutoCausal

ac = AutoCausal.from_csv("data.csv")
plan = ac.direct(
    text="Does spend cause revenue?",
    backends=["grail", "rule"],  # aliases: kineteq_grail
)
print(plan.to_markdown())
```

```bash
python -m autocausal guides list
python -m autocausal direct --csv data.csv --text "..." --guides grail,rule
```

## Insight / research loop

`InsightSuite.run` / `run_loop` soft-call `insight_grail_step` each pass (memory + graph). Results land on `InsightReport.grail` and round history.

## MCP / connective tools

Registered on `autocausal.mcp` / `AgentHook` (and skilling surface):

| Tool | Purpose |
|------|---------|
| `autocausal_grail_status` | Stub vs live backend status |
| `autocausal_grail_impute` | Imputation audit |
| `autocausal_grail_compose` | Expert chain |
| `autocausal_grail_run` | Full reflective loop |
| `autocausal_grail_memory` | Memory retrieve |
| `autocausal_grail_graph` | Graph → boost edges |

```python
from autocausal.connective import AgentHook

hook = AgentHook()
print(hook.call_tool("autocausal_grail_status", {}))
print(hook.call_tool("autocausal_grail_run", {
    "goal": "Does spend cause revenue?",
    "columns": ["campaign_spend", "revenue"],
}))
```

```bash
python -m autocausal.mcp   # stdio MCP server (needs pip install autocausal[mcp])
```

## Soft-fail policy

- Core `autocausal` never depends on a Kineteq install.
- Selecting `grail` always yields a usable stub plan when live is absent.
- MCP GRAIL tools soft-register; missing `mcp` SDK does not break `AgentHook`.
