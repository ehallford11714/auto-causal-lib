# AutoCausalLib documentation index

Package: **`auto-causal-lib`** on PyPI · import **`autocausal`** · version see `autocausal.__version__`.

> **Epistemic caveat:** Discovery, mining, NLP hints, SLM narratives, estimate/refute outputs, and agent loops are **exploratory assistance**. They do **not** guarantee causal identification. Soft-optional engines (causal-learn, DoWhy, DoubleML, EconML, lingam, gCastle, MCP SDK, torch, …) never hard-break the core numpy/pandas path.

## Start here

| Doc | Contents |
|-----|----------|
| [LIBRARY_API.md](LIBRARY_API.md) | Complete public API map: `AutoCausal`, top-level exports, pipeline |
| [MODULES.md](MODULES.md) | Every package under `src/autocausal/` — purpose, symbols, pipeline fit |
| [CLI.md](CLI.md) | All `python -m autocausal` subcommands |
| [MCP.md](MCP.md) | MCP stdio server + `AgentHook` tools |
| [CAUSAL_BACKENDS.md](CAUSAL_BACKENDS.md) | Soft discovery / estimate / refute engines |

## Area guides

| Doc | Area |
|-----|------|
| [SUITES.md](SUITES.md) | AutoCleanse / AutoEDA / AutoMine |
| [SLM_SKILLING.md](SLM_SKILLING.md) | ToolSurface, SkillRegistry, broker |
| [INSIGHT_SUITE.md](INSIGHT_SUITE.md) | InsightSuite + experiment loop |
| [AGENTIC_LOOP.md](AGENTIC_LOOP.md) | AgenticCausalLoop FSM |
| [GRAIL.md](GRAIL.md) / [GUIDES.md](GUIDES.md) | GRAIL + direction guides |
| [SUITE_TOOLS.md](SUITE_TOOLS.md) | Causal/NLP/KPI tool registry |
| [NLP_AND_BEHAVIORAL_TRACES.md](NLP_AND_BEHAVIORAL_TRACES.md) | NLP + behavioral |
| [PHYSICS_DEMO.md](PHYSICS_DEMO.md) / [ML_KPI_LOOP.md](ML_KPI_LOOP.md) | Physics + ML loops |
| [PUBLIC_SUITE.md](PUBLIC_SUITE.md) / [PUBLIC_CAUSAL_MINING.md](PUBLIC_CAUSAL_MINING.md) | Public data |
| [EXAMPLES.md](EXAMPLES.md) / [CONNECTIONS.md](CONNECTIONS.md) | Datasets + SQL |
| [LAYER_CAUSAL_IV.md](LAYER_CAUSAL_IV.md) | IntentIsolates bridge |
| [ROADMAP.md](ROADMAP.md) | Shipped / deferred |

## Canonical pipeline

```
load → [cleanse/eda] → mine → impute → discover(+soft backends)
     → estimate(DoubleML|EconML|OLS) → refute(DoWhy|placebo)
     → guide/direct → insight / agentic / physics / ml
```

Connectivity: `autocausal.engines` · CLI `engines|estimate|refute` · MCP `autocausal_list_engines|_estimate|_refute` · `AgentHook`.

## Install

```bash
pip install auto-causal-lib
# Base now includes: torch, transformers, accelerate, huggingface_hub,
# langgraph, langchain-core, mcp, scikit-learn, nltk, httpx, pyarrow
pip install "auto-causal-lib[causal-extra]"   # heavy causal engines
pip install "auto-causal-lib[bitsandbytes]"   # CUDA 4-bit only
pip install "auto-causal-lib[all]"
```

Local Qwen (hardware-selected Instruct)::

```bash
python -m autocausal slm setup-qwen
python -m autocausal slm-loop --no-slm   # rule path
# Full Qwen tests: set AUTOCAUSAL_TEST_QWEN=1
```
