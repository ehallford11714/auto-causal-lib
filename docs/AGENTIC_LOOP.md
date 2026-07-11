# Agentic causal loop (`autocausal.agentic`)

Library-first **SLM-guided** cyclic research loop for AutoCausal:

```
hypothesize → skill/tool → validate → compact → persist → route
```

Exploratory assistance only — **not** causal identification.

## Design inspiration (not paper clones)

These arXiv works informed the *API shape* and memory/compaction ideas. AutoCausal does **not** claim to reimplement them:

| Idea | arXiv | Module mapping |
|------|-------|----------------|
| Agent context compaction + distill toward SLM | [ACON 2510.00615](https://arxiv.org/abs/2510.00615) | `compact.Compactor` — lossy narrative + lossless handles |
| Constant-size agent memory | [MEM1 2506.15841](https://arxiv.org/abs/2506.15841) | `memory.EpisodicMemory` budget |
| Evolving linked memory notes | [A-MEM 2502.12110](https://arxiv.org/abs/2502.12110) | `MemoryItem.links` between episodes |
| FSM / cyclic agent orchestration | [StateFlow 2403.11322](https://arxiv.org/abs/2403.11322) | `graph_runtime.GraphRuntime` offline FSM |
| Soft LangGraph-style cycles | LangGraph (optional) | same runtime, `prefer_langgraph=True` |
| Hybrid vector + structured LTM | [HippoRAG 2405.14831](https://arxiv.org/abs/2405.14831), [Mem0 2504.19413](https://arxiv.org/abs/2504.19413) | `vector_memory.VectorStoreMemory` |

## Install / soft deps

Core loop needs only AutoCausal core (`numpy`, `pandas`). Optional:

| Extra | Effect |
|-------|--------|
| `langgraph` | Soft cyclic graph backend (else FSM stub) |
| `chromadb` / `faiss` | Soft vector backends (else in-memory TF-IDF) |
| `autocausal[slm]` | HuggingFace SLM guidance (else rule policy) |

Missing optionals **never** hard-crash the loop.

## Quick start

```python
from autocausal import load_dataset
from autocausal.agentic import AgenticCausalLoop, run_agentic_loop

df = load_dataset("iris")

# One-shot
report = run_agentic_loop(df, text="what drives petal width?", max_rounds=2, use_slm=False)
print(report.to_markdown())

# Class API + JSONL persistence
loop = AgenticCausalLoop(use_slm=False, persist_dir=".autocausal_agentic", max_rounds=3)
report = loop.run(df, text="petal drivers")
print(report.handles)          # lossless edge ids / metrics
print(report.narrative)        # lossy compaction prose
print(loop.memory.episodic)    # constant-budget episodes

# Via AutoCausal
from autocausal import AutoCausal
ac = AutoCausal.from_dataframe(df)
report = ac.agentic_loop(text="petal drivers", max_rounds=2, use_slm=False)
```

## Module tree

```
autocausal/agentic/
  __init__.py
  loop.py           # AgenticCausalLoop / run_agentic_loop
  state.py          # LoopState, Hypothesis
  memory.py         # WorkingMemory + EpisodicMemory (MEM1-inspired)
  compact.py        # Compactor (ACON-inspired)
  graph_runtime.py  # StateFlow / soft LangGraph cyclic nodes
  vector_memory.py  # VectorStoreMemory (numpy TF-IDF; soft chroma/faiss)
  persist.py        # JSONL episode store
  report.py         # AgenticLoopReport
```

## Cycle nodes

1. **hypothesize** — edge candidates, experiment recommender, vector retrieval, soft GRAIL
2. **skill** — `SLMToolBroker` / suite tools (`autocleanse.*`, `autoeda.*`, `automine.*`)
3. **validate** — QC + rediscover + soft refute + insight summary
4. **compact** — lossy narrative + lossless handles; promote to episodic + vector LTM
5. **persist** — append JSONL when `persist_dir` set
6. **route** — continue / stop (max rounds, plateau, empty edges)

## MCP / AgentHook

```python
from autocausal.connective import AgentHook

hook = AgentHook()
hook.call_tool("autocausal_load_dataset", {"dataset_id": "iris"})
out = hook.call_tool(
    "autocausal_agentic_loop",
    {"max_rounds": 2, "use_slm": False, "text": "petal drivers"},
)
print(out["ok"], out.get("n_rounds"), out.get("runtime_backend"))
```

Tool name: `autocausal_agentic_loop`.

## Relation to insight / skilling / suites

| Existing | Role in agentic loop |
|----------|----------------------|
| `insight.run_loop` / `ExperimentRecommender` | Hypotheses + experiment ideas |
| `skilling.SLMToolBroker` | Skill/tool node |
| `suites.SLMAutoDirector` | Soft director pass |
| `grail.insight_grail_step` | Soft reflective hypothesis (if present) |

Agentic loop is **additive** — it does not replace insight or suites.

## Epistemic honesty

- Edges and hypotheses are **candidates**
- Compaction narratives are **lossy**; audit via `handles`
- Vector hits are similarity, not evidence
- SLM text is generative assistance only
