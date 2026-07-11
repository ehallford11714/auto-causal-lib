# Direction guides

AutoCausalLib can steer exploratory discovery with **soft-optional** guide backends.
Missing packages never break import or the default `auto` / `mine` / `discover` path.

## Backends

| id | Class | Availability |
|----|-------|----------------|
| `rule` | `RuleGuide` | Always (default) |
| `huggingface` | `HuggingFaceSLMGuide` | `pip install autocausal[slm]` |
| `llmintent` | `LLMIntentGuide` | `pip install -e ../LLMIntent` (or `pip install llmintent`) |
| `retracement` | `RetracementGuide` | Via `llmintent.retracement`, else **documented stub** |
| `kineteq_pivot` | `KineteqPivotEmbeddingGuide` | Kineteq module / MCP, else **local `pivot_fallback`** (not Kineteq) |
| `grail` / `kineteq_grail` | `KineteqGrailGuide` | Live Kineteq GRAIL when MCP/module set; else **rich `grail_stub`** (not full GRAIL) |

Check what is live:

```bash
python -m autocausal guides list
python -m autocausal guides status
```

## DirectionPlan

Multi-backend outputs merge into a `DirectionPlan`:

- `focus_columns` — second-pass discover subset
- `candidate_z` / `treatment` / `outcome` / `confounders`
- `boost_edges` / `suppress_edges`
- `search_queries` / `next_questions`
- `related_variables` / `lag_hints`
- `rationale` / `contributions` / `unavailable`

```python
from autocausal import AutoCausal

ac = AutoCausal.from_csv("data.csv").join_public("marketing_demo")
plan = ac.direct(
    text="Does campaign spend cause revenue?",
    backends=["llmintent", "retracement", "kineteq_pivot", "rule"],
)
print(plan.to_markdown())

# Or via auto()
result = AutoCausal.auto(
    "data.csv",
    text="...",
    guide_backends=["llmintent", "kineteq_pivot"],
)
```

## LLMIntentGuide

**Found on disk:** `research/LLMIntent/` (`llmintent` package).

Uses (when importable), without rewriting upstream:

- `llmintent.heighten.retrace` — focused / retrace prompt scaffolding → `next_questions`
- `llmintent.morphemes.MorphemeExtractor` — tokens that overlap column names → focus
- Optional heavy `LLMIntentAnalyzer` only if `AUTOCAUSAL_LLMINTENT_HEAVY=1` and a model is set

Install:

```bash
pip install -e ../LLMIntent
# or
pip install llmintent
```

Env:

| Variable | Effect |
|----------|--------|
| `AUTOCAUSAL_LLMINTENT_MODEL` / `LLMINTENT_MODEL` | Model id for heavy path |
| `AUTOCAUSAL_LLMINTENT_HEAVY=1` | Load analyzer (torch); off by default |

If `llmintent` is missing, the adapter returns `backend=llmintent_stub` with column/text heuristics and a clear note.

## RetracementGuide

**On disk:** no standalone `retracement` project. Retracement lives under
`LLMIntent/src/llmintent/retracement/` (`RetracementTransformer`, modes, ablation).

When `llmintent.retracement` imports, the guide records the bound mode and applies:

- lag / lead / `_t` column name hints
- reverse-path checks on undirected edge pairs (keep higher-score direction)
- association-based reverse-causality bias (outcome←treatment flips)

If neither `llmintent.retracement` nor a top-level `retracement` package exists, a
**documented stub** still runs the same heuristics (`backend=retracement_stub`).

## KineteqPivotEmbeddingGuide

**On disk:** no `kineteq` Python package. EmotiveVision documents a Kineteq MCP bus:

- `KINETEQ_MCP_URL` / `KINETEQ_AUTH_TOKEN` (aliases: `EMOTIVEVISION_MCP_*`)
- Default public API pattern: `https://kineteq.ai/functions/publicApi`
- JSON-RPC `tools/call` for embedding tools

Adapter order:

1. Local modules `kineteq` / `kineteq_pivot` / `pivot_embeddings` with
   `embed_texts` / `pivot_embed` / `PivotEmbeddings` (future contract)
2. MCP when URL set **and** `AUTOCAUSAL_KINETEQ_MCP=1` (or `EMOTIVEVISION_LIVE_MCP=1`)
3. **Local hashing embedding fallback** labeled `pivot_fallback` — **not Kineteq**

Env:

| Variable | Effect |
|----------|--------|
| `KINETEQ_MCP_URL` / `AUTOCAUSAL_KINETEQ_MCP_URL` | MCP JSON-RPC endpoint |
| `KINETEQ_AUTH_TOKEN` / `AUTOCAUSAL_KINETEQ_TOKEN` | `x-api-key` |
| `AUTOCAUSAL_KINETEQ_MCP=1` | Enable live MCP calls |
| `AUTOCAUSAL_KINETEQ_EMBED_TOOL` | Prefer this tool name (default tries `embed_texts`, `pivot_embed`, …) |

Also needs `httpx` (`pip install autocausal[web]`) for MCP.

## KineteqGrailGuide (`grail` / `kineteq_grail`)

**Origin:** Kineteq GRAIL — Generative Reflective Agentic Imputation Loop
(`grail_impute` → `grail_compose` → `grail_run`). See [GRAIL.md](GRAIL.md).

Adapter order:

1. Optional local `kineteq` / `kineteq_grail` / `grail` module
2. Kineteq MCP `tools/call` for `grail_run` / `grail_impute` / `grail_compose` when URL + live flag set
3. **Rich offline stub** labeled `grail_stub` — structured impute/compose/fold/cycles/memory/graph; **not** live Kineteq

```python
plan = ac.direct(text="...", backends=["grail", "rule"])
```

## CLI

```bash
python -m autocausal guides list
python -m autocausal auto --csv x.csv --text "..." --guides llmintent,retracement,kineteq_pivot
python -m autocausal direct --csv x.csv --text "..." --guides llmintent
```

## Soft-fail policy

- Optional imports are try/except only; core `autocausal` never depends on them.
- Selecting an unavailable backend still contributes a stub/fallback result so
  `DirectionPlan` remains usable offline.
- `guides list` marks availability; `unavailable` on the plan lists soft backends.
