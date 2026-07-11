# AutoCausal MCP connective

Expose AutoCausalLib to other agents via the [Model Context Protocol](https://modelcontextprotocol.io/) **or** the in-process `AgentHook` broker (no MCP SDK required).

> Exploratory edges ≠ causal identification. Tool outputs are assistance for human review.

**Install:** `pip install "auto-causal-lib[mcp]"` then `import autocausal` (PyPI name is **not** `autocausal`).

## Install

```bash
pip install "auto-causal-lib[mcp]"
# or from source:
pip install -e ".[mcp]"
```

Core library imports stay intact without `mcp`. Only the stdio server needs the SDK.

## Run the server

```bash
python -m autocausal.mcp
autocausal-mcp
python -m autocausal.mcp --list-tools
python -m autocausal mcp --list-tools
```

## Cursor (`mcp.json`)

```json
{
  "mcpServers": {
    "autocausal": {
      "command": "python",
      "args": ["-m", "autocausal.mcp"],
      "env": { "PYTHONUNBUFFERED": "1" }
    }
  }
}
```

Prefer a venv `python.exe` on Windows.

## Library-first (no MCP client)

```python
from autocausal.connective import AgentHook

hook = AgentHook()
print(hook.list_names())
r = hook.call_tool("autocausal_load_dataset", {"dataset_id": "iris"})
assert r["ok"]
print(hook.call_tool("autocausal_list_engines", {}))
print(hook.call_tool("autocausal_discover", {"use_iv": False}))
print(hook.call_tool("autocausal_estimate", {"backend": "builtin_ols"}))
print(hook.call_tool("autocausal_refute", {"method": "placebo"}))
```

```python
from autocausal.connective import call_tool, list_tools
list_tools()
call_tool("autocausal_list_datasets", {})
```

## Tools

| Tool | Maps to |
|------|---------|
| `autocausal_list_datasets` | `datasets.list_datasets` |
| `autocausal_load_dataset` | `load_dataset` → session |
| `autocausal_from_csv` | `AutoCausal.from_csv` |
| `autocausal_cleanse` | `AutoCausal.cleanse` (soft → impute) |
| `autocausal_eda` | `AutoCausal.eda` (soft → QC) |
| `autocausal_mine` | automine / `mine` |
| `autocausal_discover` | `AutoCausal.discover` |
| `autocausal_list_engines` | `engines.engine_status` |
| `autocausal_estimate` | `engines.estimate` |
| `autocausal_refute` | `AutoCausal.refute` |
| `autocausal_insight_loop` | `InsightSuite` / `run_insight_loop` |
| `autocausal_agentic_loop` | `AgenticCausalLoop` |
| `autocausal_recommend_experiments` | `ExperimentRecommender` |
| `autocausal_public_mine` | `mine_public` |
| `autocausal_report` | `AutoCausal.report` |
| `autocausal_skilling_list` | `skilling.skill_catalog` |
| `autocausal_session_status` | session metadata |
| `autocausal_list_tools` | registry schemas |
| `autocausal_grail_*` | GRAIL soft tools (when registered) |

Sessions: pass `session_id` (default `"default"`) so multi-step calls share one `AutoCausal`.

## Relation to skilling

`autocausal.skilling.ToolSurface` wraps suite actions for SLM brokers.  
`autocausal.mcp` / `connective` expose library entry points for external agents.

## Epistemic note

Discovery, mining, insight narratives, estimates, and experiment suggestions are **exploratory**.

See [INDEX.md](INDEX.md), [CAUSAL_BACKENDS.md](CAUSAL_BACKENDS.md), [CLI.md](CLI.md).
