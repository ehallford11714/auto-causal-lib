# AutoCausal MCP connective

Expose AutoCausalLib to other agents via the [Model Context Protocol](https://modelcontextprotocol.io/) **or** the in-process `AgentHook` broker (no MCP SDK required).

> Exploratory edges ≠ causal identification. Tool outputs are assistance for human review.

## Install

```bash
cd research/AutoCausalLib
pip install -e ".[mcp]"
# or: pip install "mcp>=1.0" && pip install -e .
```

Core library imports stay intact without `mcp`. Only the stdio server needs the SDK.

## Run the server

```bash
python -m autocausal.mcp
# or console script:
autocausal-mcp

# list tools without starting stdio:
python -m autocausal.mcp --list-tools
```

## Cursor (`mcp.json`)

Add to Cursor MCP settings (user or project `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "autocausal": {
      "command": "python",
      "args": ["-m", "autocausal.mcp"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

If you use a venv:

```json
{
  "mcpServers": {
    "autocausal": {
      "command": "C:/path/to/venv/Scripts/python.exe",
      "args": ["-m", "autocausal.mcp"],
      "cwd": "C:/Users/you/research/AutoCausalLib",
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

Windows tip: prefer the venv’s `python.exe` and set `PYTHONUNBUFFERED=1` so stdio JSON-RPC stays line-buffered.

## Claude Desktop

Edit Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "autocausal": {
      "command": "python",
      "args": ["-m", "autocausal.mcp"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## Library-first (no MCP client)

Same tools, in-process — useful for scripts, notebooks, and non-MCP agents:

```python
from autocausal.connective import AgentHook
# alias: from autocausal.mcp.hooks import AgentHook

hook = AgentHook()
print([t["name"] for t in hook.list_tools()])

r = hook.call_tool("autocausal_load_dataset", {"dataset_id": "iris"})
assert r["ok"]
print(hook.call_tool("autocausal_mine", {"use_suite": False}))
print(hook.call_tool("autocausal_discover", {"use_iv": False, "min_abs_corr": 0.2}))
print(hook.call_tool("autocausal_report", {"format": "markdown"})["markdown"][:500])
```

Module helpers:

```python
from autocausal.connective import call_tool, list_tools

list_tools()
call_tool("autocausal_list_datasets", {})
```

## Tools

| Tool | Maps to |
|------|---------|
| `autocausal_list_datasets` | `autocausal.datasets.list_datasets` |
| `autocausal_load_dataset` | `load_dataset` → session `AutoCausal` |
| `autocausal_from_csv` | `AutoCausal.from_csv` |
| `autocausal_cleanse` | `AutoCausal.cleanse` (soft → `impute`) |
| `autocausal_eda` | `AutoCausal.eda` (soft → QC) |
| `autocausal_mine` | `automine` / `mine` |
| `autocausal_discover` | `AutoCausal.discover` |
| `autocausal_insight_loop` | `InsightSuite` / `run_insight_loop` |
| `autocausal_recommend_experiments` | `ExperimentRecommender.recommend` |
| `autocausal_public_mine` | `AutoCausal.mine_public` |
| `autocausal_report` | `AutoCausal.report` (markdown/json) |
| `autocausal_skilling_list` | `autocausal.skilling.skill_catalog` |
| `autocausal_session_status` | active session metadata |
| `autocausal_list_tools` | this registry’s schemas |

Sessions: pass `session_id` (default `"default"`) so multi-step calls share one `AutoCausal` instance.

## Relation to skilling

`autocausal.skilling.ToolSurface` wraps **suite actions** for SLM brokers (`autocleanse.impute`, …).  
`autocausal.mcp` / `autocausal.connective` expose **library entry points** for external MCP/agents. They are complementary; `autocausal_skilling_list` can enumerate the skilling catalog.

## Epistemic note

Discovery, mining, insight narratives, and experiment suggestions are **exploratory**. Do not treat tool JSON as identified causal effects.
