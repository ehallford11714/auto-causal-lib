# SLM skilling / tooling surface

Library-first structured tools for directing AutoCausal **auto\*** suites.
Skilling **wraps** dedicated suite actions — it does not replace them.

> Epistemic honesty: tool outcomes and SLM selections are generative /
> exploratory assistance — **not** causal identification.

## Install / env

```bash
pip install -e ".[dev]"
pip install -e ".[slm]"   # optional
```

| Variable | Effect |
|----------|--------|
| `AUTOCAUSAL_SLM=1` | Prefer HuggingFace when selecting tools |

Rule broker always works offline.

## Public API

```python
from autocausal.skilling import (
    ToolSurface,
    SkillRegistry,
    SLMToolBroker,
    suite_tool_surface,
    skill_catalog,
    SkillDrill,
)

from autocausal.suites.autocleanse import CleanseActions
from autocausal.suites.autoeda import EDAActions
from autocausal.suites.automine import MineActions

# Direct suite actions (library-first)
CleanseActions.impute(df, method="auto")
EDAActions.suggest_roles(df)
MineActions.mine_associations(df)

# Tool surface wrapping those actions
surface = suite_tool_surface()
broker = SLMToolBroker(surface, use_slm=True)
print(broker.list_tools(skill="skill:autocleanse")[:3])
result = broker.invoke("autocleanse.impute", {"method": "auto"}, df=df)

# Run a skill (ordered tool calls → outcomes + SkillTrace)
frame2, results, trace = broker.run_skill("skill:autoeda", df, text="prep for IV")
trace.write("eda_skill_trace.json")
```

## Concepts

| Piece | Role |
|-------|------|
| **ToolDef / ToolSurface** | JSON-schema-like tool defs (`name`, `description`, `parameters`, `suite`, `handler`) |
| **SkillRegistry** | Named skills bundling allowed tools + system prompt snippets |
| **SLMToolBroker** | `list_tools` / `invoke` / `select_tools` / `run_skill` |
| **SkillTrace** | Record `(context → tool_calls → outcomes)` for eval |
| **SkillDrill** | Offline rule-path drill + catalog markdown |

### Built-in skills

- `skill:autocleanse` — all `autocleanse.*` tools
- `skill:autoeda` — all `autoeda.*` tools
- `skill:automine` — all `automine.*` tools
- `skill:autocausal_loop` — cleanse + eda + mine + discover/insight tools

### Tool naming

Suite actions are exposed as `{suite}.{action}`:

- `autocleanse.profile_missingness`, `autocleanse.impute`, …
- `autoeda.suggest_roles`, `autoeda.correlation_matrix`, …
- `automine.mine_associations`, `automine.to_mine_report`, …
- `autocausal.discover`, `insight.run`, `insight.experiment_recommend`

## Wiring into SLMAutoDirector

`SLMAutoDirector.direct()` prefers `SLMToolBroker.select_tools()` to set
`directives.actions` and records `tools_invoked` on the directive dict.
Suites then run those actions via their registries (`CleanseActions` / …).

```python
from autocausal import AutoCleanseSuite
clean = AutoCleanseSuite(df, use_slm=True).run()
print(clean.report.actions_run)
print(clean.report.slm_directives.get("tools_invoked"))
```

## Register a custom tool

```python
from autocausal.skilling import ToolDef, ToolSurface, suite_tool_surface
from autocausal.suites.action_protocol import ActionResult

surface = suite_tool_surface()

def my_handler(df, **kwargs):
    return ActionResult(name="custom", payload={"ok": True})

surface.register(ToolDef(
    name="custom.ping",
    description="Example custom tool",
    parameters={"type": "object", "properties": {}},
    suite="custom",
    action="ping",
    handler=my_handler,
))
```

## Catalog / drill

```python
from autocausal.skilling import skill_catalog, SkillDrill

print(skill_catalog()["n_tools"])
drill = SkillDrill(skill="skill:autocleanse", use_slm=False)
trace = drill.run()
print(drill.to_markdown())
```

## CLI (thin)

```bash
python -m autocausal skilling list
python -m autocausal skilling drill --skill skill:autoeda
```

Prefer the library API in apps and notebooks.

## Related

- [SUITES.md](SUITES.md) — AutoCleanse / AutoEDA / AutoMine modules + actions
- `autocausal.slm` — RuleBackend / HuggingFaceSLM create/guide/infer
