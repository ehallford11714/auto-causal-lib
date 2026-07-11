# CLI reference (`python -m autocausal`)

Thin wrapper over library modules. Prefer importing `autocausal` in apps.

```bash
python -m autocausal --help
python -m autocausal --version
autocausal --help   # console script from pyproject
```

> Outputs are exploratory — not causal identification.

## Global source flags

Most data commands accept one of:

| Flag | Meaning |
|------|---------|
| `--csv PATH` | Load CSV |
| `--parquet PATH` | Load Parquet (needs pyarrow) |
| `--db URL` | SQLAlchemy URL (+ `--table` or `--query`) |
| `--schema` / `--limit` | SQL options |
| `--join IDS` / `--join-on KEYS` | Left-join public suite ids |

## Subcommands

### Core pipeline

| Command | Purpose |
|---------|---------|
| `discover` | Impute + discover (`--impute`, `--alpha`, `--min-corr`, `--no-iv`, `--guide`, `--slm`, `--ground`, `--format`, `-o`) |
| `mine` | Profile + associations (`--min-score`) |
| `auto` | Full: join? → mine → impute → discover → guide → ground (`--text`, `--guides`, `--physics`, …) |
| `ping` | DB/public health (`--url` \| `--public`, `--no-network`) |
| `doctor` | Environment / engine / optional-dep health (`--json`, `--production`) |
| `dialects` | Print SQLAlchemy dialect matrix JSON |
| `slm-status` | RuleBackend / HuggingFace availability |

### Guides / SLM

| Command | Purpose |
|---------|---------|
| `guide` | Rule/SLM guide after mine+discover (`--text`, `--slm`, `--guides`) |
| `direct` | DirectionPlan via backends (`--guides`, `--no-second-pass`) |
| `guides list\|status` | Guide backend registry |
| `create` | Propose questions / instruments / morphemes |
| `infer` | Narrative + caveats over discovery/IV |

### Tools / engines / estimate / refute

| Command | Purpose |
|---------|---------|
| `tools list\|validate\|invoke` | `suite_tools` registry |
| `engines list\|status` | Unified discovery/estimate/refute/package engines |
| `estimate` | ATE/CATE (`--backend builtin_ols\|doubleml\|econml\|…`, `--y/--d/--x/--z`, `--discover`) |
| `refute` | Placebo or DoWhy (`--method`, `--discover`) |
| `mcp` / `mcp --list-tools` | MCP info; run server via `python -m autocausal.mcp` |

### Suites / skilling / insight

| Command | Purpose |
|---------|---------|
| `suite cleanse\|eda\|mine` | SLM-directed AutoCleanse/EDA/Mine (`--slm`/`--no-slm`, `--text`) |
| `skilling list\|drill` | Skill catalog / offline SkillDrill |
| `insight run\|loop\|demo` | InsightSuite (see `insight.cli_hooks`) |

### Domain loops

| Command | Purpose |
|---------|---------|
| `physics loop\|rollout\|ui` | Physics predictive loop / Streamlit |
| `ml loop\|fit-imputer` | KPI-mined ML loop |
| `public list\|info\|load\|mine\|causal` | Public suite |
| `nlp extract\|features\|status\|download` | NLP helpers |
| `behavioral list\|mine` | Behavioral demos |
| `isolates-causal` | Soft IntentIsolates layer IV (`--text`) |

## Examples

```bash
python -m autocausal discover --csv data.csv --format markdown
python -m autocausal doctor
python -m autocausal doctor --json
python -m autocausal doctor --production
python -m autocausal engines status
python -m autocausal estimate --csv data.csv --backend builtin_ols --y y --d x --discover
python -m autocausal refute --csv data.csv --method placebo --discover
python -m autocausal suite mine --csv data.csv --no-slm
python -m autocausal skilling list
python -m autocausal insight demo
python -m autocausal.mcp --list-tools
```

See also: [INDEX.md](INDEX.md), [CAUSAL_BACKENDS.md](CAUSAL_BACKENDS.md), [MCP.md](MCP.md).
