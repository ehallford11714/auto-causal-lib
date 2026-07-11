"""CLI: python -m autocausal [--help] | help | discover | mine | correlate | ...

Thin consumer of library modules — prefer importing ``autocausal`` /
``autocausal.suites`` / ``autocausal.nlp`` / ``autocausal.research`` in apps.

Full module + function catalog::

    python -m autocausal help --all
    python -m autocausal help --module research
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from autocausal import AutoCausal, __version__
from autocausal.ingest import DIALECT_MATRIX


def _emit(text: str, out: Optional[str]) -> None:
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"Wrote {out}", file=sys.stderr)
    try:
        print(text)
    except UnicodeEncodeError:
        # Windows consoles (cp1252) may lack some markdown punctuation
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write(text.encode(enc, errors="replace"))
        sys.stdout.buffer.write(b"\n")


def _parse_guides(raw: Optional[str]) -> Optional[list[str]]:
    if not raw:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def _cli_use_slm(args: argparse.Namespace) -> Optional[bool]:
    """SLM guides by default; ``--no-slm`` forces rules; ``--slm`` forces try-on."""
    if getattr(args, "no_slm", False):
        return False
    if getattr(args, "slm", False):
        return True
    return None


def _load_ac(args: argparse.Namespace) -> AutoCausal:
    mode = getattr(args, "mode", None) or "exploratory"
    if getattr(args, "csv", None):
        ac = AutoCausal.from_csv(args.csv, mode=mode)
    elif getattr(args, "parquet", None):
        ac = AutoCausal.from_parquet(args.parquet, mode=mode)
    elif getattr(args, "db", None):
        if not args.table and not getattr(args, "query", None):
            raise SystemExit("--db requires --table or --query")
        ac = AutoCausal.from_sqlalchemy(
            args.db,
            table=args.table,
            query=getattr(args, "query", None),
            schema=getattr(args, "schema", None),
            limit=getattr(args, "limit", None),
            mode=mode,
        )
    else:
        raise SystemExit("Provide --csv, --parquet, or --db")
    join = getattr(args, "join", None)
    if join:
        on_raw = getattr(args, "join_on", None)
        on = [x.strip() for x in on_raw.split(",") if x.strip()] if on_raw else None
        ac.join_public(join, on=on, allow_network=False)
    return ac


def _add_source_args(p: argparse.ArgumentParser, *, require: bool = True) -> None:
    src = p.add_mutually_exclusive_group(required=require)
    src.add_argument("--csv", type=str, help="Path to CSV file")
    src.add_argument("--parquet", type=str, help="Path to Parquet file")
    src.add_argument("--db", type=str, help="SQLAlchemy engine URL")
    p.add_argument("--table", type=str, help="Table name (with --db)")
    p.add_argument("--query", type=str, help="SQL query (with --db)")
    p.add_argument("--schema", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--join",
        type=str,
        default=None,
        help="Comma-separated public suite ids to left-join",
    )
    p.add_argument("--join-on", type=str, default=None, dest="join_on", help="Join key(s), comma-separated")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autocausal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "AutoCausal library CLI — mine, discover, estimate, research, report, "
            "and inspect the full public API surface.\n\n"
            f"Package version: {__version__}\n"
            "Import in Python as `import autocausal` (PyPI name: auto-causal-lib)."
        ),
    )
    p.add_argument("--version", action="version", version=f"autocausal {__version__}")
    sub = p.add_subparsers(dest="command")

    # help — full library / module / API catalog
    help_p = sub.add_parser(
        "help",
        help="Full library help: modules, AutoCausal methods, CLI, and public symbols",
        description=(
            "Emit a catalog of AutoCausal modules and public functions/classes. "
            "Use --all for per-module symbols, --module to drill in, --api for "
            "AutoCausal methods, or --cli for command listing."
        ),
    )
    help_p.add_argument(
        "--module",
        type=str,
        default=None,
        help="Restrict to one package (e.g. research, inference, reporting)",
    )
    help_p.add_argument(
        "--api",
        action="store_true",
        help="Focus on AutoCausal session methods",
    )
    help_p.add_argument(
        "--cli",
        action="store_true",
        dest="cli_focus",
        help="Focus on CLI commands",
    )
    help_p.add_argument(
        "--all",
        action="store_true",
        dest="help_all",
        help="Include public symbols for every module",
    )
    help_p.add_argument(
        "--format",
        choices=["markdown", "json", "table"],
        default="markdown",
        dest="fmt",
    )
    help_p.add_argument("-o", "--out", type=str, default=None)

    # discover
    d = sub.add_parser("discover", help="Impute + discover causal edges")
    _add_source_args(d)
    d.add_argument("--impute", choices=["auto", "median_mode", "knn"], default="auto")
    d.add_argument("--alpha", type=float, default=0.05)
    d.add_argument("--min-corr", type=float, default=0.15, dest="min_corr")
    d.add_argument("--no-iv", action="store_true")
    d.add_argument("--guide", action="store_true", help="Run rule/SLM guide after discover")
    d.add_argument("--slm", action="store_true", help="Force HuggingFace SLM guide (default: on)")
    d.add_argument("--no-slm", action="store_true", help="Force rule-only guide")
    d.add_argument("--ground", action="store_true", help="Ground edges against glossaries")
    d.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    d.add_argument("-o", "--out", type=str, default=None)

    # mine
    m = sub.add_parser("mine", help="Profile + association mining")
    _add_source_args(m)
    m.add_argument("--min-score", type=float, default=0.15, dest="min_score")
    m.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    m.add_argument("-o", "--out", type=str, default=None)

    # ping
    ping_p = sub.add_parser("ping", help="Connection health check")
    ping_p.add_argument("--url", type=str, default=None, help="SQLAlchemy URL")
    ping_p.add_argument("--public", action="store_true", help="Ping bundled + optional public targets")
    ping_p.add_argument("--no-network", action="store_true", help="Skip optional network targets")
    ping_p.add_argument("--timeout", type=float, default=5.0)
    ping_p.add_argument("-o", "--out", type=str, default=None)

    # doctor
    doc = sub.add_parser("doctor", help="Environment / engine / optional-dep health check")
    doc.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    doc.add_argument(
        "--production",
        action="store_true",
        help="Run production safety checklist (no default auto IV, engines, qc, version)",
    )
    doc.add_argument("-o", "--out", type=str, default=None)

    # guide
    g = sub.add_parser("guide", help="SLM/rule guide from mine+discover outputs")
    _add_source_args(g)
    g.add_argument("--text", type=str, default=None, help="User question / hint")
    g.add_argument("--slm", action="store_true", help="Force HuggingFace SLM (default: on)")
    g.add_argument("--no-slm", action="store_true", help="Force rule-only guide")
    g.add_argument(
        "--guides",
        type=str,
        default=None,
        help="Comma-separated backends: llmintent,retracement,kineteq_pivot,grail,rule,huggingface",
    )
    g.add_argument("--impute", choices=["auto", "median_mode", "knn"], default="auto")
    g.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    g.add_argument("-o", "--out", type=str, default=None)

    # direct (direction steering)
    di = sub.add_parser("direct", help="Steer causal direction with guide backends -> DirectionPlan")
    _add_source_args(di)
    di.add_argument("--text", type=str, default=None)
    di.add_argument(
        "--guides",
        type=str,
        default="llmintent,retracement,kineteq_pivot,rule",
        help="Comma-separated guide backends",
    )
    di.add_argument("--slm", action="store_true", help="Force SLM (default: on)")
    di.add_argument("--no-slm", action="store_true", help="Force rule-only backends")
    di.add_argument("--no-second-pass", action="store_true")
    di.add_argument("--impute", choices=["auto", "median_mode", "knn"], default="auto")
    di.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    di.add_argument("-o", "--out", type=str, default=None)

    # guides registry
    guides = sub.add_parser("guides", help="Direction-steering guide backend registry")
    guides_sub = guides.add_subparsers(dest="guides_cmd")
    gl = guides_sub.add_parser("list", help="Show which backends are available")
    gl.add_argument("--format", choices=["json", "table"], default="table", dest="fmt")
    guides_sub.add_parser("status", help="Detailed guides + env status")

    # create (SLM-aided creation)
    cr = sub.add_parser("create", help="Propose causal questions / instruments / morphemes")
    _add_source_args(cr, require=False)
    cr.add_argument("--text", type=str, default=None)
    cr.add_argument("--slm", action="store_true", help="Force SLM (default: on)")
    cr.add_argument("--no-slm", action="store_true", help="Force rule-only create")
    cr.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    cr.add_argument("-o", "--out", type=str, default=None)

    # infer (SLM-aided inference narrative)
    inf = sub.add_parser("infer", help="Interpret discovery/IV results with caveats")
    _add_source_args(inf)
    inf.add_argument("--text", type=str, default=None)
    inf.add_argument("--slm", action="store_true", help="Force SLM (default: on)")
    inf.add_argument("--no-slm", action="store_true", help="Force rule-only interpret")
    inf.add_argument("--impute", choices=["auto", "median_mode", "knn"], default="auto")
    inf.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    inf.add_argument("-o", "--out", type=str, default=None)

    # tools suite
    tools = sub.add_parser("tools", help="Causal/NLP/KPI tool suite registry")
    tools_sub = tools.add_subparsers(dest="tools_cmd")
    tl = tools_sub.add_parser("list", help="List registered tools")
    tl.add_argument("--category", type=str, default=None, choices=["causal", "nlp", "kpi", "validation"])
    tl.add_argument("--format", choices=["json", "table"], default="table", dest="fmt")
    tv = tools_sub.add_parser("validate", help="Run validation suite on CSV/report")
    _add_source_args(tv, require=False)
    tv.add_argument("--y", type=str, default=None)
    tv.add_argument("--d", type=str, default=None)
    tv.add_argument("--z", type=str, default=None)
    tv.add_argument("--text", type=str, default="", help="Claims text for NLP checks")
    tv.add_argument("--format", choices=["markdown", "json"], default="markdown", dest="fmt")
    tv.add_argument("-o", "--out", type=str, default=None)
    ti = tools_sub.add_parser("invoke", help="Invoke a single tool by id")
    ti.add_argument("tool_id", type=str)
    ti.add_argument("--text", type=str, default="")
    ti.add_argument("--csv", type=str, default=None)
    ti.add_argument("--y", type=str, default=None)
    ti.add_argument("--d", type=str, default=None)
    ti.add_argument("--z", type=str, default=None)

    # physics suite
    phys = sub.add_parser("physics", help="Physics predictive engine / autocausal loop")
    phys_sub = phys.add_subparsers(dest="physics_cmd")
    pl = phys_sub.add_parser("loop", help="Mine -> discover -> rollout -> physical ground -> guide")
    _add_source_args(pl)
    pl.add_argument("--horizon", type=int, default=5)
    pl.add_argument("--text", type=str, default=None)
    pl.add_argument(
        "--system",
        choices=["damped_oscillator", "drift_diffusion", "linear_ode"],
        default="damped_oscillator",
    )
    pl.add_argument(
        "--domain",
        type=str,
        default="auto",
        help="mechanics-lite | markets-as-dynamics | affect-as-dynamics | auto",
    )
    pl.add_argument("--slm", action="store_true", help="Force SLM (default: on)")
    pl.add_argument("--no-slm", action="store_true", help="Force rule-only physics guide")
    pl.add_argument("--impute", choices=["auto", "median_mode", "knn"], default="auto")
    pl.add_argument("--no-second-pass", action="store_true")
    pl.add_argument("--web-ground", action="store_true", dest="web_ground")
    pl.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    pl.add_argument("-o", "--out", type=str, default=None)
    pr = phys_sub.add_parser("rollout", help="Physics-only KPI/state rollout")
    _add_source_args(pr)
    pr.add_argument("--horizon", type=int, default=5)
    pr.add_argument(
        "--system",
        choices=["damped_oscillator", "drift_diffusion", "linear_ode"],
        default="damped_oscillator",
    )
    pr.add_argument("--discover", action="store_true", help="Fit edge coupling from discover first")
    pr.add_argument("--impute", choices=["auto", "median_mode", "knn"], default="auto")
    pr.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    pr.add_argument("-o", "--out", type=str, default=None)
    pui = phys_sub.add_parser(
        "ui",
        help="Launch Streamlit physics demo (needs autocausal[ui]; default port 8518)",
    )
    pui.add_argument("--port", type=int, default=8518, help="Streamlit server port (default 8518)")
    pui.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address (default 127.0.0.1)",
    )
    pui.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a browser tab",
    )

    # auto
    a = sub.add_parser("auto", help="Full pipeline: join? mine impute discover guide ground")
    _add_source_args(a)
    a.add_argument("--text", type=str, default=None)
    a.add_argument("--slm", action="store_true", help="Force SLM (default: on)")
    a.add_argument("--no-slm", action="store_true", help="Force rule-only auto pipeline")
    a.add_argument(
        "--guides",
        type=str,
        default=None,
        help="Comma-separated guide backends (enables DirectionPlan steering)",
    )
    a.add_argument("--impute", choices=["auto", "median_mode", "knn"], default="auto")
    a.add_argument("--web-ground", action="store_true", dest="web_ground")
    a.add_argument("--no-second-pass", action="store_true")
    a.add_argument("--physics", action="store_true", help="Run physics predictive loop")
    a.add_argument("--horizon", type=int, default=5, help="Physics rollout horizon (with --physics)")
    a.add_argument(
        "--physics-system",
        choices=["damped_oscillator", "drift_diffusion", "linear_ode"],
        default="damped_oscillator",
        dest="physics_system",
    )
    a.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    a.add_argument("-o", "--out", type=str, default=None)

    # public suite
    pub = sub.add_parser("public", help="Public / demo dataset suite + causal mining")
    pub_sub = pub.add_subparsers(dest="public_cmd")
    pub_list = pub_sub.add_parser("list", help="List suite members")
    pub_list.add_argument("--offline", action="store_true")
    pub_list.add_argument("--format", choices=["json", "table"], default="table", dest="fmt")
    pub_info = pub_sub.add_parser("info", help="Show one suite member")
    pub_info.add_argument("id", type=str)
    pub_load = pub_sub.add_parser("load", help="Load suite member to CSV/stdout summary")
    pub_load.add_argument("id", type=str)
    pub_load.add_argument("--allow-network", action="store_true")
    pub_load.add_argument("-o", "--out", type=str, default=None)
    pub_mine = pub_sub.add_parser(
        "mine",
        help="Join public sources -> mine (+ optional discover)",
    )
    pub_mine.add_argument(
        "--sources",
        type=str,
        default="finance_demo,demographics_demo,health_demo",
        help="Comma-separated public suite ids",
    )
    pub_mine.add_argument("--join-on", type=str, default=None, dest="join_on")
    pub_mine.add_argument("--discover", action="store_true", help="Also run causal discovery")
    pub_mine.add_argument("--no-iv", action="store_true")
    pub_mine.add_argument("--allow-network", action="store_true")
    pub_mine.add_argument("--min-score", type=float, default=0.15, dest="min_score")
    pub_mine.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    pub_mine.add_argument("-o", "--out", type=str, default=None)
    pub_causal = pub_sub.add_parser(
        "causal",
        help="Full public corpus causal report (mine -> impute -> discover -> markdown/JSON)",
    )
    pub_causal.add_argument(
        "--sources",
        type=str,
        default="finance_demo,demographics_demo,climate_demo,health_demo",
        help="Comma-separated public suite ids",
    )
    pub_causal.add_argument("--join-on", type=str, default=None, dest="join_on")
    pub_causal.add_argument("--no-iv", action="store_true")
    pub_causal.add_argument("--validate", action="store_true", help="Light edge↔association check")
    pub_causal.add_argument("--allow-network", action="store_true")
    pub_causal.add_argument("--min-corr", type=float, default=0.12, dest="min_corr")
    pub_causal.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    pub_causal.add_argument("-o", "--out", type=str, default=None)

    # ml — KPI-mined loop + imputer fit
    ml = sub.add_parser("ml", help="KPI-mined causal loop / ML Model Hub slice")
    ml_sub = ml.add_subparsers(dest="ml_cmd")
    ml_loop = ml_sub.add_parser(
        "loop",
        help="Mine KPIs -> SLM/Rule ModelConstructPlan -> impute -> discover -> physics",
    )
    _add_source_args(ml_loop)
    ml_loop.add_argument("--text", type=str, default="")
    ml_loop.add_argument("--torch", action="store_true", help="Prefer PyTorch MLP when installed")
    ml_loop.add_argument("--slm", action="store_true", help="Force SLM guide (default: on)")
    ml_loop.add_argument("--no-slm", action="store_true", help="Force rule-only ML loop")
    ml_loop.add_argument(
        "--guides",
        type=str,
        default="rule",
        help="Comma-separated guides: rule,slm/huggingface,llmintent,kineteq_pivot,grail",
    )
    ml_loop.add_argument("--horizon", type=int, default=5)
    ml_loop.add_argument("--no-physics", action="store_true")
    ml_loop.add_argument("--epochs", type=int, default=40)
    ml_loop.add_argument(
        "--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt"
    )
    ml_loop.add_argument("-o", "--out", type=str, default=None)
    ml_fit = ml_sub.add_parser("fit-imputer", help="Fit imputer only (torch|sklearn|median)")
    _add_source_args(ml_fit)
    ml_fit.add_argument(
        "--backend",
        choices=["torch", "sklearn", "median", "torch_mlp", "iterative"],
        default="median",
    )
    ml_fit.add_argument("--epochs", type=int, default=40)
    ml_fit.add_argument(
        "--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt"
    )
    ml_fit.add_argument("-o", "--out", type=str, default=None)

    # isolates-causal — soft IntentIsolates layer IV bridge
    ic = sub.add_parser(
        "isolates-causal",
        help="Layer motifs -> indication vs IV (requires intentisolates)",
    )
    ic.add_argument("--text", type=str, required=True, help="Input text")
    ic.add_argument("--outcome-hint", type=str, default=None, dest="outcome_hint")
    ic.add_argument("--n-bootstrap", type=int, default=48, dest="n_bootstrap")
    ic.add_argument("--seed", type=int, default=17)
    ic.add_argument("--mock-iv", action="store_true", dest="mock_iv")
    ic.add_argument("--backend", type=str, default="rule")
    ic.add_argument(
        "--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt"
    )
    ic.add_argument("-o", "--out", type=str, default=None)

    # nlp — thin CLI over autocausal.nlp library
    nlp_p = sub.add_parser("nlp", help="NLTK NLP tooling (library: autocausal.nlp)")
    nlp_sub = nlp_p.add_subparsers(dest="nlp_cmd")
    nlp_ex = nlp_sub.add_parser("extract", help="Extract TextCausalHints from text")
    nlp_ex.add_argument("--text", type=str, required=True)
    nlp_ex.add_argument("--format", choices=["markdown", "json"], default="json", dest="fmt")
    nlp_ex.add_argument("-o", "--out", type=str, default=None)
    nlp_ft = nlp_sub.add_parser("features", help="Text column -> NLP feature CSV columns")
    nlp_ft.add_argument("--csv", type=str, required=True)
    nlp_ft.add_argument("--text-col", type=str, required=True, dest="text_col")
    nlp_ft.add_argument("--prefix", type=str, default="nlp_")
    nlp_ft.add_argument("-o", "--out", type=str, default=None)
    nlp_ft.add_argument("--format", choices=["json", "table"], default="table", dest="fmt")
    nlp_st = nlp_sub.add_parser("status", help="NLTK install / corpora status (offline)")
    nlp_st.add_argument("--format", choices=["json"], default="json", dest="fmt")
    nlp_dl = nlp_sub.add_parser("download", help="Soft-fail download of NLTK corpora")
    nlp_dl.add_argument("--format", choices=["json"], default="json", dest="fmt")

    # behavioral — thin CLI over autocausal.behavioral library
    beh = sub.add_parser("behavioral", help="Behavioral science traces (library: autocausal.behavioral)")
    beh_sub = beh.add_subparsers(dest="behavioral_cmd")
    beh_list = beh_sub.add_parser("list", help="List bundled demo traces")
    beh_list.add_argument("--format", choices=["json", "table"], default="table", dest="fmt")
    beh_mine = beh_sub.add_parser("mine", help="Mine demo/file traces -> hypothesized edges")
    beh_mine.add_argument("--demo", type=str, default="habit_loop", help="Demo id or omit with --csv")
    beh_mine.add_argument("--csv", type=str, default=None, help="Trace CSV path")
    beh_mine.add_argument("--discover", action="store_true")
    beh_mine.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    beh_mine.add_argument("-o", "--out", type=str, default=None)

    # insight — thin CLI over autocausal.insight (soft-optional if package incomplete)
    try:
        from autocausal.insight.cli_hooks import register_insight_parser

        register_insight_parser(sub)
    except Exception:
        pass

    # suite — AutoCleanse / AutoEDA / AutoMine (library-first; thin CLI)
    suite = sub.add_parser(
        "suite",
        help="AutoCleanse / AutoEDA / AutoMine suites (prefer library API)",
    )
    suite_sub = suite.add_subparsers(dest="suite_cmd")
    for name, help_txt in (
        ("cleanse", "SLM-directed cleanse -> CleanseReport"),
        ("eda", "SLM-directed EDA -> EDAReport"),
        ("mine", "SLM-directed mine -> MineReport"),
    ):
        sp = suite_sub.add_parser(name, help=help_txt)
        _add_source_args(sp)
        sp.add_argument("--slm", action="store_true", default=True, help="Try SLM director (default)")
        sp.add_argument("--no-slm", action="store_true", help="Force rule director")
        sp.add_argument("--text", type=str, default="", help="Optional context for director")
        sp.add_argument("--format", choices=["markdown", "json"], default="markdown", dest="fmt")
        sp.add_argument("-o", "--out", type=str, default=None)

    # skilling — SLM tool surface (library-first; thin CLI)
    skill = sub.add_parser(
        "skilling",
        help="SLM skilling / tool surface (prefer library: autocausal.skilling)",
    )
    skill_sub = skill.add_subparsers(dest="skilling_cmd")
    sk_list = skill_sub.add_parser("list", help="List skills and tools")
    sk_list.add_argument("--format", choices=["markdown", "json"], default="markdown", dest="fmt")
    sk_list.add_argument("-o", "--out", type=str, default=None)
    sk_drill = skill_sub.add_parser("drill", help="Offline rule-path skill drill")
    sk_drill.add_argument(
        "--skill",
        type=str,
        default="skill:autocleanse",
        help="Skill id (skill:autocleanse|autoeda|automine|autocausal_loop)",
    )
    sk_drill.add_argument("--format", choices=["markdown", "json"], default="markdown", dest="fmt")
    sk_drill.add_argument("-o", "--out", type=str, default=None)

    # engines — unified causal backend connectivity
    eng = sub.add_parser(
        "engines",
        help="List/status for discovery, estimate, refute, and package engines",
    )
    eng_sub = eng.add_subparsers(dest="engines_cmd")
    eng_list = eng_sub.add_parser("list", help="List engines")
    eng_list.add_argument(
        "--kind",
        choices=["discovery", "estimate", "refute", "package"],
        default=None,
    )
    eng_list.add_argument("--format", choices=["json", "table"], default="table", dest="fmt")
    eng_sub.add_parser("status", help="Full engine status + connectivity map")

    # estimate
    est = sub.add_parser("estimate", help="ATE/CATE via builtin / DoubleML / EconML")
    _add_source_args(est)
    est.add_argument(
        "--backend",
        type=str,
        default="builtin_ols",
        help="builtin_ols | doubleml | econml | econml_causal_forest | builtin_2sls",
    )
    est.add_argument("--y", type=str, default=None)
    est.add_argument("--d", type=str, default=None)
    est.add_argument("--x", type=str, default=None, help="Comma-separated controls")
    est.add_argument("--z", type=str, default=None, help="Instrument (for 2SLS)")
    est.add_argument("--discover", action="store_true", help="Mine candidates first")
    est.add_argument("-o", "--out", type=str, default=None)

    # refute
    rf = sub.add_parser("refute", help="Refute edge via placebo or DoWhy")
    _add_source_args(rf)
    rf.add_argument(
        "--method",
        type=str,
        default="placebo",
        help="placebo | random_common_cause | dowhy | dowhy_data_subset | …",
    )
    rf.add_argument("--discover", action="store_true", help="Discover first to pick an edge")
    rf.add_argument("-o", "--out", type=str, default=None)

    # mcp — hint to stdio server
    mcp_p = sub.add_parser("mcp", help="MCP server info (run: python -m autocausal.mcp)")
    mcp_p.add_argument("--list-tools", action="store_true", dest="list_tools")

    research = sub.add_parser(
        "research",
        help="Citation-grounded deep research with intensity + cross-match",
    )
    research_sub = research.add_subparsers(dest="research_cmd")
    research_plan = research_sub.add_parser(
        "plan", help="Plan intensity/agenda without retrieval"
    )
    _add_source_args(research_plan)
    research_plan.add_argument(
        "--intensity",
        choices=["quick", "standard", "deep", "exhaustive"],
        default="standard",
    )
    research_plan.add_argument("--domain", type=str, default="general")
    research_plan.add_argument(
        "--population", type=str, default=None, help="Optional PECO population context"
    )
    research_plan.add_argument("-o", "--out", type=str, default=None)
    research_run = research_sub.add_parser(
        "run", help="Run offline/local deep research after discover"
    )
    _add_source_args(research_run)
    research_run.add_argument(
        "--intensity",
        choices=["quick", "standard", "deep", "exhaustive"],
        default="standard",
    )
    research_run.add_argument("--domain", type=str, default="general")
    research_run.add_argument("--population", type=str, default=None)
    research_run.add_argument(
        "--sources-json",
        type=str,
        default=None,
        help="Optional local SourceRecord JSON list for offline retrieval",
    )
    research_run.add_argument(
        "--approval",
        action="store_true",
        help="Grant exhaustive/high-impact approval when required",
    )
    research_run.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        dest="fmt",
    )
    research_run.add_argument("-o", "--out", type=str, default=None)
    research_deepen = research_sub.add_parser(
        "deepen",
        help="Resume/deepen a prior ResearchReport JSON with higher intensity",
    )
    research_deepen.add_argument(
        "--report-json",
        type=str,
        required=True,
        help="Path to a previous ResearchReport JSON",
    )
    research_deepen.add_argument(
        "--handoff-json",
        type=str,
        required=True,
        help="Path to the original ResearchHandoff JSON",
    )
    research_deepen.add_argument(
        "--intensity",
        choices=["quick", "standard", "deep", "exhaustive"],
        default="deep",
    )
    research_deepen.add_argument(
        "--sources-json",
        type=str,
        default=None,
        help="Optional local SourceRecord JSON list",
    )
    research_deepen.add_argument("--approval", action="store_true")
    research_deepen.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        dest="fmt",
    )
    research_deepen.add_argument("-o", "--out", type=str, default=None)

    sub.add_parser("dialects", help="Print supported SQLAlchemy dialect matrix")
    sub.add_parser("slm-status", help="Show RuleBackend / HuggingFace SLM availability")

    # slm — setup-qwen / status
    slm_p = sub.add_parser("slm", help="Local SLM / Qwen setup and status")
    slm_sub = slm_p.add_subparsers(dest="slm_cmd")
    slm_sub.add_parser("status", help="Hardware + recommended Qwen + SLM readiness")
    sq = slm_sub.add_parser("setup-qwen", help="Probe hardware, pick Qwen, download to HF cache")
    sq.add_argument("--model", type=str, default=None, help="Override model id")
    sq.add_argument("--no-download", action="store_true", help="Only set env / recommend")
    sq.add_argument("--json", action="store_true", help="Emit JSON")

    # slm-loop / langgraph — LangGraph/FSM SLM chain
    sll = sub.add_parser(
        "slm-loop",
        help="Run LangGraph/FSM SLM chain (guide->skill->validate->compact->insight->route)",
    )
    _add_source_args(sll, require=False)
    sll.add_argument("--text", type=str, default="")
    sll.add_argument("--rounds", type=int, default=2)
    sll.add_argument("--slm", action="store_true", default=True)
    sll.add_argument("--no-slm", action="store_true")
    sll.add_argument("--model", type=str, default=None)
    sll.add_argument("--ensure-qwen", action="store_true")
    sll.add_argument("--no-langgraph", action="store_true")
    sll.add_argument("--format", choices=["markdown", "json"], default="markdown", dest="fmt")
    sll.add_argument("-o", "--out", type=str, default=None)
    # alias
    lg = sub.add_parser("langgraph", help="Alias for slm-loop")
    _add_source_args(lg, require=False)
    lg.add_argument("--text", type=str, default="")
    lg.add_argument("--rounds", type=int, default=2)
    lg.add_argument("--slm", action="store_true", default=True)
    lg.add_argument("--no-slm", action="store_true")
    lg.add_argument("--model", type=str, default=None)
    lg.add_argument("--ensure-qwen", action="store_true")
    lg.add_argument("--no-langgraph", action="store_true")
    lg.add_argument("--format", choices=["markdown", "json"], default="markdown", dest="fmt")
    lg.add_argument("-o", "--out", type=str, default=None)

    # correlate — descriptive association (never causal identification)
    corr = sub.add_parser(
        "correlate",
        help="Typed association / correlation matrix (descriptive, not causal)",
    )
    _add_source_args(corr)
    corr.add_argument("--x", type=str, default=None, help="First variable (pair mode)")
    corr.add_argument("--y", type=str, default=None, help="Second variable (pair mode)")
    corr.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Comma-separated columns for matrix mode",
    )
    corr.add_argument("--method", type=str, default="auto")
    corr.add_argument("--controls", type=str, default=None, help="Comma-separated controls")
    corr.add_argument("--bootstrap-n", type=int, default=0, dest="bootstrap_n")
    corr.add_argument("--format", choices=["markdown", "json"], default="json", dest="fmt")
    corr.add_argument("-o", "--out", type=str, default=None)

    # tabular-ml — leakage-safe AutoTabularML
    tml = sub.add_parser(
        "tabular-ml",
        help="Leakage-safe AutoTabularML (predictive metrics are not causal effects)",
    )
    _add_source_args(tml)
    tml.add_argument("--target", type=str, required=True)
    tml.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature columns (default: all non-target)",
    )
    tml.add_argument("--task", type=str, default=None)
    tml.add_argument("--group", type=str, default=None, dest="group_column")
    tml.add_argument("--time", type=str, default=None, dest="time_column")
    tml.add_argument("--mode", choices=["exploratory", "production"], default="exploratory")
    tml.add_argument("--calibrate", action="store_true")
    tml.add_argument("--format", choices=["markdown", "json"], default="json", dest="fmt")
    tml.add_argument("-o", "--out", type=str, default=None)

    # autoviz — analysis-aware chart planning
    av = sub.add_parser(
        "autoviz",
        help="Plan analysis-aware visualizations (descriptive; not causal proof)",
    )
    _add_source_args(av)
    av.add_argument("--discover", action="store_true", help="Run discover before planning")
    av.add_argument("--slm", action="store_true", help="Force SLM enrichment (default: on)")
    av.add_argument("--no-slm", action="store_true", help="Force rule-only viz planning")
    av.add_argument("--mode", choices=["exploratory", "production"], default="exploratory")
    av.add_argument("--format", choices=["markdown", "json"], default="json", dest="fmt")
    av.add_argument("-o", "--out", type=str, default=None)

    # report-artifact — validated PDF/Markdown/HTML via ReportEngine
    ra = sub.add_parser(
        "report-artifact",
        help="Generate a provenance-validated report artifact (PDF/MD/HTML/JSON)",
    )
    _add_source_args(ra)
    ra.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path (.pdf / .md / .html / .json)",
    )
    ra.add_argument("--discover", action="store_true", help="Run mine+discover first")
    ra.add_argument("--title", type=str, default="AutoCausal Analysis Report")
    ra.add_argument(
        "--profile",
        choices=["production", "exploratory"],
        default="production",
    )
    ra.add_argument("--format", type=str, default=None, help="Force format override")
    ra.add_argument("--slm", action="store_true", help="Force SLM report director (default: on)")
    ra.add_argument("--no-slm", action="store_true", help="Force deterministic report director")

    # integrations — lazy optional-dependency catalog / routing / install plans
    from autocausal.integrations.cli import register_integrations_parser

    register_integrations_parser(sub)

    # Expand root --help after all subcommands exist.
    from autocausal.help_catalog import root_help_epilog

    p.epilog = root_help_epilog(p)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "help":
        from autocausal.help_catalog import library_help

        text = library_help(
            module=getattr(args, "module", None),
            api=bool(getattr(args, "api", False)),
            cli=bool(getattr(args, "cli_focus", False)),
            all=bool(getattr(args, "help_all", False)),
            format=getattr(args, "fmt", "markdown"),
            parser=parser,
        )
        _emit(text, getattr(args, "out", None))
        return 0

    if args.command == "dialects":
        print(json.dumps(DIALECT_MATRIX, indent=2))
        return 0

    if args.command == "integrations":
        from autocausal.integrations.cli import handle_integrations

        return handle_integrations(args)

    if args.command == "doctor":
        from autocausal.doctor import doctor_report, format_doctor_markdown

        report = doctor_report(production=bool(getattr(args, "production", False)))
        if getattr(args, "json", False):
            text = json.dumps(report, indent=2)
        else:
            text = format_doctor_markdown(report)
        _emit(text, getattr(args, "out", None))
        # Non-zero exit when --production and checklist failed
        if getattr(args, "production", False) and not report.get("production_ok", True):
            return 1
        return 0

    if args.command == "ping":
        from autocausal.db import ping, ping_public

        if args.public:
            results = ping_public(include_network=not args.no_network, timeout=args.timeout)
            payload = [r.to_dict() for r in results]
        elif args.url:
            payload = ping(args.url, timeout=args.timeout).to_dict()
        else:
            parser.error("ping requires --url or --public")
            return 2
        text = json.dumps(payload, indent=2)
        _emit(text, args.out)
        return 0

    if args.command == "guides":
        from autocausal.guides import guides_status, list_guides

        if args.guides_cmd == "status":
            print(json.dumps(guides_status(), indent=2))
            return 0
        if args.guides_cmd in (None, "list"):
            rows = list_guides()
            fmt = getattr(args, "fmt", "table")
            if fmt == "json":
                print(json.dumps(rows, indent=2))
            else:
                print(f"{'id':16} {'available':10} {'priority':8} name / detail")
                for r in rows:
                    avail = "yes" if r["available"] else "no*"
                    print(
                        f"{r['id']:16} {avail:10} {r['priority']:<8} {r['name']} — {r['detail']}"
                    )
                print("\n* soft-optional; stubs/fallbacks still run when selected")
            return 0
        parser.parse_args(["guides", "--help"])
        return 0

    if args.command == "public":
        from autocausal.public_suite import get_public, list_public, load_public

        if args.public_cmd == "list":
            sources = list_public(offline_only=args.offline)
            if args.fmt == "json":
                print(json.dumps([s.to_dict() for s in sources], indent=2))
            else:
                print(f"{'id':20} {'access':10} {'domain':14} name")
                for s in sources:
                    print(f"{s.id:20} {s.access:10} {s.domain:14} {s.name}")
            return 0
        if args.public_cmd == "info":
            print(json.dumps(get_public(args.id).to_dict(), indent=2))
            return 0
        if args.public_cmd == "load":
            try:
                df = load_public(args.id, allow_network=args.allow_network)
            except Exception as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            if args.out:
                df.to_csv(args.out, index=False)
                print(f"Wrote {args.out} ({len(df)} rows)", file=sys.stderr)
            else:
                print(df.head(20).to_string(index=False))
                print(f"\n[{len(df)} rows x {len(df.columns)} cols]", file=sys.stderr)
            return 0
        if args.public_cmd in ("mine", "causal"):
            from autocausal.public_causal import mine_public

            src_ids = [x.strip() for x in args.sources.split(",") if x.strip()]
            on_raw = getattr(args, "join_on", None)
            on = [x.strip() for x in on_raw.split(",") if x.strip()] if on_raw else None
            do_discover = args.public_cmd == "causal" or getattr(args, "discover", False)
            report = mine_public(
                src_ids,
                join_on=on,
                allow_network=bool(getattr(args, "allow_network", False)),
                discover=do_discover,
                use_iv=not getattr(args, "no_iv", False),
                min_score=getattr(args, "min_score", 0.15),
                min_abs_corr=getattr(args, "min_corr", 0.12),
                validate=bool(getattr(args, "validate", False)),
            )
            if args.fmt == "json":
                text = report.to_json()
            elif args.fmt == "both":
                text = report.to_markdown() + "\n\n```json\n" + report.to_json() + "\n```\n"
            else:
                text = report.to_markdown()
            _emit(text, args.out)
            return 0
        parser.parse_args(["public", "--help"])
        return 0

    if args.command == "mine":
        ac = _load_ac(args)
        ac.mine(min_score=args.min_score)
        report = ac.mining
        assert report is not None
        if args.fmt == "json":
            text = report.to_json()
        elif args.fmt == "both":
            text = report.to_markdown() + "\n\n```json\n" + report.to_json() + "\n```\n"
        else:
            text = report.to_markdown()
        _emit(text, args.out)
        return 0

    if args.command == "discover":
        ac = _load_ac(args)
        result = ac.run(
            impute_method=args.impute,
            alpha=args.alpha,
            min_abs_corr=args.min_corr,
            use_iv=not args.no_iv,
        )
        if args.guide or args.slm:
            ac.guide(use_slm=_cli_use_slm(args))
        if args.ground:
            ac.ground()
        if args.fmt == "json":
            text = result.to_json()
        elif args.fmt == "both":
            text = result.to_markdown() + "\n\n```json\n" + result.to_json() + "\n```\n"
        else:
            text = result.to_markdown()
        _emit(text, args.out)
        return 0

    if args.command == "guide":
        ac = _load_ac(args)
        ac.mine()
        ac.impute(method=args.impute)
        ac.discover()
        backends = _parse_guides(args.guides)
        gres = ac.guide(text=args.text, use_slm=_cli_use_slm(args), backends=backends)
        if backends and ac.direction_plan is not None:
            payload = ac.direction_plan
            if args.fmt == "json":
                text = payload.to_json()
            elif args.fmt == "both":
                text = payload.to_markdown() + "\n\n```json\n" + payload.to_json() + "\n```\n"
            else:
                text = payload.to_markdown()
        else:
            if args.fmt == "json":
                text = gres.to_json()
            elif args.fmt == "both":
                text = gres.to_markdown() + "\n\n```json\n" + gres.to_json() + "\n```\n"
            else:
                text = gres.to_markdown()
        _emit(text, args.out)
        return 0

    if args.command == "direct":
        ac = _load_ac(args)
        ac.impute(method=args.impute)
        backends = _parse_guides(args.guides) or [
            "llmintent",
            "retracement",
            "kineteq_pivot",
            "rule",
        ]
        plan = ac.direct(
            text=args.text,
            backends=backends,
            use_slm=_cli_use_slm(args),
            second_pass=not args.no_second_pass,
        )
        if args.fmt == "json":
            text = plan.to_json()
        elif args.fmt == "both":
            text = plan.to_markdown() + "\n\n```json\n" + plan.to_json() + "\n```\n"
        else:
            text = plan.to_markdown()
        _emit(text, args.out)
        return 0

    if args.command == "create":
        from autocausal.slm import create_from_context

        if args.csv or args.parquet or args.db:
            ac = _load_ac(args)
            cres = ac.create(text=args.text, use_slm=_cli_use_slm(args))
        else:
            cres = create_from_context(
                {"text": args.text or "", "columns": []},
                use_slm=_cli_use_slm(args),
            )
        if args.fmt == "json":
            text = cres.to_json()
        elif args.fmt == "both":
            text = cres.to_markdown() + "\n\n```json\n" + cres.to_json() + "\n```\n"
        else:
            text = cres.to_markdown()
        _emit(text, args.out)
        return 0

    if args.command == "infer":
        ac = _load_ac(args)
        ac.mine()
        ac.impute(method=args.impute)
        ac.discover()
        ires = ac.interpret(text=args.text, use_slm=_cli_use_slm(args))
        if args.fmt == "json":
            text = ires.to_json()
        elif args.fmt == "both":
            text = ires.to_markdown() + "\n\n```json\n" + ires.to_json() + "\n```\n"
        else:
            text = ires.to_markdown()
        _emit(text, args.out)
        return 0

    if args.command == "slm-status":
        from autocausal.slm import slm_status

        print(json.dumps(slm_status(), indent=2))
        return 0

    if args.command == "slm":
        from autocausal.slm import ensure_local_qwen, slm_status

        if args.slm_cmd in (None, "status"):
            print(json.dumps(slm_status(), indent=2))
            return 0
        if args.slm_cmd == "setup-qwen":
            res = ensure_local_qwen(
                model_id=getattr(args, "model", None),
                download=not bool(getattr(args, "no_download", False)),
                set_env=True,
            )
            if getattr(args, "json", False):
                print(json.dumps(res, indent=2, default=str))
            else:
                print(f"model: {res.get('model_id')}")
                print(f"ok: {res.get('ok')} downloaded: {res.get('downloaded')}")
                if res.get("cache_dir"):
                    print(f"cache: {res['cache_dir']}")
                for n in res.get("notes") or []:
                    print(f"- {n}")
            return 0 if res.get("ok") or getattr(args, "no_download", False) else 1
        parser.parse_args(["slm", "--help"])
        return 0

    if args.command in ("slm-loop", "langgraph"):
        from autocausal.agentic.langgraph_chain import run_slm_langgraph_loop
        from autocausal.datasets import load_dataset

        use_slm = not bool(getattr(args, "no_slm", False))
        if getattr(args, "csv", None) or getattr(args, "parquet", None) or getattr(args, "db", None):
            ac = _load_ac(args)
            report = run_slm_langgraph_loop(
                ac=ac,
                text=getattr(args, "text", "") or "",
                max_rounds=int(getattr(args, "rounds", 2)),
                use_slm=use_slm,
                model_name=getattr(args, "model", None),
                prefer_langgraph=not bool(getattr(args, "no_langgraph", False)),
                ensure_qwen=bool(getattr(args, "ensure_qwen", False)),
            )
        else:
            report = run_slm_langgraph_loop(
                load_dataset("iris"),
                text=getattr(args, "text", "") or "iris drivers",
                max_rounds=int(getattr(args, "rounds", 2)),
                use_slm=use_slm,
                model_name=getattr(args, "model", None),
                prefer_langgraph=not bool(getattr(args, "no_langgraph", False)),
                ensure_qwen=bool(getattr(args, "ensure_qwen", False)),
            )
        text = report.to_json() if args.fmt == "json" else report.to_markdown()
        _emit(text, getattr(args, "out", None))
        return 0

    if args.command == "tools":
        from autocausal.suite_tools import invoke_tool, list_tools, tool_catalog, validate_pipeline

        if args.tools_cmd == "list":
            tools_list = list_tools(category=args.category)
            if args.fmt == "json":
                print(json.dumps([t.to_dict() for t in tools_list], indent=2))
            else:
                print(f"{'id':22} {'cat':10} {'status':10} name")
                for t in tools_list:
                    print(f"{t.id:22} {t.category:10} {t.status:10} {t.name}")
            return 0
        if args.tools_cmd == "validate":
            df = None
            if args.csv or args.parquet or args.db:
                ac = _load_ac(args)
                df = ac.df
                ac.mine()
                report = ac.mining.to_dict() if ac.mining is not None else {}
            else:
                report = {}
            vres = validate_pipeline(
                report,
                df=df,
                claims_text=args.text or "",
                y=args.y,
                d=args.d,
                z=args.z,
            )
            text = vres.to_json() if args.fmt == "json" else vres.to_markdown()
            _emit(text, args.out)
            return 0
        if args.tools_cmd == "invoke":
            kwargs: dict[str, Any] = {"text": args.text or ""}
            if args.csv:
                import pandas as pd

                kwargs["df"] = pd.read_csv(args.csv)
            if args.y:
                kwargs["y"] = args.y
            if args.d:
                kwargs["d"] = args.d
            if args.z:
                kwargs["z"] = args.z
            res = invoke_tool(args.tool_id, **kwargs)
            print(json.dumps(res.to_dict(), indent=2, default=str))
            return 0 if res.ok else 1
        print(json.dumps(tool_catalog(), indent=2))
        return 0

    if args.command == "physics":
        from autocausal.physics import PhysicsCausalSuite

        if args.physics_cmd == "loop":
            ac = _load_ac(args)
            suite = PhysicsCausalSuite.from_autocausal(ac, system=args.system)
            result = suite.loop(
                horizon=args.horizon,
                text=args.text,
                domain=args.domain,
                use_slm=_cli_use_slm(args),
                second_pass=not args.no_second_pass,
                use_web_ground=args.web_ground,
                impute_method=args.impute,
            )
            if args.fmt == "json":
                text = result.to_json()
            elif args.fmt == "both":
                text = result.to_markdown() + "\n\n```json\n" + result.to_json() + "\n```\n"
            else:
                text = result.to_markdown()
            _emit(text, args.out)
            return 0
        if args.physics_cmd == "rollout":
            ac = _load_ac(args)
            suite = PhysicsCausalSuite.from_autocausal(ac, system=args.system)
            if args.discover:
                ac.impute(method=args.impute)
                ac.discover()
            traj = suite.rollout(horizon=args.horizon, use_edges=bool(args.discover))
            if args.fmt == "json":
                text = traj.to_json()
            elif args.fmt == "both":
                text = traj.to_markdown() + "\n\n```json\n" + traj.to_json() + "\n```\n"
            else:
                text = traj.to_markdown()
            _emit(text, args.out)
            return 0
        if args.physics_cmd == "ui":
            try:
                import streamlit.web.cli as stcli  # type: ignore
            except ImportError:
                print(
                    'Streamlit not installed. Install UI extras:\n'
                    '  pip install -e ".[ui]"\n'
                    'Then: python -m autocausal physics ui --port 8518',
                    file=sys.stderr,
                )
                return 1
            from autocausal.apps import physics_demo_path

            ui_path = physics_demo_path()
            argv = [
                "streamlit",
                "run",
                ui_path,
                f"--server.port={int(args.port)}",
                f"--server.address={args.host}",
            ]
            if args.headless:
                argv.append("--server.headless=true")
            sys.argv = argv
            stcli.main()
            return 0
        parser.parse_args(["physics", "--help"])
        return 0

    if args.command == "ml":
        if args.ml_cmd == "loop":
            from autocausal.ml import KPIMinedCausalLoop

            ac = _load_ac(args)
            guides = _parse_guides(args.guides) or ["rule"]
            # map slm alias
            guides = ["huggingface" if g == "slm" else g for g in guides]
            loop = KPIMinedCausalLoop.from_autocausal(ac)
            result = loop.run(
                text=args.text or "",
                use_slm=(_cli_use_slm(args) is not False) or ("huggingface" in guides),
                use_torch=True if args.torch else None,
                guides=guides,
                horizon=args.horizon,
                physics=not args.no_physics,
                epochs=args.epochs,
            )
            if args.fmt == "json":
                text = result.to_json()
            elif args.fmt == "both":
                text = result.to_markdown() + "\n\n```json\n" + result.to_json() + "\n```\n"
            else:
                text = result.to_markdown()
            _emit(text, args.out)
            return 0
        if args.ml_cmd == "fit-imputer":
            from autocausal.ml.imputers import apply_imputer

            ac = _load_ac(args)
            backend = args.backend
            kind_map = {
                "torch": "torch_mlp",
                "torch_mlp": "torch_mlp",
                "sklearn": "iterative",
                "iterative": "iterative",
                "median": "median",
            }
            kind = kind_map.get(backend, "median")
            _out, _meta, fit = apply_imputer(ac.df, kind, epochs=args.epochs)  # type: ignore[arg-type]
            if args.fmt == "json":
                text = fit.to_json()
            elif args.fmt == "both":
                text = fit.to_markdown() + "\n\n```json\n" + fit.to_json() + "\n```\n"
            else:
                text = fit.to_markdown()
            _emit(text, args.out)
            return 0
        parser.parse_args(["ml", "--help"])
        return 0

    if args.command == "auto":
        join_on = args.join_on.split(",") if args.join_on else None
        path = args.csv or args.parquet or args.db
        if not path:
            parser.error("auto requires --csv, --parquet, or --db")
        result = AutoCausal.auto(
            path,
            table=args.table,
            query=args.query,
            text=args.text,
            use_slm=_cli_use_slm(args),
            guide_backends=_parse_guides(args.guides),
            join=args.join,
            join_on=join_on,
            use_web_ground=args.web_ground,
            impute_method=args.impute,
            second_pass=not args.no_second_pass,
            physics=bool(getattr(args, "physics", False)),
            physics_horizon=getattr(args, "horizon", 5),
            physics_system=getattr(args, "physics_system", "damped_oscillator"),
        )
        if args.fmt == "json":
            text = result.to_json()
        elif args.fmt == "both":
            text = result.to_markdown() + "\n\n```json\n" + result.to_json() + "\n```\n"
        else:
            text = result.to_markdown()
        _emit(text, args.out)
        return 0

    if args.command == "isolates-causal":
        try:
            from autocausal.isolates_bridge import run_isolates_causal
        except ImportError as e:
            print(str(e), file=sys.stderr)
            return 2
        try:
            result = run_isolates_causal(
                args.text,
                outcome_hint=args.outcome_hint,
                mock_iv=bool(args.mock_iv),
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                backend=args.backend,
            )
        except ImportError as e:
            print(str(e), file=sys.stderr)
            return 2
        if args.fmt == "json":
            text = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        elif args.fmt == "both":
            text = result.to_markdown() + "\n\n```json\n" + json.dumps(result.to_dict(), indent=2) + "\n```\n"
        else:
            text = result.to_markdown()
        _emit(text, args.out)
        return 0

    if args.command == "nlp":
        if args.nlp_cmd == "extract":
            from autocausal.nlp import extract_causal_hints_from_text

            hints = extract_causal_hints_from_text(args.text)
            if args.fmt == "markdown":
                roles = hints.roles.to_dict()
                lines = [
                    "# TextCausalHints",
                    "",
                    f"> {hints.caveat}",
                    "",
                    f"- backend: `{hints.backend}`",
                    f"- modality: {', '.join(hints.modality_markers) or '(none)'}",
                    "",
                    "## Roles",
                ]
                for role, items in roles.items():
                    lines.append(f"- **{role}**: {', '.join(items) or '(none)'}")
                text = "\n".join(lines) + "\n"
            else:
                text = json.dumps(hints.to_dict(), indent=2)
            _emit(text, args.out)
            return 0
        if args.nlp_cmd == "features":
            import pandas as pd
            from autocausal.nlp import NlpFeatureBuilder

            df = pd.read_csv(args.csv)
            out_df = NlpFeatureBuilder(prefix=args.prefix).transform_frame(df, args.text_col)
            if args.out:
                out_df.to_csv(args.out, index=False)
                print(f"Wrote {args.out} ({len(out_df)} rows)", file=sys.stderr)
            if args.fmt == "json":
                print(json.dumps({"columns": list(out_df.columns), "n_rows": len(out_df)}, indent=2))
            else:
                print(out_df.head(10).to_string(index=False))
            return 0
        if args.nlp_cmd == "status":
            from autocausal.nlp import nltk_status

            print(json.dumps(nltk_status().to_dict(), indent=2))
            return 0
        if args.nlp_cmd == "download":
            from autocausal.nlp import ensure_nltk_data

            print(json.dumps(ensure_nltk_data(), indent=2))
            return 0
        parser.parse_args(["nlp", "--help"])
        return 0

    if args.command == "behavioral":
        if args.behavioral_cmd == "list":
            from autocausal.behavioral import list_demos

            demos = list_demos()
            if args.fmt == "json":
                print(json.dumps(demos, indent=2))
            else:
                print(f"{'id':28} description")
                for d in demos:
                    print(f"{d['id']:28} {d['description']}")
            return 0
        if args.behavioral_cmd == "mine":
            from autocausal.behavioral import mine_behavioral_traces

            source = args.csv or args.demo
            result = mine_behavioral_traces(source, discover=bool(args.discover))
            if args.fmt == "json":
                text = json.dumps(result.to_dict(), indent=2, default=str)
            elif args.fmt == "both":
                text = result.to_markdown() + "\n\n```json\n" + json.dumps(result.to_dict(), indent=2, default=str) + "\n```\n"
            else:
                text = result.to_markdown()
            _emit(text, args.out)
            return 0
        parser.parse_args(["behavioral", "--help"])
        return 0

    if args.command == "insight":
        try:
            from autocausal.insight.cli_hooks import handle_insight
        except ImportError as e:
            print(f"insight module unavailable: {e}", file=sys.stderr)
            return 2
        return handle_insight(args)

    if args.command == "suite":
        use_slm = not bool(getattr(args, "no_slm", False))
        text = getattr(args, "text", "") or ""
        if args.suite_cmd == "cleanse":
            from autocausal.suites import AutoCleanseSuite

            ac = _load_ac(args)
            suite = AutoCleanseSuite(ac, use_slm=use_slm, text=text).run()
            assert suite.report is not None
            payload = suite.report.to_json() if args.fmt == "json" else suite.report.to_markdown()
            _emit(payload, args.out)
            return 0
        if args.suite_cmd == "eda":
            from autocausal.suites import AutoEDASuite

            ac = _load_ac(args)
            suite = AutoEDASuite(ac, use_slm=use_slm, text=text).run()
            assert suite.report is not None
            payload = suite.report.to_json() if args.fmt == "json" else suite.report.to_markdown()
            _emit(payload, args.out)
            return 0
        if args.suite_cmd == "mine":
            from autocausal.suites import AutoMineSuite

            ac = _load_ac(args)
            suite = AutoMineSuite(ac, use_slm=use_slm, text=text, join_public=None).run()
            assert suite.report is not None
            payload = suite.report.to_json() if args.fmt == "json" else suite.report.to_markdown()
            _emit(payload, args.out)
            return 0
        parser.parse_args(["suite", "--help"])
        return 0

    if args.command == "skilling":
        from autocausal.skilling import SkillDrill, skill_catalog

        if args.skilling_cmd == "list":
            cat = skill_catalog()
            if args.fmt == "json":
                payload = json.dumps(cat, indent=2, default=str)
            else:
                drill = SkillDrill()
                payload = drill.to_markdown()
            _emit(payload, args.out)
            return 0
        if args.skilling_cmd == "drill":
            drill = SkillDrill(skill=getattr(args, "skill", "skill:autocleanse"), use_slm=False)
            trace = drill.run()
            if args.fmt == "json":
                payload = trace.to_json()
            else:
                payload = drill.to_markdown()
            _emit(payload, args.out)
            return 0
        parser.parse_args(["skilling", "--help"])
        return 0

    if args.command == "engines":
        from autocausal.engines import engine_status, list_engines

        if args.engines_cmd == "status" or args.engines_cmd is None:
            print(json.dumps(engine_status(), indent=2, default=str))
            return 0
        if args.engines_cmd == "list":
            rows = list_engines(kind=getattr(args, "kind", None))
            if args.fmt == "json":
                print(json.dumps([r.to_dict() for r in rows], indent=2))
            else:
                print(f"{'id':28} {'kind':10} {'avail':6} description")
                for r in rows:
                    avail = "yes" if r.available else "no*"
                    print(f"{r.id:28} {r.kind:10} {avail:6} {r.description[:60]}")
                print("\n* soft-optional — install auto-causal-lib[causal-extra] or [mcp]")
            return 0
        parser.parse_args(["engines", "--help"])
        return 0

    if args.command == "estimate":
        ac = _load_ac(args)
        if args.discover:
            ac.mine()
            ac.discover(qc="off", use_iv=False)
        x = [s.strip() for s in args.x.split(",") if s.strip()] if args.x else None
        res = ac.estimate(backend=args.backend, y=args.y, d=args.d, x=x, z=args.z)
        text = json.dumps(res.to_dict(), indent=2, default=str)
        _emit(text, args.out)
        return 0 if res.ok else 1

    if args.command == "refute":
        ac = _load_ac(args)
        if args.discover:
            ac.mine()
            ac.discover(qc="off", use_iv=False)
        res = ac.refute(method=args.method)
        text = json.dumps(res.to_dict() if hasattr(res, "to_dict") else res, indent=2, default=str)
        _emit(text, args.out)
        return 0 if getattr(res, "ok", True) else 1

    if args.command == "mcp":
        from autocausal.connective import list_tools

        if args.list_tools:
            print(json.dumps(list_tools(), indent=2, default=str))
        else:
            print(
                json.dumps(
                    {
                        "run": "python -m autocausal.mcp",
                        "extra": "pip install 'auto-causal-lib[mcp]'",
                        "tools": list_tools(),
                        "note": "AgentHook works without mcp SDK via autocausal.connective",
                    },
                    indent=2,
                    default=str,
                )
            )
        return 0

    if args.command == "research":
        from autocausal.research import (
            DeepResearchSuite,
            LocalDocumentProvider,
            ResearchHandoff,
            ResearchPolicy,
            ResearchReport,
            SourceRecord,
        )

        def _load_records(path: Optional[str]) -> list[Any]:
            if not path:
                return []
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise SystemExit("--sources-json must contain a JSON list")
            return [
                item if isinstance(item, SourceRecord) else SourceRecord.from_dict(item)
                for item in payload
            ]

        if args.research_cmd == "plan":
            ac = _load_ac(args)
            ac.mine()
            ac.impute()
            ac.discover(use_iv=False, qc="off")
            context: dict[str, Any] = {}
            if getattr(args, "population", None):
                context["population"] = args.population
            handoff = ac.to_research_handoff(domain=args.domain, context=context)
            suite = DeepResearchSuite(
                policy=ResearchPolicy(allowed_providers=("local",)),
                providers=[LocalDocumentProvider([])],
            )
            plan = suite.plan(handoff, intensity=args.intensity)
            _emit(json.dumps(plan, indent=2, default=str), args.out)
            return 0

        if args.research_cmd == "run":
            ac = _load_ac(args)
            ac.mine()
            ac.impute()
            ac.discover(use_iv=False, qc="off")
            context = {}
            if getattr(args, "population", None):
                context["population"] = args.population
            records = _load_records(getattr(args, "sources_json", None))
            suite = DeepResearchSuite(
                policy=ResearchPolicy(
                    allowed_providers=("local",),
                    approval_granted=bool(getattr(args, "approval", False)),
                ),
                providers=[LocalDocumentProvider(records)],
            )
            report = ac.deep_research(
                intensity=args.intensity,
                domain=args.domain,
                context=context,
                suite=suite,
                approval_granted=bool(getattr(args, "approval", False)),
            )
            payload = report.to_json() if args.fmt == "json" else report.to_markdown()
            _emit(payload, args.out)
            return 0

        if args.research_cmd == "deepen":
            report = ResearchReport.from_json(
                Path(args.report_json).read_text(encoding="utf-8")
            )
            handoff = ResearchHandoff.from_dict(
                json.loads(Path(args.handoff_json).read_text(encoding="utf-8"))
            )
            records = _load_records(getattr(args, "sources_json", None))
            suite = DeepResearchSuite(
                policy=ResearchPolicy(
                    allowed_providers=("local",),
                    approval_granted=bool(getattr(args, "approval", False)),
                ),
                providers=[LocalDocumentProvider(records)],
            )
            deepened = suite.resume(
                report,
                handoff=handoff,
                intensity=args.intensity,
                approval_granted=bool(getattr(args, "approval", False)),
            )
            payload = (
                deepened.to_json() if args.fmt == "json" else deepened.to_markdown()
            )
            _emit(payload, args.out)
            return 0

        parser.parse_args(["research", "--help"])
        return 0

    if args.command == "correlate":
        ac = _load_ac(args)
        columns = (
            [c.strip() for c in args.columns.split(",") if c.strip()]
            if getattr(args, "columns", None)
            else None
        )
        controls = (
            [c.strip() for c in args.controls.split(",") if c.strip()]
            if getattr(args, "controls", None)
            else None
        )
        result = ac.correlate(
            x=args.x,
            y=args.y,
            columns=columns,
            method=args.method,
            controls=controls,
            bootstrap_n=int(args.bootstrap_n or 0),
        )
        if args.fmt == "markdown" and hasattr(result, "to_markdown"):
            payload = result.to_markdown()
        elif hasattr(result, "to_json"):
            payload = result.to_json()
        elif hasattr(result, "to_dict"):
            payload = json.dumps(result.to_dict(), indent=2, default=str)
        else:
            payload = json.dumps(result, indent=2, default=str)
        _emit(payload, args.out)
        return 0

    if args.command == "tabular-ml":
        ac = _load_ac(args)
        features = (
            [c.strip() for c in args.features.split(",") if c.strip()]
            if getattr(args, "features", None)
            else None
        )
        report = ac.tabular_ml(
            target=args.target,
            features=features,
            task=args.task,
            group_column=getattr(args, "group_column", None),
            time_column=getattr(args, "time_column", None),
            calibrate=bool(args.calibrate),
        )
        if args.fmt == "markdown" and hasattr(report, "to_markdown"):
            payload = report.to_markdown()
        elif hasattr(report, "to_json"):
            payload = report.to_json()
        else:
            payload = json.dumps(report.to_dict(), indent=2, default=str)
        _emit(payload, args.out)
        return 0

    if args.command == "autoviz":
        ac = _load_ac(args)
        if args.discover:
            ac.mine()
            ac.impute()
            ac.discover(use_iv=False, qc="off")
        report = ac.autoviz(use_slm=_cli_use_slm(args))
        if args.fmt == "markdown" and hasattr(report, "to_markdown"):
            payload = report.to_markdown()
        elif hasattr(report, "to_json"):
            payload = report.to_json()
        else:
            payload = json.dumps(report.to_dict(), indent=2, default=str)
        _emit(payload, args.out)
        return 0

    if args.command == "report-artifact":
        from autocausal.reporting import ReportEngine, ReportPolicy

        ac = _load_ac(args)
        if args.discover:
            ac.mine()
            ac.impute()
            ac.discover(use_iv=False, qc="off")
        policy = (
            ReportPolicy.exploratory()
            if args.profile == "exploratory"
            else ReportPolicy.production()
        )
        artifact = ReportEngine(
            use_slm=_cli_use_slm(args) is not False, policy=policy
        ).generate(
            source=ac,
            output=Path(args.output),
            format=args.format,
            title=args.title,
        )
        _emit(json.dumps(artifact.to_dict(), indent=2, default=str), None)
        print(f"Wrote {args.output}", file=sys.stderr)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
