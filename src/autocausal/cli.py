"""CLI: python -m autocausal discover|mine|ping|guide|direct|guides|create|infer|tools|auto|public|physics|ml|nlp|behavioral|insight ...

Thin consumer of library modules — prefer importing ``autocausal.nlp`` /
``autocausal.behavioral`` / ``autocausal.insight`` directly in apps and notebooks.
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


def _load_ac(args: argparse.Namespace) -> AutoCausal:
    if getattr(args, "csv", None):
        ac = AutoCausal.from_csv(args.csv)
    elif getattr(args, "parquet", None):
        ac = AutoCausal.from_parquet(args.parquet)
    elif getattr(args, "db", None):
        if not args.table and not getattr(args, "query", None):
            raise SystemExit("--db requires --table or --query")
        ac = AutoCausal.from_sqlalchemy(
            args.db,
            table=args.table,
            query=getattr(args, "query", None),
            schema=getattr(args, "schema", None),
            limit=getattr(args, "limit", None),
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
        description="Auto-impute, mine, discover, guide, and ground causal edges.",
    )
    p.add_argument("--version", action="version", version=f"autocausal {__version__}")
    sub = p.add_subparsers(dest="command")

    # discover
    d = sub.add_parser("discover", help="Impute + discover causal edges")
    _add_source_args(d)
    d.add_argument("--impute", choices=["auto", "median_mode", "knn"], default="auto")
    d.add_argument("--alpha", type=float, default=0.05)
    d.add_argument("--min-corr", type=float, default=0.15, dest="min_corr")
    d.add_argument("--no-iv", action="store_true")
    d.add_argument("--guide", action="store_true", help="Run rule/SLM guide after discover")
    d.add_argument("--slm", action="store_true", help="Enable HuggingFace SLM guide")
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

    # guide
    g = sub.add_parser("guide", help="SLM/rule guide from mine+discover outputs")
    _add_source_args(g)
    g.add_argument("--text", type=str, default=None, help="User question / hint")
    g.add_argument("--slm", action="store_true", help="Use HuggingFace SLM (needs autocausal[slm])")
    g.add_argument(
        "--guides",
        type=str,
        default=None,
        help="Comma-separated backends: llmintent,retracement,kineteq_pivot,rule,huggingface",
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
    di.add_argument("--slm", action="store_true")
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
    cr.add_argument("--slm", action="store_true")
    cr.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    cr.add_argument("-o", "--out", type=str, default=None)

    # infer (SLM-aided inference narrative)
    inf = sub.add_parser("infer", help="Interpret discovery/IV results with caveats")
    _add_source_args(inf)
    inf.add_argument("--text", type=str, default=None)
    inf.add_argument("--slm", action="store_true")
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
    pl.add_argument("--slm", action="store_true")
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
    a.add_argument("--slm", action="store_true")
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
    ml_loop.add_argument("--slm", action="store_true", help="Enable HuggingFace SLM guide")
    ml_loop.add_argument(
        "--guides",
        type=str,
        default="rule",
        help="Comma-separated guides: rule,slm/huggingface,llmintent,kineteq_pivot",
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

    sub.add_parser("dialects", help="Print supported SQLAlchemy dialect matrix")
    sub.add_parser("slm-status", help="Show RuleBackend / HuggingFace SLM availability")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "dialects":
        print(json.dumps(DIALECT_MATRIX, indent=2))
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
            ac.guide(use_slm=args.slm)
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
        gres = ac.guide(text=args.text, use_slm=args.slm, backends=backends)
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
            use_slm=args.slm,
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
            cres = ac.create(text=args.text, use_slm=args.slm)
        else:
            cres = create_from_context(
                {"text": args.text or "", "columns": []},
                use_slm=args.slm,
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
        ires = ac.interpret(text=args.text, use_slm=args.slm)
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
                use_slm=args.slm,
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
                use_slm=bool(args.slm) or "huggingface" in guides,
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
            use_slm=args.slm,
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
