"""CLI: python -m autocausal discover|mine|ping|guide|create|infer|tools|auto|public ..."""

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
    print(text)


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
    g.add_argument("--impute", choices=["auto", "median_mode", "knn"], default="auto")
    g.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    g.add_argument("-o", "--out", type=str, default=None)

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

    # auto
    a = sub.add_parser("auto", help="Full pipeline: join? mine impute discover guide ground")
    _add_source_args(a)
    a.add_argument("--text", type=str, default=None)
    a.add_argument("--slm", action="store_true")
    a.add_argument("--impute", choices=["auto", "median_mode", "knn"], default="auto")
    a.add_argument("--web-ground", action="store_true", dest="web_ground")
    a.add_argument("--no-second-pass", action="store_true")
    a.add_argument("--format", choices=["markdown", "json", "both"], default="markdown", dest="fmt")
    a.add_argument("-o", "--out", type=str, default=None)

    # public suite
    pub = sub.add_parser("public", help="Public / demo dataset suite")
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
        gres = ac.guide(text=args.text, use_slm=args.slm)
        if args.fmt == "json":
            text = gres.to_json()
        elif args.fmt == "both":
            text = gres.to_markdown() + "\n\n```json\n" + gres.to_json() + "\n```\n"
        else:
            text = gres.to_markdown()
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
            join=args.join,
            join_on=join_on,
            use_web_ground=args.web_ground,
            impute_method=args.impute,
            second_pass=not args.no_second_pass,
        )
        if args.fmt == "json":
            text = result.to_json()
        elif args.fmt == "both":
            text = result.to_markdown() + "\n\n```json\n" + result.to_json() + "\n```\n"
        else:
            text = result.to_markdown()
        _emit(text, args.out)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
