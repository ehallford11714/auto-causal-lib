"""Thin CLI handlers for ``python -m autocausal insight ...``.

Kept separate so concurrent edits to ``autocausal.cli`` stay merge-friendly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def register_insight_parser(sub: argparse._SubParsersAction) -> None:
    """Attach ``insight`` subcommand group to the root CLI parser."""
    ins = sub.add_parser(
        "insight",
        help="Insight suite / SLM research loop (library: autocausal.insight)",
    )
    ins_sub = ins.add_subparsers(dest="insight_cmd")

    run_p = ins_sub.add_parser(
        "run",
        help="Single-pass: load/join → mine → impute → discover → guide → synthesize",
    )
    src = run_p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str, help="Path to CSV")
    src.add_argument("--parquet", type=str, help="Path to Parquet")
    src.add_argument("--db", type=str, help="SQLAlchemy URL")
    run_p.add_argument("--table", type=str, default=None)
    run_p.add_argument("--query", type=str, default=None)
    run_p.add_argument(
        "--join",
        type=str,
        default=None,
        help="Comma-separated public suite ids to left-join",
    )
    run_p.add_argument("--join-on", type=str, default=None, dest="join_on")
    run_p.add_argument("--text", type=str, default="", help="Focus question for guide/SLM")
    slm = run_p.add_mutually_exclusive_group()
    slm.add_argument(
        "--slm",
        dest="use_slm",
        action="store_true",
        default=None,
        help="Enable HuggingFace SLM (soft; falls back to rule)",
    )
    slm.add_argument(
        "--no-slm",
        dest="use_slm",
        action="store_false",
        help="Force rule narrator (offline)",
    )
    run_p.add_argument(
        "--guides",
        type=str,
        default=None,
        help="Comma-separated guide backends (optional DirectionPlan path)",
    )
    run_p.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        dest="fmt",
    )
    run_p.add_argument("-o", "--out", type=str, default=None, help="Write report.md / .json")

    loop_p = ins_sub.add_parser(
        "loop",
        help="Closed research loop: mine→guide/SLM→recommend→join/remine→rediscover",
    )
    src_l = loop_p.add_mutually_exclusive_group(required=True)
    src_l.add_argument("--csv", type=str)
    src_l.add_argument("--parquet", type=str)
    src_l.add_argument("--db", type=str)
    loop_p.add_argument("--table", type=str, default=None)
    loop_p.add_argument("--query", type=str, default=None)
    loop_p.add_argument("--text", type=str, default="")
    loop_p.add_argument("--rounds", type=int, default=3, dest="rounds")
    loop_p.add_argument(
        "--join-sources",
        type=str,
        default=None,
        dest="join_sources",
        help="Comma-separated public ids for iterative joins",
    )
    slm_l = loop_p.add_mutually_exclusive_group()
    slm_l.add_argument("--slm", dest="use_slm", action="store_true", default=None)
    slm_l.add_argument("--no-slm", dest="use_slm", action="store_false")
    loop_p.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        dest="fmt",
    )
    loop_p.add_argument("-o", "--out", type=str, default=None)

    demo_p = ins_sub.add_parser("demo", help="Offline synthetic IV demo → insight research loop")
    demo_p.add_argument("--rounds", type=int, default=2, dest="rounds")
    demo_p.add_argument("--slm", action="store_true", help="Try SLM (soft)")
    demo_p.add_argument("--no-slm", dest="slm", action="store_false")
    demo_p.set_defaults(slm=False)
    demo_p.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        dest="fmt",
    )
    demo_p.add_argument("-o", "--out", type=str, default=None)


def handle_insight(args: argparse.Namespace) -> int:
    """Dispatch ``insight run|loop|demo``; return process exit code."""
    from autocausal.insight import InsightSuite, demo_insight, run_insight_loop

    def _emit(text: str, out: Optional[str]) -> None:
        if out:
            Path(out).write_text(text, encoding="utf-8")
            print(f"Wrote {out}", file=sys.stderr)
        print(text)

    def _format(report: object, fmt: str) -> str:
        if fmt == "json":
            return report.to_json()  # type: ignore[attr-defined]
        if fmt == "both":
            return (
                report.to_markdown()  # type: ignore[attr-defined]
                + "\n\n```json\n"
                + report.to_json()  # type: ignore[attr-defined]
                + "\n```\n"
            )
        return report.to_markdown()  # type: ignore[attr-defined]

    def _use_slm() -> bool:
        v = getattr(args, "use_slm", None)
        if v is None:
            return bool(getattr(args, "slm", False))
        return bool(v)

    if args.insight_cmd == "demo":
        report = demo_insight(
            use_slm=_use_slm(),
            max_rounds=int(getattr(args, "rounds", 2) or 2),
        )
        out = getattr(args, "out", None)
        text = _format(report, getattr(args, "fmt", "markdown"))
        if out:
            report.write(out)
            print(f"Wrote {out}", file=sys.stderr)
            print(
                f"insight demo: rounds={len(report.round_history)} "
                f"edges={len(report.key_edges)} experiments={len(report.experiments_recommended)}",
                file=sys.stderr,
            )
        else:
            print(text)
        return 0

    if args.insight_cmd == "loop":
        if args.csv:
            source: object = args.csv
        elif args.parquet:
            source = args.parquet
        elif args.db:
            source = args.db
        else:
            print("Provide --csv, --parquet, or --db", file=sys.stderr)
            return 2
        join_sources = None
        if getattr(args, "join_sources", None):
            join_sources = [x.strip() for x in args.join_sources.split(",") if x.strip()]
        suite = InsightSuite(use_slm=_use_slm())
        report = suite.run_loop(
            source,
            max_rounds=int(getattr(args, "rounds", 3) or 3),
            join_sources=join_sources,
            text=getattr(args, "text", "") or "",
        )
        out = getattr(args, "out", None)
        if out:
            report.write(out)
            print(f"Wrote {out}", file=sys.stderr)
            print(
                f"insight loop: rounds={len(report.round_history)} "
                f"edges={len(report.key_edges)} "
                f"experiments={len(report.experiments_recommended)}",
                file=sys.stderr,
            )
        else:
            print(_format(report, getattr(args, "fmt", "markdown")))
        return 0

    if args.insight_cmd == "run":
        join = None
        if getattr(args, "join", None):
            join = [x.strip() for x in args.join.split(",") if x.strip()]
        join_on = None
        if getattr(args, "join_on", None):
            join_on = [x.strip() for x in args.join_on.split(",") if x.strip()]
        guides = None
        if getattr(args, "guides", None):
            guides = [x.strip() for x in args.guides.split(",") if x.strip()]

        if args.csv:
            source = args.csv
        elif args.parquet:
            source = args.parquet
        elif args.db:
            source = args.db
        else:
            print("Provide --csv, --parquet, or --db", file=sys.stderr)
            return 2

        report = run_insight_loop(
            source,
            text=getattr(args, "text", "") or "",
            use_slm=_use_slm(),
            join=join,
            join_on=join_on,
            guide_backends=guides,
            table=getattr(args, "table", None),
            query=getattr(args, "query", None),
        )
        out = getattr(args, "out", None)
        if out:
            report.write(out)
            print(f"Wrote {out}", file=sys.stderr)
            print(
                f"insight: {len(report.key_edges)} edges · backend={report.guide_backend} · "
                f"slm={report.slm_used}",
                file=sys.stderr,
            )
        else:
            print(_format(report, getattr(args, "fmt", "markdown")))
        return 0

    print("usage: python -m autocausal insight {run,loop,demo} ...", file=sys.stderr)
    return 2
