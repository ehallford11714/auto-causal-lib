"""CLI: python -m autocausal discover --csv data.csv"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autocausal import AutoCausal, __version__
from autocausal.ingest import DIALECT_MATRIX


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autocausal",
        description="Auto-impute and exploratory causal discovery for CSV / SQL tables.",
    )
    p.add_argument("--version", action="version", version=f"autocausal {__version__}")
    sub = p.add_subparsers(dest="command")

    d = sub.add_parser("discover", help="Impute + discover causal edges")
    src = d.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str, help="Path to CSV file")
    src.add_argument("--parquet", type=str, help="Path to Parquet file")
    src.add_argument("--db", type=str, help="SQLAlchemy engine URL")
    d.add_argument("--table", type=str, help="Table name (with --db)")
    d.add_argument("--query", type=str, help="SQL query (with --db)")
    d.add_argument("--schema", type=str, default=None)
    d.add_argument("--limit", type=int, default=None)
    d.add_argument(
        "--impute",
        choices=["auto", "median_mode", "knn"],
        default="auto",
        help="Imputation strategy",
    )
    d.add_argument("--alpha", type=float, default=0.05, help="CI test alpha")
    d.add_argument("--min-corr", type=float, default=0.15, dest="min_corr")
    d.add_argument("--no-iv", action="store_true", help="Skip IV / 2SLS pass")
    d.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        dest="fmt",
    )
    d.add_argument("-o", "--out", type=str, default=None, help="Write report to path")

    sub.add_parser("dialects", help="Print supported SQLAlchemy dialect matrix")

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

    if args.command == "discover":
        if args.db and not args.table and not args.query:
            parser.error("--db requires --table or --query")
        if args.csv:
            ac = AutoCausal.from_csv(args.csv)
        elif args.parquet:
            ac = AutoCausal.from_parquet(args.parquet)
        else:
            ac = AutoCausal.from_sqlalchemy(
                args.db,
                table=args.table,
                query=args.query,
                schema=args.schema,
                limit=args.limit,
            )
        result = ac.run(
            impute_method=args.impute,
            alpha=args.alpha,
            min_abs_corr=args.min_corr,
            use_iv=not args.no_iv,
        )
        if args.fmt == "json":
            text = result.to_json()
        elif args.fmt == "both":
            text = result.to_markdown() + "\n\n```json\n" + result.to_json() + "\n```\n"
        else:
            text = result.to_markdown()

        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"Wrote {args.out}", file=sys.stderr)
        print(text)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
