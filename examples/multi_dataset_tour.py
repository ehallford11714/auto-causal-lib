#!/usr/bin/env python3
"""Multi-dataset tour — mine + discover on bundled real examples (offline).

Usage::

    python examples/multi_dataset_tour.py
    python examples/multi_dataset_tour.py --ids iris,wine,titanic
"""

from __future__ import annotations

import argparse
import sys


def _edge_line(e: object) -> str:
    if isinstance(e, dict):
        src = e.get("source") or e.get("from") or e.get("u")
        dst = e.get("target") or e.get("to") or e.get("v")
        score = e.get("score") or e.get("weight") or e.get("corr")
        return f"{src} -> {dst}  score={score}"
    return str(e)


def run_one(dataset_id: str, *, min_corr: float) -> dict:
    from autocausal import AutoCausal
    from autocausal.datasets import get_dataset, load_dataset

    meta = get_dataset(dataset_id)
    df = load_dataset(dataset_id, allow_network=False)
    ac = AutoCausal(df)
    ac.mine(min_score=0.08)
    result = ac.impute().discover(use_iv=False, min_abs_corr=min_corr)
    edges = list(result.edges or [])
    return {
        "id": meta.id,
        "name": meta.name,
        "rows": len(df),
        "n_edges": len(edges),
        "edges": edges[:8],
        "epistemic_note": meta.epistemic_note,
        "suggested_outcome": meta.suggested_outcome,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tour bundled real datasets")
    p.add_argument(
        "--ids",
        type=str,
        default="iris,wine,titanic,diabetes,gapminder_subset,california_housing_sample",
        help="Comma-separated dataset ids",
    )
    p.add_argument("--min-corr", type=float, default=0.15, dest="min_corr")
    args = p.parse_args(argv)

    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    print("# AutoCausalLib multi-dataset tour (offline)\n")
    print(
        "Edges are exploratory library demos — not domain scientific claims.\n"
    )

    for did in ids:
        try:
            summary = run_one(did, min_corr=args.min_corr)
        except Exception as e:  # noqa: BLE001 — tour continues
            print(f"## {did} — FAILED: {type(e).__name__}: {e}\n")
            continue
        print(f"## {summary['name']} (`{summary['id']}`)")
        print(f"rows={summary['rows']}  edges={summary['n_edges']}  "
              f"outcome_hint={summary['suggested_outcome']}")
        print(f"_{summary['epistemic_note']}_")
        for e in summary["edges"]:
            print(f"  - {_edge_line(e)}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
