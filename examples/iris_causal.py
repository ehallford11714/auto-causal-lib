#!/usr/bin/env python3
"""Iris causal demo — library-first, offline.

Loads the bundled Fisher Iris CSV, mines associations, discovers exploratory
edges, and optionally runs a short insight loop.

Epistemic note: edges among sepal/petal measurements are *illustrative*
library output — not scientific claims that one flower trait causes another.

Usage (from repo root, after ``pip install -e .``)::

    python examples/iris_causal.py
    python examples/iris_causal.py --insight
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Iris offline causal demo")
    p.add_argument("--insight", action="store_true", help="Also run insight loop")
    p.add_argument("--min-corr", type=float, default=0.2, dest="min_corr")
    p.add_argument("--no-iv", action="store_true", default=True)
    args = p.parse_args(argv)

    from autocausal import AutoCausal
    from autocausal.datasets import get_dataset, load_dataset

    meta = get_dataset("iris")
    print(f"# {meta.name}")
    try:
        print(meta.epistemic_note)
    except UnicodeEncodeError:
        print(meta.epistemic_note.encode("ascii", errors="replace").decode("ascii"))
    print()

    df = load_dataset("iris", allow_network=False)
    print(f"rows={len(df)} cols={list(df.columns)}")

    ac = AutoCausal(df)
    ac.mine(min_score=0.1)
    result = ac.impute().discover(use_iv=False, min_abs_corr=args.min_corr)
    edges = list(result.edges or [])
    print(f"\n## Exploratory edges ({len(edges)})")
    for e in edges[:20]:
        if isinstance(e, dict):
            src = e.get("source") or e.get("from") or e.get("u")
            dst = e.get("target") or e.get("to") or e.get("v")
            score = e.get("score") or e.get("weight") or e.get("corr")
            print(f"  {src} -> {dst}  score={score}")
        else:
            print(f"  {e}")

    print("\n## Report (excerpt)")
    report = ac.report()
    excerpt = "\n".join(report.splitlines()[:40])
    try:
        print(excerpt)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write(excerpt.encode(enc, errors="replace"))
        sys.stdout.buffer.write(b"\n")

    if args.insight:
        from autocausal.insight import run_insight_loop

        insight = run_insight_loop(
            df,
            text=meta.epistemic_note,
            use_slm=False,
        )
        print("\n## Insight key edges")
        for e in (insight.key_edges or [])[:12]:
            print(f"  {e}")
        print(f"guide_backend={insight.guide_backend} slm={insight.slm_used}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
