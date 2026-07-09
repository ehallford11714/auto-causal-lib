"""Soft-import test for isolates-causal bridge (skips without intentisolates)."""

from __future__ import annotations

import pytest


def test_isolates_bridge_import_or_skip():
    try:
        import intentisolates  # noqa: F401
    except ImportError:
        pytest.skip("intentisolates not installed")
    from autocausal.isolates_bridge import run_isolates_causal

    result = run_isolates_causal(
        "I want to ship. I feel blocked. I will decide using the checklist.",
        outcome_hint="decide",
        mock_iv=True,
        n_bootstrap=24,
        seed=2,
    )
    assert result.isolates
    md = result.to_markdown()
    assert "Indication" in md
    assert "Causation" in md


def test_cli_isolates_causal_help():
    from autocausal.cli import _build_parser

    p = _build_parser()
    # ensure subcommand registered
    args = p.parse_args(["isolates-causal", "--text", "hello", "--mock-iv"])
    assert args.command == "isolates-causal"
    assert args.text == "hello"
