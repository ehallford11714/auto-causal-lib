"""Coverage for the expanded library / CLI help catalog."""

from __future__ import annotations

import json

import pytest


def test_library_help_all_covers_core_modules():
    from autocausal import library_help

    text = library_help(all=True, format="markdown")
    assert "AutoCausal library help" in text
    for name in (
        "autocausal.inference",
        "autocausal.research",
        "autocausal.reporting",
        "autocausal.correlation",
        "autocausal.automl",
        "autocausal.autoviz",
        "autocausal.production",
        "autocausal.help_catalog",
    ):
        assert name in text


def test_library_help_module_research():
    from autocausal.help_catalog import library_help

    text = library_help(module="research", format="markdown")
    assert "autocausal.research" in text
    assert "DeepResearchSuite" in text or "CrossMatchEngine" in text
    assert "SearchIntensity" in text or "intensity" in text.lower()


def test_library_help_api_lists_estimate():
    from autocausal.help_catalog import library_help

    text = library_help(api=True, format="markdown")
    assert "estimate" in text
    assert "discover" in text
    assert "deep_research" in text or "correlate" in text


def test_library_help_json_schema():
    from autocausal.help_catalog import build_help_catalog

    catalog = build_help_catalog(modules=["inference", "research"])
    payload = json.loads(catalog.to_json())
    assert payload["schema"] == "AutoCausalHelpCatalog.v1"
    assert payload["n_modules"] == 2
    assert payload["n_api_methods"] >= 20
    assert any(m["name"] == "inference" for m in payload["modules"])


def test_cli_help_command_and_root_epilog():
    from autocausal.cli import _build_parser, main

    parser = _build_parser()
    assert parser.epilog
    assert "help --all" in parser.epilog
    assert "research" in parser.epilog

    choices = set()
    for action in parser._actions:
        if getattr(action, "choices", None) and isinstance(action.choices, dict):
            choices.update(action.choices.keys())
    assert "help" in choices

    rc = main(["help", "--module", "engines", "--format", "table"])
    assert rc == 0


def test_root_help_runs():
    from autocausal.cli import main

    assert main([]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0

