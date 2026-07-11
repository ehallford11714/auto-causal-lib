"""Tests for autocausal.doctor health check."""

from __future__ import annotations

from autocausal.doctor import doctor_report, format_doctor_markdown


def test_doctor_report_schema():
    r = doctor_report()
    assert r["schema"] == "AutoCausalDoctor.v1"
    assert "version" in r
    assert r["version"].startswith("0.14.")
    assert "engines" in r
    eng = r["engines"]
    assert "n" in eng
    assert "n_available" in eng
    assert "n_soft_skip" in eng
    assert "by_kind" in eng
    assert "optional_deps" in r
    assert "causallearn" in r["optional_deps"]
    assert "notes" in r
    assert isinstance(r["optional_deps"]["causallearn"], bool)


def test_format_doctor_markdown_nonempty():
    md = format_doctor_markdown()
    assert isinstance(md, str)
    assert len(md) > 40
    assert "AutoCausal doctor" in md
    assert "Engines" in md
