"""Environment / engine / optional-dep health check (library + CLI).

Soft-optional probes use ``importlib.util.find_spec`` only — never import
heavy packages. Availability ≠ causal identification.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from typing import Any

from autocausal.__version__ import __version__

__all__ = ["doctor_report", "format_doctor_markdown"]

# Soft-optional packages probed without importing
_OPTIONAL_DEPS: tuple[str, ...] = (
    "causallearn",
    "dowhy",
    "doubleml",
    "econml",
    "lingam",
    "castle",
    "mcp",
    "torch",
    "transformers",
    "nltk",
)


def _probe(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except Exception:
        return False


def doctor_report() -> dict[str, Any]:
    """Return a JSON-serializable health snapshot for CLI / MCP debugging."""
    from autocausal.engines import engine_status, list_engines

    status = engine_status()
    engines = list_engines()
    n_available = sum(1 for e in engines if e.available)
    n_soft_skip = sum(1 for e in engines if e.soft_skip)
    by_kind: dict[str, int] = {}
    for e in engines:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1

    optional = {name: _probe(name) for name in _OPTIONAL_DEPS}

    return {
        "schema": "AutoCausalDoctor.v1",
        "version": __version__,
        "python": {
            "version": sys.version.split()[0],
            "implementation": sys.implementation.name,
            "executable": sys.executable,
        },
        "engines": {
            "n": int(status.get("n") or len(engines)),
            "n_available": n_available,
            "n_soft_skip": n_soft_skip,
            "by_kind": by_kind,
            "status_schema": status.get("schema"),
        },
        "optional_deps": optional,
        "notes": [
            "Heavy deps are soft-optional; core numpy/pandas path always works.",
            "Availability ≠ causal identification.",
            "Probes use find_spec only — packages are not imported.",
        ],
    }


def format_doctor_markdown(report: dict[str, Any] | None = None) -> str:
    """Render :func:`doctor_report` as markdown."""
    r = report if report is not None else doctor_report()
    py = r.get("python") or {}
    eng = r.get("engines") or {}
    lines = [
        "# AutoCausal doctor",
        "",
        f"**Version:** `{r.get('version')}`",
        f"**Python:** `{py.get('version')}` ({py.get('implementation')})",
        "",
        "## Engines",
        "",
        f"- total: **{eng.get('n')}**",
        f"- available: **{eng.get('n_available')}**",
        f"- soft-skip: **{eng.get('n_soft_skip')}**",
        "",
        "### Counts by kind",
        "",
    ]
    for kind, n in sorted((eng.get("by_kind") or {}).items()):
        lines.append(f"- `{kind}`: {n}")
    lines += ["", "## Optional dependencies", ""]
    for name, ok in sorted((r.get("optional_deps") or {}).items()):
        mark = "yes" if ok else "no"
        lines.append(f"- `{name}`: {mark}")
    lines += ["", "## Notes", ""]
    for note in r.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)
