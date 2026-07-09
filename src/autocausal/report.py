"""Markdown report rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autocausal.api import DiscoveryResult


def render_markdown_report(result: "DiscoveryResult") -> str:
    lines: list[str] = []
    lines.append("# AutoCausal discovery report")
    lines.append("")
    lines.append(f"**Method:** `{result.method}`")
    lines.append("")
    lines.append(
        "> Exploratory heuristics only — edges are candidate relationships, "
        "not identified causal effects."
    )
    lines.append("")

    if result.imputation is not None:
        imp = result.imputation
        lines.append("## Imputation")
        lines.append("")
        lines.append(
            f"- Strategy: `{imp.method}` — "
            f"{imp.total_missing_before} missing → {imp.total_missing_after} remaining"
        )
        if imp.columns:
            lines.append("- Columns:")
            for c in imp.columns:
                lines.append(
                    f"  - `{c.column}`: {c.strategy} "
                    f"(filled {c.missing_before}, value={c.fill_value!r})"
                )
        lines.append("")

    lines.append("## Column roles")
    lines.append("")
    for col, role in result.roles.items():
        rv = role.value if hasattr(role, "value") else role
        lines.append(f"- `{col}`: {rv}")
    lines.append("")

    lines.append("## Candidates")
    lines.append("")
    for kind, cols in result.candidates.items():
        joined = ", ".join(f"`{c}`" for c in cols) if cols else "—"
        lines.append(f"- **{kind}:** {joined}")
    lines.append("")

    lines.append("## Edges")
    lines.append("")
    if not result.edges:
        lines.append("_No edges above threshold._")
    else:
        lines.append("| source | target | type | score | confidence | p-value |")
        lines.append("|---|---|---|---:|---:|---:|")
        for e in result.edges:
            lines.append(
                f"| `{e['source']}` | `{e['target']}` | {e.get('type', '')} | "
                f"{e.get('score', '')} | {e.get('confidence', '')} | {e.get('pvalue', '')} |"
            )
    lines.append("")

    if result.notes:
        lines.append("## Notes")
        lines.append("")
        for n in result.notes:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("## Graph JSON summary")
    lines.append("")
    lines.append(
        f"- Nodes: {len(result.graph.get('nodes', []))} · "
        f"Edges: {len(result.graph.get('edges', []))}"
    )
    lines.append("")
    return "\n".join(lines)
