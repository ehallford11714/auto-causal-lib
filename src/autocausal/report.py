"""Markdown report rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autocausal.results import AutoResult, DiscoveryResult


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

    if result.guide:
        lines.append("## Guide")
        lines.append("")
        lines.append(f"- Backend: `{result.guide.get('backend', '')}`")
        focus = result.guide.get("focus_columns") or []
        if focus:
            lines.append("- Focus: " + ", ".join(f"`{c}`" for c in focus[:12]))
        lines.append("")

    if result.grounding:
        lines.append("## Grounding")
        lines.append("")
        claims = result.grounding.get("claims") or []
        for c in claims[:15]:
            lines.append(
                f"- `{c.get('source')}` → `{c.get('target')}`: "
                f"**{c.get('label')}** (conf={c.get('confidence')})"
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


def render_auto_markdown(result: "AutoResult") -> str:
    lines: list[str] = ["# AutoCausal auto report", ""]
    lines.append(f"**Source:** `{result.source}`")
    lines.append("")
    if result.ping:
        lines.append(
            f"- Ping: ok={result.ping.get('ok')} "
            f"latency_ms={result.ping.get('latency_ms')} "
            f"url={result.ping.get('url_safe')}"
        )
        lines.append("")
    if result.join_log:
        lines.append("## Public joins")
        lines.append("")
        for j in result.join_log:
            lines.append(f"- `{j}`")
        lines.append("")
    if result.mining:
        lines.append("## Mining summary")
        lines.append("")
        lines.append(f"- Associations: {len(result.mining.get('associations') or [])}")
        lines.append(f"- KPIs: {result.mining.get('kpis')}")
        lines.append("")
    lines.append(render_markdown_report(result.discovery))
    if result.guide and not result.discovery.guide:
        lines.append("## Guide (auto)")
        lines.append("")
        lines.append(f"- Backend: `{result.guide.get('backend')}`")
        lines.append("")
    if result.direction_plan:
        lines.append("## Direction plan")
        lines.append("")
        lines.append(
            f"- Backends: {', '.join(result.direction_plan.get('backends') or [])}"
        )
        focus = result.direction_plan.get("focus_columns") or []
        if focus:
            lines.append(f"- Focus: {', '.join(f'`{c}`' for c in focus[:12])}")
        z = result.direction_plan.get("candidate_z") or []
        if z:
            lines.append(f"- Candidate Z: {', '.join(f'`{c}`' for c in z[:8])}")
        unavail = result.direction_plan.get("unavailable") or []
        if unavail:
            lines.append(f"- Soft-unavailable: {', '.join(f'`{u}`' for u in unavail)}")
        lines.append("")
    if getattr(result, "physics", None):
        phys = result.physics or {}
        lines.append("## Physics loop")
        lines.append("")
        lines.append(
            f"- Backend: `{phys.get('backend')}` · horizon={phys.get('horizon')} · "
            f"second_pass={phys.get('second_pass')}"
        )
        traj = phys.get("trajectory") or {}
        lines.append(
            f"- Trajectory: system=`{traj.get('system')}` · "
            f"steps={len(traj.get('points') or [])}"
        )
        pg = phys.get("physical_grounding") or {}
        lines.append(f"- Physical insights: {len(pg.get('insights') or [])}")
        lines.append("")
    if result.notes:
        lines.append("## Pipeline notes")
        lines.append("")
        for n in result.notes:
            lines.append(f"- {n}")
        lines.append("")
    return "\n".join(lines)
