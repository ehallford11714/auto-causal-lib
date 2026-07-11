"""Markdown report rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from autocausal.production import EPISTEMIC_BANNER, MATURITY, is_synthetic_instrument

if TYPE_CHECKING:
    from autocausal.results import AutoResult, DiscoveryResult


def _partition_edges(edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    assoc: list[dict[str, Any]] = []
    iv_real: list[dict[str, Any]] = []
    iv_synth: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for e in edges or []:
        t = str(e.get("type") or "")
        z = e.get("instrument")
        is_iv = t == "iv_2sls" or (z is not None and "iv" in t)
        if is_iv:
            if (
                e.get("auto_instrument")
                or e.get("synthetic")
                or e.get("identification") == "none"
                or is_synthetic_instrument(z)
            ):
                iv_synth.append(e)
            else:
                iv_real.append(e)
        elif z is not None and is_synthetic_instrument(z):
            iv_synth.append(e)
        else:
            assoc.append(e)
    return {
        "associations": assoc,
        "iv_real": iv_real,
        "iv_synth": iv_synth,
        "other": other,
    }


def _edge_table(edges: list[dict[str, Any]], *, extra_cols: bool = False) -> list[str]:
    lines: list[str] = []
    if not edges:
        lines.append("_None._")
        return lines
    if extra_cols:
        lines.append(
            "| source | target | type | instrument | evidence | identification | score | confidence | p-value |"
        )
        lines.append("|---|---|---|---|---|---|---:|---:|---:|")
        for e in edges:
            lines.append(
                f"| `{e['source']}` | `{e['target']}` | {e.get('type', '')} | "
                f"`{e.get('instrument', '')}` | {e.get('evidence_grade', 'exploratory')} | "
                f"{e.get('identification', 'unverified')} | "
                f"{e.get('score', '')} | {e.get('confidence', '')} | {e.get('pvalue', '')} |"
            )
    else:
        lines.append("| source | target | type | evidence | stability | score | confidence | p-value |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for e in edges:
            lines.append(
                f"| `{e['source']}` | `{e['target']}` | {e.get('type', '')} | "
                f"{e.get('evidence_grade', 'exploratory')} | {e.get('stability', '')} | "
                f"{e.get('score', '')} | {e.get('confidence', '')} | {e.get('pvalue', '')} |"
            )
    return lines


def render_markdown_report(result: "DiscoveryResult") -> str:
    lines: list[str] = []
    lines.append("# AutoCausal discovery report")
    lines.append("")
    lines.append(EPISTEMIC_BANNER)
    lines.append("")
    lines.append(f"**Method:** `{result.method}`")
    lines.append(f"**Mode:** `{getattr(result, 'mode', 'exploratory')}`")
    if getattr(result, "run_id", ""):
        lines.append(f"**Run ID:** `{result.run_id}`")
    lines.append("")
    lines.append(
        f"_Maturity:_ heuristic discovery `{MATURITY.get('heuristic_discovery')}`; "
        f"auto_instrument `{MATURITY.get('auto_instrument')}`."
    )
    lines.append("")

    if result.imputation is not None:
        imp = result.imputation
        redact_values = (
            getattr(result, "mode", "exploratory") == "production"
            or bool((getattr(result, "policy", {}) or {}).get("redact_sample_values"))
        )
        lines.append("## Imputation")
        lines.append("")
        lines.append(
            f"- Strategy: `{imp.method}` — "
            f"{imp.total_missing_before} missing → {imp.total_missing_after} remaining"
        )
        if imp.columns:
            lines.append("- Columns:")
            for c in imp.columns:
                if redact_values:
                    lines.append(
                        f"  - `{c.column}`: {c.strategy} "
                        f"(filled {c.missing_before}; fill value redacted)"
                    )
                else:
                    lines.append(
                        f"  - `{c.column}`: {c.strategy} "
                        f"(filled {c.missing_before}, value={c.fill_value!r})"
                    )
        lines.append("")

    gates = list(getattr(result, "evidence_gates", None) or [])
    rejected = list(getattr(result, "rejected_edges", None) or [])
    lines.append("## Evidence gates")
    lines.append("")
    lines.append(
        f"- accepted edges: **{len(result.edges or [])}** · "
        f"rejected edges: **{len(rejected)}** · failed gates: **{sum(1 for gate in gates if not gate.get('ok'))}**"
    )
    required = (getattr(result, "policy", {}) or {}).get("required_evidence")
    if required:
        lines.append(f"- required evidence: `{required}`")
    for gate in gates:
        mark = "PASS" if gate.get("ok") else "FAIL"
        lines.append(f"- [{mark}] `{gate.get('id')}` — {gate.get('detail')}")
        if gate.get("recommendation"):
            lines.append(f"  - Escalation: {gate.get('recommendation')}")
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

    parts = _partition_edges(list(result.edges or []))
    rejected_parts = _partition_edges(rejected)

    lines.append("## Exploratory associations")
    lines.append("")
    lines.append(
        "_Candidate relationships from heuristic discovery — not identified effects._"
    )
    lines.append("")
    lines.extend(_edge_table(parts["associations"]))
    lines.append("")

    lines.append("## IV (only if real Z)")
    lines.append("")
    lines.append(
        "_2SLS edges with a user-/name-provided instrument. Still **unverified** "
        "identification — exclusion/relevance are not proven by this library._"
    )
    lines.append("")
    lines.extend(_edge_table(parts["iv_real"], extra_cols=True))
    lines.append("")

    lines.append("## Synthetic IV (demo only)")
    lines.append("")
    lines.append(
        "_`auto_instrument_z` / synthetic Z — **identification=none**. "
        "Do not cite as science; plumbing/demo only._"
    )
    lines.append("")
    lines.extend(
        _edge_table(
            list(parts["iv_synth"]) + list(rejected_parts["iv_synth"]),
            extra_cols=True,
        )
    )
    lines.append("")

    if rejected:
        lines.append("## Rejected by production gates")
        lines.append("")
        lines.append(
            "_Retained for audit only; these edges are not production-eligible._"
        )
        lines.append("")
        lines.extend(_edge_table(rejected, extra_cols=False))
        lines.append("")

    manifest = getattr(result, "manifest", None)
    if manifest is not None:
        manifest_dict = (
            manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
        )
        fingerprint = manifest_dict.get("data_fingerprint") or {}
        privacy = manifest_dict.get("privacy") or {}
        lines.append("## Reproducibility and privacy")
        lines.append("")
        lines.append(
            f"- package: `{manifest_dict.get('package_version')}` · "
            f"random_state: `{manifest_dict.get('random_state')}`"
        )
        lines.append(
            f"- data fingerprint: `{str(fingerprint.get('sha256') or '')[:16]}…` "
            f"({fingerprint.get('n_rows')}×{fingerprint.get('n_columns')}; no raw values)"
        )
        lines.append(f"- stage events: {len(manifest_dict.get('events') or [])}")
        if privacy.get("pii_columns"):
            lines.append(
                "- privacy warning: potential PII columns: "
                + ", ".join(f"`{c}`" for c in privacy.get("pii_columns")[:12])
            )
        if privacy.get("high_cardinality_columns"):
            lines.append(
                "- high-cardinality columns: "
                + ", ".join(
                    f"`{c}`"
                    for c in privacy.get("high_cardinality_columns")[:12]
                )
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
    lines.append(EPISTEMIC_BANNER)
    lines.append("")
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
