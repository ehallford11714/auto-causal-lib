"""
Physics autocausal Streamlit demo.

Launch::

    python -m autocausal physics ui --port 8518
    # or: streamlit run src/autocausal/apps/physics_streamlit.py --server.port 8518

Requires: pip install -e ".[ui]"  (or ".[streamlit]")
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# Soft import — CLI checks before launch; direct `streamlit run` needs the extra.
try:
    import streamlit as st
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        'Streamlit is required for this demo. Install with: pip install -e ".[ui]"'
    ) from e

from autocausal.apps.samples import SampleKind, load_demo_frame
from autocausal.physics import PhysicsCausalSuite

CAVEAT = (
    "**Exploratory dynamics only** — not true physics identification, "
    "not conservation-law discovery, and not a substitute for domain modeling. "
    "Markets / affect domains are **analogy-labeled**. Energy and stability "
    "proxies are heuristic (½v² / ½kx² on z-scored state)."
)

DOMAIN_OPTIONS = {
    "auto (all glossaries)": "auto",
    "mechanics-lite": "mechanics-lite",
    "markets-as-dynamics": "markets-as-dynamics",
    "affect-as-dynamics": "affect-as-dynamics",
}

SYSTEM_OPTIONS = ["damped_oscillator", "drift_diffusion", "linear_ode"]

SAMPLE_LABELS: dict[str, SampleKind] = {
    "Oscillator (mechanics)": "oscillator",
    "KPI panel (generic)": "kpi_panel",
    "Markets analogy": "markets",
    "Affect analogy": "affect",
}


def _try_plotly():
    try:
        import plotly.express as px  # type: ignore

        return px
    except Exception:
        return None


def _trajectory_frame(result: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in result.trajectory.points:
        row: dict[str, Any] = {
            "t": p.t,
            "kinetic_energy": p.kinetic_energy,
            "potential_energy": p.potential_energy,
            "total_energy": p.kinetic_energy + p.potential_energy,
        }
        for name, val in zip(p.state.names, p.state.position):
            row[name] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _edges_frame(result: Any) -> pd.DataFrame:
    edges = (result.discovery or {}).get("edges") or []
    if not edges:
        return pd.DataFrame(columns=["source", "target", "type", "confidence", "score"])
    return pd.DataFrame(edges)


def _insights_frame(result: Any) -> pd.DataFrame:
    insights = result.physical_grounding.insights or []
    if not insights:
        return pd.DataFrame(
            columns=["source", "target", "mechanism", "domain", "analogy_label", "confidence"]
        )
    return pd.DataFrame([i.to_dict() for i in insights])


def _stability_note(traj_df: pd.DataFrame) -> str:
    if traj_df.empty or "total_energy" not in traj_df.columns:
        return "_No energy series._"
    e0 = float(traj_df["total_energy"].iloc[0])
    e1 = float(traj_df["total_energy"].iloc[-1])
    ke = float(traj_df["kinetic_energy"].iloc[-1])
    pe = float(traj_df["potential_energy"].iloc[-1])
    if e1 < 0.05:
        regime = "near-equilibrium (low total energy)"
    elif ke > 2 * pe:
        regime = "kinetic-dominated (transient / momentum-like)"
    elif pe > 2 * ke:
        regime = "potential-dominated (restoring / mean-reversion-like)"
    else:
        regime = "mixed energy"
    delta = e1 - e0
    return (
        f"**Energy proxy:** start={e0:.4f} → end={e1:.4f} (Δ={delta:+.4f}). "
        f"**Regime:** {regime}. "
        f"KE={ke:.4f}, PE={pe:.4f} at horizon."
    )


def _load_user_csv(upload) -> Optional[pd.DataFrame]:
    if upload is None:
        return None
    return pd.read_csv(upload)


def main() -> None:
    st.set_page_config(
        page_title="AutoCausal Physics Demo",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Physics autocausal demo")
    st.caption(
        "Mine → impute → discover → physics rollout → physical grounding "
        "· AutoCausalLib exploratory loop"
    )
    st.warning(CAVEAT)

    with st.sidebar:
        st.header("Data")
        source = st.radio(
            "Source",
            ["Bundled / synthetic sample", "Upload CSV"],
            index=0,
        )
        df: Optional[pd.DataFrame] = None
        if source == "Upload CSV":
            up = st.file_uploader("CSV file", type=["csv"])
            df = _load_user_csv(up)
        else:
            sample_label = st.selectbox("Sample", list(SAMPLE_LABELS.keys()), index=0)
            kind = SAMPLE_LABELS[sample_label]
            df = load_demo_frame(kind)

        st.header("Loop controls")
        horizon = st.slider("Horizon", min_value=1, max_value=30, value=5)
        system = st.selectbox("Dynamics system", SYSTEM_OPTIONS, index=0)
        domain_label = st.selectbox("Grounding domain", list(DOMAIN_OPTIONS.keys()), index=0)
        domain = DOMAIN_OPTIONS[domain_label]
        text = st.text_input(
            "Text hint",
            value="what drives outcome?",
            help="Passed to the rule/SLM guide after discovery.",
        )
        second_pass = st.checkbox("Second-pass (guide focus → rediscover)", value=True)
        use_slm = st.checkbox("Use SLM guide (needs autocausal[slm])", value=False)
        impute = st.selectbox("Impute method", ["auto", "median_mode", "knn"], index=0)
        min_corr = st.slider("Min |corr| for discovery", 0.0, 0.5, 0.1, 0.01)
        run = st.button("Run physics loop", type="primary", use_container_width=True)

    if df is None:
        st.info("Upload a CSV or keep the bundled sample selected.")
        return

    st.subheader("Input frame")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Numeric", int(df.select_dtypes("number").shape[1]))
    st.dataframe(df.head(20), use_container_width=True)

    if not run:
        st.info("Adjust controls in the sidebar, then click **Run physics loop**.")
        return

    with st.spinner("Running mine → impute → discover → rollout → ground → guide…"):
        suite = PhysicsCausalSuite.from_dataframe(df, system=system, prefer_nfs=True)
        result = suite.loop(
            horizon=horizon,
            text=text or None,
            domain=domain,
            system=system,
            use_slm=use_slm,
            second_pass=second_pass,
            impute_method=impute,
            min_abs_corr=min_corr,
        )

    st.success(
        f"Done · backend=`{result.backend}` · system=`{result.trajectory.system}` · "
        f"second_pass={result.second_pass}"
    )

    notes = result.notes or []
    if notes:
        with st.expander("Pipeline notes", expanded=False):
            for n in notes:
                st.write(f"- {n}")

    traj_df = _trajectory_frame(result)
    state_cols = [
        c
        for c in traj_df.columns
        if c not in ("t", "kinetic_energy", "potential_energy", "total_energy")
    ]

    tab_traj, tab_energy, tab_edges, tab_phys, tab_md = st.tabs(
        ["Trajectory", "Energy / stability", "Causal edges", "Physical insights", "Full markdown"]
    )

    with tab_traj:
        st.markdown("#### State over horizon (z-scored proxies)")
        if traj_df.empty:
            st.write("_Empty trajectory._")
        else:
            px = _try_plotly()
            plot_df = traj_df.melt(
                id_vars=["t"],
                value_vars=state_cols,
                var_name="variable",
                value_name="value",
            )
            if px is not None:
                fig = px.line(
                    plot_df,
                    x="t",
                    y="value",
                    color="variable",
                    markers=True,
                    title="Physics rollout trajectory",
                )
                fig.update_layout(height=420, legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(traj_df.set_index("t")[state_cols])
            st.dataframe(traj_df, use_container_width=True)
            if result.trajectory.predictions:
                st.markdown("##### Horizon predictions (± band stub)")
                st.json(result.trajectory.predictions)

    with tab_energy:
        st.markdown(_stability_note(traj_df))
        if not traj_df.empty:
            px = _try_plotly()
            ecols = ["kinetic_energy", "potential_energy", "total_energy"]
            if px is not None:
                edf = traj_df.melt(
                    id_vars=["t"],
                    value_vars=ecols,
                    var_name="proxy",
                    value_name="value",
                )
                fig = px.line(
                    edf,
                    x="t",
                    y="value",
                    color="proxy",
                    markers=True,
                    title="Energy proxies (heuristic)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(traj_df.set_index("t")[ecols])
        for note in result.trajectory.notes or []:
            st.caption(note)

    with tab_edges:
        edges_df = _edges_frame(result)
        st.dataframe(edges_df, use_container_width=True)
        if edges_df.empty:
            st.caption("No discovery edges — try lowering min |corr| or a richer sample.")

    with tab_phys:
        insights_df = _insights_frame(result)
        st.dataframe(insights_df, use_container_width=True)
        st.markdown(result.physical_grounding.to_markdown())
        if result.guide:
            st.markdown("#### Guide")
            st.json(result.guide)

    with tab_md:
        st.markdown(result.to_markdown())
        with st.expander("JSON"):
            st.code(result.to_json(), language="json")


if __name__ == "__main__":
    main()
