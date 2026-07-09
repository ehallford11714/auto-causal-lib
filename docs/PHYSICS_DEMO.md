# Physics Streamlit demo

Interactive demo for the **physics autocausal loop**: mine → impute → discover → analytic rollout → physical insight grounding → guide.

> **Caveat:** exploratory dynamics only. This is **not** true physics identification, system ID, or conservation-law discovery. Markets / affect glossaries are **analogy-labeled**. Energy / stability numbers are heuristic proxies on z-scored state.

## Install

```bash
cd research/AutoCausalLib
pip install -e ".[ui]"          # streamlit + plotly
# alias:
pip install -e ".[streamlit]"
```

Core physics (`PhysicsEngine`, `PhysicsCausalSuite`) does **not** require Streamlit.

## Launch

```bash
# recommended (port 8518 — avoids 8765 / 8800)
python -m autocausal physics ui --port 8518

# headless / custom bind
python -m autocausal physics ui --port 8518 --host 127.0.0.1 --headless

# direct streamlit
streamlit run src/autocausal/apps/physics_streamlit.py --server.port 8518
```

Open http://127.0.0.1:8518

## Demo controls

| Control | Effect |
|---------|--------|
| Sample / CSV upload | Oscillator, KPI panel, markets, or affect synthetic frames; or your CSV |
| Horizon | Rollout steps (1–30) |
| Dynamics system | `damped_oscillator` / `drift_diffusion` / `linear_ode` |
| Grounding domain | `mechanics-lite` / `markets-as-dynamics` / `affect-as-dynamics` / `auto` |
| Text hint | Passed to rule (or optional SLM) guide |
| Second-pass | Guide focus columns → rediscover + rerollout |
| Min \|corr\| | Discovery edge threshold |

## What you see

1. **Trajectory** — line plots of z-scored state variables over the horizon (+ prediction bands stub)
2. **Energy / stability** — KE / PE / total energy proxies and a simple regime note
3. **Causal edges** — discovery edge table
4. **Physical insights** — glossary-linked mechanisms (literal vs analogy) + markdown
5. **Full markdown / JSON** — `PhysicsLoopResult` report

## Programmatic equivalent

```python
from autocausal.physics import PhysicsCausalSuite
from autocausal.apps.samples import load_demo_frame

df = load_demo_frame("oscillator")
result = PhysicsCausalSuite.from_dataframe(df).loop(
    horizon=5,
    text="what drives outcome?",
    domain="mechanics-lite",
)
print(result.to_markdown())
```

CLI without UI:

```bash
python -m autocausal physics loop --csv data.csv --horizon 5 --text "what drives outcome?"
python -m autocausal physics rollout --csv data.csv --horizon 5
```

## Soft NextFrameSeq NPE

If `nextframeseq.physics.npe.NeuralPhysicsEngine` is importable, the engine notes `nfs_available` on the backend string. Tabular demo rollouts still use the local numpy analytic engine (graph NPE needs morpheme graphs).

## Related

- [SOTA physics world-model notes](SOTA_PHYSICS_WORLD_MODEL_AUTOCAUSAL.md)
- Package: `autocausal.physics`, `autocausal.apps.physics_streamlit`
