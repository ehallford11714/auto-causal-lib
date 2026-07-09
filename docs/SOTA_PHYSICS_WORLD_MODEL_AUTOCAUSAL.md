# SOTA: Physical World Models + Autocausal Looping

Research brief for AutoCausalLib’s **physics predictive engine** and how it couples to NextFrameSeq’s neural physics engine (NPE), IV lift, and the broader Causal Fabric.

**Related:** [SOTA.md](./SOTA.md) · NextFrameSeq [WORLD_MODEL.md](../../NextFrameSeq/docs/WORLD_MODEL.md) · [HIERARCHICAL_PIXEL_CAUSAL.md](../../NextFrameSeq/docs/HIERARCHICAL_PIXEL_CAUSAL.md) · [HEATMAP_DRIFT_CAUSAL.md](../../NextFrameSeq/docs/HEATMAP_DRIFT_CAUSAL.md) · [GLOBAL_SYSTEM.md](../../docs/GLOBAL_SYSTEM.md)

---

## 1. Why physical world models (not free pixel / KPI AR)

Standard generative video and tabular forecasters learn **spurious correlations**. At long horizons they violate conservation, melt objects, or invent mean-reversion that is not identified. SOTA physical world modeling instead:

1. Compress observations into **state / latents** (morphemes, mesh nodes, PDE fields).
2. **Identify** structure (IV, causal discovery, physics priors).
3. **Roll out** under dynamical constraints (ODE/PDE/GNN message passing).
4. Decode to pixels / KPIs only at the leaves.

AutoCausal’s tabular loop mirrors that stack: mine KPIs → discover/IV → analytic (or NPE) rollout → ground insights → re-observe / second-pass guide.

---

## 2. SOTA map (physical world modeling)

### 2.1 Mesh / GNN simulators

| Line | Idea | Link |
|------|------|------|
| **MeshGraphNets** (Pfaff et al., ICLR 2021) | Message passing on meshes; learn continuum dynamics | [arXiv:2010.03409](https://arxiv.org/abs/2010.03409) |
| **Graph Network Simulators** (Sanchez-Gonzalez et al.) | Particle GNS for fluids/solids | [arXiv:2002.09405](https://arxiv.org/abs/2002.09405) |
| **Neural Operator / FNO** | Learn solution operators for PDEs | [arXiv:2010.08895](https://arxiv.org/abs/2010.08895) |

**Takeaway for us:** Edges = forces; nodes = state. NextFrameSeq NPE already does a lite GNN+PDE step on morpheme graphs; AutoCausal uses discovery edges as soft coupling weights on KPI state.

### 2.2 Neural ODEs & continuous dynamics

| Line | Idea | Link |
|------|------|------|
| **Neural ODEs** (Chen et al., NeurIPS 2018) | Continuous-depth residual nets as ODEs | [arXiv:1806.07366](https://arxiv.org/abs/1806.07366) |
| **Latent ODEs** | Stochastic latent dynamics for irregular time series | [arXiv:1907.03907](https://arxiv.org/abs/1907.03907) |

**Takeaway:** `linear_ode` / damped oscillator in `PhysicsEngine` are demo-scale cousins; roadmap = learned \(A\) / Neural ODE heads on mined features.

### 2.3 Latent video / agent world models

| Line | Idea | Link |
|------|------|------|
| **DreamerV3** (Hafner et al.) | RSSM latent dynamics + actor-critic | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) |
| **Genie** (Bruce et al., 2024) | Latent action world model from video | [DeepMind Genie](https://deepmind.google/discover/blog/genie-generative-interactive-environments/) |
| **V-JEPA / V-JEPA 2** | Predict in representation space, not pixels | [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) · Meta V-JEPA2 |

**Takeaway:** Predict **latents / physics state**, decode last — matches NFS WORLD_MODEL and AutoCausal KPI rollout (not free AR on columns).

### 2.4 Physics-informed & causal discovery

| Line | Idea | Link |
|------|------|------|
| **PINNs** (Raissi et al.) | Soft PDE residuals in loss | [arXiv:1711.10561](https://arxiv.org/abs/1711.10561) |
| **SINDy** (Brunton et al.) | Sparse identification of nonlinear dynamics | [PNAS 2016](https://www.pnas.org/doi/10.1073/pnas.1517384113) |
| **Deep IV** (Hartford et al., ICML 2017) | Neural 2SLS | [PMLR](https://proceedings.mlr.press/v70/hartford17a.html) |
| **Causal discovery** (PC / NOTEARS / …) | Structure from data | See [SOTA.md](./SOTA.md) |

**Takeaway:** Autocausal looping is where **identification** (IV / PC-lite) meets **dynamics** (rollout). Without IV, physics rollout amplifies confounded edges.

---

## 3. Autocausal looping (observe → … → re-observe)

```text
observe / load tabular (or frame KPIs)
    → mine KPIs & associations
    → impute
    → discover / IV-identify edges
    → physics rollout (analytic ODE or NFS NPE)
    → ground physical insights (+ domain glossaries)
    → guide / SLM (focus columns, instruments)
    → optional second pass (re-discover → re-rollout)
    → re-observe (new data / next window)
```

| Step | AutoCausalLib | NextFrameSeq |
|------|---------------|--------------|
| Observe | CSV / SQL / public suite | FrameKPI / morphemes |
| Mine | `mining.mine` | KPI miner / SAE hooks |
| Identify | `discover` + optional 2SLS | `CausalLiftEstimator` β_IV |
| Rollout | `PhysicsEngine.rollout` | `NeuralPhysicsEngine.rollout` |
| Ground | `ground_physical` + `ground_edges` | Scene / next-moment narration |
| Guide | `guide` / direction backends | Text / affect as instruments |
| Loop | `PhysicsCausalSuite.loop` | World model `predict` + IV loop |

This is the same fabric layer as GLOBAL_SYSTEM’s **world model** + **data plane**: Bridge can later move `Trajectory` + `InsightPack` artifacts across products.

---

## 4. Mapping to NextFrameSeq NPE + AutoCausal discover/guide

### Newtonian proxy (shared)

\[
x'' + c\,x' + k\,x = \sum_j \beta_{ij}\, x_j
\]

- NFS: \(\beta_{ij}\) from IV lift on morpheme graph; Euler step in `physics/npe.py`.
- AutoCausal: \(\beta_{ij}\) from discovery edge scores; same damping/stiffness defaults (`0.85` / `0.15`); tabular state from z-scored KPIs.

### Soft integration

```python
from autocausal.physics import try_nextframeseq_npe, PhysicsEngine

PhysicsEngine(prefer_nfs=True)  # notes if NeuralPhysicsEngine importable
# Tabular path always uses local numpy engine (NPE needs MorphemeGraph)
```

No hard dependency: if `nextframeseq` is absent, the suite stays offline-demo ready.

### Hierarchical / heatmap context (NFS docs)

- **HIERARCHICAL_PIXEL_CAUSAL:** L2 morphemes → L3 IV → L4 PDE — AutoCausal owns the **tabular analogue** of L2–L4 (KPI state → edges → rollout).
- **HEATMAP_DRIFT_CAUSAL:** latent drift of \((H,V)\) — future hook: treat permission/activation columns as state dims in `PhysicsEngine`.

---

## 5. What AutoCausal implements vs roadmap

### Implemented (v1)

| Piece | Status |
|-------|--------|
| `autocausal.physics.PhysicsEngine` | Analytic damped oscillator / drift-diffusion / linear ODE |
| Edge-coupled forces from discover | Yes |
| Uncertainty band stub (\(\propto\sqrt{t}\)) | Yes (not calibrated) |
| `ground_physical` + domain glossaries | mechanics-lite · markets-as-dynamics · affect-as-dynamics (analogy-labeled) |
| Merge with `autocausal.grounding` | Yes |
| `PhysicsCausalSuite.loop` / `.rollout` | Yes |
| CLI `physics loop` / `physics rollout` | Yes |
| `AutoCausal.physics_loop()` / `auto(..., physics=True)` | Yes |
| Soft NFS NPE import | Yes (availability note; graph path not required) |
| Offline tests | Yes |

### Roadmap

| Item | Notes |
|------|-------|
| Learned Neural ODE / SINDy on mined series | Fit \(A\) or sparse library from windows |
| Deep IV / EconML adapters into edge weights | Stronger \(\beta\) than score edges |
| True NFS graph bridge | Build `MorphemeGraph` from KPI columns when NFS present |
| Calibrated predictive intervals | Conformal / bootstrap bands |
| PINN residual checks | Soft constraint loss on rollout |
| MeshGraphNets / DiffTaichi | Only if vision/mesh domains land in AutoCausal |
| Bridge `Trajectory` contract | Align with GLOBAL_SYSTEM portable artifacts |

---

## 6. Caveats (read before claiming “physics”)

1. **Analogies are labeled.** Markets/affect glossaries use `analogy_label="analogy"` — not conservation laws.
2. **Exploratory edges ≠ identified effects.** Rollout amplifies whatever discover returns; use IV + domain ground.
3. **Uncertainty bands are stubs.** Do not use for risk limits without calibration.
4. **NPE parity is soft.** Full GNN+PDE on morphemes remains in NextFrameSeq; AutoCausal is the tabular twin.

---

## 7. Quick citations

- Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021. https://arxiv.org/abs/2010.03409  
- Chen et al., *Neural Ordinary Differential Equations*, NeurIPS 2018. https://arxiv.org/abs/1806.07366  
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3). https://arxiv.org/abs/2301.04104  
- Assran et al., *V-JEPA*. https://arxiv.org/abs/2404.08471  
- Raissi et al., *Physics-informed neural networks*. https://arxiv.org/abs/1711.10561  
- Hartford et al., *Deep IV*, ICML 2017. https://proceedings.mlr.press/v70/hartford17a.html  
- Brunton et al., *SINDy*, PNAS 2016. https://www.pnas.org/doi/10.1073/pnas.1517384113  
