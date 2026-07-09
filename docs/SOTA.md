# SOTA context (brief)

`autocausal` ships **exploratory** heuristics. It is useful for triage and candidate graphs, not for claiming identified causal effects.

## Causal discovery (landscape)

| Family | Idea | Typical output | Caveat |
|--------|------|----------------|--------|
| **PC / FCI** | Conditional independence tests → prune undirected graph → orient v-structures | CPDAG / PAG | Needs faithfulness; CI tests fragile in finite samples |
| **GES / FGES** | Score-based search (BIC etc.) over DAGs / equivalence classes | CPDAG | Combinatorial; sensitive to score & sample size |
| **NOTEARS / GOLEM / DAG-GNN** | Continuous optimization with acyclicity constraints | Weighted adjacency | Often assumes linear SEM; hyperparameter-sensitive |
| **LiNGAM** | Non-Gaussian ICA-style identification | Directed edges | Needs non-Gaussian noise assumptions |

**What we implement:** a lightweight **PC-style stub** (pairwise + small conditioning sets via partial correlation / Fisher-z) plus **score-based orientation** (compare simple regression \(R^2\)). Optional **IV / 2SLS** edges when treatment, outcome, and instrument candidates exist (CausalIVSuite if importable, else NumPy 2SLS lite).

## Imputation (landscape)

| Method | Role here |
|--------|-----------|
| Median / mode | Default robust baseline; fully reported |
| KNN-lite | Optional distance-weighted fill on numeric columns |
| MICE / MissForest / deep imputers | Out of scope (keep the library focused) |

Missingness mechanisms (MCAR / MAR / MNAR) are **not** diagnosed. Imputation can itself induce spurious associations — treat post-impute graphs cautiously.

## Why auto-DAG is heuristic

1. **Observational equivalence** — many DAGs share the same independencies; data alone often cannot pick one.
2. **Unmeasured confounding** — latent variables break naive CI orientation (FCI-style PAGs are needed for honesty).
3. **Selection & measurement** — sampling bias and noisy proxies change the graph.
4. **Multiple testing** — many CI tests inflate false edges without correction.
5. **Encoding choices** — categoricals-as-codes and imputation alter dependence structure.

**Bottom line:** use `autocausal` to propose edges, roles, and IV candidates; validate with domain knowledge, experiments, or stronger identification strategies before acting.
