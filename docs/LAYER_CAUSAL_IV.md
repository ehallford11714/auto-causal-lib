# Layer motifs ↔ IV (IntentIsolates bridge)

Soft adapter: `autocausal.isolates_bridge` → `intentisolates.causal`.

Primary docs and implementation: **IntentIsolates**
[`docs/LAYER_CAUSAL_IV.md`](https://github.com/ehallford11714/intent-isolates/blob/main/docs/LAYER_CAUSAL_IV.md).

## CLI

```bash
pip install intentisolates
python -m autocausal isolates-causal --text "I want X; I feel Y; I will do Z" --outcome-hint decision
```

Prefer the primary CLI when working from isolates:

```bash
python -m intentisolates causal --text "..." --outcome-hint decision
```

## API

```python
from autocausal.isolates_bridge import run_isolates_causal

result = run_isolates_causal("...", outcome_hint="decision")
print(result.to_markdown())
```

## Indication vs causation

- **Indication:** Pearson association of layer motif/isolate features with Y.
- **Causation:** IV / 2SLS with lower-layer Z instrumenting mid/late X → Y
  (`causaliv` when installed, else AutoCausal numpy 2SLS lite, else Wald).

See IntentIsolates docs for epistemic caveats (indication ≠ causation; IV assumptions;
bootstrap rows from a single text are exploratory).
