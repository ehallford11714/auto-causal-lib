# NLP tooling & behavioral science traces

**Library modules first.** `autocausal.nlp` and `autocausal.behavioral` are importable Python packages for apps, notebooks, and other products. The CLI (`python -m autocausal nlp|behavioral …`) is a thin consumer of the same APIs.

> **Epistemic honesty:** NLP role hints are linguistic cues only. Behavioral stimulus→response / habit→outcome edges are exploratory. Neither identifies causation.

## Library vs CLI

| Use case | Prefer |
|----------|--------|
| Embed in an app, notebook, or another library | `from autocausal.nlp import …` / `from autocausal.behavioral import …` |
| One-off terminal exploration | `python -m autocausal nlp …` / `behavioral …` |
| Facade on an existing `AutoCausal` frame | `AutoCausal.from_text_hints`, `apply_text_features`, `mine_behavioral_traces` |

---

## 1. NLP — Python API

### Install (optional)

```bash
pip install -e ".[nlp]"   # nltk + gensim; core works without it (regex/lexicon fallbacks)
```

Corpora are soft-optional. Without them, tokenize/POS/sentiment fall back to regex and a small lexicon.

```python
from autocausal.nlp import ensure_nltk_data, nltk_status

print(nltk_status().to_dict())          # offline inspection
# ensure_nltk_data()                    # soft-fail download of punkt, tagger, vader, wordnet
```

### Extract causal role hints from text

```python
from autocausal.nlp import (
    TextCausalHints,
    extract_causal_hints_from_text,
)

text = (
    "A randomized nudge treatment leads to higher compliance and retention, "
    "associated with baseline age as a confounder."
)

hints = extract_causal_hints_from_text(text)
# or: hints = TextCausalHints.extract(text)

print(hints.roles.to_dict())
# {'treatment': [...], 'outcome': [...], 'confounder': [...], 'instrument': [...]}

print(hints.modality_markers)   # e.g. ['leads_to', 'associated_with']
print(hints.sentiment)          # VADER or lexicon stub
print(hints.caveat)

# Feed guide / discover pipelines
ctx = hints.to_guide_context()
print(ctx["candidates"])
print(ctx["focus_columns"])
```

Top-level re-exports (lazy)::

```python
from autocausal import TextCausalHints, extract_causal_hints_from_text, NlpFeatureBuilder
```

### Text → feature columns for mining

```python
import pandas as pd
from autocausal.nlp import NlpFeatureBuilder, analyze, polarity, tokenize

builder = NlpFeatureBuilder(prefix="nlp_")
row = builder.row("Because the intervention increases sales")
print(row["nlp_polarity"], row["nlp_mod_because"], row["nlp_role_treatment"])

df = pd.DataFrame({"notes": ["treatment improves outcome", "random lottery assignment"]})
featured = builder.transform_frame(df, text_col="notes")
print(featured.columns.tolist()[:12])

# Low-level helpers also work without NLTK installed
print(tokenize("leads to higher revenue"))
print(analyze("policy affects churn").backend)
print(polarity("great improvement").compound)
```

### End-to-end: text → hints → features → discover

```python
import pandas as pd
from autocausal import AutoCausal
from autocausal.nlp import NlpFeatureBuilder, extract_causal_hints_from_text

# 1) Linguistic hints (for guides / focus)
hints = extract_causal_hints_from_text(
    "Does the marketing treatment cause revenue, controlling for season?"
)

# 2) Tabular text features joined into a panel
panel = pd.DataFrame({
    "notes": [
        "treatment campaign leads to sales",
        "no intervention; baseline confounder season",
        "randomized assignment associated with revenue",
    ],
    "y": [1.2, 0.4, 1.5],
    "spend": [10, 2, 12],
})
featured = NlpFeatureBuilder().transform_frame(panel, "notes")

# 3) Mine / discover on numeric + NLP columns
ac = AutoCausal.from_dataframe(featured, source="nlp_demo")
ac.mine()
# Focus on continuous cols; NLP flags are exploratory covariates
result = ac.discover()
print(result.to_markdown())

# Facade helpers
ac2 = AutoCausal.from_text_hints(hints.text)
print(ac2.nlp_hints.roles.to_dict())
```

---

## 2. Behavioral traces — Python API

### Trace schema

Each event: `subject_id`, `timestamp`, `action` (stimulus), `response`, optional `context` covariates, `reward`, `outcome`, `trial`.

### Load demos / build a store

```python
from autocausal.behavioral import (
    BehavioralTraceStore,
    TraceEvent,
    list_demos,
    load_demo,
    mine_behavioral_traces,
)

print(list_demos())
# habit_loop | nudge_ab | reinforcement_schedule

store = BehavioralTraceStore.from_demo("habit_loop")
panel = store.to_panel()                 # subject-level panel
events = store.events_frame(engineer=True)  # lag, habit_strength, compliance, exposure

print(panel.head())
print(store.mineable_columns())
```

### End-to-end: traces → panel → mine → edges

```python
from autocausal.behavioral import BehavioralTraceStore, mine_behavioral_traces

# One-shot pipeline
result = mine_behavioral_traces("habit_loop", discover=True)
print(result.report.to_markdown())
for edge in result.report.edges:
    print(edge.kind, edge.source, "→", edge.target, edge.score)

# Or via the store (same library objects the CLI uses)
store = BehavioralTraceStore.from_demo("nudge_ab")
result = store.mine(discover=True)
print(result.panel.shape)
print(result.report.caveat)

# Custom events
from autocausal.behavioral import TraceEvent, BehavioralTraceStore

store = BehavioralTraceStore.from_events(
    [
        TraceEvent("S1", "t0", "cue", "routine", reward=1.0, outcome=0.8, trial=0),
        TraceEvent("S1", "t1", "cue", "routine", reward=1.0, outcome=0.85, trial=1),
        TraceEvent("S2", "t0", "cue", "skip", reward=0.0, outcome=0.2, trial=0),
    ],
    name="tiny_habit",
)
print(store.to_panel())
```

Top-level re-exports::

```python
from autocausal import BehavioralTraceStore, mine_behavioral_traces

result = AutoCausal.mine_behavioral_traces("habit_loop", discover=True)
print(result.report.to_markdown())
```

### Soft IntentIsolates / ReasonTrace hooks

```python
from autocausal.behavioral import soft_isolates_annotate, soft_reason_trace_hook

print(soft_isolates_annotate("nudge improves habit"))  # soft-skip if missing
print(soft_reason_trace_hook([{"action": "cue"}]))
```

---

## 3. CLI (secondary)

```bash
python -m autocausal nlp extract --text "Randomized treatment leads to revenue"
python -m autocausal nlp features --csv notes.csv --text-col notes -o featured.csv
python -m autocausal nlp status
python -m autocausal behavioral list
python -m autocausal behavioral mine --demo habit_loop --discover
```

---

## Public API surface

### `autocausal.nlp`

| Symbol | Role |
|--------|------|
| `TextCausalHints` / `extract_causal_hints_from_text` | Text → roles / modality / guide context |
| `NlpFeatureBuilder` | Text → feature columns / DataFrame transform |
| `analyze`, `tokenize`, `pos_tag`, `lemmatize`, `polarity` | Soft-optional NLTK primitives |
| `ensure_nltk_data`, `nltk_status` | Offline status + soft download helper |
| `dataframe_text_features`, `texts_to_features` | Functional feature API |

### `autocausal.behavioral`

| Symbol | Role |
|--------|------|
| `BehavioralTraceStore` | Load/generate/store traces; `to_panel()`, `mine()` |
| `mine_behavioral_traces` | Traces → panel → mine/discover → `BehavioralReport` |
| `TraceEvent`, `TraceCollection` | Schema |
| `list_demos`, `load_demo`, `generate_demo` | Bundled offline demos |
| `engineer_trace_features`, `subject_panel` | Lag / habit / compliance / exposure |
| `BehavioralReport`, `BehavioralEdge` | Hypothesized edges + caveats |

---

## Optional dependency

```toml
# pyproject.toml
nlp = ["nltk>=3.8", "gensim>=4.3"]
```

Core `pip install -e .` never requires NLTK. Behavioral demos ship as offline CSV under `autocausal/data/behavioral/`.
