# AutoNLP

AutoNLP builds on the existing `autocausal.nlp` tokenization, POS, lemma, and
sentiment fallbacks without changing their APIs.

```python
from autocausal.autonlp import AutoNLPSuite

report = AutoNLPSuite(
    df,
    mode="production",
    text_columns=["notes", "transcript"],
    target="outcome_label",
).run()

print(report.report())
guide_context = report.to_guide_context()
```

Profiles

Text detection and profiles include missingness, duplication, length
distribution, lightweight language hints, and aggregate PII/secret risk counts.
When a target is supplied, AutoNLP flags target-name and label-value leakage
risk. Profiles never include sample values.

Claims and roles

Connector-based extraction returns `CausalClaim` objects with treatment/outcome
phrases, modality, negation, uncertainty, evidence spans, and character
offsets. Treatment, outcome, instrument, and confounder candidates are exposed
as `RoleHypothesis` values. Guide context marks every NLP-derived role and claim
as a hypothesis.

Features

```python
from autocausal.autonlp import FoldSafeTextVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("text", FoldSafeTextVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("model", LogisticRegression(max_iter=1000)),
])
```

The vectorizer combines deterministic TF-IDF/ngrams with readability,
sentiment, modality, negation, uncertainty, and causal-language counts. Its
vocabulary is learned only in `fit`, so placing it inside a CV pipeline keeps
features fold-safe. `audit()` returns the fit-document count and a vocabulary
hash, not text.

`aggregate_text_features(...)` creates document/trace aggregates for panel
analysis. `TextRetriever` provides local sklearn TF-IDF/cosine search and omits
documents from results unless requested.

Privacy and optional adapters

PII and secret patterns include email, phone, SSN, payment-card-like values,
IP addresses, API-key assignments, JWTs, cloud keys, and private-key headers.
Production serialization redacts matches in evidence spans while preserving
original offsets.

Raw text is not sent to SLMs, MCP tools, embedding services, or vector stores by
default. External enrichment and vector-store adapters require
`allow_external_text=True`; an attached AutoCausal production policy may still
deny the request. Production enrichment sends redacted text.

Epistemic boundary and integration

Causal wording in text is not causal evidence. Negation, uncertainty, quoted
claims, reporting bias, and document timing all require review. The stable
entry point is `from autocausal.autonlp import AutoNLPSuite`. A fluent
`AutoCausal.autonlp()` adapter was deferred to avoid conflicts with concurrent
core-facade work; it can later attach `report.to_guide_context()` explicitly.
