# AutoCausal tool suite (`autocausal.suite_tools`)

Shared registry of causal, NLP, KPI-mining, and validation adapters. Soft-imports optional packages; never hard-fails on missing extras.

## CLI

```bash
python -m autocausal tools list
python -m autocausal tools list --category nlp
python -m autocausal tools validate --csv data.csv --y y --d d --z z
python -m autocausal tools invoke text_z --text "lottery randomized rainfall"
python -m autocausal slm-status
```

## Catalog

### Causal (built-in)

| id | Status |
|----|--------|
| `builtin_2sls` | Always — numpy 2SLS |
| `builtin_did` | Always — simple 2×2 DiD |

### Causal (optional soft)

| id | Install |
|----|---------|
| `causaliv` | `pip install -e ../CausalIVSuite` |
| `dowhy` | `pip install dowhy` |
| `econml` | `pip install econml` |
| `causalml` | `pip install causalml` |

### NLP

| id | Notes |
|----|-------|
| `nltk` | Tokenize / stopwords / POS; regex fallback if missing |
| `gensim` | Similarity / LDA; bag-of-words cosine stub if missing |
| `spacy` | Optional NER; blank English if model missing |
| `text_z` | Built-in text → instrument/role tags |

```bash
pip install 'autocausal[nlp]'
```

### KPI mining

| id | Notes |
|----|-------|
| `autocausal_mining` | Built-in profiles + associations + KPI hints |
| `tabular_kpi_profile` | Built-in numeric candidates |
| `datamine` | Soft — DataMineLib when present |
| `vision_kpi` | Soft — VisionKPIMiner / EmotiveVision |

### Validation

`validate_pipeline(report, df=..., y=..., d=..., z=..., claims_text=...)` combines:

1. KPI coverage  
2. Weak-IV first-stage F (≥10 heuristic)  
3. Placebo stub (shuffle Z)  
4. NLP claim consistency (NLTK/regex + text_z)  
5. Edge presence  

## SLM

See `autocausal.slm`:

- **Creation:** `create_from_context` / `AutoCausal.create` / `python -m autocausal create`
- **Inference:** `infer_from_results` / `AutoCausal.interpret` / `python -m autocausal infer`
- **Guide:** `guide_pipeline` / `AutoCausal.guide`

`RuleBackend` always works. `HuggingFaceSLM` behind `pip install 'autocausal[slm]'` and `AUTOCAUSAL_SLM=1` or `--slm`. Default test model: `sshleifer/tiny-gpt2`. Import never requires torch.
