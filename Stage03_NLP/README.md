# Stage 03 — Natural Language Processing

Free-text emergency message triage: given an unstructured report from a
dispatcher log or a social feed, predict **how urgent it is**, **what hazard it
describes**, and **extract the operational entities** a responder needs
(location, resource required, headcount).

---

## 1. Problem statement

Stage 01 consumes numeric sensor readings and Stage 02 consumes images and time
series. Neither can read the thing that actually arrives first in a real
incident: a person describing what they can see. Stage 03 turns that text into
structured fields that a dispatcher can sort and act on.

## 2. Tasks and label spaces

| Task | Type | Classes |
| --- | --- | --- |
| Urgency | 4-class classification | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` |
| Hazard | 12-class classification | Flood, Cyclone, Heavy Rainfall, Landslide, Earthquake, Lightning, Wildfire, Tsunami, Drought, Medical Emergency, Rescue Emergency, Road Blockage |
| Entities | Token-level BIO tagging | `LOCATION`, `RESOURCE`, `HEADCOUNT` |

## 3. Data — read this before quoting any metric

**Both corpora are synthetically generated.** That is a real limitation and it
bounds what the numbers mean.

| Corpus | Rows | Source |
| --- | --- | --- |
| Dispatcher log | 60,000 | Rendered from `data/raw/Dispatcher_Log_Master_60000_RAW.csv` |
| Social feeds | 5,000 | Generated from per-hazard templates over Indian place names |
| Safety SOP | 1,697 | Used for NER annotation only |

### Two label-leakage defects were found and fixed

**1. The urgency label used to be written verbatim into the text.** Every
dispatcher message ended with a severity-specific phrase:

| Severity | Phrase (old generator) | Appeared in |
| --- | --- | --- |
| LOW | `no major injuries reported` | 100.00% of LOW rows, 0% of others |
| MEDIUM | `minor injuries reported` | 100.00% of MEDIUM rows, 0% of others |
| HIGH | `some injuries reported` | 100.00% of HIGH rows, 0% of others |
| CRITICAL | `urgent evacuation needed` | 100.00% of CRITICAL rows, 0% of others |

A TF-IDF trigram model recovers the label by string match. The ~96% urgency
accuracy previously reported measured lookup, not language understanding.

*Fix:* impact language is now sampled from a **shared pool of ten clauses**
where severity only shifts the sampling weights. Every clause occurs at every
severity; the most class-diagnostic clause now reaches only ~52% concentration
in its dominant class. Urgency is correlated with the wording, not determined
by it — which is how real dispatch text behaves.

**2. The hazard label was a keyword rule applied to the text it labelled.**
`detect_hazard()` returned `Flood` if `"flood"` appeared in the string, and that
output *became* the training label. The classifier was trained to reproduce a
lookup table over the same text the lookup read, which is why hazard precision,
recall and F1 were all exactly `1.0000` on all twelve classes.

*Fix:* hazard is now the **generative parameter** that produced the text —
genuine ground truth recorded at creation time. `detect_hazard()` is retained
as a reported baseline (it agrees with ground truth ~80.6% of the time), so the
evaluation can state what the trained model adds over a plain lookup.

**3. Social-feed urgency was partly a coin flip.** Labels came from
`random.choice(...)` gated on four trigger words, making a large share of the
label pure noise uncorrelated with the text. Urgency is now sampled from a prior
attached to the **semantic content** of the template that generated the post
(trapped people skew CRITICAL, advisories skew LOW), so the noise sits around a
signal that actually exists in the sentence.

## 4. Models

**Urgency — two candidates, benchmarked and selected on validation macro F1:**

- **A: TF-IDF (1–2 grams, 50k features) + Logistic Regression**, class-balanced.
- **B: DistilBERT** (`distilbert-base-uncased`) fine-tuned for sequence
  classification.

Whichever wins on **validation** is written to `urgency_model_meta.json` and is
the backend that inference loads. This matters: an earlier revision fine-tuned a
transformer, saved it, and then never used it — the inference branch tested for
a tokenizer key that nothing ever set, so every prediction silently came from
the old TF-IDF model while the metrics file claimed a hardcoded `0.98`.

**Hazard:** TF-IDF (1–3 grams, 10k features) + Logistic Regression, reported
against the keyword baseline.

**Entities:** token-level BIO classifier — `DictVectorizer` over word shape,
prefix/suffix and ±1 context features, then class-balanced Logistic Regression —
plus a controlled regex/gazetteer fallback layer for location, resource and
headcount when the tagger returns nothing.

## 5. Metrics

**Macro F1 is the primary metric.** The urgency classes are imbalanced, and
`CRITICAL` is simultaneously the smallest class and the one whose errors are
most costly. Accuracy would let a model ignore it and still score well.

NER is reported at **entity level** (exact type *and* span match) as well as
token level. Token accuracy alone is misleading here: the tag distribution is
dominated by `O`, so a model predicting `O` everywhere already scores ~0.95+.
Both are published; entity F1 is the one to quote.

Current measured results: `data/outputs/nlp_evaluation_metrics.json` and
`data/outputs/evaluation/evaluation_report.md`.

## 6. Preprocessing

`clean_text_basic` (used for NER) strips URLs, `@mentions` and `#` while
preserving case and numeric structure — case matters for entity detection.
`clean_text_for_classification` additionally lowercases and removes non-
alphanumerics.

**Both urgency backends now receive identically cleaned text.** The transformer
branch previously received the raw string while TF-IDF received the cleaned one,
a train/inference skew that would have surfaced the moment the transformer was
wired up.

Known preprocessing limitation: lowercasing and stripping punctuation discards
emphasis (`HELP!!!` and `help` become identical) and all emoji. That is a
deliberate trade for TF-IDF sparsity, and it is a genuine weakness on real
social-media text.

## 7. Running it

```bash
# 1. Regenerate the corpora and BIO annotations
python Stage03_NLP/01_data_engineer.py

# 2. Train (benchmarks TF-IDF vs DistilBERT, selects on validation macro F1)
python Stage03_NLP/03_nlp_engineer.py

#    TF-IDF only -- much faster on CPU:
python Stage03_NLP/03_nlp_engineer.py --skip-transformer

# 3. Evaluate on the held-out test set + controlled robustness cases
python Stage03_NLP/04_evaluation_engineer.py

# 4. Tests
pytest Stage03_NLP/test/ -v
```

Training is **never** triggered automatically. `_get_artifacts()` raises if a
model is missing rather than training inside the caller: it previously kicked
off a full 63k-record retrain plus a DistilBERT fine-tune from inside a Flask
request handler, and when that run failed it truncated the deployed NER model to
2 bytes. Artifacts are now written atomically (temp file + `os.replace`), so a
failed save cannot destroy a working model.

## 8. Serving contract

```python
engine = NLPIntegrationEngine()
engine.analyze("Three people are trapped near the railway bridge, send a boat.")
```

```json
{
  "status": "ok",
  "urgency": "CRITICAL",
  "confidence": 0.71,
  "hazard_type": "Rescue Emergency",
  "hazard_confidence": 0.64,
  "location": "railway bridge",
  "resource_needed": ["rescue boat"],
  "headcount": 3,
  "entities": {"location": ["railway bridge"], "resource_needed": ["rescue boat"], "headcount": 3}
}
```

`health_check()` runs a real analysis and validates the returned label. It
previously returned `"healthy"` whenever the module merely imported, which is
how the dashboard advertised "Stage 03 API (NLP): Online" while failing every
single request.

## 9. Known limitations

1. **The corpus is synthetic.** Even with both leakage defects fixed, template
   text is far more regular than real emergency messages. Treat the scores as an
   upper bound on real-world performance.
2. **Hazard classification is near-ceiling by construction** — the templates
   name the hazard. The keyword baseline is reported precisely so this is visible.
3. **Negation and hypotheticals are not handled.** "NO FLOOD HERE" and "if
   flooding happens" carry flood vocabulary and are scored on it.
4. **Emphasis and emoji are discarded** by classification preprocessing.
5. **No entity linking** — an extracted location is a string, not a coordinate.
6. **Nothing consumes these predictions downstream.** Stage 03 does not feed
   Stage 01 or Stage 02; see `docs/ARCHITECTURE.md`.

## 10. A note on the transformer artifact

The fine-tuned DistilBERT is **~1 GB and is gitignored** (`.gitignore` excludes
`Stage03_NLP/data/models/transformer_urgency/`). A fresh clone therefore has
`urgency_model_meta.json` selecting the transformer but no weights on disk.

Rather than taking the stage offline, `_get_artifacts()` **degrades to the
TF-IDF classifier** (always persisted by training) and prints a warning naming
the reason. `analyze_text()` reports which backend actually served the
prediction in its `urgency_backend` field, and `health_check()` surfaces the
same. To restore the selected model, run training again:

```bash
python Stage03_NLP/03_nlp_engineer.py
```

Expect a measurable quality drop while on the fallback: test macro F1 0.662
(TF-IDF) versus 0.722 (DistilBERT).
