# System Architecture

## 1. What this system actually is

Three **independent** predictors, each answering a different question from a
different modality, plus a **fusion layer** that combines whichever of them have
evidence into a single prioritised decision.

The stages themselves are still parallel, not a pipeline — Stage 03 does not
consume Stage 01's output. The fusion layer sits *above* all three and is the
only component that sees more than one modality. This document is explicit about
that because an earlier project diagram implied a sequential chain
(`ML -> DL -> NLP -> decision`) that has never existed in the code.

Each stage answers a different question from a different input modality:

| Stage | Input | Question answered | Model |
| --- | --- | --- | --- |
| 01 ML | 12 numeric/categorical sensor fields | How severe is the flood risk in this zone? | XGBoost (selected from 5 candidates) |
| 02 DL (CNN) | One camera/drone image | Is this scene flooded? | ResNet18 transfer learning |
| 02 DL (LSTM) | 72 hourly water-level readings | What is the river level for the next 6 hours? | 2-layer LSTM, 4 features |
| 03 NLP | One free-text emergency message | How urgent is it, what hazard, which entities? | TF-IDF/DistilBERT + hazard classifier + BIO NER |

## 2. Real request flow

```
                         Browser (single page, 4 tabs)
                                     |
                                  app.py
                                     |
      +---------------+--------------+--------------+----------------+
      |               |                             |                |
      v               v                             v                v
POST /api/         POST /api/                  POST /api/       POST /api/
predict/ml         predict/dl/image            predict/dl/lstm  predict/nlp
      |               |                             |                |
      v               v                             v                v
Stage01            Stage02                     Stage02          Stage03
Integration        Integration                 Integration      Integration
Engine             Engine                      Engine           Engine
      |               |                             |                |
      v               v                             v                v
ml_pipeline        disaster_cnn.pt             water_level_     urgency + hazard
.joblib            (cached, ResNet18)          lstm.pt +        + NER artifacts
(preprocessor      224x224 -> P(flooded)       scaler           (backend chosen
 + XGBoost)        vs saved threshold          (cached)          at train time)
      |               |                             |                |
      v               v                             v                v
 Low/Moderate      flooded /                   6-step recursive  urgency, hazard,
 /Severe           unflooded                   forecast +        location,
 + per-row         + confidence                OOD flag +        resource,
 top factors                                   measured MAE      headcount
      |               |                             |                |
      +---------------+--------------+--------------+----------------+
                                     |
                                     v
                          JSON response -> browser
```

The four endpoints above serve each stage in isolation. `POST /api/assess` sits
on top of them:

```
                   POST /api/assess   {sensors?, text?, image_path?, water_levels?}
                              |
                   fusion/decision_engine.py
                              |
        +---------------+-----+-------+----------------+
        v               v             v                v
  Stage01 risk    Stage03 urgency  Stage02 CNN    Stage02 LSTM
        |               |             |                |
        +---------------+------+------+----------------+
                               v
                 project onto one 0-3 severity scale
                 weight by each stage's own confidence
                 renormalise over AVAILABLE sources only
                               v
                 one-directional escalation rules
                 (they may only ever RAISE priority)
                               v
        ROUTINE / ELEVATED / URGENT / CRITICAL
        + per-source evidence and provenance
        + explicit conflict report (never averaged away)
        + human_review_required flag and reasons
        + recommended actions
```

## 3a. How fusion decides

**It is a deterministic policy, not a learned model.** Learning the fusion would
require a corpus of incidents where sensor readings, imagery, water levels and a
text report all describe the *same* event with a known outcome. No such joint
dataset exists here, and fabricating one would put a trained-looking number on
top of invented supervision — the exact defect Stage 03 had to have removed.
Every weight in `fusion/decision_engine.py` is a named constant a domain expert
can inspect and change.

**Evidence weights** (`EVIDENCE_WEIGHTS`): sensor risk 0.35, text urgency 0.30,
visual confirmation 0.20, forecast 0.15. Sensor telemetry is weighted highest
because it is the only signal validated against held-out ground-truth labels;
the text model is weighted almost as highly but is capped because it is trained
on synthetic text.

**Rules that matter:**

- *Escalations are one-directional.* Every rule may raise the priority; none may
  lower it. Under-responding to a real emergency costs more than over-responding
  to a false alarm. An escalation is only reported when it actually changed the
  outcome.
- *A human reporting CRITICAL sets an URGENT floor*, even when the sensors
  disagree. A person on the scene is not out-voted by a weighted mean.
- *Corroboration escalates.* Imagery confirming flooding **and** sensors reading
  high jumps to CRITICAL.
- *The forecast cannot lower a present-tense assessment.* It describes the
  future, so a flat projection is not evidence that conditions are calm now. It
  is excluded from the base score unless it warns of a rise, and it is excluded
  from conflict detection entirely — two different points in time cannot
  contradict each other.
- *An out-of-distribution forecast is suppressed outright*, not down-weighted.
- *Low confidence is floored, not discarded* (`MIN_CONFIDENCE_FLOOR = 0.25`), so
  a low-confidence CRITICAL report still carries weight.
- *Conflicts are surfaced, never averaged away.* A ≥2-level disagreement between
  present-tense sources is reported with both readings and marked
  "Not auto-resolved"; the higher assessment sets the floor and a human
  adjudicates.
- *Human review is mandatory* when sources conflict, when only one source was
  available, when any evidence is low-confidence, or when priority is URGENT or
  above.
- *No evidence never reads as safe.* Zero usable inputs returns
  `insufficient_evidence` with `human_review_required`, not `ROUTINE`.

## 3. Module loading

Stage entry points are named with a leading digit (`05_integration_engineer.py`),
which makes them invalid Python module names and therefore not importable with a
normal `import`. `app.py` loads them through
`importlib.util.spec_from_file_location`. This is a consequence of the stage
file-naming convention, and it is the root cause of several historical
fragilities (notably the Stage 03 NER artifact being pickled against whatever
synthetic module name the file happened to be loaded under — now avoided by
serialising the model's fitted components instead of the object itself).

## 4. Readiness reporting

Every integration engine exposes `health_check()`, and each one **runs a real
prediction** rather than only confirming that a file loaded:

- Stage 01 scores a fixed reference record.
- Stage 02 forecasts one step from a flat series at the training mean.
- Stage 03 analyses a fixed reference message and validates the returned label.

`GET /health` aggregates all three. The dashboard status line is derived from
these results, so a stage cannot report "Online" while failing every request.

## 5. Data lineage

```
Stage01_ML/data/processed/Master_Dataset.csv          (source, labelled)
        |  01_data_engineer.py   (dedupe, impute, IQR cap; labels PRESERVED)
        v
Master_Processed_Dataset.csv
        |  02_eda_engineer.py    (pinned input, no auto-fallback)
        v
data/outputs/pattern/eda_checked_dataset.csv
        |  03_ml_engineer.py     (chronological 70/15/15, OOF stacking benchmark)
        v
data/models/ml_pipeline.joblib + data/test/{X,y}_test.csv
        |  04_evaluation_engineer.py  (held-out test only)
        v
eval_final_report.json, evaluation_summary.json, calibration_report.json
```

```
Stage03_NLP/data/raw/Dispatcher_Log_Master_60000_RAW.csv
        |  01_data_engineer.py  (renders text; impact clause SAMPLED, not
        v                        switched on severity; hazard = generative param)
data/processed/*.csv + data/outputs/*_ner_bio.jsonl
        |  03_nlp_engineer.py   (stratified 70/15/15; urgency candidates
        v                        benchmarked and selected on VALIDATION macro F1)
data/models/* + urgency_model_meta.json + nlp_evaluation_metrics.json
        |  04_evaluation_engineer.py
        v
data/outputs/evaluation/*
```

## 6. Design decisions worth defending

**Why chronological splitting in Stage 01, stratified elsewhere.** Stage 01's
rows are a time series of zone observations, so a random split would let the
model see the future. Stage 02's images and Stage 03's messages have no
meaningful ordering, so stratification is the right choice there — it keeps rare
classes present in every partition.

**Why macro F1 is the primary metric everywhere.** All three tasks are
imbalanced, and in every case the *minority* class is the one that matters
operationally (Severe zones, flooded scenes, CRITICAL messages). Accuracy
rewards ignoring them.

**Why selection never touches the test set.** Stage 01 selects on validation,
Stage 02 selects the CNN architecture on validation, and Stage 03 selects the
urgency backend on validation. The test partition is scored once, for reporting.

**Why the LSTM reports an out-of-distribution flag.** The model was trained on a
series with mean 3.12 m and std 1.56 m. Given inputs several sigma outside that,
it extrapolates and produces physically implausible output. Rather than hide
this, `forecast_water_levels` returns `in_distribution` and a warning string.

**Why the confidence is calibrated but the decision is not.** Stage 01 fits an
isotonic calibrator on the *validation* partition and serves the calibrated
probability as `confidence` (raw score kept as `raw_confidence`). Measured on
the held-out test set, this cuts expected calibration error from 0.0120 to
0.0042.

The class decision still comes from the raw model. Taking argmax over the
calibrated probabilities instead was measured and it moved the operating point:
macro F1 rose 0.915 -> 0.928 and Low recall rose 0.710 -> 0.774, but Severe
recall fell 0.9866 -> 0.9786 and missed Severe zones rose from 15 to 24. Model
selection deliberately chose XGBoost on Severe recall, so letting a post-hoc
calibrator overturn that rule for a better aggregate metric would undo the
selection criterion. Calibration fixes the probability, not the boundary.
Confidence is additionally clipped to [0.01, 0.99] -- isotonic regression will
emit exactly 1.0, and no flood warning should display 100% certainty.

**Why the CNN is cross-validated as well as held-out tested.** The single split
evaluates 75 images with 15 flooded positives, which is too few to place a
meaningful interval around. `06_cross_validation.py` runs stratified k-fold so
every image is scored once by a model that never trained on it, and reports
fold-to-fold standard deviation plus bootstrap confidence intervals. The
held-out evaluation still measures the exact deployed checkpoint; the CV
measures the architecture and training procedure, which is the thing an interval
can legitimately be placed around.

## 7. Known limitations

1. **Fusion is a hand-written policy, not a learned model** -- for the reason in
   section 3a. Its weights are defensible defaults, not empirically optimal ones.
2. **Datasets are synthetic or small.** The Stage 03 corpora are generated; the
   CNN has 500 images (75 test, 15 of them flooded), so its confidence interval
   is wide.
3. **Recursive forecasting degrades with horizon.** Measured per step and
   published; step 6 is materially worse than step 1.
4. **Stage 01's Low class is weak** (recall ~0.71 on 31 test samples).
5. **No authentication, single-process dev server.** Demo-grade deployment only.
6. **Stages 04-06 are empty placeholders.**
