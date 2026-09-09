# System Architecture

## 1. What this system actually is

Three **independent** predictors served behind one Flask dashboard. They are not
a pipeline, and this document says so plainly because an earlier version of the
project diagram implied a chain (`ML -> DL -> NLP -> fusion -> decision`) that
does not exist in the code.

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

**There is no fusion layer.** No stage consumes another stage's output. Building
one is listed under Known Limitations rather than implied by the diagram.

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

**Why confidence is labelled as uncalibrated.** Stage 01 measures Brier score and
expected calibration error and publishes the reliability curve. The model is
overconfident in its middle bins, so the UI states that the number is a ranking
signal, not a probability of being correct.

## 7. Known limitations

1. **The three stages do not combine.** There is no joint decision.
2. **Datasets are synthetic or small.** The Stage 03 corpora are generated; the
   CNN has 500 images (75 test, 15 of them flooded), so its confidence interval
   is wide.
3. **Recursive forecasting degrades with horizon.** Measured per step and
   published; step 6 is materially worse than step 1.
4. **Stage 01's Low class is weak** (recall ~0.71 on 31 test samples).
5. **No authentication, single-process dev server.** Demo-grade deployment only.
6. **Stages 04-06 are empty placeholders.**
