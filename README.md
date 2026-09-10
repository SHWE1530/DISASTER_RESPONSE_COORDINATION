# Disaster Response Coordination

A decision-support system for urban flood response. Three independent AI models
— classical ML, deep learning, and NLP — served behind one Flask dashboard.

> **This is decision support for a human responder, not a verified assessment.**
> Stage 01's confidence is isotonic-calibrated and clipped so it never claims
> certainty; the forecasting endpoint flags inputs outside its trained range; and
> the fusion layer forces human review whenever sources conflict, evidence is
> thin, or priority is URGENT or above.

---

## 1. Problem statement

During an urban flood, information arrives in three incompatible forms at once:
numeric sensor telemetry (river gauges, rain gauges, call volumes), imagery from
cameras and drones, and free text from dispatchers and the public. A coordinator
has to triage all three under time pressure.

This project builds one model per modality and exposes them through a single
interface so they can be compared and used side by side.

## 2. Proposed solution

| Stage | Modality | Task | Model |
| --- | --- | --- | --- |
| **01 ML** | 12 tabular sensor fields | 3-class zone risk (Low/Moderate/Severe) | XGBoost, selected from 5 candidates |
| **02 DL** | Camera/drone image | Binary flooded/unflooded | ResNet18 transfer learning |
| **02 DL** | 72h water-level series | 6-hour river forecast | 2-layer LSTM |
| **03 NLP** | Free-text emergency message | 4-class urgency + 12-class hazard + entity extraction | DistilBERT + TF-IDF LogReg + BIO tagger |
| **Fusion** | Any subset of the above | One prioritised incident decision | Deterministic, auditable policy (not a learned model) |

## 3. System architecture

The three stages are **parallel and independent** — no stage consumes another's
output. A **fusion layer sits above all three** and is the only component that
sees more than one modality. Full detail and the honest data-flow diagram:
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

```
Browser ──► app.py ──┬──► /api/predict/ml         ──► Stage 01 (XGBoost)
                     ├──► /api/predict/dl/image   ──► Stage 02 (ResNet18)
                     ├──► /api/predict/dl/lstm    ──► Stage 02 (LSTM)
                     ├──► /api/predict/nlp        ──► Stage 03 (DistilBERT + NER)
                     │
                     └──► /api/assess  ──► fusion/decision_engine.py
                                            └─► calls whichever stages have evidence
                                                └─► ROUTINE / ELEVATED / URGENT / CRITICAL
                                                    + evidence provenance
                                                    + conflict report
                                                    + human-review flag
```

### The fusion layer

`POST /api/assess` accepts any subset of `{sensors, text, image_path,
water_levels}` and returns one prioritised decision. It is a **deterministic
policy, not a learned model** — training one would need a corpus of incidents
where all four modalities describe the same event with a known outcome, which
does not exist here. Every weight is a named constant in
`fusion/decision_engine.py`.

Properties worth knowing:

- **Escalations only ever raise priority.** Under-responding costs more than over-responding.
- **A human reporting CRITICAL sets an URGENT floor**, even against calmer sensors.
- **A flat forecast cannot lower a present-tense assessment** — it describes a different point in time.
- **An out-of-distribution forecast is suppressed entirely**, not down-weighted.
- **Conflicts are surfaced, never averaged away**, and marked "not auto-resolved".
- **No evidence never reads as safe** — it returns `insufficient_evidence`, not `ROUTINE`.
- **Human review is mandatory** on conflict, single-source, low confidence, or URGENT+.

## 4. Installation

Requires Python 3.11.

```bash
git clone <repo-url>
cd DISASTER_RESPONSE_COORDINATION

python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
source .venv/bin/activate        # Linux / macOS

pip install -r requirements.txt
```

Versions in `requirements.txt` are **pinned to the versions the committed model
artifacts were produced with**. This matters: the Stage 03 artifacts were once
pickled under scikit-learn 1.9.0 and failed at predict time under 1.6.0, which
took the entire NLP endpoint offline while the dashboard still reported it as
online. Do not float `scikit-learn` without retraining Stages 01 and 03.

## 5. Running the dashboard

```bash
python app.py
```

Then open <http://127.0.0.1:5000>. Startup prints each stage's real status, and
`GET /health` returns a per-stage report. Status comes from a **live prediction**
in each stage, not from whether a module imported.

Debug mode is off by default (it exposes the Werkzeug interactive debugger, which
is a remote code execution console). Enable deliberately with `FLASK_DEBUG=1`.

## 6. Reproducing the results

Each stage runs end to end from the committed raw data.

```bash
# ---- Stage 01: ML ----
python Stage01_ML/01_data_engineer.py       # clean; labels are PRESERVED, not regenerated
python Stage01_ML/02_eda_engineer.py        # EDA + eda_checked_dataset.csv
python Stage01_ML/03_ml_engineer.py         # benchmark 5 candidates, train, calibrate
python Stage01_ML/04_evaluation_engineer.py # held-out test evaluation

# ---- Stage 02: DL ----   (~80 min on CPU)
python Stage02_DL/03_dl_engineer.py
python Stage02_DL/04_evaluation_engineer.py

# ---- Stage 03: NLP ----  (~25 min on CPU)
python Stage03_NLP/01_data_engineer.py
python Stage03_NLP/03_nlp_engineer.py                     # or --skip-transformer for a fast run
python Stage03_NLP/04_evaluation_engineer.py

# ---- Tests ----
pytest Stage01_ML/test/ Stage02_DL/test/ Stage03_NLP/test/ test/ -v
```

All seeds are fixed at 42. Stage 02 writes split manifests
(`cnn_split_manifest.json`, `lstm_split_manifest.json`) recording the held-out
partition **by file path**, and the evaluation script replays them rather than
re-deriving the split from a seed — filesystem ordering is not a guarantee.

## 7. Results

**Macro F1 is the primary metric throughout.** All three tasks are imbalanced,
and in each the minority class is the one that matters operationally.

### Stage 01 — Zone risk (held-out test, n = 1,500)

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Low | 1.000 | 0.710 | 0.830 | 31 |
| Moderate | 0.931 | 0.934 | 0.932 | 347 |
| Severe | 0.980 | 0.987 | 0.983 | 1,122 |
| **Macro** | 0.970 | 0.877 | **0.915** | 1,500 |

Accuracy 0.969. Severe recall 0.987 with 15 missed Severe cases.

**Calibration (held-out test).** An isotonic calibrator is fitted on the
*validation* partition and serves the reported confidence:

| | Brier (mean) | ECE |
| --- | ---: | ---: |
| Raw model score | 0.0163 | 0.0120 |
| **Isotonic-calibrated (served)** | **0.0159** | **0.0042** |

The **class decision still comes from the raw model**. Using calibrated argmax
instead was measured: macro F1 rose 0.915 → 0.928 and Low recall 0.710 → 0.774,
but Severe recall fell 0.9866 → 0.9786 and missed Severe zones went **15 → 24**.
Model selection deliberately optimised Severe recall, so a post-hoc calibrator is
not allowed to overturn it for a better aggregate. Calibration fixes the
probability, not the boundary. Confidence is clipped to [0.01, 0.99] — isotonic
regression emits exactly 1.0, and no flood warning should display 100%.

Candidate benchmark (validation): Logistic Regression macro F1 0.911, LightGBM
0.868, **XGBoost 0.845 (selected)**, Stacking 0.857, Random Forest 0.673.
XGBoost was selected on Severe recall (0.984 vs LR's 0.979) — a ~8-sample
difference bought at a 6.6-point macro F1 cost. That trade is debatable and is
flagged as such rather than hidden.

### Stage 02 — CNN

**Quote the cross-validated figures, not the single split.** The held-out split
evaluates 75 images with only 15 flooded positives, which cannot support a
precise claim. 5-fold stratified CV scores all 500 images exactly once, each by a
model that never trained on it — 100 flooded positives in total.

| Metric | Single split (n=75) | **5-fold CV (n=500)** | Fold std | **95% CI (bootstrap)** |
| --- | ---: | ---: | ---: | :---: |
| Accuracy | 0.960 | **0.9580** | ±0.0319 | [0.9400, 0.9740] |
| **Macro F1** | 0.939 | **0.9326** | ±0.0510 | [0.9029, 0.9583] |
| Flooded recall | 0.933 | **0.8600** | ±0.0962 | [0.7884, 0.9239] |
| Flooded precision | 1.000 | **0.9247** | ±0.0836 | [0.8700, 0.9717] |
| Flooded ROC-AUC | 0.986 | 0.9792 | ±0.0212 | — |
| Flooded PR-AUC | 0.958 | 0.9509 | ±0.0433 | — |

The correction matters: the single split's **0.933 flooded recall sits above the
cross-validated 95% upper bound of 0.924**. It was a lucky partition. The honest
figure is **0.86 recall, ±0.10 across folds**. Per-fold recall ranged 0.75–1.00,
which is the real measure of how much any single number here should be trusted.

Reproduce with `python Stage02_DL/06_cross_validation.py`; full output in
`data/outputs/cnn_cross_validation.json`.

### Stage 02 — LSTM water-level forecast

| Horizon | MAE (m) | RMSE (m) |
| --- | ---: | ---: |
| +1 h | 0.170 | 0.296 |
| +3 h | 0.509 | 1.201 |
| **+6 h (what the API serves)** | **0.731** | **1.598** |

Error grows ~4.3× from one step to six. Only the 1-step figure used to be
published; the served horizon is now measured and returned to the caller.

### Stage 03 — NLP (held-out test, n = 9,750)

| Task | Metric | Result |
| --- | --- | ---: |
| Urgency | **Macro F1** | **0.722** |
| Urgency | Accuracy | 0.718 |
| Hazard | Macro F1 | 1.000 (see caveat) |
| Hazard | Keyword-lookup baseline | 0.987 |
| NER | Entity-level micro F1 | 0.996 (see caveat) |
| NER | Token accuracy | 0.999 |

Urgency backend selected on validation macro F1:

| Candidate | Val macro F1 | Test macro F1 |
| --- | ---: | ---: |
| TF-IDF + LogReg | 0.665 | 0.662 |
| **DistilBERT (selected)** | **0.733** | **0.722** |

Per-class urgency F1: CRITICAL 0.794 (precision 0.989, **recall 0.663**),
HIGH 0.773, LOW 0.694, MEDIUM 0.629. The model is conservative about declaring
CRITICAL — a dangerous-underestimation failure mode, and the most important
thing to fix next.

> **Why these numbers are much lower than earlier versions of this project.**
> Two label-leakage defects were found and fixed. The urgency label used to be
> written verbatim into the training text (a severity-specific phrase in
> **100.00%** of rows), and the hazard label was a keyword rule applied to the
> same text it labelled. The previously reported 96.3% urgency accuracy and
> 1.0000 hazard F1 measured string matching, not language understanding.
> **0.722 macro F1 on honest labels is a better result than 0.963 on leaked
> ones.** See [`Stage03_NLP/README.md`](Stage03_NLP/README.md).

## 8. Limitations

1. **Fusion is a hand-written policy, not a learned model.** Its weights are
   defensible defaults, not empirically optimal ones — see `docs/ARCHITECTURE.md`
   §3a for why learning them is not possible with the available data.
2. **Stage 03's corpora are synthetic.** Even with leakage removed, template text is far more regular than real messages — treat the scores as an upper bound. Hazard F1 of 1.000 and NER F1 of 0.996 are **near-ceiling by construction** (the templates name the hazard and fill entities from slots), not evidence of a hard problem solved.
3. **The CNN has only 500 images.** Cross-validation and bootstrap intervals now
   quantify that properly rather than hiding it, but ±0.10 fold-to-fold variation
   on flooded recall is a data-volume limit, not something better methodology can
   remove.
4. **Stage 01's Low class is weak** (recall 0.710 on 31 samples); genuinely calm inputs are often returned as Moderate.
5. **No negation handling in NLP.** "NO FLOOD HERE" is scored on its flood vocabulary.
6. **Confidence is uncalibrated** and labelled as such in the UI.
7. **Recursive forecasting degrades sharply** beyond ~3 hours.
8. **Demo-grade deployment**: single-process dev server, no authentication, no rate limiting.
9. **Stages 04–06 (SLM, GenAI, Agentic AI) are empty placeholders.**

## 9. Repository layout

```
├── app.py                      Flask dashboard + 4 prediction endpoints + /health
├── requirements.txt            Pinned dependencies
├── docs/ARCHITECTURE.md        Real architecture, data lineage, design rationale
├── test/test_app.py            End-to-end HTTP tests
├── Stage01_ML/                 Tabular zone-risk ML     (README, 5 scripts, tests)
├── Stage02_DL/                 CNN + LSTM               (README, 5 scripts, tests)
├── Stage03_NLP/                Urgency + hazard + NER   (README, 5 scripts, tests)
└── Stage04_SLM, Stage05_GenAI, Stage06_AgenticAI/   (placeholders, not implemented)
```

Within each stage, scripts run in numeric order: `01_data_engineer` →
`02_eda_engineer` → `03_*_engineer` (train) → `04_evaluation_engineer` →
`05_integration_engineer` (serving adapter used by `app.py`).
