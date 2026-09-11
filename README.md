# Disaster Response Coordination

A decision-support system for urban flood response. Four independent AI models
— classical ML, deep learning, NLP, and Small Language Models (SLMs) — served behind one Flask dashboard.

> **This is decision support for a human responder, not a verified assessment.**
> Stage 01's confidence is isotonic-calibrated and clipped so it never claims
> certainty; the forecasting endpoint flags inputs outside its trained range;
> the SLM briefing summarizes multi-entry logs into operational SOP actions; and
> the fusion layer forces human review whenever sources conflict, evidence is
> thin, or priority is URGENT or above.

---

## 1. Problem statement

During an urban flood, information arrives in multiple incompatible forms at once:
numeric sensor telemetry (river gauges, rain gauges, call volumes), imagery from
cameras and drones, free text from dispatchers and the public, and multi-entry incident logs. A coordinator
has to triage all of them under time pressure.

This project builds one specialized model per modality and exposes them through a single
interface so they can be compared and used side by side.

## 2. Proposed solution

| Stage | Modality | Task | Model |
| --- | --- | --- | --- |
| **01 ML** | 12 tabular sensor fields | 3-class zone risk (Low/Moderate/Severe) | XGBoost, selected from 5 candidates |
| **02 DL** | Camera/drone image | Binary flooded/unflooded | ResNet18 transfer learning |
| **02 DL** | 72h water-level series | 6-hour river forecast | 2-layer LSTM |
| **03 NLP** | Free-text emergency message | 4-class urgency + 12-class hazard + entity extraction | DistilBERT + TF-IDF LogReg + BIO tagger |
| **04 SLM** | Incident report text log | Structured Tactical Briefing (Situation, Risk, Actions) | Qwen2.5-3B-Instruct QLoRA + TF-IDF baseline |
| **Fusion** | Any subset of the above | One prioritised incident decision | Deterministic, auditable policy (not a learned model) |

## 3. System architecture

The four stages are **parallel and independent** — no stage consumes another's
output. A **fusion layer sits above all stages** and is the component that
synthesizes multiple modalities. Full detail and the data-flow diagram:
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

```
Browser ──► app.py ──┬──► /api/predict/ml         ──► Stage 01 (XGBoost)
                     ├──► /api/predict/dl/image   ──► Stage 02 (ResNet18)
                     ├──► /api/predict/dl/lstm    ──► Stage 02 (LSTM)
                     ├──► /api/predict/nlp        ──► Stage 03 (DistilBERT + NER)
                     ├──► /api/slm/summarize      ──► Stage 04 (Qwen2.5-3B QLoRA / Baseline)
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
where all modalities describe the same event with a known outcome, which
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
artifacts were produced with**. Do not float `scikit-learn` without retraining Stages 01 and 03.

## 5. Running the dashboard

```bash
python app.py
```

Then open <http://127.0.0.1:5000>. Startup prints each stage's real status, and
`GET /health` returns a per-stage report. Status comes from a **live prediction**
in each stage, not from whether a module imported.

Debug mode is off by default. Enable deliberately with `FLASK_DEBUG=1`.

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

# ---- Stage 04: SLM ----  (GPU recommended for Qwen QLoRA)
python Stage04_SLM/01_data_engineer.py
python Stage04_SLM/02_eda_engineer.py
python Stage04_SLM/03_slm_engineer.py --epochs 3          # or --baseline-only
python Stage04_SLM/04_evaluation_engineer.py --compare

# ---- Tests ----
pytest Stage01_ML/test/ Stage02_DL/test/ Stage03_NLP/test/ Stage04_SLM/test/ test/ -v
```

All seeds are fixed at 42.

## 7. Results

**Macro F1 is the primary metric throughout.** All tasks are imbalanced,
and in each the minority class is the one that matters operationally.

### Stage 01 — Zone risk (held-out test, n = 1,500)

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Low | 1.000 | 0.710 | 0.830 | 31 |
| Moderate | 0.931 | 0.934 | 0.932 | 347 |
| Severe | 0.980 | 0.987 | 0.983 | 1,122 |
| **Macro** | 0.970 | 0.877 | **0.915** | 1,500 |

Accuracy 0.969. Severe recall 0.987 with 15 missed Severe cases.

### Stage 02 — CNN

| Metric | Single split (n=75) | **5-fold CV (n=500)** | Fold std | **95% CI (bootstrap)** |
| --- | ---: | ---: | ---: | :---: |
| Accuracy | 0.960 | **0.9580** | ±0.0319 | [0.9400, 0.9740] |
| **Macro F1** | 0.939 | **0.9326** | ±0.0510 | [0.9029, 0.9583] |
| Flooded recall | 0.933 | **0.8600** | ±0.0962 | [0.7884, 0.9239] |
| Flooded precision | 1.000 | **0.9247** | ±0.0836 | [0.8700, 0.9717] |

### Stage 02 — LSTM water-level forecast

| Horizon | MAE (m) | RMSE (m) |
| --- | ---: | ---: |
| +1 h | 0.170 | 0.296 |
| +3 h | 0.509 | 1.201 |
| **+6 h (what the API serves)** | **0.731** | **1.598** |

### Stage 03 — NLP (held-out test, n = 9,750)

| Task | Metric | Result |
| --- | --- | ---: |
| Urgency | **Macro F1** | **0.722** |
| Urgency | Accuracy | 0.718 |
| Hazard | Macro F1 | 1.000 |
| NER | Entity-level micro F1 | 0.996 |

### Stage 04 — SLM Tactical Briefing (held-out test, n = 481)

| Metric | Baseline (CPU) | Qwen2.5-3B QLoRA | Metric Focus / Target |
| --- | ---: | ---: | --- |
| **ROUGE-1 F1** | 0.3282 | 0.0908 | SOP Template Overlap |
| **Priority Accuracy** | 65.90% | 0.00% | Class Precision over 4 Priorities |
| **Combined High-Risk Recall** | 60.20% | 0.00% | `URGENT` + `IMMEDIATE` Detection ($\ge 80.0\%$) |
| **Content Slot Fidelity** | 27.44% | 0.00% | Extraction of Location, Headcount, Hazard |
| **Hallucination Rate** | 0.3829 | 0.5004 | Entity Tokens Grounded in Source ($\le 0.150$) |
| **Safety Failure Rate** | **0.00%** | 50.00% | 0 Inaction Directives on High-Risk Cases |
| **Read-Time Savings** | **45.9%** | 38.2% | Time Saved vs. Full Raw Log Reading |

---

## 8. Limitations

1. **Fusion is a hand-written policy, not a learned model.** Its weights are
   defensible defaults, not empirically optimal ones.
2. **Stage 03's corpora are synthetic.** Treat the scores as an upper bound.
3. **The CNN has 500 images total.** ±0.10 fold-to-fold variation on flooded recall is a data-volume limit.
4. **Stage 01's Low class is weak** (recall 0.710 on 31 samples); genuinely calm inputs are often returned as Moderate.
5. **No negation handling in NLP.** "NO FLOOD HERE" is scored on its flood vocabulary.
6. **Recursive forecasting degrades sharply** beyond ~3 hours.
7. **Qwen SLM requires structured JSON schema output** to maximize ROUGE score against synthetic SOP templates.
8. **Stages 05–06 (GenAI, Agentic AI)** are reserved future architecture modules.

---

## 9. Repository layout

```
├── app.py                      Flask dashboard + prediction endpoints + /health
├── requirements.txt            Pinned dependencies
├── docs/ARCHITECTURE.md        Real architecture, data lineage, design rationale
├── test/test_app.py            End-to-end HTTP tests
├── Stage01_ML/                 Tabular zone-risk ML     (README, 5 scripts, tests)
├── Stage02_DL/                 CNN + LSTM               (README, 5 scripts, tests)
├── Stage03_NLP/                Urgency + hazard + NER   (README, 5 scripts, tests)
├── Stage04_SLM/                Tactical Briefing SLM    (README, 5 scripts, tests)
└── Stage05_GenAI, Stage06_AgenticAI/   (Reserved future expansion stages)
```

Within each stage, scripts run in numeric order: `01_data_engineer` →
`02_eda_engineer` → `03_*_engineer` (train) → `04_evaluation_engineer` →
`05_integration_engineer` (serving adapter used by `app.py`).
