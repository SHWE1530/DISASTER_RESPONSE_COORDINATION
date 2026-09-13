# Disaster Response Coordination

A decision-support system for urban flood response, built as six AI stages behind one Flask dashboard:
classical ML, deep learning, NLP, small language models, generative AI, and an agentic AI coordinator.

> **This is decision support for a human responder, not a verified assessment.**
> Stage 01's confidence is isotonic-calibrated and clipped so it never claims
> certainty; the forecasting endpoint flags inputs outside its trained range;
> the SLM briefing summarizes multi-entry logs into operational SOP actions; the
> fusion layer forces human review whenever sources conflict, evidence is thin,
> or priority is URGENT or above; and the Stage 06 agent cannot commit a dispatch,
> evacuation or recall without a human decision.

---

## 1. Problem statement

During an urban flood, information arrives in multiple incompatible forms at once:
numeric sensor telemetry (river gauges, rain gauges, call volumes), imagery from
cameras and drones, free text from dispatchers and the public, and multi-entry
incident logs. A coordinator has to triage all of them under time pressure, across
many zones at once, with too few boats and ambulances.

Two further problems follow. Major disasters are rare, so the historical record
barely rehearses the situations that matter most. And reading every model output,
then splitting scarce resources between zones in seconds, is more than one person
can do reliably.

This project builds one specialized model per modality and fuses them into one
decision. It then stress-tests the pipeline with generated disaster scenarios, and
puts an agent on top that coordinates the whole incident with a human in control.

## 2. Proposed solution

| Stage | Input | Task | Approach |
| --- | --- | --- | --- |
| **01 ML** | 12 tabular sensor fields | 3-class zone risk (Low/Moderate/Severe) | XGBoost, selected from 5 candidates |
| **02 DL** | Camera/drone image | Binary flooded/unflooded | ResNet18 transfer learning |
| **02 DL** | 72h water-level series | 6-hour river forecast | 2-layer LSTM |
| **03 NLP** | Free-text emergency message | 4-class urgency + 12-class hazard + entity extraction | DistilBERT + TF-IDF LogReg + BIO tagger |
| **04 SLM** | Incident report text log | Structured tactical briefing (situation, risk, actions) | Qwen2.5-3B-Instruct QLoRA + TF-IDF baseline |
| **Fusion** | Any subset of the above | One prioritised incident decision | Deterministic, auditable policy (not a learned model) |
| **05 GenAI** | Seed conditions + prompt library | Synthetic multi-zone disaster scenarios that stress-test Stages 01–04 and fusion | 3.6M-parameter domain SLM (sequence generation), CVAE fallback, optional Qwen/Gemini LLM |
| **06 Agentic AI** | A multi-zone incident + finite inventory | Coordinate the response: assess every zone, reason through trade-offs, allocate scarce units, escalate to a human | Multi-agent coordinator (ReAct, plan-and-execute, tree of thoughts, debate, reflexion, memory) over 18 MCP tools |

Every stage follows the same five roles: data (or knowledge) engineer → EDA
(or workflow) engineer → model/agent engineer → evaluation engineer → integration engineer.

## 3. System architecture

Stages 01–04 are **parallel and independent**; no stage consumes another's output. A
**fusion layer sits above them** and synthesizes the modalities. **Stage 05** feeds
generated incidents into that pipeline to find where it breaks. **Stage 06** calls the
stages and fusion as tools to coordinate a whole incident. Full detail:
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

```
Browser ──► app.py ──┬──► /api/predict/ml         ──► Stage 01 (XGBoost)
                     ├──► /api/predict/dl/image   ──► Stage 02 (ResNet18)
                     ├──► /api/predict/dl/lstm    ──► Stage 02 (LSTM)
                     ├──► /api/predict/nlp        ──► Stage 03 (DistilBERT + NER)
                     ├──► /api/slm/summarize      ──► Stage 04 (Qwen2.5-3B QLoRA / Baseline)
                     │
                     ├──► /api/assess  ──► fusion/decision_engine.py
                     │                      └─► ROUTINE / ELEVATED / URGENT / CRITICAL
                     │                          + evidence, conflicts, human-review flag
                     │
                     ├──► /genai, /api/genai/*  ──► Stage 05: generate a scenario
                     │                               └─► stress-test it through Stages 01–04 + fusion
                     │
                     └──► /agent, /api/agent/*  ──► Stage 06: coordination agent
                                                     ├─► tools: Stages 01–04, fusion, SOP knowledge base
                                                     ├─► plan → perceive → rank → allocate → debate
                                                     │   → audit → act → reflect
                                                     ├─► human: approve / override / recall / emergency stop
                                                     └─► MCP server (POST /api/agent/mcp or --mcp stdio)
```

### The fusion layer

`POST /api/assess` accepts any subset of `{sensors, text, image_path, water_levels}`
and returns one prioritised decision. It is a **deterministic policy, not a learned
model**. Training one would need a corpus of incidents where all modalities
describe the same event with a known outcome, and none exists here. Every weight is a
named constant in `fusion/decision_engine.py`.

- **Escalations only ever raise priority.** Under-responding costs more than over-responding.
- **A human reporting CRITICAL sets an URGENT floor**, even against calmer sensors.
- **A flat forecast cannot lower a present-tense assessment**, because it describes a different point in time.
- **An out-of-distribution forecast is suppressed entirely**, not down-weighted.
- **Conflicts are surfaced, never averaged away**, and marked "not auto-resolved".
- **No evidence never reads as safe.** It returns `insufficient_evidence`, not `ROUTINE`.
- **Human review is mandatory** on conflict, single-source, low confidence, or URGENT+.

### Stage 05: generative stress testing

Stage 05 measures 17 blind spots in the historical data (11 absent, 2 rare), writes a
20-scenario prompt library plus a wildcard aimed at them, and generates each zone's
sensor row and dispatcher message **together in one autoregressive pass** with a
small language model trained on the project's own corpus. A scenario audit scores
realism, diversity coverage and overconfidence. Each scenario is then run through
the real Stage 01–04 models and fusion, and every zone is checked against its
ground-truth expectations. Details: [`Stage05_GenAI/README.md`](Stage05_GenAI/README.md).

### Stage 06: agentic coordination

Stage 06 is a multi-agent system: a Coordinator, a Tactical Dispatcher, a Resource
Allocator and a Safety Auditor, plus the human commander.

- **Plan and perceive.** It writes and checks its plan first, then reads each zone through the stage models.
- **Reason through trade-offs.** Between equally severe zones, emergency calls and people decide, and the reasoning is written out.
- **Allocate scarce units.** It compares allocation plans (tree of thoughts). Zones bid for contested units, and the Safety Auditor challenges the proposal before anything is reserved.
- **Cover shortfalls.** Unmet need is rerouted to secondary responders.
- **Keep a human in control.** Every URGENT+ dispatch, evacuation and low-confidence decision waits for a person. The dispatch and recall tools refuse to act without a one-time token that only a human decision issues. An emergency stop halts the run and recalls committed dispatches.

Details: [`Stage06_AgenticAI/README.md`](Stage06_AgenticAI/README.md).

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
artifacts were produced with**. Do not float `scikit-learn` without retraining
Stages 01 and 03. `google-genai` is only needed for the optional Gemini backends in
Stages 05 and 06.

## 5. Running the dashboard

```bash
python app.py
```

Then open <http://127.0.0.1:5000>. The sidebar has a tab for every stage, including
**GenAI Stress Test (Stage 05)** and **Agent Coordination (Stage 06)**. Startup prints
each stage's real status, and `GET /health` returns a per-stage report. Status comes
from a **live prediction** in each stage (for Stage 06, a model-free probe run of the
whole agent loop), not from whether a module imported.

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

# ---- Stage 05: GenAI ----  (SLM training ~3.5 min on GPU)
python Stage05_GenAI/01_data_engineer.py --skip-imagery
python Stage05_GenAI/02_eda_engineer.py
python Stage05_GenAI/03_genai_engineer.py                 # CVAE fallback generator
python Stage05_GenAI/03c_slm_sequence_generator.py --train
python Stage05_GenAI/04_evaluation_engineer.py            # scenario audit + stress test

# ---- Stage 06: Agentic AI ----  (~30 s after model load)
python Stage06_AgenticAI/01_knowledge_engineer.py         # SOP knowledge base + tool registry
python Stage06_AgenticAI/02_workflow_engineer.py          # state graph, task and tool mapping
python Stage06_AgenticAI/03_agent_engineer.py             # demo run on the wildcard incident
python Stage06_AgenticAI/04_evaluation_engineer.py        # suites, decision probes, ablations, faults
python Stage06_AgenticAI/05_integration_engineer.py       # self-test; --serve or --mcp

# ---- Tests ----
pytest Stage01_ML/test/ Stage02_DL/test/ Stage03_NLP/test/ Stage04_SLM/test/ Stage05_GenAI/test/ Stage06_AgenticAI/test/ test/ -v
```

All seeds are fixed at 42.

## 7. Results

**Macro F1 is the primary metric for the predictive stages.** All tasks are
imbalanced, and in each the minority class is the one that matters operationally.

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

### Stage 05 — Generative stress test (20 scenarios + wildcard, 51 zones)

| Metric | Domain-SLM suite | CVAE suite |
| --- | ---: | ---: |
| Scenarios passed | 16/20 | 17/20 |
| Zones passed | 44/51 | 47/51 |
| **Critical misses** | **0** | **0** |
| Wildcard zones passed | **4/4** | 3/4 |
| Realism score | **0.989** | 0.984 |

The domain SLM reaches validation perplexity 1.96. The CVAE's classifier two-sample
test gives ROC-AUC 0.763 (0.5 means indistinguishable from real data), against 0.831
for a naive baseline. Blind-spot coverage is 1.0.

### Stage 06 — Agentic coordination (agent never sees ground truth)

| Metric | CVAE suite (development) | SLM suite (held-out) |
| --- | ---: | ---: |
| Scenario goal success | **0.857** | 0.762 |
| Zone goal success | 0.927 | 0.855 |
| Critical misses | **0** | **0** |
| **Unsafe commits** (consequential action without a human) | **0** | **0** |
| Bad trade-off risk (confident and wrong) | 0.000 | 0.018 |
| Tool precision / recall | 0.987 / 1.000 | 0.982 / 1.000 |
| High-severity need coverage | 0.910 | 0.910 |
| Workflow violations | 0 | 0 |

- **Decision probes:** 4 hand-written one-ambulance trade-offs, including two severe zones with 38 vs 15 emergency calls, all pass on the real models.
- **Human-in-the-loop:** all 78 commit attempts without a valid token were blocked, and every approved dispatch was recalled by an emergency stop.
- **Ablations:** allocating in arrival order covers 64% of high-severity needs, against 91% for the agent. Removing the Safety Auditor lets 39 dispatches commit with no human.

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
8. **Stage 05's local Qwen checkpoint is corrupt**, so the LLM generation path runs only through the Gemini backend or its fallback. The generated scenarios are synthetic, so stress-test results measure behaviour on generated incidents, not real floods.
9. **Stage 06's evaluated planner is rule-based.** Its reasoning text is templated from observed values. The Gemini planner was blocked by a free-tier quota of 20 requests per day, and planning ratios and confidence factors are policy constants.
10. **Demo-grade deployment:** single-process development server, no authentication, no rate limiting. Alerts, reroutes and dispatches are recorded, never sent to a real system.

---

## 9. Repository layout

```
├── app.py                      Flask dashboard, prediction endpoints, /health, Stage 05/06 blueprints
├── requirements.txt            Pinned dependencies
├── docs/ARCHITECTURE.md        Real architecture, data lineage, design rationale
├── fusion/decision_engine.py   Cross-stage fusion policy
├── test/                       End-to-end HTTP tests + fusion policy tests
├── Stage01_ML/                 Tabular zone-risk ML          (README, 5 scripts, tests)
├── Stage02_DL/                 CNN + LSTM                    (README, 5 scripts, tests)
├── Stage03_NLP/                Urgency + hazard + NER        (README, 5 scripts, tests)
├── Stage04_SLM/                Tactical briefing SLM         (README, 5 scripts, tests)
├── Stage05_GenAI/              Scenario generation + stress test (README, 7 scripts, tests)
└── Stage06_AgenticAI/          Coordination agent + MCP      (README, 5 scripts, tests)
```

Within each stage, scripts run in numeric order: `01_data_engineer` (or
`01_knowledge_engineer`) → `02_eda_engineer` (or `02_workflow_engineer`) →
`03_*_engineer` → `04_evaluation_engineer` → `05_integration_engineer`
(the serving adapter used by `app.py`).
