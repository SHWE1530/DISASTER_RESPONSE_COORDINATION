# Stage 05 — Generative AI (GenAI)

**Status: implemented. Stress test run end to end against the real Stage 01–04 models.**

| Script | Role | Description |
| --- | --- | --- |
| `01_data_engineer.py` | Data Engineer | Reference baselines from real history: sensor distributions, dispatcher phrasing, real gauge dynamics, and a mask-labelled imagery bank |
| `02_eda_engineer.py` | EDA / Prompt Engineer | Measures 17 blind spots in the historical data and writes the 20-scenario prompt library plus the wildcard, each prompt mapped to the blind spots it targets |
| `03_genai_engineer.py` | GenAI Engineer | Conditional VAE over sensor readings and a prompt-driven, multi-zone compound scenario generator |
| `04_evaluation_engineer.py` | Evaluation Engineer | Realism audit and a 20-scenario stress test through the live pipeline, plus a Stage 02 forecast probe |
| `05_integration_engineer.py` | Integration Engineer | Stress-test dashboard, run history, and a Flask blueprint (standalone `--serve`) |

---

## 1. Mission

Major disasters are rare, so the historical record barely rehearses the situations that matter most. Stage 05 **synthesises compound, multi-zone disaster scenarios** and runs them through the *existing* pipeline: Stage 01 risk, the Stage 02 CNN and LSTM, and Stage 03 NLP, fused by `fusion/decision_engine.py`, plus a Stage 04 briefing for each zone. The goal is to find where the pipeline breaks *before* a real flood does.

**Scope guarantee.** Stage 05 only *reads* earlier stages' data and calls their public adapters. Nothing in `Stage01_ML/`–`Stage04_SLM/`, `fusion/`, `app.py` or the root docs is modified.

---

## 2. Architecture

```
 Stage 01/02/03 data (read-only)
          │
  01_data_engineer ──► sensor_reference.csv, reference_distributions.json,
          │            text_phrase_bank.json, water_level_reference.json, imagery bank
          ▼
  02_eda_engineer ───► blind_spot_report.json ──► scenario_prompts.json (20 + wildcard)
          ▼
  03_genai_engineer
     ├─ SensorCVAE  p(9 sensor fields | risk class), trained on 10,000 real rows
     └─ ScenarioGenerator  prompt spec ─► per zone:
            sensors (CVAE + stress modifiers) · 72 h gauge history (real rise rates)
            dispatcher/citizen text (real Stage 03 phrasing + implicit/negated/noisy variants)
            Stage 04 INCIDENT LOG · imagery · ground-truth expectations
          ▼
  04_evaluation_engineer
     ├─ realism audit   KS, correlation, C2ST vs a naive baseline, Stage 01/03 cross-checks
     ├─ stress test     Stage 01 + 02 + 03 ─► fusion ─► priority/review/conflicts ; Stage 04 briefing
     └─ forecast probe  flat rivers ─► Stage 02 LSTM
          ▼
  05_integration_engineer ─► stress_test_dashboard.html · /genai · /api/genai/*
```

---

## 3. Blind spots found in the historical data (`02_eda_engineer.py`)

Every blind spot is **measured**. Structural gaps state why the data cannot represent them. Result: **11 absent, 2 rare, 4 covered**, and every rare or absent blind spot is targeted by at least one prompt.

| ID | Blind spot | Status | Evidence |
| --- | --- | --- | --- |
| BS01 | Low-risk (calm) zones | rare | 179 / 10,000 rows (1.79%) |
| BS02 | Rainfall beyond the record | absent (**censored**) | 295 rows (2.95%) sit *exactly* on the 104.02 mm ceiling: the tail was capped during cleaning |
| BS03 | Urban waterlogging (heavy rain, low river) | rare | 171 rows (1.71%) |
| BS04 | Silent rise (river high, few calls) | covered | 202 rows (2.02%) |
| BS05–07 | Receding-but-severe, dry-season severe, night onset | covered | 7.4% / 52.8% / 25.0% |
| BS08 | ≥3 Severe zones at the same moment | absent | every row is one zone in isolation |
| BS09 | Missing or faulty telemetry | absent | 0 missing values |
| BS10 | Infrastructure failure (power/telecom) | absent | no such field |
| BS11 | Cross-modal conflict | absent | no joint sensor/image/text dataset |
| BS12 | Noisy or code-mixed text | absent | 100% of sampled dispatcher texts fit one template |
| BS13 | Negated hazard language | absent | 0 matches |
| BS14 | Water-level change beyond the record | absent (**censored**) | 30 rows pinned at the 1.575 m ceiling |
| BS15 | Mass-casualty headcounts (>150) | absent | max headcount 120 |
| BS16 | Multi-hazard messages | absent | every message carries one hazard label |
| BS17 | Forward-looking labels | absent | labels describe the present only |

---

## 4. The generator (`03_genai_engineer.py`)

**SensorCVAE** is a conditional variational autoencoder (latent 6, hidden 64, KL warm-up, inverse-√frequency class sampling) that learns the *joint* distribution of the nine Stage 01 sensor fields given a risk class. The CVAE is the GenAI component. The alternative was an LLM, but a CVAE runs offline on CPU in about 30 s and can be measured quantitatively.

- Samples include the model's measured **observation noise**. A first version decoded the mean only. That put all nine fields on a 6-D surface, inflated the rainfall-to-calls correlation to 0.996 (real value 0.824), and made samples trivially detectable (C2ST AUC 0.966). Adding observation noise fixed it (section 5).
- Plain samples are clipped to the historical record. Only named **stress modifiers** (`flash_flood`, `dam_release`, `mass_casualty`, …) may push past it, and each push is logged per zone in `provenance.sensor_adjustments` / `beyond_record_fields`. In the suite, 10 of 51 zones extrapolate, and each one is disclosed.
- Gauge histories use **real CWC rise rates**: 101 stations and 653k hourly readings, after removing 155k sentinel values. Ordinary histories stay inside the LSTM's trained range. Only the deliberate `ood_spike` trips its out-of-distribution guard.
- Text recombines real Stage 03 phrasing and adds what the corpus lacks: implicit urgency (no "urgent" or "critical" words), negation, prank messages, outage messages, and comms noise (typos, caps, Hinglish, truncated SMS).
- **Ground truth is the scenario designer's intent**, encoded per zone as pass criteria *before* the run. Expectations were never relaxed after seeing results.

---

## 5. Results (seed 42; `data/outputs/evaluation/stress_test_report.md`)

### Realism audit

| Check | CVAE | Naive baseline (independent marginals) |
| --- | ---: | ---: |
| C2ST ROC-AUC (0.5 = indistinguishable from real) | **0.763** | 0.831 |
| Rainfall-to-calls correlation (real 0.824) | **0.797** | 0.088 |
| Stage 01 recovers the conditioning class (macro F1) | 0.784 | 0.756 |
| Mean KS statistic / max \|Δr\| | 0.053 / 0.072 | — |

| Stage 03 reading generated text | Clean | Degraded (comms noise) |
| --- | ---: | ---: |
| Hazard accuracy | 1.000 | 0.988 |
| Urgency macro F1 | 0.870 | 0.870 |
| Headcount exact | 1.000 | 0.806 |

Implicit-urgency messages scored HIGH or CRITICAL by Stage 03: 100%.

### Stress test: 20 scenarios, 51 zones

| Metric | Result |
| --- | ---: |
| Scenarios passed | **17 / 20** |
| Zones passed | 47 / 51 |
| **Critical misses** (true URGENT+ called ≤ ELEVATED) | **0 / 32** |
| Pipeline crashes | 0 |
| Priority exact / within one level | 70.6% / 90.2% |
| Over-triage (calm zones called 2+ levels high) | 26.3% |
| Human-review recall | 100% |
| Modality conflicts surfaced | 66.7% |
| Mean zone-ranking Kendall τ (17 multi-level scenarios) | 0.871 |
| Fusion latency p50 / p95 | 56 / 67 ms |

### What the stress test found

1. **Stage 02's LSTM projects large rises from flat, low rivers, and fusion escalates calm zones to URGENT.** This causes all 4 failing suite zones (S05, S09-Z2, S18-Z2, S18-Z3). In each one, Stage 01 says Low at 0.99 confidence and the text and imagery are calm. A direct probe with a *perfectly flat* 72 h history gives:

   | Flat level | 1 m | 2 m | 3 m | 4 m | 5 m | 6–9 m |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | Projected 6 h change | **+2.74** | **+1.03** | −0.11 | **+1.57** | −1.43 | −0.23 to −0.49 |

   Real gauges rise more than 1.55 m in 6 h in only 1% of windows. Fusion escalates on a projected rise of ≥0.9 m. The pattern looks like reversion toward the training mean (3.12 m), and it's non-monotonic. **Suggested follow-up (Stage 02 / fusion owners):** require a real recent rise before a forecast may escalate, or cap projected rises at the observed gauge tail.
2. **Stage 03 has no negation handling** (a documented limitation, now reproduced). "No flooding here, please ignore the forwarded message" reads as MEDIUM urgency.
3. **The overall over-triage rate is 26.3%.** The system errs toward over-response, which matches the fusion design ("under-responding costs more"), but it's the price of finding 1.
4. **The Stage 04 baseline briefing is longer than short logs.** Read-time savings average −9.1% on 1–4-entry logs, compared with +45.9% on Stage 04's 139–378-word logs. The >80% target only applies to long logs. The baseline also over-calls priority (40.8% exact) while keeping 100% high-risk recall.
5. **Single-seed sensitivity.** S20 (slow onset) flipped from FAIL to PASS when the CVAE was retrained, so individual scenario outcomes depend on the random draw. Treat the suite rates as indicative, and re-run with other `--seed` values before drawing conclusions from one scenario.

---

## 6. The wildcard: flash flood + municipal blackout (capstone)

**Scenario (invented by the team):** at 2 AM a flash flood hits an Assam district *while the municipal grid fails*. Cameras need mains power and go dark. Gauges survive only on batteries. Cell towers fail, so fewer, shorter, garbled messages get through. One neighbourhood goes completely silent. Resources: one ambulance and one 40-bed shelter, against **323 people reported** across the zones.

| Zone | Truth | Evidence left | Decision | Result |
| --- | --- | --- | --- | --- |
| Riverside settlement (battery gauge) | CRITICAL | sensors, text, gauge history | CRITICAL | PASS |
| Low-lying colony (grid down) | CRITICAL | one garbled text | **ROUTINE** (human review forced) | **FAIL** |
| District hospital (generator) | URGENT | everything | URGENT | PASS |
| Char island (no signal) | CRITICAL | nothing | `insufficient_evidence` (not ROUTINE) | PASS |

**Defence: why it matters.**
- *Silence is not safety:* the zone with zero evidence is refused and escalated to a human. It is never scored as calm. This is the fusion layer's most important safety property, and the wildcard proves it works under total blackout.
- *The failure is structural, not random.* When the only evidence is one low-confidence report, fusion multiplies the **severity level** by confidence (`level × max(conf, 0.25)`), so a lone MEDIUM report at 0.37 confidence rounds to ROUTINE. Human review is still forced (single source, low confidence), which is the safety net that caught it. A sparse, garbled signal from a blacked-out zone is *more* alarming than its words suggest, so outage context should raise priority rather than lower it. That's a concrete requirement for Stage 06.
- *It hands Stage 06 its dilemma:* two CRITICAL zones and one ambulance, with a 323-person demand against 40 beds.

---

## 7. How to run

```bash
python Stage05_GenAI/01_data_engineer.py          # --skip-imagery to reuse the committed imagery bank
python Stage05_GenAI/02_eda_engineer.py
python Stage05_GenAI/03_genai_engineer.py         # trains the CVAE (~30 s CPU); --skip-train to reuse it
python Stage05_GenAI/04_evaluation_engineer.py    # --slm qwen | none, --skip-realism, --seed N
python Stage05_GenAI/05_integration_engineer.py   # self-test + static dashboard
python Stage05_GenAI/05_integration_engineer.py --serve   # live dashboard: http://127.0.0.1:5005/genai
pytest Stage05_GenAI/test/ -v                     # 32 tests, ~10 s
```

`04_evaluation_engineer.py` uses the Stage 04 **baseline** by default, because the Qwen adapter takes about 10 s per briefing. Each run appends to `stress_test_history.csv`, and the dashboard plots the pass-rate and critical-miss trend across runs.

**Dependencies:** no new pins. The scripts use packages already in `requirements.txt` (torch, pandas, scikit-learn, scipy, xgboost, flask). The one exception is **rebuilding** the imagery bank, which needs `datasets` (installed here, not pinned). The bank itself is committed, so `--skip-imagery` avoids it. The C2ST uses XGBoost rather than scikit-learn's forests, because on this machine an application-control policy blocks a scikit-learn extension that `sklearn.ensemble` imports.

### Wiring into the main dashboard (optional, not applied)

Stage 05 deliberately leaves `app.py` untouched. To mount it there:

```python
stage05 = load_module("stage05_api", BASE_DIR / "Stage05_GenAI" / "05_integration_engineer.py")
app.register_blueprint(stage05.create_blueprint())   # serves /genai and /api/genai/*
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/genai` | GET | Live dashboard (pick a prompt and seed, then generate and stress-test) |
| `/api/genai/health` | GET | Generator readiness via a real probe scenario |
| `/api/genai/prompts` | GET | Prompt library |
| `/api/genai/scenario` | POST | `{prompt_id, seed}` or `{spec, seed}` returns a scenario |
| `/api/genai/stress-test` | POST | Same input; returns scored zones (loads Stage 01–04 on first call) |
| `/api/genai/report` | GET | Latest full report |

---

## 8. Artifacts

| Path | Content |
| --- | --- |
| `data/processed/` | Reference baselines, phrase bank, gauge reference, imagery manifest, `scenario_prompts.json` |
| `data/raw/imagery/` | 24 scenes (12 flooded, 12 unflooded) labelled from water masks |
| `data/models/sensor_cvae.pt` | CVAE bundle (weights-only loadable) |
| `data/outputs/eda/` | Blind-spot report, coverage grid, prompt × blind-spot matrix |
| `data/outputs/scenarios/` | `scenario_suite.json` (20), `wildcard_scenario.json`, `scenario_zones.csv` |
| `data/outputs/evaluation/` | Stress-test report (MD/JSON), per-zone and per-scenario CSVs, realism report, run history |
| `data/outputs/stress_test_dashboard.html` | Self-contained dashboard (no external scripts) |

## 9. Limitations

1. **Ground truth is designer intent**, not observed outcomes. Pass rates measure agreement with documented expectations.
2. **The imagery is likely the Stage 02 CNN's own training pool.** It's the only labelled imagery available, so visual evidence in the stress test is optimistic.
3. **The CVAE is still distinguishable from real data** (C2ST 0.763): it's better than the naive baseline but not indistinguishable. Observation noise pushes 3.6–11.7% of raw values past the record edges, and those are clipped.
4. **Text realism is bounded by Stage 03's own templated corpus.** Clean generated text scores near-perfect partly because it shares that template.
5. **Scenario results come from a single seed** (see finding 5).
