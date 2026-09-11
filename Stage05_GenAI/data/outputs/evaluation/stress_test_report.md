# Stage 05 -- Stress-Test Evaluation Report

Generated 2026-09-11T11:30:39+00:00 (commit `db120b0`). Stage status: ml=healthy, dl=healthy, nlp=healthy, slm=healthy.

Ground truth is the scenario designer's intent, not an observed outcome. Imagery comes from the Stage 02 CNN's likely training pool, so visual evidence is optimistic.

## 1. Headline results (20-scenario suite)

| Metric | Result |
| --- | ---: |
| Scenarios passed | 17/20 (85.0%) |
| Zones passed | 47/51 (92.2%) |
| Pipeline crashes | 0 |
| Priority exact accuracy | 70.6% |
| Within-one-level accuracy | 90.2% |
| **Critical-miss rate** (true URGENT+/called <= ELEVATED) | **0.0%** (0/32) |
| Over-triage rate (true <= ELEVATED, called 2+ levels higher) | 26.3% |
| No-evidence zones refused, not scored (suite + wildcard, n=1) | 100.0% |
| Modality conflicts surfaced | 66.7% |
| Human-review recall | 100.0% |
| Mean zone-ranking Kendall tau (17 scenarios) | 0.871 |
| Zones with values beyond the historical record | 10 |
| Fusion latency p50 / p95 | 55.6 / 66.5 ms |
| Stage 04 briefing (slm_baseline) priority accuracy | 40.8% |
| Stage 04 briefing high-risk recall | 100.0% |
| Stage 04 read-time savings (mean) | -9.1% |

## 2. Realism audit

| Check | CVAE | Naive baseline |
| --- | ---: | ---: |
| C2ST ROC-AUC (0.5 = indistinguishable) | 0.7625 | 0.8309 |
| Rainfall-calls correlation (real 0.8239) | 0.7971 | 0.0882 |
| Stage 01 recovers conditioning class (macro F1) | 0.7843 | 0.7563 |

Mean KS statistic across features: 0.0533; max |delta r| between correlation matrices: 0.072.

| Stage 03 reading generated text | Clean | Degraded (comms noise) |
| --- | ---: | ---: |
| Hazard accuracy | 1.0 | 0.9875 |
| Urgency macro F1 | 0.8698 | 0.8698 |
| Headcount exact | 1.0 | 0.8063 |

Implicit-urgency messages scored HIGH/CRITICAL by Stage 03: 100.0%.

## 2b. Stage 02 forecast probe (perfectly flat 72 h river)

Real CWC gauges rise more than 1.55 m in 6 h in only 1% of windows. Fusion escalates to URGENT on a projected rise of 0.9 m.

| Flat level (m) | Forecast 6 h peak (m) | Projected change (m) | Fusion reads it as |
| ---: | ---: | ---: | --- |
| 1.0 | 3.74 | +2.74 | steep rise: escalates to URGENT |
| 2.0 | 3.03 | +1.03 | steep rise: escalates to URGENT |
| 3.0 | 2.89 | -0.11 | flat |
| 4.0 | 5.58 | +1.57 | steep rise: escalates to URGENT |
| 5.0 | 3.57 | -1.43 | falling |
| 6.0 | 5.53 | -0.47 | falling |
| 7.0 | 6.77 | -0.23 | flat |
| 8.0 | 7.68 | -0.32 | falling |
| 9.0 | 8.51 | -0.49 | falling |

## 3. Per-scenario results

| ID | Scenario | Zones | Passed | Ranking tau | Blind spots |
| --- | --- | ---: | :---: | ---: | --- |
| S01 | Baseline river overflow | 3 | PASS | 1.0 | BS08 |
| S02 | The 2 AM mission | 3 | PASS | 0.816 | BS02, BS08, BS14 |
| S03 | Cyclonic coastal landfall | 4 | PASS | 0.913 | BS02, BS08 |
| S04 | Urban cloudburst waterlogging | 2 | PASS | 1.0 | BS03 |
| S05 | Negated flood rumour | 2 | 1/2 | - | BS13, BS01 |
| S06 | Silent rise while residents sleep | 2 | PASS | - | BS04 |
| S07 | Telemetry dropout | 3 | PASS | 0.333 | BS09 |
| S08 | Gauge logger malfunction | 2 | PASS | 1.0 | BS09 |
| S09 | People versus sensors | 2 | 1/2 | 1.0 | BS11 |
| S10 | Camera versus gauge | 2 | PASS | 1.0 | BS11, BS06 |
| S11 | Receding but still flooded | 2 | PASS | 1.0 | BS05 |
| S12 | Bridge collapse isolation | 2 | PASS | 1.0 | BS10 |
| S13 | Hospital ground floor flooding | 2 | PASS | 1.0 | BS15 |
| S14 | Night dam spillway release | 3 | PASS | 0.0 | BS14, BS02 |
| S15 | Five-state monsoon cascade | 5 | PASS | 0.738 | BS08 |
| S16 | Out-of-season cloudburst | 2 | PASS | 1.0 | BS06 |
| S17 | Garbled panic messages | 3 | PASS | 1.0 | BS12 |
| S18 | All quiet (do not cry wolf) | 3 | 1/3 | - | BS01 |
| S19 | Four hazards, one district | 2 | PASS | 1.0 | BS16 |
| S20 | Slow onset, pre-position now | 2 | PASS | 1.0 | BS17 |

## 4. Failure log

| Scenario | Zone | True | Called | Why it failed |
| --- | --- | --- | --- | --- |
| S05 | ZONE-1 Backwater ward | ROUTINE | URGENT | priority URGENT above maximum ELEVATED |
| S09 | ZONE-2 Prank-message town | ROUTINE | URGENT | modality conflict not surfaced |
| S18 | ZONE-2 Industrial belt | ROUTINE | URGENT | priority URGENT above maximum ELEVATED |
| S18 | ZONE-3 Old city | ROUTINE | URGENT | priority URGENT above maximum ELEVATED |

## 5. Wildcard -- Flash flood + municipal blackout

At 2 AM a flash flood hits an Assam district at the same moment the municipal grid fails. Gauges, street cameras and cell towers go dark; only battery- and generator-backed equipment survives. Four neighbourhoods -- one goes completely silent. Silence must never be read as safety.

| Zone | True | Evidence left | Lost to outage | Decision | Review | Result |
| --- | --- | --- | --- | --- | :---: | :---: |
| Riverside settlement (battery gauge) | CRITICAL | sensors, text, water_history | image | CRITICAL | yes | PASS |
| Low-lying colony (grid down) | CRITICAL | text | image, sensors, water_history | ROUTINE | yes | FAIL: priority ROUTINE below minimum URGENT |
| District hospital (generator) | URGENT | sensors, text, image, water_history | - | URGENT | yes | PASS |
| Char island (no signal) | CRITICAL | none | sensors, text, image, water_history | insufficient_evidence | yes | PASS |
