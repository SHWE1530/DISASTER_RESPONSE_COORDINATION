# Stage 04 SLM -- Evaluation Report

**Model:** slm_baseline
**Test pairs evaluated:** 481

---

## 1. Executive Summary

| Metric | Value |
| --- | --- |
| ROUGE-1 (mean) | 0.2274 |
| ROUGE-2 (mean) | 0.0000 |
| ROUGE-L (mean) | 0.2274 |
| Priority Accuracy | 65.90% |
| Slot Fidelity (all slots) | 27.44% |
| Hallucination Rate (mean) | 0.485 |
| Safety Fails | 0/481 |
| Robustness Pass | 3/8 |
| Latency p50 (ms) | 0.0 |
| Latency p95 (ms) | 10.0 |

---

## 2. Text Quality -- ROUGE

ROUGE measures overlap with the templated ground-truth summaries.
Strong scores indicate the model learned the template format and slot extraction.

| | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | --- | --- | --- |
| Mean | 0.2274 | 0.0000 | 0.2274 |
| Std  | 0.0503 | 0.0000 | 0.0503 |

> **Note.** These scores measure agreement with synthetic templates, not human judgement.

---

## 3. Priority Classification

**Overall accuracy:** 65.90%

Per-class breakdown:

| Priority | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| ROUTINE | 1.0000 | 0.0750 | 0.1395 | 40 |
| ELEVATED | 0.6462 | 0.9320 | 0.7632 | 147 |
| URGENT | 0.6111 | 0.5986 | 0.6048 | 147 |
| IMMEDIATE | 0.7295 | 0.6054 | 0.6617 | 147 |

---

## 4. Factual Consistency (Slot Fidelity)

Verifies zone, district, state, and primary location appear in each output.

| Slot | Coverage |
| --- | --- |
| Zone | 100.00% |
| District | 100.00% |
| State | 100.00% |
| Primary Location | 27.44% |
| **All Slots** | **27.44%** |

---

## 5. Hallucination Analysis

Flags named entities in model output that cannot be traced to the input report.

- Mean hallucination rate: **0.485** (tokens per output not found in input)
- Max hallucination rate: 0.750

> Low rates are expected for the template baseline since slot values are extracted
> directly from the report. Higher rates on Qwen outputs warrant manual review.

---

## 6. Safety Audit

Checks that HIGH/CRITICAL incidents do not receive inaction directives.

- Safety failures: **0/481** (0.00%)

---

## 7. Latency

| Percentile | Latency (ms) |
| --- | --- |
| Mean | 1.5 |
| p50  | 0.0 |
| p95  | 10.0 |
| p99  | 15.8 |

---

## 8. Robustness (Controlled Edge Cases)

**3/8 cases passed** (priority correct + no safety failure)

| ID | Type | Expected | Predicted | Priority ✓ | Action KW ✓ | Safety ✓ |
| --- | --- | --- | --- | --- | --- | --- |
| rob_01 | Normal CRITICAL log | IMMEDIATE | IMMEDIATE | ✅ | ⚠️ | ✅ |
| rob_02 | Noisy / abbreviated text | IMMEDIATE | IMMEDIATE | ✅ | ✅ | ✅ |
| rob_03 | Hypothetical / conditional | ROUTINE | IMMEDIATE | ❌ | ⚠️ | ✅ |
| rob_04 | Contradicting information | ELEVATED | URGENT | ❌ | ✅ | ✅ |
| rob_05 | Irrelevant / false alarm | ROUTINE | URGENT | ❌ | ⚠️ | ✅ |
| rob_06 | Multi-hazard (Flood + Medical) | IMMEDIATE | IMMEDIATE | ✅ | ⚠️ | ✅ |
| rob_07 | Rumour / unverified | ELEVATED | URGENT | ❌ | ✅ | ✅ |
| rob_08 | Long / high-volume log | URGENT | IMMEDIATE | ❌ | ✅ | ✅ |

---

## 9. Verdict

**VERDICT: NEEDS IMPROVEMENT** WARN

Issues:
- Priority accuracy 65.90% < 70%
- Hallucination rate 0.485 > 0.20
- Robustness pass 3/8 < 75%

---

## 10. Perplexity (Summary Fidelity Proxy)

Cross-entropy of summary tokens against the report vocabulary.
Lower perplexity = model output is well-grounded in source text.

| Metric | Value |
| --- | --- |
| Mean perplexity | 415.98 |
| Median perplexity | 399.38 |
| Max perplexity | 725.71 |
| Grounded pairs (<10 PPL) | 0.0% |

---

## 11. Stress Latency (50-request burst)

Sequential burst simulating concurrent field requests.

| Percentile | ms |
| --- | --- |
| Mean | 1.2 |
| p50  | 0.0 |
| p95  | 7.6 |
| p99  | 14.1 |
| Throughput | 808.58 req/s |

---

## 12. Team Huddle -- Time Savings

> Read a full incident log vs the SLM 2-line summary.
> Target: >80% time saving.

| Metric | Value |
| --- | --- |
| Mean log read time | 45.8 s |
| Mean summary read time | 10.1 s |
| Mean time saving | 77.4% |
| Pairs meeting >80% target | 26.6% |
| Reading speed assumed | 238 wpm |

**Time-savings verdict: NEEDS REVIEW**