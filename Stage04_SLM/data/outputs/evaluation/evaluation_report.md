# Stage 04 SLM -- Evaluation & Capstone Audit Report

**Model Evaluated:** `slm_baseline`
**Test Pairs Evaluated:** 481
**Evaluation Timestamp:** `2026-09-11 05:25:26`

---

## 1. Executive Summary

| Metric | Measured Value | Capstone Target | Status |
| --- | --- | --- | --- |
| Priority Accuracy | 65.90% | $\ge 70.0\%$ | ❌ FAIL |
| Combined High-Risk Recall | 60.20% | $\ge 80.0\%$ | ❌ FAIL |
| High-Risk Safety Failures | 0 fails | 0 fails | ✅ PASS |
| Content Slot Fidelity (All Slots) | 27.44% | $\ge 80.0\%$ | ❌ FAIL |
| Hallucination Rate (mean) | 0.3829 | $\le 0.1500$ | ❌ FAIL |
| Robustness Pass Rate | 1/8 (12.50%) | $\ge 75.0\%$ | ❌ FAIL |
| Latency p50 | 1.4 ms | Performance Benchmark | Evaluated |

**FINAL CAPSTONE VERDICT: `NEEDS IMPROVEMENT`**

---

## 2. Text Quality -- ROUGE

ROUGE measures text overlap against template-grounded reference summaries.

| Metric | ROUGE-1 | ROUGE-2 | ROUGE-L | Calculator Type |
| --- | --- | --- | --- | --- |
| Mean Score | 0.3282 | 0.1452 | 0.2416 | LCS-based Fallback |
| Std Dev | 0.0464 | 0.0333 | 0.0415 | - |

> **Note:** ROUGE scores reflect agreement with synthetic SOP templates, not open-ended human style.

---

## 3. Priority Classification & High-Risk Recall

- **Overall Accuracy:** 65.90%
- **Macro F1:** 0.5423
- **Weighted F1:** 0.6319
- **URGENT Recall:** 59.86%
- **IMMEDIATE Recall:** 60.54%
- **Combined High-Risk Recall (URGENT + IMMEDIATE):** **60.20%**

Per-Class Breakdown:

| Priority | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| ROUTINE | 1.0000 | 0.0750 | 0.1395 | 40 |
| ELEVATED | 0.6462 | 0.9320 | 0.7632 | 147 |
| URGENT | 0.6111 | 0.5986 | 0.6048 | 147 |
| IMMEDIATE | 0.7295 | 0.6054 | 0.6617 | 147 |

---

## 4. Factual Consistency / Slot Fidelity

Verifies presence and numerical correctness of content slots (separated from decision quality).

| Slot Category | Coverage / Accuracy |
| --- | --- |
| Zone | 100.00% |
| District | 100.00% |
| State | 100.00% |
| Primary Location | 27.44% |
| Hazard Type | 99.38% |
| Headcount Numerical Correctness | 100.00% |
| **All Content Slots Preserved Rate** | **27.44%** |

---

## 5. Hallucination Analysis

Detects entity tokens in generated output that cannot be grounded in the source report.

- **Mean Hallucination Rate:** `0.3829`
- **Max Hallucination Rate:** `0.5714`

> **Methodology Note:** Standard SOP action verbs (e.g. `evacuate`, `deploy`, `triage`, `NDRF`, `SDRF`) are explicitly exempted from hallucination flags.

---

## 6. Safety & Guardrail Audit

Audits for dangerous inaction directives on high-risk incidents (URGENT / IMMEDIATE).

- **Total High-Risk Safety Failures:** `0/481`
- **Safety Failure Rate:** `0.00%`

---

## 7. Embedded Latency & Concurrency Stress Test

| Metric / Percentile | Latency / Throughput |
| --- | --- |
| Mean Latency | 1.5 ms |
| p50 Latency | 1.4 ms |
| p95 Latency | 2.4 ms |
| p99 Latency | 3.0 ms |
| Burst Sequential Throughput | 645.32 req/s |
| 5-Worker Concurrent Throughput | 853.99 req/s |

---

## 8. Robustness Audit (8 Controlled Edge Cases)

**Passed:** 1/8 cases (12.50%)

| ID | Case Type | Expected | Predicted | Priority ✓ | Action Keyword ✓ | Safety ✓ | Verdict | Failure Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rob_01 | Normal CRITICAL / IMMEDIATE log | IMMEDIATE | IMMEDIATE | ✅ | ❌ | ✅ | **FAIL** | missing action keyword 'evacuate' |
| rob_02 | Noisy / abbreviated text | IMMEDIATE | IMMEDIATE | ✅ | ✅ | ✅ | **PASS** | None |
| rob_03 | Hypothetical / conditional alert | ROUTINE | IMMEDIATE | ❌ | ❌ | ✅ | **FAIL** | wrong priority (IMMEDIATE != ROUTINE); missing action keyword 'monitor' |
| rob_04 | Contradictory information | ELEVATED | URGENT | ❌ | ❌ | ✅ | **FAIL** | wrong priority (URGENT != ELEVATED); missing action keyword 'assess' |
| rob_05 | Irrelevant / false alarm | ROUTINE | URGENT | ❌ | ❌ | ✅ | **FAIL** | wrong priority (URGENT != ROUTINE); missing action keyword 'monitor' |
| rob_06 | Multi-hazard (Flood + Medical) | IMMEDIATE | IMMEDIATE | ✅ | ❌ | ✅ | **FAIL** | missing action keyword 'evacuate' |
| rob_07 | Rumour / unverified report | ELEVATED | URGENT | ❌ | ❌ | ✅ | **FAIL** | wrong priority (URGENT != ELEVATED); missing action keyword 'assess' |
| rob_08 | Long / high-volume log | URGENT | IMMEDIATE | ❌ | ✅ | ✅ | **FAIL** | wrong priority (IMMEDIATE != URGENT) |

---

## 9. Generated Output Vocabulary Grounding Diagnostic

Unigram cross-entropy grounding index of generated summaries against report text.

- Mean Grounding Index: `846.34`
- Grounded Pairs (<10 index): `0.0%`

> **Diagnostic Note:** This index evaluates unigram source grounding of generated output and is NOT neural model perplexity.

---

## 10. Team Huddle -- Read-Time Savings Benchmark

- Mean Incident Log Read Time: `45.8 s`
- Mean SLM Summary Read Time: `24.1 s`
- **Mean Read-Time Savings:** **`45.9%`**
- Target (>80% savings) Pass Rate: `0.0%`
- Assumed Silent Reading Speed: `238 WPM`

---

## 11. Baseline vs Qwen 2.5 3B QLoRA Comparison

| Metric | Baseline (CPU) | Qwen 2.5 3B QLoRA | Delta / Improvement | Preference |
| --- | --- | --- | --- | --- |
| ROUGE-1 F1 | 0.3282 | 0.0908 | -0.2374 | Higher is better |
| Priority Accuracy | 65.90% | 0.00% | -65.9% | Higher is better |
| High-Risk Recall | 60.20% | 0.00% | -60.2% | Higher is better |
| Slot Fidelity | 27.44% | 0.00% | -27.4% | Higher is better |
| Hallucination Rate | 0.3829 | 0.5004 | +0.1175 | Lower is better |
| Safety Failure Rate | 0.00% | 50.00% | +50.0% | Lower is better |
| Robustness Pass Rate | 12.50% | 0.00% | -12.5% | Higher is better |
| Latency p50 | 1.0 ms | 9889.5 ms | +9888.5 ms | Lower is better |

---

## 12. Limitations & Scope

1. **Template Reference Bias:** Ground truth summaries in the dataset were generated from SOP templates. High ROUGE scores reflect template fidelity.
2. **Sequential Load Testing:** Hardware latency benchmarks reflect single-threaded embedded CPU inference.
3. **Deterministic Entity Grounding:** Entity extraction relies on deterministic token matching with SOP action verb exemptions, not full LLM-as-a-judge reasoning.

---

## 13. Final Capstone Verdict

**VERDICT: `NEEDS IMPROVEMENT`**
