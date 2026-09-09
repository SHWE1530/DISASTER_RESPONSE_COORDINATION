# Stage 03 NLP Quality & Evaluation Report

## 1. Executive Summary
- **Urgency Macro F1 (primary metric):** 0.7224
- **Urgency Accuracy (secondary):** 71.77%

Per-class F1:
  - **LOW**: F1 0.6944 (precision 0.5945, recall 0.8347, support 1500)
  - **MEDIUM**: F1 0.6291 (precision 0.6319, recall 0.6263, support 2938)
  - **HIGH**: F1 0.7726 (precision 0.7817, recall 0.7637, support 3821)
  - **CRITICAL**: F1 0.7936 (precision 0.9890, recall 0.6626, support 1491)
- **Robustness Tests Passed:** 4/8

### Corpus caveat
The Stage 03 corpora are synthetically generated. Impact language is sampled from a shared pool with severity-dependent weights (it is correlated with urgency, not determined by it), and hazard labels are the generative parameter rather than a keyword rule applied to the rendered text. An earlier revision violated both of those properties, which is why it reported ~96% urgency accuracy and exactly 1.0000 hazard F1 on every class. Those numbers measured string matching. The figures below are lower and are the real ones.

## 2. Critical Failure Analysis
Disaster-response specific failures that could cost lives or waste resources:

### Test Case: Noisy communication
- **Input:** "pls hlp!!! flood water everywhere near main rd"
- **Error Category:** Severity Underestimation
- **True Urgency:** HIGH -> **Predicted:** MEDIUM (Conf: 0.41)

### Test Case: Spelling errors
- **Input:** "2 ppl trapped in flud water"
- **Error Category:** Severity Underestimation
- **True Urgency:** CRITICAL -> **Predicted:** MEDIUM (Conf: 0.47)

### Test Case: Hypothetical
- **Input:** "If flooding happens, people may need rescue near the river bank."
- **Error Category:** False Positive (Overreaction)
- **True Urgency:** LOW -> **Predicted:** HIGH (Conf: 0.38)

### Test Case: Rumour/Ambiguous
- **Input:** "Someone said 5 people are trapped near the old temple."
- **Error Category:** Severity Underestimation
- **True Urgency:** HIGH -> **Predicted:** MEDIUM (Conf: 0.42)

## 3. Confusion Matrix Analysis (Urgency)
Analyzing dangerous confusions (Predicting LOW when actual is HIGH/CRITICAL):
- **HIGH misclassified as LOW/MEDIUM:** 898 times (Dangerous Underestimation)
- **CRITICAL misclassified as LOW/MEDIUM:** 70 times (Extremely Dangerous Underestimation)

## 4. Final Verdict
**VERDICT: NEEDS IMPROVEMENT**

*Reasoning:* The model exhibits severity underestimation on edge cases (e.g. typos, informal text) and failed to extract location from critical noisy alerts. A fallback mechanism or robustness retraining is recommended.