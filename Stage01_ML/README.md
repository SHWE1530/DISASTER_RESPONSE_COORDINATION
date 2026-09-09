# Stage 01: Machine Learning (Disaster Response Coordination)

This directory contains the machine learning pipeline for generating Zone Risk Scores.

## FIELD BRIEFING SHEET: ZONE RISK SCORE MODEL

### Overview
This reference guide helps Emergency Response Teams understand the **Zone Risk Score** output provided by the Disaster Response Coordination System (Stage 01).

Our model turns continuous sensor data (river levels, rainfall), weather patterns, and emergency-call volumes into an instant risk score (Low, Moderate, Severe) for every city zone, allowing you to prioritize deployments effectively.

---

### The Risk Categories

#### 🟢 LOW RISK
- **Definition:** Normal conditions or minor weather events. Infrastructure is handling the load. 
- **Typical Indicators:** Rainfall < 20mm, River levels well below danger thresholds, Emergency call volumes within normal daily averages.
- **Action:** Routine monitoring. No immediate field deployment required.

#### 🟡 MODERATE RISK
- **Definition:** Elevated stress on infrastructure. Potential for localized waterlogging, isolated road closures, or minor disruptions.
- **Typical Indicators:** Continuous rainfall, River levels approaching warning thresholds, noticeable uptick in emergency calls.
- **Action:** Alert field teams. Pre-position equipment in vulnerable zones. Monitor for escalation.

#### 🔴 SEVERE RISK (HIGH PRIORITY)
- **Definition:** Imminent or ongoing critical disaster event. High probability of widespread flooding, bridge/road closures, and significant population impact.
- **Typical Indicators:** River level at or above the district danger threshold, sustained high call volume, heavy rainfall.
- **Action:** Immediate full-scale deployment. Execute evacuation protocols for affected populations. Coordinate with central command for bridge/road closures.

> **How the label is defined — important.** `zone_risk` is the **observed label
> carried through from the source dataset**. It is *not* computed from a
> threshold rule.
>
> An earlier revision of `01_data_engineer.py` overwrote `zone_risk` with a
> hand-written rule (`river_level_m > river_level_threshold_m`, `emergency_calls
> >= 35 and rainfall_mm >= 40`, ...). Those inputs are model features, and the
> engineered feature `river_level_margin_m` is literally that rule's decision
> variable — so training on those labels is target leakage by construction, and
> the reported accuracy would approach 100% while measuring nothing.
>
> The rule is retained in the script as `reference_risk_rule()` and reported
> only as an **agreement rate** (currently **0.7932** against the observed
> labels). That 21% gap is the part of real-world risk assessment a fixed
> threshold does not capture, and it is the reason a learned model is worth
> having at all.

---

### Top Risk Factors (What the Model Looks At)

When assessing a zone, the intelligence system prioritizes these signals in order:

1. **River Level (m):** The absolute height of the river compared to historical baselines.
2. **Rainfall (mm):** The rolling accumulation of rain in the district.
3. **Bridge Closures:** Infrastructure failure is a leading indicator of severe impact.
4. **Population Affected:** Density of people in the impacted zone.
5. **Road Closures:** Disruptions to transport networks.

---

### How to use the Dashboard

1. **Live Assessment:** Enter the latest readings from the field (Rainfall, River Level, Call Volumes) into the "Live risk assessment" panel to get an instant classification.
2. **Confidence Score:** The confidence percentage is the model's **uncalibrated** score — it is a ranking signal, not a probability of being correct. Calibration is measured and published in `data/outputs/calibration_report.json` and `calibration_reliability_curve.png`: mean Brier score **0.0197**, expected calibration error **0.0174**. The model is well calibrated in its top bin (confidence 0.995 vs observed accuracy 0.988) but **overconfident in the middle bins** — in the 0.6–0.7 band it is right only about 43% of the time. Treat mid-range confidence with suspicion and escalate to a human.
3. **Human oversight:** Every output is decision support for a responder, not a verified assessment. The Low class in particular is weak (recall ~0.71 on 31 held-out samples), so a "Moderate" result on visibly calm conditions is a known failure mode.
4. **Monitor Trends:** Use the "Risk Trend" and "Risk by Region" panels to see which districts have the highest frequency of Severe events.

---

## EVALUATION ENGINEER (Model Audit & Feedback Loop)

The Evaluation Engineer stage (`04_evaluation_engineer.py`) acts as the final gatekeeper before a model is integrated into the web application.

### Purpose
To rigorously audit the finalized Machine Learning model on **independent, held-out test data** that the model has never seen during training or tuning. This ensures honest metrics and uncovers real-world failure modes.

### Key Terminology
- **Prediction/Inference:** The act of the model generating a guess (Low/Moderate/Severe) for a set of data.
- **Evaluation / Testing:** The rigorous, automated process of comparing the model's predictions against known "ground truth" answers.
- **Validation:** An earlier step performed by the ML Engineer to tune the model.
- **Error Analysis:** Deep-diving into specific mistakes (e.g., missed severe floods) to find data gaps.
- **Model Approval:** The final business decision (Pass/Needs Improvement) based on established thresholds.

### Workflow & Inputs
1. **Inputs:** Loads the serialized model (`ml_pipeline.joblib`) and the independent test sets (`X_test.csv` and `y_test.csv`).
2. **Feature Engineering:** Re-creates the exact features (e.g., `river_level_margin_m`, `is_monsoon`) so inference works perfectly.
3. **Inference:** Generates predictions without retraining the model.

### Key Outputs
- `eval_final_report.json`: Classification metrics (Accuracy, F1, Precision, Recall).
- `eval_confusion_matrix.png`: Visual heatmap of where the model is making classification mistakes.
- `missed_severe_cases.csv`: A strict log of every single instance where a "Severe" flood occurred but the model missed it.
- `critical_errors.csv`: High-priority errors where a Severe flood was confidently predicted as "Low".
- `district_generalization.csv`: Stress-testing the model's recall across different geographic regions to prevent bias.
- `evaluation_summary.json`: The final PASS/FAIL verdict based on criteria (Severe Recall >= 0.70 & Macro F1 >= 0.65).

### The ML Feedback Loop
If the Evaluation Engineer determines the model **NEEDS IMPROVEMENT** or is a **CONDITIONAL PASS**, the model is NOT integrated. Instead, it triggers a feedback loop:
1. **Evaluation Engine** generates the missed severe reports.
2. **Feedback to ML Engineer:** The ML Engineer inspects `missed_severe_cases.csv` to find patterns (e.g., "The model struggles in Nashik during June").
3. **Model Improvement:** The ML Engineer tweaks hyperparameters, adds new features, or cleans data.
4. **Retrain & Re-Evaluate:** The ML Engineer outputs a new model, and the Evaluation Engineer runs the audit again.

### Handoff to Integration
Once the model passes the evaluation thresholds (PASS), the Integration Engineer (`05_integration_engineer.py`) wraps the `.joblib` model into a clean API that the Flask dashboard can easily query without duplicating any feature engineering logic.
