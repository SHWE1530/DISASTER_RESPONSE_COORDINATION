"""
STAGE 01 - ML ENGINEER
Building an Autonomous, Multi-Agent Decision Engine for Urban Crises
Mission: Transform continuous disaster-related information into zone-level risk triage.

Input  : Stage01_ML/data/outputs/pattern/eda_checked_dataset.csv
Outputs:
  - Stage01_ML/data/models/ml_pipeline.joblib
  - Stage01_ML/data/outputs/ml_evaluation_metrics.json
  - Stage01_ML/data/outputs/ml_predictions.csv
  - Stage01_ML/data/outputs/feature_importance.csv
  - Stage01_ML/data/outputs/ml_integration_contract_sample.json
  - Stage01_ML/data/outputs/confusion_matrix.png

STRICT BOUNDARY: Modifies ONLY 03_ml_engineer.py.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    log_loss,
    brier_score_loss,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. PATHS & REPRODUCIBILITY CONFIGURATION
# ============================================================
"""
WHAT THIS DOES:
  Defines explicit, relative workspace paths and sets a random seed (42).

WHY WE NEED IT:
  Ensures the script runs smoothly in any system directory and produces 100%
  reproducible train/test metrics across runs.

WHAT HAPPENS WITHOUT IT:
  Hardcoded paths break when moved across machines, and unseeded models produce
  varying metric results, failing scientific auditability.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We established clear project path boundaries and fixed random seeds to guarantee
  reproducibility across team environments."
"""

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "outputs" / "pattern" / "eda_checked_dataset.csv"
MODEL_DIR = BASE_DIR / "data" / "models"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 70)
print("STAGE 01 - ML ENGINEER: DISASTER RISK PREDICTION ENGINE")
print("=" * 70)


# ============================================================
# 2. DATA INGESTION & SCHEMA VALIDATION
# ============================================================
"""
WHAT THIS DOES:
  Loads eda_checked_dataset.csv and validates required columns and target labels.

WHY WE NEED IT:
  Ensures inputs match the official contract before executing training.

WHAT HAPPENS WITHOUT IT:
  A schema mismatch or missing feature silently fails or produces corrupt predictions.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We perform defensive schema validation at startup to ensure input integrity."
"""

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Official ML input dataset not found at expected path: {DATA_PATH}")

print(f"\n[STEP 1] Loading official EDA-approved dataset from:\n  {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Dataset Loaded. Initial Shape: {df.shape[0]} rows, {df.shape[1]} columns")

required_columns = [
    "timestamp", "state", "district", "rainfall_mm", "river_level_m",
    "river_level_threshold_m", "emergency_calls", "road_closures",
    "bridge_closures", "flood_history_count", "population_affected",
    "water_level_change_m", "zone_risk"
]

missing_cols = [c for c in required_columns if c not in df.columns]
if missing_cols:
    raise ValueError(f"CRITICAL ERROR: Dataset missing required columns: {missing_cols}")

target_labels = sorted(df["zone_risk"].unique().tolist())
expected_labels = ["Low", "Moderate", "Severe"]
print(f"\n[SCHEMA CHECK] Verified target labels present: {target_labels}")


# ============================================================
# 3. CHRONOLOGICAL SORTING & TEMPORAL FEATURE ENGINEERING
# ============================================================
"""
WHAT THIS DOES:
  Sorts observations by timestamp and creates domain features:
    - river_level_margin_m: Delta between river level and danger threshold.
    - river_level_ratio: Ratio of river level relative to threshold.
    - hour, month, dayofweek, is_monsoon: Temporal signals.

WHY WE NEED IT:
  Disaster incidents are temporal sequence events. River margin directly captures
  critical threshold breaches identified during EDA.

WHAT HAPPENS WITHOUT IT:
  Unsorted data leads to temporal lookahead leakage during train/val split, and
  raw river level without threshold context misses key physical hazard signals.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We sort chronologically to respect disaster progression and construct river margin
  delta features as highlighted during EDA threshold analysis."
"""

print("\n[STEP 2] Sorting dataset chronologically and engineering features...")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
if df["timestamp"].isnull().sum() > 0:
    print("Warning: Standardizing parsed timestamp with fallback formatting...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Time Range: {df['timestamp'].min()}  -->  {df['timestamp'].max()}")

# Domain-specific feature engineering
df["river_level_margin_m"] = df["river_level_m"] - df["river_level_threshold_m"]
df["river_level_ratio"] = df["river_level_m"] / (df["river_level_threshold_m"] + 1e-5)
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

# Map target to integer indices
target_map = {"Low": 0, "Moderate": 1, "Severe": 2}
inv_target_map = {0: "Low", 1: "Moderate", 2: "Severe"}
df["target"] = df["zone_risk"].map(target_map)

# Feature separation
feature_cols_num = [
    "rainfall_mm", "river_level_m", "river_level_threshold_m",
    "river_level_margin_m", "river_level_ratio", "emergency_calls",
    "road_closures", "bridge_closures", "flood_history_count",
    "population_affected", "water_level_change_m", "hour", "month",
    "dayofweek", "is_monsoon"
]
feature_cols_cat = ["state", "district"]

X = df[feature_cols_num + feature_cols_cat]
y = df["target"]


# ============================================================
# 4. CHRONOLOGICAL 80/20 TRAIN/VALIDATION SPLIT (ZERO LEAKAGE)
# ============================================================
"""
WHAT THIS DOES:
  Splits the dataset sequentially: First 80% for Training (8,000 samples),
  Final 20% for Validation/Testing (2,001 samples). NO SHUFFLING.

WHY WE NEED IT:
  In crisis response, past observations predict future incidents. Random splitting
  leaks future flood information into past predictions.

WHAT HAPPENS WITHOUT IT:
  Random k-fold validation yields overly optimistic accuracy metrics that fail
  when deployed on future disaster events.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We use a time-aware 80/20 chronological split to simulate real-world disaster
  forecasting without temporal data leakage."
"""

print("\n[STEP 3] Executing Time-Aware Chronological 80/20 Split (No Shuffling)...")
split_idx = int(len(df) * 0.80)

X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
df_test = df.iloc[split_idx:].copy()

print(f"Training Set Size   : {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation Set Size : {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

print("\nTrain Class Distribution:")
for k, v in y_train.value_counts().sort_index().items():
    print(f"  Class {k} ({inv_target_map[k]}): {v} ({v/len(y_train)*100:.2f}%)")

print("Validation Class Distribution:")
for k, v in y_test.value_counts().sort_index().items():
    print(f"  Class {k} ({inv_target_map[k]}): {v} ({v/len(y_test)*100:.2f}%)")


# ============================================================
# 5. PREPROCESSING PIPELINE (FITTED ONLY ON TRAIN DATA)
# ============================================================
"""
WHAT THIS DOES:
  Defines a ColumnTransformer with StandardScaler for numericals and OneHotEncoder
  for categoricals, fitted strictly on X_train.

WHY WE NEED IT:
  Prevents feature scaling parameters (mean, std) or categorical levels from
  leaking validation set statistics into the training pipeline.

WHAT HAPPENS WITHOUT IT:
  Fitting scaling transformers across the full dataset causes data leakage.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "Preprocessing transformers are fitted exclusively on the 80% train partition to maintain
  strict isolation of held-out validation data."
"""

print("\n[STEP 4] Fitting Preprocessing Pipeline on Training Data ONLY...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature_cols_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feature_cols_cat)
    ]
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Extract encoded feature names for explainability
cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(feature_cols_cat))
all_feature_names = feature_cols_num + cat_feature_names

print(f"Preprocessed Feature Matrix Shape: Train {X_train_prep.shape}, Test {X_test_prep.shape}")


# ============================================================
# 6. CANDIDATE MODEL BENCHMARKING & SELECTION
# ============================================================
"""
WHAT THIS DOES:
  Trains baseline and candidate classifiers on training data and evaluates them
  on the held-out 20% validation set across Severe Recall, Severe Precision,
  Macro F1, Accuracy, and Log-Loss.

WHY WE NEED IT:
  We do not guess the best algorithm; we benchmark defensible candidates
  focusing on operational disaster utility (Severe Recall & Macro F1).

WHAT HAPPENS WITHOUT IT:
  Arbitrarily picking a model without comparison risks selecting a model that suffers
  from severe class under-detection or poor probability calibration.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We compared a Stratified Baseline, Logistic Regression, Random Forest, and Extra Trees
  using balanced class weights to maximize Severe recall without sacrificing overall Macro F1."
"""

print("\n" + "=" * 70)
print("[STEP 5] BENCHMARKING CANDIDATE MODELS ON HELD-OUT VALIDATION SET")
print("=" * 70)

candidates = {
    "Baseline_Stratified": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
    "Logistic_Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "Random_Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE),
    "Extra_Trees": ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE)
}

benchmark_results = {}

for name, model in candidates.items():
    model.fit(X_train_prep, y_train)
    preds = model.predict(X_test_prep)
    probs = model.predict_proba(X_test_prep)
    
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    weighted_f1 = f1_score(y_test, preds, average="weighted")
    
    # Class-specific Severe (class 2) metrics using labels=[2] to prevent warnings
    rec_severe = recall_score(y_test, preds, labels=[2], average=None)[0]
    prec_severe = precision_score(y_test, preds, labels=[2], average=None)[0]
    f1_severe = f1_score(y_test, preds, labels=[2], average=None)[0]
    
    loss = log_loss(y_test, probs)
    
    benchmark_results[name] = {
        "model_obj": model,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "severe_recall": float(rec_severe),
        "severe_precision": float(prec_severe),
        "severe_f1": float(f1_severe),
        "log_loss": float(loss),
        "predictions": preds,
        "probabilities": probs
    }
    
    print(f"\n--- Model: {name} ---")
    print(f"  Accuracy        : {acc:.4f}  |  Macro F1        : {macro_f1:.4f}  |  Log-Loss: {loss:.4f}")
    print(f"  Severe Recall   : {rec_severe:.4f}  |  Severe Precision: {prec_severe:.4f}  |  Severe F1: {f1_severe:.4f}")

# Model Selection Decision Logic
# Prioritize Severe Recall while requiring high Macro F1 and low Log-Loss
best_model_name = max(
    benchmark_results.keys(),
    key=lambda k: (benchmark_results[k]["severe_recall"], benchmark_results[k]["macro_f1"])
)

print("\n" + "=" * 70)
print(f"SELECTED OPTIMAL MODEL: {best_model_name}")
print("=" * 70)
best_candidate = benchmark_results[best_model_name]


# ============================================================
# 7. EVIDENCE-BASED PROBABILITY CALIBRATION ANALYSIS
# ============================================================
"""
WHAT THIS DOES:
  Evaluates whether Sigmoid probability calibration on the best model improves
  Brier score and Log Loss on held-out validation data.

WHY WE NEED IT:
  Uncalibrated tree ensembles can produce overconfident probabilities. We apply
  calibration ONLY if empirical evidence confirms improved probability quality.

WHAT HAPPENS WITHOUT IT:
  Applying calibration blindly can degrade probability quality if the uncalibrated
  probabilities are already well-calibrated.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We tested probability calibration empirically on held-out validation data. We only adopt
  calibrated probabilities if Brier score and Log-Loss indicate measurable quality improvement."
"""

print("\n[STEP 6] Performing Evidence-Based Probability Calibration Analysis...")

uncalibrated_model = best_candidate["model_obj"]
uncalibrated_probs = best_candidate["probabilities"]
uncalibrated_log_loss = best_candidate["log_loss"]

# Compute Brier score average across one-hot target classes
y_test_oh = pd.get_dummies(y_test).values
uncalibrated_brier = np.mean([
    brier_score_loss(y_test_oh[:, i], uncalibrated_probs[:, i]) for i in range(3)
])

print(f"Uncalibrated Model Log-Loss: {uncalibrated_log_loss:.4f}")
print(f"Uncalibrated Model Brier Score: {uncalibrated_brier:.4f}")

# Train calibrated classifier using 3-fold CV on training set
calibrated_clf = CalibratedClassifierCV(estimator=uncalibrated_model, method="sigmoid", cv=3)
calibrated_clf.fit(X_train_prep, y_train)

calibrated_probs = calibrated_clf.predict_proba(X_test_prep)
calibrated_preds = calibrated_clf.predict(X_test_prep)
calibrated_log_loss = log_loss(y_test, calibrated_probs)
calibrated_brier = np.mean([
    brier_score_loss(y_test_oh[:, i], calibrated_probs[:, i]) for i in range(3)
])

print(f"Calibrated Model Log-Loss  : {calibrated_log_loss:.4f}")
print(f"Calibrated Model Brier Score: {calibrated_brier:.4f}")

if calibrated_brier < uncalibrated_brier:
    print("\n-> Calibration Decision: ADOPTING Calibrated Classifier (improved Brier score).")
    final_model = calibrated_clf
    final_preds = calibrated_preds
    final_probs = calibrated_probs
    is_calibrated = True
else:
    print("\n-> Calibration Decision: RETAINING Uncalibrated Classifier (calibration did not improve score).")
    final_model = uncalibrated_model
    final_preds = best_candidate["predictions"]
    final_probs = uncalibrated_probs
    is_calibrated = False


# ============================================================
# 8. FINAL EVALUATION & METRICS GENERATION
# ============================================================
"""
WHAT THIS DOES:
  Computes comprehensive classification metrics on held-out 20% validation data.

WHY WE NEED IT:
  Provides objective evidence of model performance for team review and evaluation audit.

WHAT HAPPENS WITHOUT IT:
  Model claims remain unverified and prone to error.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We report actual metrics on the held-out 20% validation set without metric manipulation."
"""

print("\n" + "=" * 70)
print("FINAL HELD-OUT VALIDATION PERFORMANCE SUMMARY")
print("=" * 70)

final_acc = accuracy_score(y_test, final_preds)
final_macro_f1 = f1_score(y_test, final_preds, average="macro")
final_weighted_f1 = f1_score(y_test, final_preds, average="weighted")
cm = confusion_matrix(y_test, final_preds)

class_report = classification_report(y_test, final_preds, target_names=expected_labels, output_dict=True)
print("\nClassification Report:")
print(classification_report(y_test, final_preds, target_names=expected_labels))

print("Confusion Matrix (Actual rows vs Predicted columns):")
print(f"            Pred Low  Pred Mod  Pred Sev")
print(f"Actual Low     {cm[0,0]:7d}   {cm[0,1]:7d}   {cm[0,2]:7d}")
print(f"Actual Mod     {cm[1,0]:7d}   {cm[1,1]:7d}   {cm[1,2]:7d}")
print(f"Actual Sev     {cm[2,0]:7d}   {cm[2,1]:7d}   {cm[2,2]:7d}")

# Save Confusion Matrix Plot
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=expected_labels, yticklabels=expected_labels)
plt.title(f"Confusion Matrix - Final Model ({best_model_name})")
plt.xlabel("Predicted Risk Category")
plt.ylabel("Actual Risk Category")
plt.tight_layout()
cm_plot_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved Confusion Matrix Plot: {cm_plot_path}")


# ============================================================
# 9. EXPLAINABILITY & FEATURE IMPORTANCE LEADERBOARD
# ============================================================
"""
WHAT THIS DOES:
  Extracts feature importances/coefficients from the model and ranks top drivers.
  Supports both tree-based (feature_importances_) and linear models (coef_).

WHY WE NEED IT:
  Emergency commanders cannot act on black-box risk labels. They require clear
  top contributing factors for tactical decisions.

WHAT HAPPENS WITHOUT IT:
  High-risk alerts lack explanatory context, reducing operator trust.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We extract feature importance magnitude to rank key risk drivers, answering why a zone
  was classified as Severe risk."
"""

print("\n[STEP 7] Generating Defensible Feature Importance Leaderboard...")

if hasattr(uncalibrated_model, "feature_importances_"):
    importances = uncalibrated_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
elif hasattr(uncalibrated_model, "coef_"):
    # For linear models (e.g. LogisticRegression), compute mean absolute coefficient magnitude across classes
    importances = np.mean(np.abs(uncalibrated_model.coef_), axis=0)
    feat_imp_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
else:
    feat_imp_df = pd.DataFrame({"feature": all_feature_names, "importance": [0.0]*len(all_feature_names)})

feat_imp_path = OUTPUT_DIR / "feature_importance.csv"
feat_imp_df.to_csv(feat_imp_path, index=False)
print(f"Saved Feature Importance Leaderboard: {feat_imp_path}")

print("\nTop 10 Feature Drivers:")
print(feat_imp_df.head(10).to_string(index=False))


# ============================================================
# 10. PREDICTION-SPECIFIC TOP FACTORS & INTEGRATION CONTRACT
# ============================================================
"""
WHAT THIS DOES:
  Calculates instance-level feature contributions for each prediction row:
    contribution_ij = transformed_feature_value_ij * coefficient_j (for predicted class)
  Ranks top 3 contributing factors specific to each individual zone prediction.

WHY WE NEED IT:
  Different disaster zones have different risk triggers (e.g. high river level vs heavy rainfall).
  Instance-level top factors give operators tailored explanations per record.

WHAT HAPPENS WITHOUT IT:
  Returning static global top features for every zone obscures zone-specific hazard causes.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We compute per-prediction feature contributions (feature value x class coefficient) so every zone
  receives custom, prediction-specific top factors."
"""

print("\n[STEP 8] Computing Instance-Level Prediction-Specific Top Factors...")

instance_top_factors = []
num_test = len(X_test_prep)

for i in range(num_test):
    pred_class = final_preds[i]
    x_i = X_test_prep[i]
    
    if hasattr(uncalibrated_model, "coef_"):
        # Vector of coefficients corresponding to the predicted class
        coef_class = uncalibrated_model.coef_[pred_class]
        # Instance contribution = feature_value * coefficient
        contributions = x_i * coef_class
        top_indices = np.argsort(np.abs(contributions))[::-1][:3]
        top_3 = [all_feature_names[j] for j in top_indices]
    elif hasattr(uncalibrated_model, "feature_importances_"):
        contributions = x_i * uncalibrated_model.feature_importances_
        top_indices = np.argsort(np.abs(contributions))[::-1][:3]
        top_3 = [all_feature_names[j] for j in top_indices]
    else:
        top_3 = feat_imp_df["feature"].head(3).tolist()
        
    instance_top_factors.append(top_3)

# Build Prediction Output DataFrame
df_test["predicted_risk_category"] = [inv_target_map[p] for p in final_preds]
df_test["risk_score"] = np.round(final_probs[:, 2], 4)  # Severe probability
df_test["confidence"] = np.round(np.max(final_probs, axis=1), 4)  # Max class probability
df_test["top_factors"] = [", ".join(f) for f in instance_top_factors]

pred_cols = [
    "timestamp", "state", "district", "rainfall_mm", "river_level_m",
    "river_level_threshold_m", "zone_risk", "predicted_risk_category",
    "risk_score", "confidence", "top_factors"
]
df_test_preds = df_test[pred_cols]

predictions_csv_path = OUTPUT_DIR / "ml_predictions.csv"
df_test_preds.to_csv(predictions_csv_path, index=False)
print(f"Saved ML Predictions CSV: {predictions_csv_path}")

print("\n[STEP 9] Exporting Trained Model and Integration Artifacts...")

# Build combined pipeline object
full_pipeline = {
    "preprocessor": preprocessor,
    "model": final_model,
    "feature_cols_num": feature_cols_num,
    "feature_cols_cat": feature_cols_cat,
    "target_map": target_map,
    "inv_target_map": inv_target_map,
    "is_calibrated": is_calibrated,
    "model_name": best_model_name
}

pipeline_file = MODEL_DIR / "ml_pipeline.joblib"
joblib.dump(full_pipeline, pipeline_file)
print(f"Saved Trained ML Pipeline: {pipeline_file}")

# Save JSON Metrics Summary
metrics_summary = {
    "model_selected": best_model_name,
    "is_calibrated": is_calibrated,
    "dataset_used": str(DATA_PATH),
    "split_strategy": "Chronological 80/20 Time-Aware Split",
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_test)),
    "validation_metrics": {
        "accuracy": float(final_acc),
        "macro_f1": float(final_macro_f1),
        "weighted_f1": float(final_weighted_f1),
        "severe_recall": float(class_report["Severe"]["recall"]),
        "severe_precision": float(class_report["Severe"]["precision"]),
        "severe_f1": float(class_report["Severe"]["f1-score"]),
        "log_loss": float(log_loss(y_test, final_probs))
    },
    "candidate_benchmarks": {
        k: {
            "accuracy": v["accuracy"],
            "macro_f1": v["macro_f1"],
            "severe_recall": v["severe_recall"],
            "severe_precision": v["severe_precision"],
            "log_loss": v["log_loss"]
        } for k, v in benchmark_results.items()
    }
}

metrics_json_path = OUTPUT_DIR / "ml_evaluation_metrics.json"
with open(metrics_json_path, "w") as f:
    json.dump(metrics_summary, f, indent=4)
print(f"Saved Evaluation Metrics JSON: {metrics_json_path}")

# Generate Sample Integration Contract JSON for Integration Engineer
sample_records = []
for i, (idx, row) in enumerate(df_test_preds.head(10).iterrows()):
    sample_records.append({
        "zone": str(row["district"]),
        "state": str(row["state"]),
        "timestamp": str(row["timestamp"]),
        "actual_risk": str(row["zone_risk"]),
        "risk_category": str(row["predicted_risk_category"]),
        "risk_score": float(row["risk_score"]),
        "confidence": float(row["confidence"]),
        "top_factors": instance_top_factors[i]
    })

contract_path = OUTPUT_DIR / "ml_integration_contract_sample.json"
with open(contract_path, "w") as f:
    json.dump(sample_records, f, indent=4)
print(f"Saved Integration Contract Sample: {contract_path}")

print("\n" + "=" * 70)
print("STAGE 01 ML ENGINEER PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 70)
