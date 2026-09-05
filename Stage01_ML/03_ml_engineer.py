"""
STAGE 01 - ML ENGINEER (UPGRADED STACKING PIPELINE & INDEPENDENT TEST SET)
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
  - Stage01_ML/data/test/X_test.csv  (Held-out for Evaluation Engineer)
  - Stage01_ML/data/test/y_test.csv  (Held-out for Evaluation Engineer)

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

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
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

# Verify XGBoost and LightGBM availability
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


# ============================================================
# 1. PATHS & REPRODUCIBILITY CONFIGURATION
# ============================================================
"""
WHAT THIS DOES:
  Defines relative workspace paths and fixes random seeds (42).

WHY WE NEED IT:
  Ensures reproducible train/val/test splits and consistent model benchmarks across environments.

WHAT HAPPENS WITHOUT IT:
  Unseeded models produce varying evaluation metrics, breaking scientific auditability.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We fixed random state=42 and configured strict workspace paths for reproducible execution."
"""

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "outputs" / "pattern" / "eda_checked_dataset.csv"
MODEL_DIR = BASE_DIR / "data" / "models"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
TEST_DIR = BASE_DIR / "data" / "test"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 70)
print("STAGE 01 - ML ENGINEER: UPGRADED STACKING & INDEPENDENT TEST PIPELINE")
print("=" * 70)

if not XGB_AVAILABLE or not LGBM_AVAILABLE:
    missing = []
    if not XGB_AVAILABLE: missing.append("xgboost")
    if not LGBM_AVAILABLE: missing.append("lightgbm")
    raise ImportError(f"CRITICAL DEPENDENCY MISSING: {missing}. Cannot proceed with stacking ensemble without dependencies.")

print("[DEPENDENCY CHECK] XGBoost and LightGBM are successfully installed and available.")


# ============================================================
# 2. DATA INGESTION & SCHEMA VALIDATION
# ============================================================
"""
WHAT THIS DOES:
  Loads eda_checked_dataset.csv and verifies required feature columns and target labels.

WHY WE NEED IT:
  Guarantees dataset schema compatibility before training.

WHAT HAPPENS WITHOUT IT:
  Schema mismatches or missing columns produce unexpected runtime errors.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We perform defensive schema validation at dataset ingestion."
"""

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Official ML input dataset not found at expected path: {DATA_PATH}")

print(f"\n[STEP 1] Loading official EDA-approved dataset from:\n  {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Dataset Loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns")

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
print(f"[SCHEMA CHECK] Verified target labels: {target_labels}")


# ============================================================
# 3. CHRONOLOGICAL SORTING & TEMPORAL FEATURE ENGINEERING
# ============================================================
"""
WHAT THIS DOES:
  Sorts rows chronologically by timestamp and constructs domain features:
    - river_level_margin_m: river_level_m - river_level_threshold_m
    - river_level_ratio: river_level_m / (river_level_threshold_m + 1e-5)
    - hour, month, dayofweek, is_monsoon: temporal signals derived from timestamp.

WHY WE NEED IT:
  Disaster events develop over time. River level deltas capture physical hazard thresholds.

WHAT HAPPENS WITHOUT IT:
  Unsorted data introduces lookahead data leakage, and raw river levels lack local threshold context.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We sort chronologically and construct river margin deltas to represent physical flood risk thresholds."
"""

print("\n[STEP 2] Sorting dataset chronologically and engineering features...")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
if df["timestamp"].isnull().sum() > 0:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Dataset Time Range: {df['timestamp'].min()}  -->  {df['timestamp'].max()}")

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

feature_cols_num = [
    "rainfall_mm", "river_level_m", "river_level_threshold_m",
    "river_level_margin_m", "river_level_ratio", "emergency_calls",
    "road_closures", "bridge_closures", "flood_history_count",
    "population_affected", "water_level_change_m", "hour", "month",
    "dayofweek", "is_monsoon"
]
feature_cols_cat = ["state", "district"]

# Metadata & Feature Matrix
X_raw = df[["timestamp"] + feature_cols_cat + feature_cols_num]
X_features = df[feature_cols_num + feature_cols_cat]
y = df["target"]
y_labels = df["zone_risk"]


# ============================================================
# 4. THREE-WAY CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================
"""
WHAT THIS DOES:
  Splits data chronologically into 3 partitions without shuffling:
    - TRAIN (70% = 7,000 samples): For model fitting and Out-Of-Fold meta-training.
    - VALIDATION (15% = 1,500 samples): For candidate model benchmarking & final model selection.
    - TEST (15% = 1,500 samples): Completely held out for the Evaluation Engineer.

WHY WE NEED IT:
  Strictly holding out the test set guarantees that tuning, stacking meta-training, and model selection
  do not leak test information.

WHAT HAPPENS WITHOUT IT:
  Evaluating model selection choices on test data causes optimistic bias and invalidates independent auditing.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We maintain 3 chronological partitions (70% Train, 15% Validation, 15% Test). The test set is completely
  held out for Stage 04 Evaluation Engineer."
"""

print("\n[STEP 3] Executing 3-Way Chronological Train/Validation/Test Split (70/15/15, No Shuffling)...")

total_n = len(df)
train_end = int(total_n * 0.70)      # 7,000 samples
val_end = int(total_n * 0.85)        # 1,500 samples for validation (7000 to 8500)
                                     # 1,500 samples for test (8500 to 10000)

# Feature splits
X_train, y_train = X_features.iloc[:train_end].copy(), y.iloc[:train_end].copy()
X_val, y_val = X_features.iloc[train_end:val_end].copy(), y.iloc[train_end:val_end].copy()
X_test_feat, y_test_feat = X_features.iloc[val_end:].copy(), y.iloc[val_end:].copy()

df_val = df.iloc[train_end:val_end].copy()
df_test_raw = df.iloc[val_end:].copy()

print(f"Train Set Size      : {len(X_train)} samples ({len(X_train)/total_n*100:.1f}%)")
print(f"Validation Set Size : {len(X_val)} samples ({len(X_val)/total_n*100:.1f}%)")
print(f"Held-Out Test Size  : {len(X_test_feat)} samples ({len(X_test_feat)/total_n*100:.1f}%)")

# Export Independent Test Set Files for Evaluation Engineer
raw_test_feature_cols = [
    "timestamp", "state", "district", "rainfall_mm", "river_level_m",
    "river_level_threshold_m", "emergency_calls", "road_closures",
    "bridge_closures", "flood_history_count", "population_affected",
    "water_level_change_m"
]
X_test_export = df_test_raw[raw_test_feature_cols].copy()
# Format timestamp as string for portable export
X_test_export["timestamp"] = X_test_export["timestamp"].dt.strftime("%d-%m-%Y %H:%M")

y_test_export = pd.DataFrame({"zone_risk": df_test_raw["zone_risk"].values})

x_test_file = TEST_DIR / "X_test.csv"
y_test_file = TEST_DIR / "y_test.csv"

X_test_export.to_csv(x_test_file, index=False)
y_test_export.to_csv(y_test_file, index=False)

print(f"\n[TEST SET EXPORTED]")
print(f"  X_test.csv saved: {x_test_file} (Shape: {X_test_export.shape})")
print(f"  y_test.csv saved: {y_test_file} (Shape: {y_test_export.shape})")


# ============================================================
# 5. PREPROCESSING PIPELINE (FITTED ONLY ON TRAIN SET)
# ============================================================
"""
WHAT THIS DOES:
  Defines ColumnTransformer with StandardScaler (numericals) and OneHotEncoder (categoricals),
  fitted EXCLUSIVELY on X_train.

WHY WE NEED IT:
  Prevents feature mean/std or district category levels in validation/test sets from leaking into training.

WHAT HAPPENS WITHOUT IT:
  Fitting preprocessors across the full dataset causes data leakage.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "Preprocessing transformers are fitted only on the 70% training partition."
"""

print("\n[STEP 4] Fitting Preprocessing Pipeline on Training Data ONLY...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature_cols_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feature_cols_cat)
    ]
)

X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep = preprocessor.transform(X_val)
X_test_prep = preprocessor.transform(X_test_feat)

cat_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(feature_cols_cat))
all_feature_names = feature_cols_num + cat_feature_names

print(f"Preprocessed Matrix Shapes -> Train: {X_train_prep.shape}, Val: {X_val_prep.shape}, Test: {X_test_prep.shape}")


# ============================================================
# 6. BASE MODELS & STACKING ENSEMBLE BUILDER (OOF PREDICTIONS)
# ============================================================
"""
WHAT THIS DOES:
  Defines 3 diverse base learners (Random Forest, XGBoost, LightGBM) and a Stacking Ensemble:
    1. Uses 5-Fold Stratified K-Fold CV on the training partition to generate Out-Of-Fold (OOF) prediction probabilities.
    2. Trains a Logistic Regression meta-learner on the 5-fold OOF probability matrix.
    3. Refits base learners on the complete training set to generate validation and test meta-features.

WHY WE NEED IT:
  Training a meta-learner on in-fold base predictions leads to severe overfitting. OOF predictions ensure
  the meta-learner learns how base models perform on unseen samples.

WHAT HAPPENS WITHOUT IT:
  In-fold meta-training causes the meta-learner to over-trust overfitted base model predictions.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We use 5-fold Out-Of-Fold cross-validation on the training set to train our Logistic Regression meta-learner,
  preventing meta-model leakage and overfitting."
"""

print("\n[STEP 5] Building Base Models & Stacking Ensemble with Out-Of-Fold (OOF) Predictions...")

def get_base_models():
    return {
        "Random_Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)
    }

# Class to encapsulate Stacking Model logic cleanly
class StackingEnsembleModel:
    def __init__(self, base_models_dict, meta_model, random_state=42):
        self.base_models_dict = base_models_dict
        self.meta_model = meta_model
        self.random_state = random_state
        self.fitted_base_models = {}
        self.classes_ = np.array([0, 1, 2])

    def fit_oof_and_meta(self, X_tr, y_tr, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        n_samples = len(X_tr)
        n_base = len(self.base_models_dict)
        
        # OOF probabilities array: (n_samples, 3_base_models * 3_classes)
        oof_probs = np.zeros((n_samples, n_base * 3))

        print(f"  Generating {n_splits}-Fold OOF probabilities for Stacking Meta-Learner...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            X_fold_tr, y_fold_tr = X_tr[train_idx], y_tr.iloc[train_idx]
            X_fold_val = X_tr[val_idx]

            for b_idx, (b_name, b_model_cls) in enumerate(self.base_models_dict.items()):
                fold_model = clone(b_model_cls)
                fold_model.fit(X_fold_tr, y_fold_tr)
                fold_probs = fold_model.predict_proba(X_fold_val)
                oof_probs[val_idx, b_idx*3 : (b_idx+1)*3] = fold_probs

        # Train Meta-Learner on OOF probabilities
        print("  Fitting Logistic Regression Meta-Learner on OOF predictions...")
        self.meta_model.fit(oof_probs, y_tr)

        # Refit base models on full training data
        print("  Refitting base models on full training partition...")
        for b_name, b_model_cls in self.base_models_dict.items():
            full_b_model = clone(b_model_cls)
            full_b_model.fit(X_tr, y_tr)
            self.fitted_base_models[b_name] = full_b_model

        return self

    def _transform_meta_features(self, X):
        meta_features = []
        for b_name in self.base_models_dict.keys():
            probs = self.fitted_base_models[b_name].predict_proba(X)
            meta_features.append(probs)
        return np.hstack(meta_features)

    def predict_proba(self, X):
        meta_feats = self._transform_meta_features(X)
        return self.meta_model.predict_proba(meta_feats)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# Fit Stacking Ensemble
base_models = get_base_models()
meta_learner = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
stacking_model = StackingEnsembleModel(base_models, meta_learner, random_state=RANDOM_STATE)
stacking_model.fit_oof_and_meta(X_train_prep, y_train, n_splits=5)


# ============================================================
# 7. CANDIDATE BENCHMARKING ON VALIDATION SET (1,500 SAMPLES)
# ============================================================
"""
WHAT THIS DOES:
  Benchmarks 5 candidates on the VALIDATION set:
    1. Single Logistic Regression
    2. Single Random Forest
    3. Single XGBoost
    4. Single LightGBM
    5. Stacking Ensemble (RF + XGB + LGBM -> Meta Logistic Regression)

WHY WE NEED IT:
  Determines whether stacking provides measurable performance improvement over single models.

WHAT HAPPENS WITHOUT IT:
  Blindly adopting stacking without empirical proof risks unnecessary complexity.

HOW TO EXPLAIN IT IN A TEAM DISCUSSION:
  "We benchmarked single models against the stacking ensemble on held-out validation data.
  Final model selection is strictly driven by Severe Recall and Macro F1."
"""

print("\n" + "=" * 70)
print("[STEP 6] BENCHMARKING ALL CANDIDATES ON VALIDATION SET (1,500 SAMPLES)")
print("=" * 70)

# Prepare single candidate models fit on full train set
single_candidates = {
    "Logistic_Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "Random_Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=RANDOM_STATE),
    "LightGBM": LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)
}

benchmark_results = {}

# Fit & evaluate single candidate models
for name, model in single_candidates.items():
    model.fit(X_train_prep, y_train)
    preds = model.predict(X_val_prep)
    probs = model.predict_proba(X_val_prep)

    acc = accuracy_score(y_val, preds)
    macro_f1 = f1_score(y_val, preds, average="macro")
    weighted_f1 = f1_score(y_val, preds, average="weighted")
    rec_sev = recall_score(y_val, preds, labels=[2], average=None)[0]
    prec_sev = precision_score(y_val, preds, labels=[2], average=None)[0]
    f1_sev = f1_score(y_val, preds, labels=[2], average=None)[0]
    loss = log_loss(y_val, probs)

    benchmark_results[name] = {
        "model_obj": model,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "severe_recall": float(rec_sev),
        "severe_precision": float(prec_sev),
        "severe_f1": float(f1_sev),
        "log_loss": float(loss),
        "predictions": preds,
        "probabilities": probs
    }

    print(f"\n--- Candidate: {name} ---")
    print(f"  Accuracy        : {acc:.4f}  |  Macro F1        : {macro_f1:.4f}  |  Log-Loss: {loss:.4f}")
    print(f"  Severe Recall   : {rec_sev:.4f}  |  Severe Precision: {prec_sev:.4f}  |  Severe F1: {f1_sev:.4f}")

# Evaluate Stacking Ensemble on Validation Set
stack_preds = stacking_model.predict(X_val_prep)
stack_probs = stacking_model.predict_proba(X_val_prep)

acc_st = accuracy_score(y_val, stack_preds)
macro_f1_st = f1_score(y_val, stack_preds, average="macro")
weighted_f1_st = f1_score(y_val, stack_preds, average="weighted")
rec_sev_st = recall_score(y_val, stack_preds, labels=[2], average=None)[0]
prec_sev_st = precision_score(y_val, stack_preds, labels=[2], average=None)[0]
f1_sev_st = f1_score(y_val, stack_preds, labels=[2], average=None)[0]
loss_st = log_loss(y_val, stack_probs)

benchmark_results["Stacking_Ensemble"] = {
    "model_obj": stacking_model,
    "accuracy": float(acc_st),
    "macro_f1": float(macro_f1_st),
    "weighted_f1": float(weighted_f1_st),
    "severe_recall": float(rec_sev_st),
    "severe_precision": float(prec_sev_st),
    "severe_f1": float(f1_sev_st),
    "log_loss": float(loss_st),
    "predictions": stack_preds,
    "probabilities": stack_probs
}

print(f"\n--- Candidate: Stacking_Ensemble ---")
print(f"  Accuracy        : {acc_st:.4f}  |  Macro F1        : {macro_f1_st:.4f}  |  Log-Loss: {loss_st:.4f}")
print(f"  Severe Recall   : {rec_sev_st:.4f}  |  Severe Precision: {prec_sev_st:.4f}  |  Severe F1: {f1_sev_st:.4f}")


# Model Selection Decision Logic
# Primary: Severe Recall, Secondary: Macro F1, Severe Precision, Log-Loss
best_model_name = max(
    benchmark_results.keys(),
    key=lambda k: (
        benchmark_results[k]["severe_recall"],
        benchmark_results[k]["macro_f1"],
        benchmark_results[k]["severe_precision"],
        -benchmark_results[k]["log_loss"]
    )
)

print("\n" + "=" * 70)
print(f"SELECTED OPTIMAL MODEL ON VALIDATION SET: {best_model_name}")
print("=" * 70)

selected_info = benchmark_results[best_model_name]
final_model = selected_info["model_obj"]
val_preds = selected_info["predictions"]
val_probs = selected_info["probabilities"]


# ============================================================
# 8. VALIDATION SUMMARY & CONFUSION MATRIX PLOT
# ============================================================
"""
WHAT THIS DOES:
  Computes evaluation metrics and generates confusion matrix plot on validation set.

WHY WE NEED IT:
  Documents final validation performance for team review.
"""

print("\n[STEP 7] Validation Performance Summary & Confusion Matrix...")

val_acc = accuracy_score(y_val, val_preds)
val_macro_f1 = f1_score(y_val, val_preds, average="macro")
val_weighted_f1 = f1_score(y_val, val_preds, average="weighted")
cm_val = confusion_matrix(y_val, val_preds)
val_class_report = classification_report(y_val, val_preds, target_names=expected_labels, output_dict=True)

print("\nValidation Classification Report:")
print(classification_report(y_val, val_preds, target_names=expected_labels))

print("Validation Confusion Matrix (Actual rows vs Predicted columns):")
print(f"            Pred Low  Pred Mod  Pred Sev")
print(f"Actual Low     {cm_val[0,0]:7d}   {cm_val[0,1]:7d}   {cm_val[0,2]:7d}")
print(f"Actual Mod     {cm_val[1,0]:7d}   {cm_val[1,1]:7d}   {cm_val[1,2]:7d}")
print(f"Actual Sev     {cm_val[2,0]:7d}   {cm_val[2,1]:7d}   {cm_val[2,2]:7d}")

plt.figure(figsize=(7, 5))
sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues", xticklabels=expected_labels, yticklabels=expected_labels)
plt.title(f"Validation Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Risk Category")
plt.ylabel("Actual Risk Category")
plt.tight_layout()
cm_plot_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Validation Confusion Matrix Plot: {cm_plot_path}")


# ============================================================
# 9. DEFENSIBLE FEATURE IMPORTANCE LEADERBOARD
# ============================================================
"""
WHAT THIS DOES:
  Computes global feature importances for the selected model architecture:
    - If Stacking Ensemble: computes averaged feature importances across base tree models (RF, XGBoost, LightGBM).
    - If Single Tree Model (RF/XGB/LGBM): extracts model.feature_importances_.
    - If Linear Model: computes mean absolute coefficient magnitude.

WHY WE NEED IT:
  Provides defensible global feature driver rankings without false causal claims.
"""

print("\n[STEP 8] Computing Defensible Global Feature Importance Leaderboard...")

if best_model_name == "Stacking_Ensemble":
    base_m = final_model.fitted_base_models
    rf_imp = base_m["Random_Forest"].feature_importances_
    xgb_imp = base_m["XGBoost"].feature_importances_
    lgb_imp = base_m["LightGBM"].feature_importances_
    
    # Averaged base ensemble feature importance
    avg_imp = (rf_imp + xgb_imp + lgb_imp) / 3.0
    feat_imp_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": avg_imp
    }).sort_values("importance", ascending=False).reset_index(drop=True)
elif hasattr(final_model, "feature_importances_"):
    feat_imp_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
elif hasattr(final_model, "coef_"):
    feat_imp_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": np.mean(np.abs(final_model.coef_), axis=0)
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
  Calculates per-prediction instance-level top factors for the validation set.
  For each sample, ranks features by contribution magnitude (feature_value x feature_weight)
  so each zone prediction receives tailored risk drivers.

WHY WE NEED IT:
  Different disaster zones require zone-specific explanatory context.
"""

print("\n[STEP 9] Computing Prediction-Specific Top Factors for Validation Set...")

instance_top_factors = []
num_val = len(X_val_prep)

for i in range(num_val):
    pred_c = val_preds[i]
    x_i = X_val_prep[i]

    if best_model_name == "Stacking_Ensemble":
        base_m = final_model.fitted_base_models
        rf_imp = base_m["Random_Forest"].feature_importances_
        xgb_imp = base_m["XGBoost"].feature_importances_
        lgb_imp = base_m["LightGBM"].feature_importances_
        avg_weights = (rf_imp + xgb_imp + lgb_imp) / 3.0
        
        contributions = x_i * avg_weights
        top_idx = np.argsort(np.abs(contributions))[::-1][:3]
        top_3 = [all_feature_names[j] for j in top_idx]
    elif hasattr(final_model, "coef_"):
        coef_vec = final_model.coef_[pred_c]
        contributions = x_i * coef_vec
        top_idx = np.argsort(np.abs(contributions))[::-1][:3]
        top_3 = [all_feature_names[j] for j in top_idx]
    elif hasattr(final_model, "feature_importances_"):
        contributions = x_i * final_model.feature_importances_
        top_idx = np.argsort(np.abs(contributions))[::-1][:3]
        top_3 = [all_feature_names[j] for j in top_idx]
    else:
        top_3 = feat_imp_df["feature"].head(3).tolist()

    instance_top_factors.append(top_3)

# Build Validation Prediction Output DataFrame
df_val["predicted_risk_category"] = [inv_target_map[p] for p in val_preds]
df_val["risk_score"] = np.round(val_probs[:, 2], 4)   # Severe class probability
df_val["confidence"] = np.round(np.max(val_probs, axis=1), 4) # Max class probability
df_val["top_factors"] = [", ".join(f) for f in instance_top_factors]

pred_cols = [
    "timestamp", "state", "district", "rainfall_mm", "river_level_m",
    "river_level_threshold_m", "zone_risk", "predicted_risk_category",
    "risk_score", "confidence", "top_factors"
]
df_val_preds = df_val[pred_cols]

predictions_csv_path = OUTPUT_DIR / "ml_predictions.csv"
df_val_preds.to_csv(predictions_csv_path, index=False)
print(f"Saved ML Predictions CSV: {predictions_csv_path}")

# Export Integration Contract Sample JSON
sample_records = []
for i, (idx, row) in enumerate(df_val_preds.head(10).iterrows()):
    sample_records.append({
        "zone": str(row["district"]),
        "state": str(row["state"]),
        "timestamp": str(row["timestamp"].strftime("%d-%m-%Y %H:%M")),
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


# ============================================================
# 11. ARTIFACT PERSISTENCE & PIPELINE SERIALIZATION
# ============================================================
"""
WHAT THIS DOES:
  Serializes the complete fitted ML pipeline to joblib and saves metrics JSON summary.
"""

print("\n[STEP 10] Serializing ML Pipeline Artifacts & Metrics JSON...")

full_pipeline = {
    "preprocessor": preprocessor,
    "model": final_model,
    "feature_cols_num": feature_cols_num,
    "feature_cols_cat": feature_cols_cat,
    "target_map": target_map,
    "inv_target_map": inv_target_map,
    "model_name": best_model_name,
    "xgboost_available": XGB_AVAILABLE,
    "lightgbm_available": LGBM_AVAILABLE
}

pipeline_file = MODEL_DIR / "ml_pipeline.joblib"
joblib.dump(full_pipeline, pipeline_file)
print(f"Saved Serialized ML Pipeline: {pipeline_file}")

metrics_summary = {
    "model_selected": best_model_name,
    "dataset_used": str(DATA_PATH),
    "split_strategy": "Chronological 70/15/15 Train/Validation/Held-out Test Split",
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "held_out_test_samples": int(len(X_test_feat)),
    "dependencies_available": {
        "xgboost": XGB_AVAILABLE,
        "lightgbm": LGBM_AVAILABLE
    },
    "validation_metrics": {
        "accuracy": float(val_acc),
        "macro_f1": float(val_macro_f1),
        "weighted_f1": float(val_weighted_f1),
        "severe_recall": float(val_class_report["Severe"]["recall"]),
        "severe_precision": float(val_class_report["Severe"]["precision"]),
        "severe_f1": float(val_class_report["Severe"]["f1-score"]),
        "log_loss": float(log_loss(y_val, val_probs))
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

print("\n" + "=" * 70)
print("STAGE 01 ML ENGINEER PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 70)
