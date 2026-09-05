"""
STAGE 01 - EVALUATION ENGINEER
Evaluates the trained ML model for disaster risk prediction, focusing on Severe recall, 
overconfidence, and generalization to unseen disasters.
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

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. PATHS & CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
TEST_X_PATH = BASE_DIR / "data" / "test" / "X_test.csv"
TEST_Y_PATH = BASE_DIR / "data" / "test" / "y_test.csv"
TRAIN_DATA_PATH = BASE_DIR / "data" / "outputs" / "pattern" / "eda_checked_dataset.csv"
MODEL_PATH = BASE_DIR / "data" / "models" / "ml_pipeline.joblib"
METRICS_PATH = BASE_DIR / "data" / "outputs" / "ml_evaluation_metrics.json"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STAGE 01 - EVALUATION ENGINEER: MODEL EVALUATION & STRESS TESTING")
print("=" * 70)

# ============================================================
# 2. DATA LEAKAGE CHECK
# ============================================================
def check_data_leakage(test_df):
    """
    Checks for temporal data leakage.
    Ensures that test data is strictly held out and chronologically after training data.
    """
    print("\n[STEP 1] Performing Data Leakage Checks...")
    
    # Load the original full data used for training
    if not TRAIN_DATA_PATH.exists():
        print("WARNING: Original dataset not found. Cannot perform rigorous temporal leakage check.")
        return False
        
    full_df = pd.read_csv(TRAIN_DATA_PATH)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    if full_df["timestamp"].isnull().sum() > 0:
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], errors="coerce")
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)
    
    # Identify split index from ML Engineer (85% index)
    total_n = len(full_df)
    val_end = int(total_n * 0.85)
    
    train_val_timestamps = full_df["timestamp"].iloc[:val_end]
    
    # Test timestamps
    test_timestamps = pd.to_datetime(test_df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    if test_timestamps.isnull().sum() > 0:
        test_timestamps = pd.to_datetime(test_df["timestamp"], errors="coerce")
        
    max_train_val_time = train_val_timestamps.max()
    min_test_time = test_timestamps.min()
    
    print(f"  Max Train/Val Timestamp: {max_train_val_time}")
    print(f"  Min Test Timestamp     : {min_test_time}")
    
    if min_test_time >= max_train_val_time:
        print("  [PASS] Temporal Data Leakage Check: Test set strictly follows Train/Val set in time.")
        return True
    else:
        print("  [FAIL] Temporal Data Leakage Check: Overlapping timestamps detected between Train/Val and Test!")
        return False


# ============================================================
# 3. MODEL & DATA LOADING
# ============================================================
def load_assets():
    if not TEST_X_PATH.exists() or not TEST_Y_PATH.exists():
        raise FileNotFoundError(f"Independent test sets not found in {TEST_X_PATH.parent}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model pipeline not found: {MODEL_PATH}")
        
    print(f"\n[STEP 2] Loading official independent test set...")
    X_test_raw = pd.read_csv(TEST_X_PATH)
    y_test_raw = pd.read_csv(TEST_Y_PATH)
    print(f"Loaded X_test shape: {X_test_raw.shape}, y_test shape: {y_test_raw.shape}")
    
    print(f"[STEP 2] Loading model pipeline from {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    
    return X_test_raw, y_test_raw, pipeline

# ============================================================
# 4. MODEL COMPARISON (FROM ML ENGINEER)
# ============================================================
def display_model_comparison():
    print("\n[STEP 3] ML Engineer Validation Model Comparison...")
    if not METRICS_PATH.exists():
        print(f"Metrics file not found at {METRICS_PATH}. Skipping comparison.")
        return
        
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        
    benchmarks = metrics.get("candidate_benchmarks", {})
    if not benchmarks:
        print("No candidate benchmarks found in metrics.")
        return
        
    print(f"{'Model':<20} | {'Macro F1':<10} | {'Severe Recall':<15} | {'Log Loss':<10}")
    print("-" * 65)
    for model_name, res in benchmarks.items():
        print(f"{model_name:<20} | {res['macro_f1']:<10.4f} | {res['severe_recall']:<15.4f} | {res['log_loss']:<10.4f}")
        
    print(f"\nModel Selected by ML Engineer: {metrics.get('model_selected', 'Unknown')}")

# ============================================================
# 5. TEST DATA PREPARATION
# ============================================================
def prepare_test_data(X_test_raw, y_test_raw, pipeline):
    """
    Applies the domain feature engineering exactly as expected by the ML pipeline.
    Does NOT refit any preprocessing transformers.
    """
    print("\n[STEP 4] Engineering domain features on independent test set...")
    df = X_test_raw.copy()
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    if df["timestamp"].isnull().sum() > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
    df["river_level_margin_m"] = df["river_level_m"] - df["river_level_threshold_m"]
    df["river_level_ratio"] = df["river_level_m"] / (df["river_level_threshold_m"] + 1e-5)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    
    target_map = pipeline["target_map"]
    y_test = y_test_raw["zone_risk"].map(target_map)
    
    feature_cols = pipeline["feature_cols_num"] + pipeline["feature_cols_cat"]
    X_test_feat = df[feature_cols].copy()
    
    return X_test_feat, y_test, df

# ============================================================
# 6. PREDICTION GENERATION
# ============================================================
def generate_predictions(X_test, pipeline):
    print("\n[STEP 5] Generating predictions using the official model pipeline...")
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]
    
    X_test_prep = preprocessor.transform(X_test)
    preds = model.predict(X_test_prep)
    
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_prep)
        
    return preds, probs

# ============================================================
# 7. METRICS EVALUATION
# ============================================================
def evaluate_metrics(y_test, preds, pipeline):
    print("\n[STEP 6] Evaluating Classification Metrics on Independent Test Set...")
    
    inv_target_map = pipeline["inv_target_map"]
    labels = [0, 1, 2]
    target_names = [inv_target_map[0], inv_target_map[1], inv_target_map[2]]
    
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    
    report_dict = classification_report(y_test, preds, target_names=target_names, output_dict=True)
    
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Overall Macro F1 : {macro_f1:.4f}")
    print(f"Severe Recall    : {report_dict['Severe']['recall']:.4f}")
    
    print("\nTest Classification Report:")
    print(classification_report(y_test, preds, target_names=target_names))
    
    cm = confusion_matrix(y_test, preds, labels=labels)
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted Risk")
    plt.ylabel("Actual Risk")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "eval_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Confusion Matrix Plot: {cm_path}")
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }

# ============================================================
# 8. OVERCONFIDENCE & DANGEROUS ERRORS ANALYSIS
# ============================================================
def analyze_overconfidence(df_test, y_test, preds, probs, pipeline):
    """
    Identifies high-confidence false negatives, particularly where Actual=Severe but Pred=Low.
    """
    print("\n[STEP 7] Conducting Overconfidence & Severe Error Analysis...")
    
    if probs is None:
        print("Model does not provide probabilities. Overconfidence analysis is unavailable.")
        return None
        
    target_map = pipeline["target_map"]
    inv_target_map = pipeline["inv_target_map"]
    severe_idx = target_map["Severe"]
    
    confidence = np.max(probs, axis=1)
    
    analysis_df = df_test.copy()
    analysis_df["actual"] = y_test.map(inv_target_map).values
    analysis_df["predicted"] = pd.Series(preds).map(inv_target_map).values
    analysis_df["confidence"] = confidence
    analysis_df["severe_prob"] = probs[:, severe_idx]
    
    # Find missed Severe cases
    missed_severe = analysis_df[(analysis_df["actual"] == "Severe") & (analysis_df["predicted"] != "Severe")]
    
    # Critical Errors: Actual=Severe, Predicted=Low with High Confidence (>0.80)
    critical_errors = analysis_df[
        (analysis_df["actual"] == "Severe") & 
        (analysis_df["predicted"] == "Low") & 
        (analysis_df["confidence"] >= 0.80)
    ]
    
    print(f"Total Missed Severe Predictions: {len(missed_severe)}")
    print(f"Critical Overconfident Errors (Actual Severe -> Pred Low, Conf >= 80%): {len(critical_errors)}")
    
    if len(critical_errors) > 0:
        print("\nWARNING: Critical Overconfident Errors Detected!")
        print(critical_errors[["timestamp", "district", "actual", "predicted", "confidence", "severe_prob"]].head())
        
    return {
        "missed_severe_count": len(missed_severe),
        "critical_overconfident_errors": len(critical_errors)
    }

# ============================================================
# 9. UNSEEN DISASTER / STRESS TEST
# ============================================================
def stress_test_unseen_disasters(df_test, y_test, preds, pipeline):
    """
    Tests generalization by evaluating on specific unseen groups in the test set.
    """
    print("\n[STEP 8] Performing Unseen Disaster / Generalization Stress Test...")
    
    inv_target_map = pipeline["inv_target_map"]
    severe_idx = pipeline["target_map"]["Severe"]
    
    # Group by district in the test set
    districts = df_test["district"].unique()
    
    print(f"Testing generalization across {len(districts)} distinct districts in the test set.")
    
    generalization_results = {}
    for district in districts:
        idx = df_test["district"] == district
        y_d = y_test[idx]
        p_d = preds[idx]
        
        if len(y_d) < 10:
            continue # Skip very small samples
            
        acc = accuracy_score(y_d, p_d)
        severe_recall = recall_score(y_d, p_d, labels=[severe_idx], average=None)[0] if severe_idx in y_d.values else None
        
        generalization_results[district] = {
            "samples": len(y_d),
            "accuracy": acc,
            "severe_recall": severe_recall
        }
    
    # Find the worst performing district for Severe recall
    valid_districts = {k: v for k, v in generalization_results.items() if v["severe_recall"] is not None}
    
    if valid_districts:
        worst_district = min(valid_districts.keys(), key=lambda k: valid_districts[k]["severe_recall"])
        worst_recall = valid_districts[worst_district]["severe_recall"]
        print(f"\nGeneralization Stress Test - Weakest Region for Severe Risk:")
        print(f"District: {worst_district} | Samples: {valid_districts[worst_district]['samples']} | Severe Recall: {worst_recall:.4f}")
    else:
        print("Not enough Severe cases per district to effectively stress-test Severe recall by region.")
        
    return generalization_results

# ============================================================
# 10. EXECUTION & INTEGRATION CONTRACT SAVING
# ============================================================
def main():
    try:
        X_test_raw, y_test_raw, pipeline = load_assets()
        
        check_data_leakage(X_test_raw)
        
        display_model_comparison()
        
        X_test_feat, y_test, df_test = prepare_test_data(X_test_raw, y_test_raw, pipeline)
        
        preds, probs = generate_predictions(X_test_feat, pipeline)
        
        metrics_report = evaluate_metrics(y_test, preds, pipeline)
        
        overconfidence_report = analyze_overconfidence(df_test, y_test, preds, probs, pipeline)
        
        generalization_report = stress_test_unseen_disasters(df_test, y_test, preds, pipeline)
        
        # Integration Criteria: 
        # Pass if Severe Recall >= 0.70 and Macro F1 >= 0.65
        severe_recall = metrics_report["classification_report"]["Severe"]["recall"]
        macro_f1 = metrics_report["macro_f1"]
        passed_integration = (severe_recall >= 0.70) and (macro_f1 >= 0.65)
        
        print(f"\n[INTEGRATION CHECK] Severe Recall: {severe_recall:.4f}, Macro F1: {macro_f1:.4f}")
        if passed_integration:
            print(">>> MODEL PASSED EVALUATION. READY FOR INTEGRATION. <<<")
        else:
            print(">>> MODEL FAILED EVALUATION. DO NOT INTEGRATE. <<<")
        
        # Save evaluation report / Integration Contract
        eval_report = {
            "integration_status": {
                "ready_for_integration": passed_integration,
                "criteria": "Severe Recall >= 0.70 AND Macro F1 >= 0.65"
            },
            "metrics": metrics_report,
            "overconfidence_analysis": overconfidence_report,
            "generalization_stress_test": generalization_report
        }
        
        report_path = OUTPUT_DIR / "test_evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(eval_report, f, indent=4)
        
        print(f"\n[STEP 9] Saved Final Evaluation Report to {report_path}")
        print("\nEVALUATION COMPLETE.")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR IN EVALUATION PIPELINE: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
