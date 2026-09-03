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
DATA_PATH = BASE_DIR / "data" / "outputs" / "pattern" / "eda_checked_dataset.csv"
MODEL_PATH = BASE_DIR / "data" / "models" / "ml_pipeline.joblib"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STAGE 01 - EVALUATION ENGINEER: MODEL EVALUATION & STRESS TESTING")
print("=" * 70)

# ============================================================
# 2. MODEL & DATA LOADING
# ============================================================
def load_assets():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model pipeline not found: {MODEL_PATH}")
        
    print(f"[STEP 1] Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    print(f"[STEP 1] Loading model pipeline from {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    
    return df, pipeline

# ============================================================
# 3. TEST DATA PREPARATION
# ============================================================
def prepare_test_data(df, pipeline):
    """
    Recreates the exact preprocessing steps and chronological split
    used by the ML Engineer to extract the test set.
    """
    print("\n[STEP 2] Preparing test dataset...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    if df["timestamp"].isnull().sum() > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    df["river_level_margin_m"] = df["river_level_m"] - df["river_level_threshold_m"]
    df["river_level_ratio"] = df["river_level_m"] / (df["river_level_threshold_m"] + 1e-5)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    
    target_map = pipeline["target_map"]
    df["target"] = df["zone_risk"].map(target_map)
    
    feature_cols = pipeline["feature_cols_num"] + pipeline["feature_cols_cat"]
    X = df[feature_cols]
    y = df["target"]
    
    split_idx = int(len(df) * 0.80)
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"Extracted chronological test set: {len(X_test)} samples")
    return X_test, y_test, df_test

# ============================================================
# 4. PREDICTION GENERATION
# ============================================================
def generate_predictions(X_test, pipeline):
    print("\n[STEP 3] Generating predictions using the official model pipeline...")
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]
    
    X_test_prep = preprocessor.transform(X_test)
    preds = model.predict(X_test_prep)
    
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_prep)
        
    return preds, probs

# ============================================================
# 5. METRICS EVALUATION
# ============================================================
def evaluate_metrics(y_test, preds, pipeline):
    print("\n[STEP 4] Evaluating Classification Metrics...")
    
    inv_target_map = pipeline["inv_target_map"]
    labels = [0, 1, 2]
    target_names = [inv_target_map[0], inv_target_map[1], inv_target_map[2]]
    
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    
    report_dict = classification_report(y_test, preds, target_names=target_names, output_dict=True)
    
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Overall Macro F1 : {macro_f1:.4f}")
    
    print("\nClassification Report:")
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
# 6. OVERCONFIDENCE & DANGEROUS ERRORS ANALYSIS
# ============================================================
def analyze_overconfidence(df_test, y_test, preds, probs, pipeline):
    """
    Identifies high-confidence false negatives, particularly where Actual=Severe but Pred=Low.
    """
    print("\n[STEP 5] Conducting Overconfidence & Severe Error Analysis...")
    
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
# 7. UNSEEN DISASTER / STRESS TEST
# ============================================================
def stress_test_unseen_disasters(df_test, y_test, preds, pipeline):
    """
    Tests generalization by evaluating on specific unseen groups in the test set.
    Since we don't have explicit disaster event IDs, we'll use state/district groupings 
    to see if performance degrades significantly in certain regions.
    """
    print("\n[STEP 6] Performing Unseen Disaster / Generalization Stress Test...")
    
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
# 8. EXECUTION & SAVING
# ============================================================
def main():
    try:
        df, pipeline = load_assets()
        X_test, y_test, df_test = prepare_test_data(df, pipeline)
        
        preds, probs = generate_predictions(X_test, pipeline)
        
        metrics_report = evaluate_metrics(y_test, preds, pipeline)
        
        overconfidence_report = analyze_overconfidence(df_test, y_test, preds, probs, pipeline)
        
        generalization_report = stress_test_unseen_disasters(df_test, y_test, preds, pipeline)
        
        # Save evaluation report
        eval_report = {
            "metrics": metrics_report,
            "overconfidence_analysis": overconfidence_report,
            "generalization_stress_test": generalization_report
        }
        
        report_path = OUTPUT_DIR / "eval_final_report.json"
        with open(report_path, "w") as f:
            json.dump(eval_report, f, indent=4)
        
        print(f"\n[STEP 7] Saved Final Evaluation Report to {report_path}")
        print("\nEVALUATION COMPLETE.")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR IN EVALUATION PIPELINE: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
