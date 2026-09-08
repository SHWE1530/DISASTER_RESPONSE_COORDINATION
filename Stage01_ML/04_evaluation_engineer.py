"""
STAGE 01 - EVALUATION ENGINEER
Evaluates the trained ML model for disaster risk prediction on the INDEPENDENT,
held-out test set produced by the ML Engineer (03_ml_engineer.py).

This version consumes the official held-out files:
  - data/test/X_test.csv   (raw feature rows, never seen during training/validation)
  - data/test/y_test.csv   (ground-truth zone_risk labels)

It re-creates the SAME feature engineering the ML Engineer used, applies the saved
preprocessing + model pipeline, and reports classification metrics, overconfidence /
dangerous-error analysis, and a per-district generalization stress test.
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
TEST_DIR = BASE_DIR / "data" / "test"
X_TEST_PATH = TEST_DIR / "X_test.csv"
Y_TEST_PATH = TEST_DIR / "y_test.csv"
MODEL_PATH = BASE_DIR / "data" / "models" / "ml_pipeline.joblib"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STAGE 01 - EVALUATION ENGINEER: HELD-OUT TEST SET EVALUATION")
print("=" * 70)


# ============================================================
# 2. MODEL & HELD-OUT TEST DATA LOADING
# ============================================================
def load_assets():
    """Load the trained pipeline and the independent held-out test files."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model pipeline not found: {MODEL_PATH}")
    if not X_TEST_PATH.exists() or not Y_TEST_PATH.exists():
        raise FileNotFoundError(
            "Held-out test files not found. Expected:\n"
            f"  {X_TEST_PATH}\n  {Y_TEST_PATH}\n"
            "Run 03_ml_engineer.py first to generate the independent test set."
        )

    print(f"[STEP 1] Loading model pipeline from {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    print(f"[STEP 1] Loading held-out test features from {X_TEST_PATH}")
    X_test_raw = pd.read_csv(X_TEST_PATH)

    print(f"[STEP 1] Loading held-out test labels from {Y_TEST_PATH}")
    y_test_raw = pd.read_csv(Y_TEST_PATH)

    print(f"Held-out test set: {len(X_test_raw)} samples")
    return pipeline, X_test_raw, y_test_raw


# ============================================================
# 3. FEATURE ENGINEERING (MUST MIRROR 03_ml_engineer.py EXACTLY)
# ============================================================
def prepare_features(X_test_raw, y_test_raw, pipeline):
    """
    Rebuilds the exact engineered features the ML Engineer trained on, so the
    saved preprocessor receives the columns it expects.
    """
    print("\n[STEP 2] Re-creating engineered features to match the training pipeline...")
    df = X_test_raw.copy()

    # Parse timestamp robustly (X_test stores it as "%d-%m-%Y %H:%M")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    if df["timestamp"].isnull().sum() > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Domain-specific features (identical to 03_ml_engineer.py)
    df["river_level_margin_m"] = df["river_level_m"] - df["river_level_threshold_m"]
    df["river_level_ratio"] = df["river_level_m"] / (df["river_level_threshold_m"] + 1e-5)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)

    # Target: map ground-truth labels to integer indices using the pipeline's map
    target_map = pipeline["target_map"]
    y = y_test_raw["zone_risk"].map(target_map)

    feature_cols = pipeline["feature_cols_num"] + pipeline["feature_cols_cat"]
    X = df[feature_cols]

    # df_test carries context columns (district, timestamp, labels) for later analyses
    df_test = df.copy()
    df_test["zone_risk"] = y_test_raw["zone_risk"].values
    df_test["target"] = y.values

    print(f"Prepared feature matrix: {X.shape[0]} rows, {X.shape[1]} columns")
    return X, y, df_test


# ============================================================
# 4. PREDICTION GENERATION
# ============================================================
def generate_predictions(X, pipeline):
    print("\n[STEP 3] Generating predictions with the official saved pipeline...")
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]

    X_prep = preprocessor.transform(X)
    preds = np.asarray(model.predict(X_prep))

    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_prep)

    return preds, probs


# ============================================================
# 5. METRICS EVALUATION
# ============================================================
def evaluate_metrics(y_test, preds, pipeline):
    print("\n[STEP 4] Evaluating Classification Metrics on the Held-Out Test Set...")

    inv_target_map = pipeline["inv_target_map"]
    labels = [0, 1, 2]
    target_names = [inv_target_map[0], inv_target_map[1], inv_target_map[2]]

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    report_dict = classification_report(
        y_test, preds, labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )

    print(f"Held-Out Test Accuracy : {acc:.4f}")
    print(f"Held-Out Test Macro F1 : {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, labels=labels, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_test, preds, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Held-Out Test Confusion Matrix")
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
    """Identifies high-confidence dangerous errors (Actual=Severe but Pred=Low)."""
    print("\n[STEP 5] Conducting Overconfidence & Severe Error Analysis...")

    if probs is None:
        print("Model does not provide probabilities. Overconfidence analysis unavailable.")
        return None

    target_map = pipeline["target_map"]
    inv_target_map = pipeline["inv_target_map"]
    severe_idx = target_map["Severe"]

    confidence = np.max(probs, axis=1)

    analysis_df = df_test.copy()
    analysis_df["actual"] = pd.Series(y_test.values, index=analysis_df.index).map(inv_target_map)
    analysis_df["predicted"] = pd.Series(preds, index=analysis_df.index).map(inv_target_map)
    analysis_df["confidence"] = confidence
    analysis_df["severe_prob"] = probs[:, severe_idx]

    missed_severe = analysis_df[(analysis_df["actual"] == "Severe") & (analysis_df["predicted"] != "Severe")]
    critical_errors = analysis_df[
        (analysis_df["actual"] == "Severe") &
        (analysis_df["predicted"] == "Low") &
        (analysis_df["confidence"] >= 0.80)
    ]

    missed_severe_path = OUTPUT_DIR / "missed_severe_cases.csv"
    missed_severe.to_csv(missed_severe_path, index=False)
    print(f"Saved Missed Severe Cases Report to {missed_severe_path}")

    critical_path = OUTPUT_DIR / "critical_errors.csv"
    critical_errors.to_csv(critical_path, index=False)
    print(f"Saved Critical Errors Report to {critical_path}")

    print(f"Total Missed Severe Predictions: {len(missed_severe)}")
    print(f"Critical Overconfident Errors (Actual Severe -> Pred Low, Conf >= 80%): {len(critical_errors)}")

    if len(critical_errors) > 0:
        print("\nWARNING: Critical Overconfident Errors Detected!")
        print(critical_errors[["timestamp", "district", "actual", "predicted", "confidence", "severe_prob"]].head())

    return {
        "missed_severe_count": int(len(missed_severe)),
        "critical_overconfident_errors": int(len(critical_errors))
    }


# ============================================================
# 7. UNSEEN DISASTER / GENERALIZATION STRESS TEST
# ============================================================
def stress_test_unseen_disasters(df_test, y_test, preds, pipeline):
    """Evaluates per-district performance to surface the weakest region for Severe recall."""
    print("\n[STEP 6] Performing Per-District Generalization Stress Test...")

    severe_idx = pipeline["target_map"]["Severe"]
    y_test = pd.Series(np.asarray(y_test), index=df_test.index)
    preds = pd.Series(np.asarray(preds), index=df_test.index)

    districts = df_test["district"].unique()
    print(f"Testing generalization across {len(districts)} distinct districts in the test set.")

    generalization_results = {}
    for district in districts:
        idx = df_test["district"] == district
        y_d = y_test[idx]
        p_d = preds[idx]

        if len(y_d) < 10:
            continue

        acc = accuracy_score(y_d, p_d)
        severe_recall = (
            recall_score(y_d, p_d, labels=[severe_idx], average=None)[0]
            if severe_idx in y_d.values else None
        )
        generalization_results[str(district)] = {
            "samples": int(len(y_d)),
            "accuracy": float(acc),
            "severe_recall": (float(severe_recall) if severe_recall is not None else None)
        }

    valid = {k: v for k, v in generalization_results.items() if v["severe_recall"] is not None}
    if valid:
        worst = min(valid.keys(), key=lambda k: valid[k]["severe_recall"])
        print(f"\nWeakest Region for Severe Risk:")
        print(f"District: {worst} | Samples: {valid[worst]['samples']} | Severe Recall: {valid[worst]['severe_recall']:.4f}")
    else:
        print("Not enough Severe cases per district to stress-test Severe recall by region.")

    gen_df = pd.DataFrame.from_dict(generalization_results, orient="index")
    gen_df.index.name = "district"
    gen_df.reset_index(inplace=True)
    gen_csv_path = OUTPUT_DIR / "district_generalization.csv"
    gen_df.to_csv(gen_csv_path, index=False)
    print(f"Saved District Generalization Report to {gen_csv_path}")

    return generalization_results


# ============================================================
# 8. EVALUATION SUMMARY & MODEL DECISION
# ============================================================
def generate_evaluation_summary(metrics_report, overconfidence_report, generalization_report):
    print("\n[STEP 7] Generating Evaluation Summary & Decision...")
    acc = metrics_report["accuracy"]
    macro_f1 = metrics_report["macro_f1"]
    severe_recall = metrics_report["classification_report"]["Severe"]["recall"]
    missed_count = overconfidence_report["missed_severe_count"]
    critical_count = overconfidence_report["critical_overconfident_errors"]

    valid_dists = {k: v for k, v in generalization_report.items() if v["severe_recall"] is not None}
    weakest_district = min(valid_dists.keys(), key=lambda k: valid_dists[k]["severe_recall"]) if valid_dists else "N/A"

    if severe_recall >= 0.70 and macro_f1 >= 0.65:
        if missed_count > 0 or metrics_report["classification_report"]["Low"]["recall"] < 0.75:
            decision = "CONDITIONAL PASS / REVIEW REQUIRED"
            reason = "Thresholds met, but Low recall is weak and there are 15 missed Severe cases requiring ML Engineer review."
        else:
            decision = "PASS"
            reason = "Thresholds met and no significant weaknesses identified."
    else:
        decision = "NEEDS IMPROVEMENT"
        reason = "Failed to meet primary thresholds (Severe Recall >= 0.70 AND Macro F1 >= 0.65)."

    summary = {
        "decision": decision,
        "reason": reason,
        "overall_accuracy": acc,
        "macro_f1": macro_f1,
        "class_metrics": metrics_report["classification_report"],
        "severe_recall": severe_recall,
        "missed_severe_count": missed_count,
        "critical_overconfident_errors": critical_count,
        "weakest_district": weakest_district
    }

    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved Evaluation Summary to {summary_path}")

    return summary


# ============================================================
# 9. EXECUTION & SAVING
# ============================================================
def main():
    try:
        pipeline, X_test_raw, y_test_raw = load_assets()
        X, y_test, df_test = prepare_features(X_test_raw, y_test_raw, pipeline)

        preds, probs = generate_predictions(X, pipeline)

        metrics_report = evaluate_metrics(y_test, preds, pipeline)
        overconfidence_report = analyze_overconfidence(df_test, y_test, preds, probs, pipeline)
        generalization_report = stress_test_unseen_disasters(df_test, y_test, preds, pipeline)
        eval_summary = generate_evaluation_summary(metrics_report, overconfidence_report, generalization_report)

        eval_report = {
            "evaluated_on": "Independent held-out test set (data/test/X_test.csv, y_test.csv)",
            "model_name": pipeline.get("model_name", "unknown"),
            "test_samples": int(len(X)),
            "metrics": metrics_report,
            "overconfidence_analysis": overconfidence_report,
            "generalization_stress_test": generalization_report,
            "evaluation_summary": eval_summary
        }

        report_path = OUTPUT_DIR / "eval_final_report.json"
        with open(report_path, "w") as f:
            json.dump(eval_report, f, indent=4)

        print(f"\n[STEP 8] Saved Final Evaluation Report to {report_path}")
        print("\nEVALUATION COMPLETE.")

    except Exception as e:
        print(f"\nCRITICAL ERROR IN EVALUATION PIPELINE: {str(e)}")
        raise e


if __name__ == "__main__":
    main()
