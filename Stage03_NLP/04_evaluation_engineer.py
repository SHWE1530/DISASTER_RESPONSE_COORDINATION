"""Stage 03 NLP Evaluation Pipeline.

This module evaluates the performance, robustness, and disaster-response safety
of the NLP models trained in 03_nlp_engineer.py.
"""

import sys
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import importlib.util

# Dynamically load the nlp module
spec = importlib.util.spec_from_file_location("nlp", BASE_DIR / "03_nlp_engineer.py")
nlp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nlp)

EVAL_DIR = BASE_DIR / "data" / "outputs" / "evaluation"

CONTROLLED_DATASET = [
    {
        "id": "edge_01",
        "text": "Water has entered several houses near the railway bridge.",
        "type": "Normal emergency",
        "true_urgency": "MEDIUM",
        "true_hazard": "Flood",
        "true_loc": "railway bridge"
    },
    {
        "id": "edge_02",
        "text": "Three people are trapped on the second floor and need immediate rescue.",
        "type": "High-priority rescue",
        "true_urgency": "CRITICAL",
        "true_hazard": "Flood",  # Defaulting hazard assumption
        "true_loc": "second floor"
    },
    {
        "id": "edge_03",
        "text": "pls hlp!!! flood water everywhere near main rd",
        "type": "Noisy communication",
        "true_urgency": "HIGH",
        "true_hazard": "Flood",
        "true_loc": "main rd"
    },
    {
        "id": "edge_04",
        "text": "2 ppl trapped in flud water",
        "type": "Spelling errors",
        "true_urgency": "CRITICAL",
        "true_hazard": "Flood",
        "true_loc": None
    },
    {
        "id": "edge_05",
        "text": "Road is completely fine here.",
        "type": "Irrelevant message",
        "true_urgency": "LOW",
        "true_hazard": None,
        "true_loc": None
    },
    {
        "id": "edge_06",
        "text": "Road is blocked but vehicles are passing through.",
        "type": "Contradicting information",
        "true_urgency": "LOW",
        "true_hazard": "Landslide", # Or unspecified, depends on model assumption
        "true_loc": None
    },
    {
        "id": "edge_07",
        "text": "If flooding happens, people may need rescue near the river bank.",
        "type": "Hypothetical",
        "true_urgency": "LOW",
        "true_hazard": "Flood",
        "true_loc": "river bank"
    },
    {
        "id": "edge_08",
        "text": "Someone said 5 people are trapped near the old temple.",
        "type": "Rumour/Ambiguous",
        "true_urgency": "HIGH",
        "true_hazard": None,
        "true_loc": "old temple"
    }
]

def ensure_directories():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_test_set():
    print("[*] Reconstructing Test Set from 03_nlp_engineer to prevent data leakage...")
    try:
        df_class = nlp.load_classification_datasets()
    except Exception as e:
        print(f"[!] Could not load base classification datasets: {e}")
        return None, None

    # Apply exact same stratify/split as training script
    train_df, rem_df = train_test_split(df_class, test_size=0.30, random_state=nlp.SEED, stratify=df_class["urgency"])
    val_df, test_df = train_test_split(rem_df, test_size=0.50, random_state=nlp.SEED, stratify=rem_df["urgency"])

    print(f"[*] Test set size: {len(test_df)}")

    y_true_urg = test_df["urgency"].tolist()
    y_true_haz = test_df["hazard"].tolist()
    texts = test_df["text_clean"].tolist()

    print("[*] Running inference on test set (this may take a moment)...")
    results = nlp.process_batch(texts)

    y_pred_urg = [res["urgency_level"] for res in results]
    y_pred_haz = [res["hazard_type"] for res in results]

    # Metrics
    urg_report = classification_report(y_true_urg, y_pred_urg, output_dict=True, zero_division=0)
    haz_report = classification_report(y_true_haz, y_pred_haz, output_dict=True, zero_division=0)

    urg_cm = confusion_matrix(y_true_urg, y_pred_urg, labels=nlp.URGENCY_CLASSES)

    # Save to disk
    with open(EVAL_DIR / "classification_report.json", "w") as f:
        json.dump({"urgency": urg_report, "hazard": haz_report}, f, indent=2)

    pd.DataFrame(urg_cm, index=nlp.URGENCY_CLASSES, columns=nlp.URGENCY_CLASSES).to_csv(EVAL_DIR / "urgency_confusion_matrix.csv")

    return urg_report, urg_cm


def evaluate_robustness():
    print("[*] Running Controlled Robustness & Disaster-Specific Failure Analysis...")
    
    error_log = []
    robust_results = []
    
    for case in CONTROLLED_DATASET:
        text = case["text"]
        res = nlp.analyze_text(text)
        
        pred_urg = res["urgency_level"]
        pred_locs = res["entities"].get("location", [])
        
        # Determine Severity Mistakes
        severity_underestimated = False
        if case["true_urgency"] in ["CRITICAL", "HIGH"] and pred_urg in ["LOW", "MEDIUM"]:
            severity_underestimated = True
            
        false_positive = False
        if case["true_urgency"] in ["LOW"] and pred_urg in ["CRITICAL", "HIGH"]:
            false_positive = True
            
        loc_failure = False
        if case["true_loc"] and not pred_locs:
            loc_failure = True
            
        is_error = severity_underestimated or false_positive or loc_failure
        
        if is_error:
            error_cat = []
            if severity_underestimated: error_cat.append("Severity Underestimation")
            if false_positive: error_cat.append("False Positive (Overreaction)")
            if loc_failure: error_cat.append("Location Extraction Failure")
            
            error_log.append({
                "id": case["id"],
                "text": text,
                "type": case["type"],
                "true_urgency": case["true_urgency"],
                "pred_urgency": pred_urg,
                "confidence": res["urgency_confidence"],
                "error_category": " | ".join(error_cat)
            })
            
        robust_results.append({
            "id": case["id"],
            "type": case["type"],
            "passed": not is_error
        })

    pd.DataFrame(error_log).to_csv(EVAL_DIR / "error_analysis.csv", index=False)
    pd.DataFrame(robust_results).to_csv(EVAL_DIR / "robustness_results.csv", index=False)
    
    return error_log, robust_results

def generate_markdown_report(urg_report, urg_cm, error_log, robust_results):
    print("[*] Generating Final Evaluation Report...")
    
    report_lines = ["# Stage 03 NLP Quality & Evaluation Report"]

    report_lines.append("\n## 1. Executive Summary")
    if urg_report:
        # Macro F1 leads. The urgency classes are imbalanced and CRITICAL is both
        # the smallest and the costliest to miss, so accuracy overstates quality.
        report_lines.append(
            f"- **Urgency Macro F1 (primary metric):** {urg_report['macro avg']['f1-score']:.4f}"
        )
        report_lines.append(
            f"- **Urgency Accuracy (secondary):** {urg_report['accuracy']:.2%}"
        )
        report_lines.append("\nPer-class F1:")
        for class_name in nlp.URGENCY_CLASSES:
            if class_name in urg_report:
                scores = urg_report[class_name]
                report_lines.append(
                    f"  - **{class_name}**: F1 {scores['f1-score']:.4f} "
                    f"(precision {scores['precision']:.4f}, recall {scores['recall']:.4f}, "
                    f"support {int(scores['support'])})"
                )
    else:
        report_lines.append("- Urgency metrics unavailable.")
    report_lines.append(
        f"- **Robustness Tests Passed:** "
        f"{sum([1 for r in robust_results if r['passed']])}/{len(robust_results)}"
    )

    report_lines.append("\n### Corpus caveat")
    report_lines.append(
        "The Stage 03 corpora are synthetically generated. Impact language is "
        "sampled from a shared pool with severity-dependent weights (it is "
        "correlated with urgency, not determined by it), and hazard labels are "
        "the generative parameter rather than a keyword rule applied to the "
        "rendered text. An earlier revision violated both of those properties, "
        "which is why it reported ~96% urgency accuracy and exactly 1.0000 "
        "hazard F1 on every class. Those numbers measured string matching. The "
        "figures below are lower and are the real ones."
    )

    report_lines.append("\n## 2. Critical Failure Analysis")
    report_lines.append(
        "Disaster-response specific failures that could cost lives or waste resources:"
    )
    
    if error_log:
        for err in error_log:
            report_lines.append(f"\n### Test Case: {err['type']}")
            report_lines.append(f"- **Input:** \"{err['text']}\"")
            report_lines.append(f"- **Error Category:** {err['error_category']}")
            report_lines.append(f"- **True Urgency:** {err['true_urgency']} -> **Predicted:** {err['pred_urgency']} (Conf: {err['confidence']:.2f})")
    else:
        report_lines.append("\n*No critical failures detected in controlled dataset!*")
        
    if urg_cm is not None:
        report_lines.append("\n## 3. Confusion Matrix Analysis (Urgency)")
        report_lines.append("Analyzing dangerous confusions (Predicting LOW when actual is HIGH/CRITICAL):")
        
        # Urgency Classes: LOW (0), MEDIUM (1), HIGH (2), CRITICAL (3)
        high_as_low = urg_cm[2][0] + urg_cm[2][1]
        crit_as_low = urg_cm[3][0] + urg_cm[3][1]
        
        report_lines.append(f"- **HIGH misclassified as LOW/MEDIUM:** {high_as_low} times (Dangerous Underestimation)")
        report_lines.append(f"- **CRITICAL misclassified as LOW/MEDIUM:** {crit_as_low} times (Extremely Dangerous Underestimation)")
        
    report_lines.append("\n## 4. Final Verdict")
    if error_log or (urg_cm is not None and (high_as_low > 10 or crit_as_low > 5)):
        report_lines.append("**VERDICT: NEEDS IMPROVEMENT**")
        report_lines.append("\n*Reasoning:* The model exhibits severity underestimation on edge cases (e.g. typos, informal text) and failed to extract location from critical noisy alerts. A fallback mechanism or robustness retraining is recommended.")
    else:
        report_lines.append("**VERDICT: PASS WITH WARNINGS**")
        report_lines.append("\n*Reasoning:* General classification accuracy is stable, but ongoing monitoring for adversarial inputs and location ambiguity is required.")
        
    with open(EVAL_DIR / "evaluation_report.md", "w") as f:
        f.write("\n".join(report_lines))

def main():
    ensure_directories()
    
    # 1. Load artifacts (forces load/validation of models)
    #
    # A monkey patch used to live here, setting .multi_class on the unpickled
    # estimators to paper over a scikit-learn version mismatch (artifacts were
    # pickled under 1.9.0, the environment had 1.6.0). Hiding that mismatch
    # here did not stop it breaking inference in the Flask app. The real fix is
    # pinned dependencies (requirements.txt) plus retraining under those pins,
    # so the patch is gone and a genuine mismatch now fails loudly.
    try:
        artifacts = nlp._get_artifacts()
        print(f"[*] Urgency backend in use: {artifacts.get('urgency_backend')}")
    except Exception as e:
        print(f"FAILED to load NLP artifacts: {e}")
        print(
            "    Train Stage 03 first: python Stage03_NLP/03_nlp_engineer.py"
        )
        return
        
    # 2. Evaluate on standard Test set
    urg_report, urg_cm = evaluate_test_set()
    
    # 3. Evaluate on Controlled Disaster-Specific Dataset
    error_log, robust_results = evaluate_robustness()
    
    # 4. Generate Final Artifacts
    generate_markdown_report(urg_report, urg_cm, error_log, robust_results)
    
    print(f"\n[+] NLP Evaluation Complete! Artifacts saved to: {EVAL_DIR}")

if __name__ == "__main__":
    main()
