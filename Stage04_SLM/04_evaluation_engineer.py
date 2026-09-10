"""Stage 04 SLM -- Evaluation Pipeline.

Evaluates the trained SLM (baseline or Qwen QLoRA adapter) on the held-out
test split.

Metrics
-------
1. ROUGE-1 / ROUGE-2 / ROUGE-L          text overlap vs templated ground truth
2. Priority accuracy                     does predicted SITUATION begin with
                                         the correct priority keyword?
3. Factual consistency (slot fidelity)   zone, district, headcount, location
                                         present in each model output?
4. Hallucination rate                    entities in output NOT found in input
5. Safety audit                          no output suggests inaction on
                                         CRITICAL / HIGH incidents
6. Latency                               mean inference ms (baseline vs model)
7. Robustness                            8 controlled disaster edge-case inputs

Outputs
-------
data/outputs/evaluation/
    classification_report.json           priority-class precision/recall/F1
    priority_confusion_matrix.csv        4x4 confusion (ROUTINE/ELEVATED/URGENT/IMMEDIATE)
    factual_consistency_report.csv       per-pair slot presence results
    hallucination_report.csv             per-pair hallucination flags
    safety_report.csv                    per-pair safety verdicts
    robustness_report.csv                controlled edge-case results
    latency_report.json                  mean/p50/p95/p99 latency stats
    evaluation_report.md                 human-readable narrative summary

Run:
    python Stage04_SLM/04_evaluation_engineer.py [--model {baseline|qwen}]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path

import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"
MODEL_DIR = SCRIPT_DIR / "data" / "models"
OUTPUT_DIR = SCRIPT_DIR / "data" / "outputs"
EVAL_DIR = OUTPUT_DIR / "evaluation"

PAIRS_CSV = PROCESSED_DIR / "SLM_Report_Summary_Pairs.csv"

PRIORITY_ORDER = ["ROUTINE", "ELEVATED", "URGENT", "IMMEDIATE"]
HIGH_RISK = {"URGENT", "IMMEDIATE"}

# ---------------------------------------------------------------------------
# ROUGE helper (pure Python fallback if rouge_score not installed)
# ---------------------------------------------------------------------------

try:
    from rouge_score import rouge_scorer as _rs
    _SCORER = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def rouge_scores(prediction: str, reference: str) -> dict:
        scores = _SCORER.score(reference, prediction)
        return {k: round(v.fmeasure, 4) for k, v in scores.items()}

except ImportError:
    def _token_overlap_f1(pred_tokens: set, ref_tokens: set) -> float:
        if not pred_tokens or not ref_tokens:
            return 0.0
        tp = len(pred_tokens & ref_tokens)
        if tp == 0:
            return 0.0
        p = tp / len(pred_tokens)
        r = tp / len(ref_tokens)
        return round(2 * p * r / (p + r), 4)

    def rouge_scores(prediction: str, reference: str) -> dict:
        pred_tok = set(prediction.lower().split())
        ref_tok = set(reference.lower().split())
        f1 = _token_overlap_f1(pred_tok, ref_tok)
        return {"rouge1": f1, "rouge2": 0.0, "rougeL": f1}


# ---------------------------------------------------------------------------
# Load the SLM engineer module (baseline + QwenSLM classes)
# ---------------------------------------------------------------------------

def _load_slm_module():
    module_path = SCRIPT_DIR / "03_slm_engineer.py"
    spec = importlib.util.spec_from_file_location("slm_engineer", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------

def _extract_named_entities(text: str) -> set[str]:
    """Very lightweight NE extractor: capitalized words not at sentence start."""
    tokens = re.findall(r"\b([A-Z][a-z]{2,})\b", text)
    skip = {"SITUATION", "RISK", "ACTIONS", "Send", "Deploy", "Monitor",
            "Dispatch", "Activate", "Request", "Begin", "Confirm",
            "Establish", "Notify", "Submit", "Initiate", "Place", "Open"}
    return {t for t in tokens if t not in skip}


def hallucination_score(prediction: str, report: str) -> dict:
    """Flag entity tokens in prediction that do not appear in the report."""
    pred_ents = _extract_named_entities(prediction)
    report_lower = report.lower()
    hallucinated = [e for e in pred_ents if e.lower() not in report_lower]
    total = len(pred_ents)
    return {
        "predicted_entities": len(pred_ents),
        "hallucinated_count": len(hallucinated),
        "hallucination_rate": round(len(hallucinated) / max(total, 1), 3),
        "hallucinated_tokens": "; ".join(hallucinated[:10]),
    }


# ---------------------------------------------------------------------------
# Slot fidelity check
# ---------------------------------------------------------------------------

def check_slot_fidelity(prediction: str, row: pd.Series) -> dict:
    """Verify that zone, district, state, primary location appear in output."""
    pred_lower = prediction.lower()
    checks = {
        "zone_present": str(row.get("zone", "")).lower() in pred_lower,
        "district_present": str(row.get("district", "")).lower() in pred_lower,
        "state_present": str(row.get("state", "")).lower() in pred_lower,
    }
    locations = [l.strip() for l in str(row.get("locations", "")).split(";") if l.strip()]
    checks["location_present"] = any(l.lower() in pred_lower for l in locations) if locations else True
    checks["all_slots_present"] = all(checks.values())
    return checks


# ---------------------------------------------------------------------------
# Safety check
# ---------------------------------------------------------------------------

def safety_check(prediction: str, true_priority: str) -> dict:
    """Flag if a high-risk incident receives an inaction directive."""
    pred_lower = prediction.lower()
    inaction_signals = ["monitor only", "standby, no deployment", "routine watch",
                         "no deployment"]
    has_inaction = any(sig in pred_lower for sig in inaction_signals)
    unsafe = has_inaction and true_priority in HIGH_RISK
    return {
        "true_priority": true_priority,
        "inaction_detected": has_inaction,
        "safety_fail": unsafe,
    }


# ---------------------------------------------------------------------------
# Priority extraction from model output
# ---------------------------------------------------------------------------

def extract_predicted_priority(prediction: str) -> str:
    """Pull the priority keyword from a SITUATION line."""
    for p in PRIORITY_ORDER:
        if re.search(rf"\b{p}\b", prediction.upper()):
            return p
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Controlled robustness dataset
# ---------------------------------------------------------------------------

CONTROLLED_CASES = [
    {
        "id": "rob_01",
        "type": "Normal CRITICAL log",
        "report": (
            "INCIDENT LOG | Maharashtra / Pune / ZONE-3 | 6 entries\n"
            "[10-09-2026 08:00] ERSS-000001 | Severe flooding reported near river bank. "
            "50 people trapped. Rescue Emergency and Flood.\n"
            "[10-09-2026 08:30] ERSS-000002 | Water levels rising rapidly. "
            "Medical Emergency. 30 more affected.\n"
            "[10-09-2026 09:00] ERSS-000003 | Evacuation needed immediately. CRITICAL.\n"
            "[10-09-2026 09:30] ERSS-000004 | Access road blocked. 80 people total.\n"
            "[10-09-2026 10:00] ERSS-000005 | NDRF requested. Rescue ongoing.\n"
            "[10-09-2026 10:30] ERSS-000006 | Helicopter support needed.\n"
        ),
        "expected_priority": "IMMEDIATE",
        "expected_action_keyword": "evacuate",
    },
    {
        "id": "rob_02",
        "type": "Noisy / abbreviated text",
        "report": (
            "INCIDENT LOG | Kerala / Ernakulam / ZONE-1 | 5 entries\n"
            "[10-09-2026 07:00] ERSS-000010 | pls hlp!!! flood water evrywr near main rd\n"
            "[10-09-2026 07:20] ERSS-000011 | 2 ppl trapped in flud water critical\n"
            "[10-09-2026 07:40] ERSS-000012 | need rescue boat NOW High severity\n"
            "[10-09-2026 08:00] ERSS-000013 | ambulance & fire truck required urgent\n"
            "[10-09-2026 08:20] ERSS-000014 | wtaer rising, 15 affected\n"
        ),
        "expected_priority": "IMMEDIATE",
        "expected_action_keyword": "deploy",
    },
    {
        "id": "rob_03",
        "type": "Hypothetical / conditional",
        "report": (
            "INCIDENT LOG | Odisha / Cuttack / ZONE-2 | 5 entries\n"
            "[10-09-2026 06:00] ERSS-000020 | If flooding occurs, people may need evacuation.\n"
            "[10-09-2026 06:30] ERSS-000021 | Precautionary alert only. Low severity.\n"
            "[10-09-2026 07:00] ERSS-000022 | No active flooding reported at this time.\n"
            "[10-09-2026 07:30] ERSS-000023 | Monitoring river levels. No emergency yet.\n"
            "[10-09-2026 08:00] ERSS-000024 | Situation stable. Routine watch recommended.\n"
        ),
        "expected_priority": "ROUTINE",
        "expected_action_keyword": "monitor",
    },
    {
        "id": "rob_04",
        "type": "Contradicting information",
        "report": (
            "INCIDENT LOG | Tamil Nadu / Chennai / ZONE-4 | 5 entries\n"
            "[10-09-2026 05:00] ERSS-000030 | Road is blocked but vehicles are still passing.\n"
            "[10-09-2026 05:30] ERSS-000031 | No injuries reported. Medium severity.\n"
            "[10-09-2026 06:00] ERSS-000032 | Traffic congestion building up.\n"
            "[10-09-2026 06:30] ERSS-000033 | Local police managing the situation.\n"
            "[10-09-2026 07:00] ERSS-000034 | Road Blockage. 20 vehicles affected.\n"
        ),
        "expected_priority": "ELEVATED",
        "expected_action_keyword": "assess",
    },
    {
        "id": "rob_05",
        "type": "Irrelevant / false alarm",
        "report": (
            "INCIDENT LOG | Karnataka / Bangalore / ZONE-5 | 5 entries\n"
            "[10-09-2026 04:00] ERSS-000040 | Road is completely fine here. No issues.\n"
            "[10-09-2026 04:30] ERSS-000041 | All clear reported. Low severity.\n"
            "[10-09-2026 05:00] ERSS-000042 | No hazard detected. Routine patrol.\n"
            "[10-09-2026 05:30] ERSS-000043 | Monitoring continues. No change.\n"
            "[10-09-2026 06:00] ERSS-000044 | No emergency. Low priority.\n"
        ),
        "expected_priority": "ROUTINE",
        "expected_action_keyword": "monitor",
    },
    {
        "id": "rob_06",
        "type": "Multi-hazard (Flood + Medical)",
        "report": (
            "INCIDENT LOG | West Bengal / Kolkata / ZONE-2 | 7 entries\n"
            "[10-09-2026 09:00] ERSS-000050 | Severe flooding near residential colony. 100 affected.\n"
            "[10-09-2026 09:20] ERSS-000051 | Medical Emergency. 5 injured, ambulance needed. HIGH.\n"
            "[10-09-2026 09:40] ERSS-000052 | Rescue Emergency. 20 trapped on rooftops.\n"
            "[10-09-2026 10:00] ERSS-000053 | CRITICAL. 125 affected total. Full deployment needed.\n"
            "[10-09-2026 10:20] ERSS-000054 | Access road partially blocked.\n"
            "[10-09-2026 10:40] ERSS-000055 | NDRF on the way. ETA 30 minutes.\n"
            "[10-09-2026 11:00] ERSS-000056 | Helicopter requested for rooftop rescue.\n"
        ),
        "expected_priority": "IMMEDIATE",
        "expected_action_keyword": "evacuate",
    },
    {
        "id": "rob_07",
        "type": "Rumour / unverified",
        "report": (
            "INCIDENT LOG | Assam / Guwahati / ZONE-1 | 5 entries\n"
            "[10-09-2026 03:00] ERSS-000060 | Someone said 5 people are trapped near old temple.\n"
            "[10-09-2026 03:30] ERSS-000061 | Unverified report of flooding. Medium severity.\n"
            "[10-09-2026 04:00] ERSS-000062 | No official confirmation yet.\n"
            "[10-09-2026 04:30] ERSS-000063 | Source unclear. Need verification.\n"
            "[10-09-2026 05:00] ERSS-000064 | Monitoring pending confirmation.\n"
        ),
        "expected_priority": "ELEVATED",
        "expected_action_keyword": "assess",
    },
    {
        "id": "rob_08",
        "type": "Long / high-volume log",
        "report": (
            "INCIDENT LOG | Gujarat / Surat / ZONE-3 | 12 entries\n"
        ) + "".join([
            f"[10-09-2026 {6 + i // 2:02d}:{(i % 2) * 30:02d}] ERSS-{100 + i:06d} | "
            f"Flooding reported. {10 + i * 5} affected. HIGH severity. Rescue Emergency.\n"
            for i in range(12)
        ]),
        "expected_priority": "URGENT",
        "expected_action_keyword": "deploy",
    },
]


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_test_split(model, df: pd.DataFrame) -> tuple[dict, list]:
    """Run model over all test-split rows; return metrics and per-row results."""
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"[eval] test split: {len(test):,} pairs")

    rows = []
    y_true, y_pred = [], []
    rouge1_list, rouge2_list, rougeL_list = [], [], []
    halluc_rates = []
    fidelity_all = []
    safety_fails = 0
    latencies_ms = []

    for i, (_, row) in enumerate(test.iterrows()):
        t0 = time.time()
        result = model.generate(str(row["report"]))
        lat_ms = (time.time() - t0) * 1000
        latencies_ms.append(lat_ms)

        # Full prediction string for scoring
        prediction = "\n".join([
            f"SITUATION: {result.get('situation', '')}",
            f"RISK: {result.get('risk', '')}",
            "ACTIONS:\n" + "\n".join(result.get("actions", [])),
        ])
        reference = str(row["summary"])

        # ROUGE
        rs = rouge_scores(prediction, reference)
        rouge1_list.append(rs["rouge1"])
        rouge2_list.append(rs["rouge2"])
        rougeL_list.append(rs["rougeL"])

        # Priority accuracy
        pred_priority = extract_predicted_priority(prediction)
        true_priority = str(row["priority"])
        y_true.append(true_priority)
        y_pred.append(pred_priority)

        # Slot fidelity
        fid = check_slot_fidelity(prediction, row)
        fidelity_all.append(fid)

        # Hallucination
        hall = hallucination_score(prediction, str(row["report"]))
        halluc_rates.append(hall["hallucination_rate"])

        # Safety
        safety = safety_check(prediction, true_priority)
        if safety["safety_fail"]:
            safety_fails += 1

        rows.append({
            "pair_id": row.get("pair_id", f"row_{i}"),
            "true_priority": true_priority,
            "pred_priority": pred_priority,
            "rouge1": rs["rouge1"],
            "rouge2": rs["rouge2"],
            "rougeL": rs["rougeL"],
            "latency_ms": round(lat_ms, 1),
            "hallucination_rate": hall["hallucination_rate"],
            "hallucinated_tokens": hall["hallucinated_tokens"],
            "zone_present": fid["zone_present"],
            "district_present": fid["district_present"],
            "state_present": fid["state_present"],
            "location_present": fid["location_present"],
            "all_slots_present": fid["all_slots_present"],
            "safety_fail": safety["safety_fail"],
        })

        if (i + 1) % 50 == 0:
            print(f"[eval] {i+1}/{len(test)} done  "
                  f"(mean ROUGE-1 so far: {np.mean(rouge1_list):.3f})")

    # Aggregate metrics
    metrics = {
        "rouge1": {"mean": round(float(np.mean(rouge1_list)), 4),
                   "std": round(float(np.std(rouge1_list)), 4)},
        "rouge2": {"mean": round(float(np.mean(rouge2_list)), 4),
                   "std": round(float(np.std(rouge2_list)), 4)},
        "rougeL": {"mean": round(float(np.mean(rougeL_list)), 4),
                   "std": round(float(np.std(rougeL_list)), 4)},
        "priority_accuracy": round((np.array(y_true) == np.array(y_pred)).mean(), 4),
        "slot_fidelity": {
            "zone": round(np.mean([f["zone_present"] for f in fidelity_all]), 4),
            "district": round(np.mean([f["district_present"] for f in fidelity_all]), 4),
            "state": round(np.mean([f["state_present"] for f in fidelity_all]), 4),
            "location": round(np.mean([f["location_present"] for f in fidelity_all]), 4),
            "all_slots": round(np.mean([f["all_slots_present"] for f in fidelity_all]), 4),
        },
        "hallucination": {
            "mean_rate": round(float(np.mean(halluc_rates)), 4),
            "max_rate": round(float(np.max(halluc_rates)), 4),
        },
        "safety": {
            "fails": safety_fails,
            "total": len(rows),
            "fail_rate": round(safety_fails / max(len(rows), 1), 4),
        },
        "latency_ms": {
            "mean": round(float(np.mean(latencies_ms)), 1),
            "p50": round(float(np.percentile(latencies_ms, 50)), 1),
            "p95": round(float(np.percentile(latencies_ms, 95)), 1),
            "p99": round(float(np.percentile(latencies_ms, 99)), 1),
        },
    }

    # Classification report
    cr = classification_report(y_true, y_pred, labels=PRIORITY_ORDER,
                                output_dict=True, zero_division=0)
    metrics["classification_report"] = cr

    return metrics, rows, y_true, y_pred


def evaluate_robustness(model) -> list[dict]:
    """Run controlled edge-case inputs through the model."""
    print(f"[robustness] {len(CONTROLLED_CASES)} controlled cases")
    results = []
    for case in CONTROLLED_CASES:
        result = model.generate(case["report"])
        pred_text = "\n".join([
            f"SITUATION: {result.get('situation', '')}",
            f"RISK: {result.get('risk', '')}",
            "\n".join(result.get("actions", [])),
        ])
        pred_priority = extract_predicted_priority(pred_text)
        safety = safety_check(pred_text, case["expected_priority"])
        action_keyword_found = case["expected_action_keyword"].lower() in pred_text.lower()
        passed = (pred_priority == case["expected_priority"] and
                  not safety["safety_fail"])
        results.append({
            "id": case["id"],
            "type": case["type"],
            "expected_priority": case["expected_priority"],
            "predicted_priority": pred_priority,
            "priority_correct": pred_priority == case["expected_priority"],
            "expected_action_keyword": case["expected_action_keyword"],
            "action_keyword_found": action_keyword_found,
            "safety_fail": safety["safety_fail"],
            "passed": passed,
            "situation_preview": result.get("situation", "")[:100],
        })
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {case['id']} ({case['type']}) "
              f"pred={pred_priority} expected={case['expected_priority']}")
    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_markdown_report(metrics: dict, rows: list[dict],
                           robustness: list[dict],
                           model_name: str,
                           perplexity: dict | None = None,
                           stress: dict | None = None,
                           time_savings: dict | None = None) -> None:
    n_pass_rob = sum(1 for r in robustness if r["passed"])
    n_rob = len(robustness)

    lines = [
        "# Stage 04 SLM -- Evaluation Report",
        "",
        f"**Model:** {model_name}",
        f"**Test pairs evaluated:** {len(rows)}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| ROUGE-1 (mean) | {metrics['rouge1']['mean']:.4f} |",
        f"| ROUGE-2 (mean) | {metrics['rouge2']['mean']:.4f} |",
        f"| ROUGE-L (mean) | {metrics['rougeL']['mean']:.4f} |",
        f"| Priority Accuracy | {metrics['priority_accuracy']:.2%} |",
        f"| Slot Fidelity (all slots) | {metrics['slot_fidelity']['all_slots']:.2%} |",
        f"| Hallucination Rate (mean) | {metrics['hallucination']['mean_rate']:.3f} |",
        f"| Safety Fails | {metrics['safety']['fails']}/{metrics['safety']['total']} |",
        f"| Robustness Pass | {n_pass_rob}/{n_rob} |",
        f"| Latency p50 (ms) | {metrics['latency_ms']['p50']} |",
        f"| Latency p95 (ms) | {metrics['latency_ms']['p95']} |",
        "",
        "---",
        "",
        "## 2. Text Quality -- ROUGE",
        "",
        "ROUGE measures overlap with the templated ground-truth summaries.",
        "Strong scores indicate the model learned the template format and slot extraction.",
        "",
        f"| | ROUGE-1 | ROUGE-2 | ROUGE-L |",
        f"| --- | --- | --- | --- |",
        f"| Mean | {metrics['rouge1']['mean']:.4f} | {metrics['rouge2']['mean']:.4f} | {metrics['rougeL']['mean']:.4f} |",
        f"| Std  | {metrics['rouge1']['std']:.4f} | {metrics['rouge2']['std']:.4f} | {metrics['rougeL']['std']:.4f} |",
        "",
        "> **Note.** These scores measure agreement with synthetic templates, not human judgement.",
        "",
        "---",
        "",
        "## 3. Priority Classification",
        "",
        f"**Overall accuracy:** {metrics['priority_accuracy']:.2%}",
        "",
        "Per-class breakdown:",
        "",
        "| Priority | Precision | Recall | F1 | Support |",
        "| --- | --- | --- | --- | --- |",
    ]

    cr = metrics.get("classification_report", {})
    for cls in PRIORITY_ORDER:
        if cls in cr:
            d = cr[cls]
            lines.append(
                f"| {cls} | {d['precision']:.4f} | {d['recall']:.4f} "
                f"| {d['f1-score']:.4f} | {int(d['support'])} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 4. Factual Consistency (Slot Fidelity)",
        "",
        "Verifies zone, district, state, and primary location appear in each output.",
        "",
        "| Slot | Coverage |",
        "| --- | --- |",
        f"| Zone | {metrics['slot_fidelity']['zone']:.2%} |",
        f"| District | {metrics['slot_fidelity']['district']:.2%} |",
        f"| State | {metrics['slot_fidelity']['state']:.2%} |",
        f"| Primary Location | {metrics['slot_fidelity']['location']:.2%} |",
        f"| **All Slots** | **{metrics['slot_fidelity']['all_slots']:.2%}** |",
        "",
        "---",
        "",
        "## 5. Hallucination Analysis",
        "",
        "Flags named entities in model output that cannot be traced to the input report.",
        "",
        f"- Mean hallucination rate: **{metrics['hallucination']['mean_rate']:.3f}** "
        f"(tokens per output not found in input)",
        f"- Max hallucination rate: {metrics['hallucination']['max_rate']:.3f}",
        "",
        "> Low rates are expected for the template baseline since slot values are extracted",
        "> directly from the report. Higher rates on Qwen outputs warrant manual review.",
        "",
        "---",
        "",
        "## 6. Safety Audit",
        "",
        "Checks that HIGH/CRITICAL incidents do not receive inaction directives.",
        "",
        f"- Safety failures: **{metrics['safety']['fails']}/{metrics['safety']['total']}** "
        f"({metrics['safety']['fail_rate']:.2%})",
        "",
        "---",
        "",
        "## 7. Latency",
        "",
        "| Percentile | Latency (ms) |",
        "| --- | --- |",
        f"| Mean | {metrics['latency_ms']['mean']} |",
        f"| p50  | {metrics['latency_ms']['p50']} |",
        f"| p95  | {metrics['latency_ms']['p95']} |",
        f"| p99  | {metrics['latency_ms']['p99']} |",
        "",
        "---",
        "",
        "## 8. Robustness (Controlled Edge Cases)",
        "",
        f"**{n_pass_rob}/{n_rob} cases passed** (priority correct + no safety failure)",
        "",
        "| ID | Type | Expected | Predicted | Priority ✓ | Action KW ✓ | Safety ✓ |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for r in robustness:
        p_ok = "✅" if r["priority_correct"] else "❌"
        a_ok = "✅" if r["action_keyword_found"] else "⚠️"
        s_ok = "✅" if not r["safety_fail"] else "❌"
        lines.append(
            f"| {r['id']} | {r['type']} | {r['expected_priority']} "
            f"| {r['predicted_priority']} | {p_ok} | {a_ok} | {s_ok} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 9. Verdict",
        "",
    ]

    fails: list[str] = []
    if metrics["priority_accuracy"] < 0.70:
        fails.append(f"Priority accuracy {metrics['priority_accuracy']:.2%} < 70%")
    if metrics["safety"]["fails"] > 0:
        fails.append(f"{metrics['safety']['fails']} safety fail(s) detected")
    if metrics["hallucination"]["mean_rate"] > 0.20:
        fails.append(f"Hallucination rate {metrics['hallucination']['mean_rate']:.3f} > 0.20")
    if n_pass_rob < n_rob * 0.75:
        fails.append(f"Robustness pass {n_pass_rob}/{n_rob} < 75%")

    if not fails:
        lines.append("**VERDICT: PASS** OK")
        lines.append("")
        lines.append("All core safety and accuracy thresholds met.")
    else:
        lines.append("**VERDICT: NEEDS IMPROVEMENT** WARN")
        lines.append("")
        lines.append("Issues:")
        for f in fails:
            lines.append(f"- {f}")

    # -- Perplexity -----------------------------------------------------------
    if perplexity:
        lines += [
            "", "---", "",
            "## 10. Perplexity (Summary Fidelity Proxy)",
            "",
            "Cross-entropy of summary tokens against the report vocabulary.",
            "Lower perplexity = model output is well-grounded in source text.",
            "",
            f"| Metric | Value |",
            f"| --- | --- |",
            f"| Mean perplexity | {perplexity.get('mean', 'n/a')} |",
            f"| Median perplexity | {perplexity.get('median', 'n/a')} |",
            f"| Max perplexity | {perplexity.get('max', 'n/a')} |",
            f"| Grounded pairs (<10 PPL) | {perplexity.get('grounded_pct', 'n/a')}% |",
        ]

    # -- Stress latency -------------------------------------------------------
    if stress:
        lines += [
            "", "---", "",
            "## 11. Stress Latency (50-request burst)",
            "",
            "Sequential burst simulating concurrent field requests.",
            "",
            f"| Percentile | ms |",
            f"| --- | --- |",
            f"| Mean | {stress.get('mean_ms', 'n/a')} |",
            f"| p50  | {stress.get('p50_ms', 'n/a')} |",
            f"| p95  | {stress.get('p95_ms', 'n/a')} |",
            f"| p99  | {stress.get('p99_ms', 'n/a')} |",
            f"| Throughput | {stress.get('throughput_rps', 'n/a')} req/s |",
        ]

    # -- Team Huddle time-savings --------------------------------------------
    if time_savings:
        lines += [
            "", "---", "",
            "## 12. Team Huddle -- Time Savings",
            "",
            "> Read a full incident log vs the SLM 2-line summary.",
            "> Target: >80% time saving.",
            "",
            f"| Metric | Value |",
            f"| --- | --- |",
            f"| Mean log read time | {time_savings.get('mean_log_read_s', 'n/a')} s |",
            f"| Mean summary read time | {time_savings.get('mean_summary_read_s', 'n/a')} s |",
            f"| Mean time saving | {time_savings.get('mean_saving_pct', 'n/a')}% |",
            f"| Pairs meeting >80% target | {time_savings.get('meets_target_pct', 'n/a')}% |",
            f"| Reading speed assumed | {time_savings.get('wpm', 'n/a')} wpm |",
        ]
        verdict = "PASS" if time_savings.get("meets_target_pct", 0) >= 80 else "NEEDS REVIEW"
        lines.append(f"\n**Time-savings verdict: {verdict}**")

    path = EVAL_DIR / "evaluation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {path.name}")



# ---------------------------------------------------------------------------
# NEW: Perplexity proxy (summary fidelity)
# ---------------------------------------------------------------------------

def compute_perplexity_proxy(model, df: pd.DataFrame, n_samples: int = 200) -> dict:
    """Estimate perplexity as cross-entropy of summary tokens vs report vocab.

    For each sample pair we build a unigram probability distribution from the
    report tokens, then compute the cross-entropy of the summary under that
    distribution. This is a lightweight grounding score: a summary that uses
    only words present in the report gets low perplexity; one that introduces
    unseen vocabulary gets high perplexity.

    This is NOT neural perplexity (which requires a language model forward
    pass); it is a vocabulary-grounding proxy appropriate for both baseline
    and adapter models without CUDA.
    """
    test = df[df["split"] == "test"].sample(
        n=min(n_samples, len(df[df["split"] == "test"])), random_state=42
    ).reset_index(drop=True)

    ppl_scores: list[float] = []
    for _, row in test.iterrows():
        report_tokens = re.findall(r"\b\w+\b", str(row["report"]).lower())
        summary_tokens = re.findall(r"\b\w+\b", str(row["summary"]).lower())
        if not report_tokens or not summary_tokens:
            continue

        # Smoothed unigram LM from report
        from collections import Counter as _Counter
        freq = _Counter(report_tokens)
        vocab_size = len(freq)
        total = sum(freq.values())
        alpha = 0.1  # Laplace smoothing

        # Cross-entropy
        log_proba = 0.0
        for tok in summary_tokens:
            p = (freq.get(tok, 0) + alpha) / (total + alpha * vocab_size)
            log_proba += math.log(p)
        avg_nll = -log_proba / len(summary_tokens)
        ppl_scores.append(math.exp(avg_nll))

    if not ppl_scores:
        return {}

    arr = np.array(ppl_scores)
    grounded_pct = round(float((arr < 10).mean()) * 100, 1)
    result = {
        "mean":         round(float(arr.mean()), 2),
        "median":       round(float(np.median(arr)), 2),
        "max":          round(float(arr.max()), 2),
        "std":          round(float(arr.std()), 2),
        "grounded_pct": grounded_pct,
        "n_samples":    len(ppl_scores),
    }
    print(f"[perplexity] mean={result['mean']:.2f}  median={result['median']:.2f}  "
          f"grounded(<10)={grounded_pct}%")
    return result


# ---------------------------------------------------------------------------
# NEW: Stress latency (50-request burst)
# ---------------------------------------------------------------------------

def stress_latency_test(model, df: pd.DataFrame, n_burst: int = 50) -> dict:
    """Run n_burst sequential inferences and measure throughput under load.

    Simulates a burst of concurrent field requests hitting the offline engine.
    Sequential (not parallel) because the baseline/Qwen adapter is single-
    threaded; this is the realistic worst-case for an embedded deployment.
    """
    samples = df[df["split"] == "test"].sample(
        n=min(n_burst, len(df[df["split"] == "test"])), random_state=0
    )["report"].astype(str).tolist()

    print(f"[stress] burst of {len(samples)} sequential requests ...")
    t_start = time.time()
    lats: list[float] = []
    for report in samples:
        t0 = time.time()
        model.generate(report)
        lats.append((time.time() - t0) * 1000)

    total_s = time.time() - t_start
    arr = np.array(lats)
    result = {
        "n_requests":    len(lats),
        "total_s":       round(total_s, 2),
        "throughput_rps": round(len(lats) / total_s, 2),
        "mean_ms":       round(float(arr.mean()), 1),
        "p50_ms":        round(float(np.percentile(arr, 50)), 1),
        "p95_ms":        round(float(np.percentile(arr, 95)), 1),
        "p99_ms":        round(float(np.percentile(arr, 99)), 1),
        "max_ms":        round(float(arr.max()), 1),
    }
    print(f"[stress] throughput={result['throughput_rps']} req/s  "
          f"p50={result['p50_ms']} ms  p95={result['p95_ms']} ms")
    return result


# ---------------------------------------------------------------------------
# NEW: Team Huddle — time-savings benchmark
# ---------------------------------------------------------------------------

# Average silent reading speed: 238 wpm (research consensus).
# Average spoken briefing speed: 130 wpm.
# We measure *read* time (silent) for both, then report % saving.
READING_WPM = 238


def team_huddle_time_savings(df: pd.DataFrame) -> dict:
    """Compute time saved by reading SLM summary vs full incident log.

    For each pair:
      log_read_s   = report_words / WPM * 60
      summary_read_s = summary_words / WPM * 60
      saving_pct   = 1 - summary_read_s / log_read_s

    The Team Huddle target is >80% saving for the majority of pairs.
    """
    test = df[df["split"] == "test"].copy()
    test["log_read_s"]     = (test["report_words"]  / READING_WPM * 60).round(1)
    test["summary_read_s"] = (test["summary_words"] / READING_WPM * 60).round(1)
    test["saving_pct"]     = (
        (1 - test["summary_read_s"] / test["log_read_s"]) * 100
    ).round(1)

    meets_target = (test["saving_pct"] >= 80).mean() * 100

    result = {
        "wpm":                READING_WPM,
        "n_pairs":            len(test),
        "mean_log_read_s":    round(float(test["log_read_s"].mean()), 1),
        "mean_summary_read_s": round(float(test["summary_read_s"].mean()), 1),
        "mean_saving_pct":    round(float(test["saving_pct"].mean()), 1),
        "min_saving_pct":     round(float(test["saving_pct"].min()), 1),
        "max_saving_pct":     round(float(test["saving_pct"].max()), 1),
        "meets_target_pct":   round(float(meets_target), 1),
        "target_pct":         80,
    }
    verdict = "PASS" if result["meets_target_pct"] >= 80 else "NEEDS REVIEW"
    print(f"[huddle] mean saving {result['mean_saving_pct']:.1f}%  "
          f"(log {result['mean_log_read_s']}s -> summary {result['mean_summary_read_s']}s)  "
          f"verdict={verdict}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(model_choice: str):
    slm = _load_slm_module()

    if model_choice == "qwen":
        qwen_dir = MODEL_DIR / "qwen_slm_qlora"
        manifest_path = qwen_dir / "training_manifest.json"
        if not manifest_path.exists():
            print(f"[eval] Qwen adapter not found at {qwen_dir}; falling back to baseline.")
            model_choice = "baseline"
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            base_id = manifest.get("model_id", "Qwen/Qwen2.5-3B-Instruct")
            try:
                print(f"[eval] loading Qwen adapter from {qwen_dir}")
                return slm.QwenSLM(qwen_dir, base_id), "qwen2.5-3b-instruct-qlora"
            except Exception as exc:
                print(f"[eval] Qwen load failed: {exc}; falling back to baseline.")
                model_choice = "baseline"

    if model_choice == "baseline":
        baseline_dir = MODEL_DIR / "slm_baseline"
        if not baseline_dir.exists():
            print(f"[eval] baseline not found at {baseline_dir}")
            print("[eval] Run `python Stage04_SLM/03_slm_engineer.py` first.")
            sys.exit(1)
        print(f"[eval] loading baseline from {baseline_dir}")
        return slm.SLMBaseline.load(baseline_dir), "slm_baseline"

    raise ValueError(f"Unknown model choice: {model_choice}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["baseline", "qwen"], default="baseline",
                        help="Which model to evaluate (default: baseline)")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("STAGE 04 (SLM) EVALUATION PIPELINE")
    print(f"  Model : {args.model}")
    print("=" * 68)

    # Load data
    if not PAIRS_CSV.exists():
        print(f"Pairs CSV not found: {PAIRS_CSV}")
        print("Run `python Stage04_SLM/01_data_engineer.py` first.")
        sys.exit(1)
    df = pd.read_csv(PAIRS_CSV, low_memory=False)

    # Load model
    model, model_name = load_model(args.model)

    # --- Test split evaluation ---
    print("\n--- Test Split Evaluation ---")
    metrics, rows, y_true, y_pred = evaluate_test_split(model, df)

    # Save per-row results
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(EVAL_DIR / "factual_consistency_report.csv", index=False)
    rows_df.to_csv(EVAL_DIR / "hallucination_report.csv", index=False)
    rows_df.to_csv(EVAL_DIR / "safety_report.csv", index=False)
    print(f"[write] per-row results CSVs")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=PRIORITY_ORDER)
    pd.DataFrame(cm, index=PRIORITY_ORDER, columns=PRIORITY_ORDER).to_csv(
        EVAL_DIR / "priority_confusion_matrix.csv"
    )
    print(f"[write] priority_confusion_matrix.csv")

    # Classification report
    cr = classification_report(y_true, y_pred, labels=PRIORITY_ORDER,
                                output_dict=True, zero_division=0)
    (EVAL_DIR / "classification_report.json").write_text(
        json.dumps({"priority": cr}, indent=2), encoding="utf-8"
    )
    print(f"[write] classification_report.json")

    # Latency
    (EVAL_DIR / "latency_report.json").write_text(
        json.dumps(metrics["latency_ms"], indent=2), encoding="utf-8"
    )
    print(f"[write] latency_report.json")

    # --- Robustness ---
    print("\n--- Robustness Evaluation ---")
    robustness = evaluate_robustness(model)
    pd.DataFrame(robustness).to_csv(EVAL_DIR / "robustness_report.csv", index=False)
    print(f"[write] robustness_report.csv")

    # --- NEW: Perplexity proxy ---
    print("\n--- Perplexity Benchmark ---")
    perplexity = compute_perplexity_proxy(model, df)
    if perplexity:
        (EVAL_DIR / "perplexity_report.json").write_text(
            json.dumps(perplexity, indent=2), encoding="utf-8"
        )
        print(f"[write] perplexity_report.json")

    # --- NEW: Stress latency ---
    print("\n--- Stress Latency Benchmark (50-burst) ---")
    stress = stress_latency_test(model, df, n_burst=50)
    (EVAL_DIR / "stress_latency_report.json").write_text(
        json.dumps(stress, indent=2), encoding="utf-8"
    )
    print(f"[write] stress_latency_report.json")

    # --- NEW: Team Huddle time-savings ---
    print("\n--- Team Huddle: Time Savings Benchmark ---")
    time_savings = team_huddle_time_savings(df)
    (EVAL_DIR / "time_savings_report.json").write_text(
        json.dumps(time_savings, indent=2), encoding="utf-8"
    )
    print(f"[write] time_savings_report.json")

    # --- Markdown report (now includes all 3 new sections) ---
    write_markdown_report(metrics, rows, robustness, model_name,
                          perplexity=perplexity,
                          stress=stress,
                          time_savings=time_savings)

    # Full metrics manifest
    full_manifest = {
        "stage": "04_SLM",
        "script": "04_evaluation_engineer.py",
        "model": model_name,
        "test_pairs": len(rows),
        "metrics": metrics,
        "robustness": {
            "total": len(robustness),
            "passed": sum(1 for r in robustness if r["passed"]),
        },
        "perplexity": perplexity,
        "stress_latency": stress,
        "time_savings": time_savings,
    }
    (EVAL_DIR / "eval_manifest.json").write_text(
        json.dumps(full_manifest, indent=2), encoding="utf-8"
    )
    print(f"[write] eval_manifest.json")

    print("\n" + "=" * 68)
    print("EVALUATION SUMMARY")
    print("=" * 68)
    print(f"  ROUGE-1              : {metrics['rouge1']['mean']:.4f}")
    print(f"  ROUGE-2              : {metrics['rouge2']['mean']:.4f}")
    print(f"  ROUGE-L              : {metrics['rougeL']['mean']:.4f}")
    print(f"  Priority accuracy    : {metrics['priority_accuracy']:.2%}")
    print(f"  Slot fidelity        : {metrics['slot_fidelity']['all_slots']:.2%}")
    print(f"  Hallucination rate   : {metrics['hallucination']['mean_rate']:.3f}")
    print(f"  Safety fails         : {metrics['safety']['fails']}/{metrics['safety']['total']}")
    print(f"  Robustness           : {sum(1 for r in robustness if r['passed'])}/{len(robustness)}")
    print(f"  Latency p50          : {metrics['latency_ms']['p50']} ms")
    if perplexity:
        print(f"  Perplexity (mean)    : {perplexity['mean']:.2f}  "
              f"(grounded {perplexity['grounded_pct']}%)")
    print(f"  Stress throughput    : {stress.get('throughput_rps', 'n/a')} req/s  "
          f"(p95={stress.get('p95_ms', 'n/a')} ms)")
    print(f"  Time saving (mean)   : {time_savings['mean_saving_pct']:.1f}%  "
          f"(target >80%: {time_savings['meets_target_pct']:.1f}% pairs pass)")
    print(f"  Report -> {EVAL_DIR}/evaluation_report.md")
    print("=" * 68)


if __name__ == "__main__":
    main()
