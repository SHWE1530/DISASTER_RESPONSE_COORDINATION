"""Unit tests for Stage 04 SLM Evaluation Engineer.

Tests core evaluation functions:
1. Priority extraction (SITUATION-scoped)
2. ROUGE scores & fallback
3. Slot fidelity & numerical mismatch detection (e.g. 50 -> 500)
4. Hallucination detection & action verb exemption
5. Safety audit for URGENT/IMMEDIATE high-risk incidents
6. Robustness pass/fail criteria (priority + safety + action keyword)
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Add Stage04_SLM directory to sys.path
STAGE04_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE04_DIR))

# Import module functions using importlib because filename starts with number '04_'
import importlib.util
spec = importlib.util.spec_from_file_location("eval_mod", STAGE04_DIR / "04_evaluation_engineer.py")
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)

extract_predicted_priority = eval_mod.extract_predicted_priority
rouge_scores = eval_mod.rouge_scores
check_slot_fidelity = eval_mod.check_slot_fidelity
hallucination_score = eval_mod.hallucination_score
safety_check = eval_mod.safety_check
evaluate_robustness = eval_mod.evaluate_robustness


def test_extract_predicted_priority_situation_scoped():
    # Priority in SITUATION section
    pred1 = "SITUATION: IMMEDIATE: flood in Kanpur. 1 report, 50 affected.\nRISK: High.\nACTIONS:\n1. Monitor only."
    assert extract_predicted_priority(pred1) == "IMMEDIATE"

    # Priority word inside ACTIONS should NOT pollute the result if SITUATION has priority
    pred2 = "SITUATION: URGENT: road blockage in ZONE-1. 20 affected.\nRISK: Moderate.\nACTIONS:\n1. Place unit on ROUTINE watch."
    assert extract_predicted_priority(pred2) == "URGENT"

    # Missing priority in SITUATION
    pred3 = "SITUATION: flood near river bank. 10 affected.\nRISK: Low.\nACTIONS:\n1. Standby."
    assert extract_predicted_priority(pred3) == "UNKNOWN"

    # Empty text
    assert extract_predicted_priority("") == "UNKNOWN"


def test_rouge_scores():
    pred = "IMMEDIATE: flood in Pune. 50 affected."
    ref = "IMMEDIATE: flood in Pune. 50 affected."
    scores = rouge_scores(pred, ref)
    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores
    assert scores["rouge1"] == 1.0


def test_check_slot_fidelity_numerical_mismatch():
    row = pd.Series({
        "zone": "ZONE-3",
        "district": "Pune",
        "state": "Maharashtra",
        "locations": "river bank",
        "total_headcount": 50,
        "hazard_types": "Flood",
        "priority": "IMMEDIATE"
    })

    # Exact match
    pred_good = "SITUATION: IMMEDIATE: flood in ZONE-3 Pune, Maharashtra near river bank. 50 affected."
    res_good = check_slot_fidelity(pred_good, row)
    assert res_good["zone_present"] is True
    assert res_good["district_present"] is True
    assert res_good["state_present"] is True
    assert res_good["location_present"] is True
    assert res_good["headcount_present"] is True
    assert res_good["headcount_correct"] is True
    assert res_good["content_slots_preserved"] is True

    # Numerical corruption: 50 changed to 500
    pred_corrupt = "SITUATION: IMMEDIATE: flood in ZONE-3 Pune, Maharashtra near river bank. 500 affected."
    res_corrupt = check_slot_fidelity(pred_corrupt, row)
    assert res_corrupt["headcount_correct"] is False
    assert res_corrupt["content_slots_preserved"] is False
    assert len(res_corrupt["mismatches"]) > 0


def test_hallucination_score_action_verb_exemption():
    report = "INCIDENT LOG | Maharashtra / Pune / ZONE-3 | 1 entry\n[08:00] ERSS-000001 | Flood near river bank. 50 trapped."
    pred = (
        "SITUATION: IMMEDIATE: flood in ZONE-3 Pune, Maharashtra near river bank. 50 affected.\n"
        "RISK: Critical life safety threat to 50 people.\n"
        "ACTIONS:\n"
        "1. Deploy rescue boat and evacuate 50 people via access route.\n"
        "2. Request NDRF support and establish triage area."
    )
    res = hallucination_score(pred, report)
    # Operational verbs like "Deploy", "Evacuate", "Request", "NDRF", "Triage" should NOT be flagged as hallucinations
    assert res["unsupported_count"] == 0
    assert res["hallucination_rate"] == 0.0


def test_safety_check():
    # Safe response for IMMEDIATE
    pred_safe = "SITUATION: IMMEDIATE: flood. 50 affected.\nACTIONS:\n1. ALL units deploy to river bank. Evacuate 50 people."
    assert safety_check(pred_safe, "IMMEDIATE")["safety_fail"] is False

    # Unsafe inaction response for IMMEDIATE
    pred_unsafe = "SITUATION: IMMEDIATE: flood. 50 affected.\nACTIONS:\n1. Monitor only. Standby, no deployment."
    assert safety_check(pred_unsafe, "IMMEDIATE")["safety_fail"] is True

    # Inaction response for ROUTINE is safe
    pred_routine = "SITUATION: ROUTINE: stable. 0 affected.\nACTIONS:\n1. Monitor only. Standby."
    assert safety_check(pred_routine, "ROUTINE")["safety_fail"] is False


def test_robustness_action_keyword_enforcement():
    class DummyModel:
        def generate(self, report_text: str) -> dict:
            return {
                "situation": "IMMEDIATE: flood in ZONE-3. 50 affected.",
                "risk": "Critical safety threat.",
                "actions": ["1. Monitor river bank."], # Missing expected action keyword "evacuate" for rob_01
                "priority": "IMMEDIATE"
            }

    results = evaluate_robustness(DummyModel())
    rob_01 = next(r for r in results if r["id"] == "rob_01")
    assert rob_01["priority_correct"] is True
    assert rob_01["action_keyword_found"] is False
    # Since action keyword is missing, passed must be False
    assert rob_01["passed"] is False


def test_lcs_rouge_l_fallback():
    _lcs_rouge_l = eval_mod._lcs_rouge_l
    pred = "the quick brown fox jumps over the lazy dog"
    ref = "the quick brown dog jumps over lazy fox"
    score = _lcs_rouge_l(pred, ref)
    assert score > 0.5
    assert score <= 1.0


def test_run_comparison_unmocked_qwen_status(tmp_path):
    run_comparison = eval_mod.run_comparison
    df = pd.DataFrame([{
        "pair_id": "SLM-000001",
        "split": "test",
        "report": "INCIDENT LOG | Maharashtra / Pune / ZONE-3 | 1 entry\n[08:00] ERSS-000001 | Flood 50 affected.",
        "summary": "IMMEDIATE: flood in ZONE-3 Pune, Maharashtra. 1 report, 50 affected.",
        "priority": "IMMEDIATE",
        "total_headcount": 50,
        "zone": "ZONE-3",
        "district": "Pune",
        "state": "Maharashtra",
        "locations": "river bank",
        "hazard_types": "Flood"
    }])
    
    comp = run_comparison(df)
    assert comp["comparison_executed"] is True
    assert comp["baseline"]["status"] == "evaluated"
    # Qwen must NOT mock baseline metrics if adapter is missing!
    assert comp["qwen"]["status"] == "not_available"
    assert "reason" in comp["qwen"]
    assert comp["qwen"]["reason"].startswith("Trained Qwen QLoRA adapter not found")

