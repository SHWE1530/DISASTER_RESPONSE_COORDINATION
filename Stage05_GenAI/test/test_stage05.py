"""Tests for Stage 05 GenAI.

Fast by design: the CVAE is retrained for a few epochs into a temp directory,
and the stress-test logic runs against stub stage engines (the pattern used by
test/test_fusion.py), so what is under test is the generator and scoring
LOGIC, not model quality. Requires the committed data/processed/ artifacts
from 01_data_engineer.py.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STAGE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = STAGE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
PROCESSED = STAGE_DIR / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "sensor_reference.csv").exists(),
    reason="run Stage05_GenAI/01_data_engineer.py first",
)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, STAGE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENAI = _load("stage05_genai_engineer_under_test", "03_genai_engineer.py")
EDA = _load("stage05_eda_engineer_under_test", "02_eda_engineer.py")
EVAL = _load("stage05_evaluation_engineer_under_test", "04_evaluation_engineer.py")
INTEGRATION = _load("stage05_integration_engineer_under_test", "05_integration_engineer.py")

STAGE01_REQUIRED = {
    "timestamp", "state", "district", "rainfall_mm", "river_level_m", "river_level_threshold_m",
    "emergency_calls", "road_closures", "bridge_closures", "flood_history_count",
    "population_affected", "water_level_change_m",
}
LSTM_OOD_BOUND_M = 3.12 + 4 * 1.56


@pytest.fixture(scope="module")
def generator(tmp_path_factory):
    reference = pd.read_csv(PROCESSED / "sensor_reference.csv")
    subset = reference.groupby("zone_risk", group_keys=False).sample(frac=0.2, random_state=0)
    bundle = tmp_path_factory.mktemp("cvae") / "sensor_cvae.pt"
    GENAI.train_cvae(subset, epochs=3, output_path=bundle, verbose=False)
    return GENAI.ScenarioGenerator(bundle_path=bundle)


def one_zone_spec(**zone):
    base = {"label": "Test zone", "true_severity": "URGENT", "hazards": ["Flood"], "modifiers": [],
            "evidence": {"sensors": True, "text": "dispatcher", "image": None,
                         "water_history": "steady_rise"}}
    base.update(zone)
    return {"id": "T01", "name": "Test", "archetype": "test", "state": "Assam",
            "start_time": "14-08-2026 02:00", "global_modifiers": [], "zones": [base]}


# ------------------------------------------------------------- generator ---

def test_generation_is_deterministic(generator):
    spec = EDA.SCENARIO_PROMPTS[1]
    first = json.dumps(generator.generate(spec, seed=7), sort_keys=True)
    again = json.dumps(generator.generate(spec, seed=7), sort_keys=True)
    other = json.dumps(generator.generate(spec, seed=8), sort_keys=True)
    assert first == again
    assert first != other


def test_every_prompt_realises_stage01_ready_sensors(generator):
    """Each generated sensor payload must be scorable by Stage 01 as-is."""
    for spec in EDA.SCENARIO_PROMPTS + [EDA.WILDCARD_PROMPT]:
        for zone in generator.generate(spec, seed=1)["zones"]:
            sensors = zone["inputs"]["sensors"]
            if sensors is None:
                continue
            assert set(sensors) == STAGE01_REQUIRED
            assert re.fullmatch(r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}", sensors["timestamp"])
            for feature, (lo, hi) in GENAI.PHYSICAL_LIMITS.items():
                assert np.isfinite(sensors[feature])
                assert lo <= sensors[feature] <= hi, (spec["id"], feature, sensors[feature])
            for feature in GENAI.INTEGER_FEATURES:
                assert isinstance(sensors[feature], int)


def test_plain_samples_never_leave_the_historical_record(generator):
    sampler = generator.sampler
    for risk_class in GENAI.RISK_CLASSES:
        frame, _ = sampler.sample(risk_class, 500, temperature=2.0, seed=3)
        for feature in GENAI.NUMERIC_FEATURES:
            assert frame[feature].min() >= sampler.hist_min[feature] - 1e-9
            assert frame[feature].max() <= sampler.hist_max[feature] + 1e-9


def test_extrapolation_is_always_disclosed(generator):
    """A value beyond the record must be listed in beyond_record_fields."""
    for spec in EDA.SCENARIO_PROMPTS:
        for zone in generator.generate(spec, seed=2)["zones"]:
            sensors = zone["inputs"]["sensors"]
            if not sensors:
                continue
            outside = {f for f in GENAI.NUMERIC_FEATURES
                       if sensors[f] > generator.sampler.hist_max[f] + 1e-6
                       or sensors[f] < generator.sampler.hist_min[f] - 1e-6}
            assert outside <= set(zone["provenance"]["beyond_record_fields"])


@pytest.mark.parametrize("shape", sorted(GENAI.WATER_SHAPES - {"ood_spike"}))
def test_ordinary_water_histories_stay_inside_the_lstm_range(generator, shape):
    levels = generator._water_history(6.5, shape, np.random.default_rng(0))
    assert len(levels) == GENAI.LOOKBACK_HOURS
    assert max(levels) < LSTM_OOD_BOUND_M
    assert levels[-1] == pytest.approx(6.5)


def test_ood_spike_trips_the_lstm_guard(generator):
    levels = generator._water_history(5.0, "ood_spike", np.random.default_rng(0))
    assert max(levels) > LSTM_OOD_BOUND_M


def test_wildcard_outage_removes_exactly_the_right_evidence(generator):
    zones = generator.generate(EDA.WILDCARD_PROMPT, seed=142)["zones"]
    battery, grid_down, generator_backed, silent = (z["provenance"]["evidence_available"] for z in zones)
    assert battery["sensors"] and battery["water_history"] and not battery["image"]
    assert not grid_down["sensors"] and not grid_down["image"] and grid_down["text"]
    assert all(generator_backed.values())
    assert not any(silent.values())
    assert zones[3]["expected"]["status"] == "insufficient_evidence"


def test_incident_log_matches_stage04_header_format(generator):
    zone = generator.generate(one_zone_spec(), seed=5)["zones"][0]
    header = zone["inputs"]["incident_log"].splitlines()[0]
    assert re.fullmatch(r"INCIDENT LOG \| .+? / .+? / ZONE-\d+ \| \d+ entries", header)


def test_implicit_urgency_templates_never_state_urgency():
    explicit = re.compile(r"\b(urgent|critical|emergency|immediate)", re.IGNORECASE)
    for templates in GENAI.IMPLICIT_TEMPLATES.values():
        assert not any(explicit.search(t) for t in templates)


def test_degrade_text_is_deterministic_and_changes_the_message():
    text = "Water entering the house, please help, people trapped near the river bank quickly."
    first = GENAI.degrade_text(text, np.random.default_rng(1), truncate_probability=0.0)
    again = GENAI.degrade_text(text, np.random.default_rng(1), truncate_probability=0.0)
    assert first == again
    assert first != text


# ------------------------------------------------------------ validation ---

@pytest.mark.parametrize("mutate, message", [
    (lambda s: s.update(zones=[]), "zones"),
    (lambda s: s.update(start_time="2026-08-14"), "start_time"),
    (lambda s: s["zones"][0].update(true_severity="PANIC"), "true_severity"),
    (lambda s: s["zones"][0].update(hazards=["Volcano"]), "hazards"),
    (lambda s: s["zones"][0].update(modifiers=["alien_invasion"]), "modifiers"),
    (lambda s: s["zones"][0]["evidence"].update(image="blurry"), "evidence.image"),
    (lambda s: s["zones"][0].update(expect={"min_priority": "HUGE"}), "min_priority"),
    (lambda s: s.update(global_modifiers=["earthquake"]), "global modifiers"),
])
def test_validate_spec_rejects_malformed_specs(mutate, message):
    spec = one_zone_spec()
    mutate(spec)
    with pytest.raises(ValueError, match=message):
        GENAI.validate_spec(spec)


def test_every_library_prompt_is_valid_and_targets_a_blind_spot():
    for spec in EDA.SCENARIO_PROMPTS + [EDA.WILDCARD_PROMPT]:
        GENAI.validate_spec(spec)
        assert spec["blind_spots"]
    assert len(EDA.SCENARIO_PROMPTS) == 20


def test_default_expectations_follow_ground_truth():
    all_evidence = {"sensors": True, "text": True, "image": False, "water_history": False}
    assert GENAI.build_expectations("CRITICAL", all_evidence)["min_priority"] == "URGENT"
    assert GENAI.build_expectations("CRITICAL", all_evidence)["human_review"] is True
    assert GENAI.build_expectations("ELEVATED", all_evidence)["min_priority"] is None
    silent = GENAI.build_expectations("CRITICAL", dict.fromkeys(all_evidence, False))
    assert silent["status"] == "insufficient_evidence"


# ------------------------------------------------------------ stress test ---

class StubML:
    def __init__(self, category="Severe", confidence=0.9):
        self.category, self.confidence = category, confidence

    def predict(self, record):
        return {"risk_category": self.category, "confidence": self.confidence,
                "zone": record.get("district"), "top_factors": ["river_level_m"]}


class StubNLP:
    def __init__(self, urgency="HIGH", confidence=0.8):
        self.urgency, self.confidence = urgency, confidence

    def analyze(self, _text):
        return {"status": "ok", "urgency": self.urgency, "confidence": self.confidence,
                "hazard_type": "Flood", "entities": {"headcount": 12}}


class StubDL:
    def predict_image(self, _path):
        return {"label": "flooded", "confidence": 0.9, "flooded_probability": 0.9}

    def forecast_water_levels(self, levels, horizon=6):
        return {"forecast_water_levels": [levels[-1]] * horizon, "in_distribution": True,
                "expected_mae_at_horizon": 0.73, "distribution_check": {}}


def test_check_expectations_catches_each_failure_mode():
    ok = {"status": "ok", "priority": "ROUTINE", "priority_index": 0,
          "human_review_required": False, "conflicts": []}
    assert EVAL.check_expectations({"status": "ok", "min_priority": "URGENT"}, ok)
    assert EVAL.check_expectations({"status": "ok", "human_review": True}, ok)
    assert EVAL.check_expectations({"status": "ok", "conflict": True}, ok)
    assert EVAL.check_expectations({"status": "insufficient_evidence"}, ok)
    assert not EVAL.check_expectations({"status": "ok", "max_priority": "ELEVATED"}, ok)
    refused = {"status": "insufficient_evidence", "human_review_required": True}
    assert not EVAL.check_expectations({"status": "insufficient_evidence", "human_review": True}, refused)


def test_stress_tester_runs_the_wildcard_without_crashing(generator):
    tester = EVAL.StressTester(ml=StubML(), dl=StubDL(), nlp=StubNLP())
    result = tester.run_scenario(generator.generate(EDA.WILDCARD_PROMPT, seed=142))
    statuses = [z["status"] for z in result["zones"]]
    assert "crash" not in statuses
    assert statuses[3] == "insufficient_evidence"
    assert result["zones"][3]["passed"]


def test_a_missed_critical_zone_is_flagged(generator):
    """Calm sensors and no other evidence for a truly CRITICAL zone = critical miss."""
    spec = one_zone_spec(true_severity="CRITICAL",
                         evidence={"sensors": True, "text": None, "image": None, "water_history": None})
    tester = EVAL.StressTester(ml=StubML(category="Low", confidence=0.95))
    row = tester.run_scenario(generator.generate(spec, seed=3))["zones"][0]
    assert row["critical_miss"] is True
    assert row["passed"] is False


def test_ranking_tau():
    perfect = [{"true_index": i, "priority_index": i, "score": float(i)} for i in range(4)]
    assert EVAL.ranking_tau(perfect) == pytest.approx(1.0)
    same_truth = [{"true_index": 2, "priority_index": i, "score": 1.0} for i in range(3)]
    assert EVAL.ranking_tau(same_truth) is None


def test_imagery_adapter_refuses_paths_outside_the_bank():
    adapter = EVAL.ImageryAwareDLAdapter(StubDL())
    with pytest.raises(ValueError, match="imagery"):
        adapter.predict_image("../../app.py")


# ----------------------------------------------------------- integration ---

def test_dashboard_data_cannot_close_the_script_tag():
    """Untrusted strings must not be able to break out of the embedded data.

    An opening <script> inside a JS string literal is inert; only a closing
    </script> could end the data block early. The payload may add none.
    """
    engine = INTEGRATION.GenAIIntegrationEngine()
    hostile = "</script><script>alert(1)</script>"
    html = engine.render_dashboard(report={"generated_at": hostile, "summary": {}, "scenarios": []})
    template_closers = INTEGRATION.DASHBOARD_TEMPLATE.count("</script>")
    assert html.count("</script>") == template_closers
    assert "<\\/script><script>alert(1)<\\/script>" in html


def test_blueprint_validates_requests(generator):
    flask = pytest.importorskip("flask")
    engine = INTEGRATION.GenAIIntegrationEngine()
    engine._generator = generator
    app = flask.Flask(__name__)
    app.register_blueprint(INTEGRATION.create_blueprint(engine))
    client = app.test_client()

    assert client.post("/api/genai/scenario", json={"prompt_id": "NOPE"}).status_code == 400
    assert client.post("/api/genai/scenario", json={"prompt_id": "S01", "seed": "x"}).status_code == 400
    assert client.post("/api/genai/scenario", data="not json").status_code == 400
    bad_spec = one_zone_spec(true_severity="PANIC")
    assert client.post("/api/genai/scenario", json={"spec": bad_spec}).status_code == 400

    response = client.post("/api/genai/scenario", json={"prompt_id": "S02", "seed": 3})
    assert response.status_code == 200
    assert len(response.get_json()["zones"]) == 3
    assert client.get("/api/genai/health").get_json()["status"] == "healthy"


# --------------------------------------------------- LLM sequence generation ---
#
# The LLM itself is never loaded here: Qwen2.5-3B needs a GPU and a minute to
# start, which does not belong in a unit test. What IS tested is everything
# around it -- prompt construction, constrained decoding, the physics validator
# and the fallback -- driven by a fake backend that returns canned text. That
# is where the bugs live.

LLMGEN = _load("stage05_llm_generator_under_test", "03b_llm_scenario_generator.py")


class ScriptedBackend:
    """Stands in for Qwen. Returns queued answers, records the prompts it saw."""

    model_id = "scripted-test-model"
    use_adapter = False

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, system, user, temperature, top_p, seed, max_new_tokens=0):
        self.prompts.append(user)
        return self.answers.pop(0) if self.answers else self.answers_exhausted()

    @staticmethod
    def answers_exhausted():
        return "no json here"


def llm_answer(message="Flooding reported near the river bank in Nagaon, Assam. "
                       "42 people affected, requesting rescue boat and ambulance.",
               **overrides):
    payload = {"rainfall_mm": 96.0, "river_level_m": 7.4, "river_level_threshold_m": 6.0,
               "emergency_calls": 58, "road_closures": 3, "bridge_closures": 1,
               "flood_history_count": 6, "population_affected": 4200,
               "water_level_change_m": 0.42, "message": message}
    payload.update(overrides)
    return "REASONING: heavy rain has pushed the river past its danger mark.\n" \
           "JSON: " + json.dumps(payload)


def test_extract_json_survives_fences_and_trailing_prose():
    raw = ("REASONING: rising fast.\nJSON:\n```json\n"
           '{"rainfall_mm": 90, "message": "x",}\n```\nThat is my answer.')
    assert LLMGEN.extract_json(raw)["rainfall_mm"] == 90


def test_extract_json_rejects_an_answer_with_no_object():
    with pytest.raises(ValueError):
        LLMGEN.extract_json("REASONING: I could not decide.")


def test_validate_payload_clamps_out_of_range_values_instead_of_failing():
    payload = json.loads(llm_answer().split("JSON: ", 1)[1])
    payload["rainfall_mm"] = 9_000.0
    zone = {"true_severity": "URGENT"}
    sensors, _, clamped = LLMGEN.validate_payload(payload, zone)
    assert clamped == ["rainfall_mm"]
    assert sensors["rainfall_mm"] == GENAI.PHYSICAL_LIMITS["rainfall_mm"][1]


@pytest.mark.parametrize("mutate, fragment", [
    (lambda p: p.pop("river_level_m"), "missing keys"),
    (lambda p: p.update(emergency_calls="many"), "not a number"),
    (lambda p: p.update(message="too short"), "shorter than 8 words"),
    (lambda p: p.update(river_level_m=2.0), "below river_level_threshold_m"),
])
def test_validate_payload_rejects_unusable_generations(mutate, fragment):
    payload = json.loads(llm_answer().split("JSON: ", 1)[1])
    mutate(payload)
    with pytest.raises(ValueError, match=fragment):
        LLMGEN.validate_payload(payload, {"true_severity": "CRITICAL"})


def test_prompt_carries_retrieved_examples_schema_and_reasoning_step():
    zone = {"location": "river bank", "district": "Nagaon", "state": "Assam",
            "true_severity": "CRITICAL", "hazards": ["Flood"], "modifiers": ["flash_flood"]}
    prompt = LLMGEN.build_user_prompt(
        zone, {"prompt": "night flash flood", "start_time": "14-08-2026 02:00",
               "global_modifiers": ["night"]},
        sensor_shots=[{"rainfall_mm": 101.2}], text_shots=["Flooding reported near the ghat."])
    assert "REAL SENSOR ROWS" in prompt and "101.2" in prompt        # few-shot + retrieval
    assert "REAL DISPATCHER MESSAGES" in prompt                      # retrieval, text side
    assert "STEP 1 -- REASONING" in prompt                           # chain of thought
    assert all(f'"{f}"' in prompt for f in GENAI.NUMERIC_FEATURES)   # constrained schema
    assert "(night)" in prompt and "flash_flood" in prompt           # spec reached the prompt


def test_repair_retry_quotes_the_rejection_back_to_the_model(generator):
    backend = ScriptedBackend(["REASONING: done.\nJSON: {}", llm_answer()])
    llm = LLMGEN.LLMScenarioGenerator(backend, processed_dir=PROCESSED, max_retries=1)
    llm.base = generator
    scenario = llm.generate(one_zone_spec(), seed=3)
    zone = scenario["zones"][0]
    assert zone["provenance"]["llm"]["status"] == "ok"
    assert zone["provenance"]["llm"]["attempts"] == 2
    assert "CORRECTION" in backend.prompts[1] and "missing keys" in backend.prompts[1]


def test_llm_sensors_and_message_land_in_the_zone_together(generator):
    llm = LLMGEN.LLMScenarioGenerator(ScriptedBackend([llm_answer()]), processed_dir=PROCESSED)
    llm.base = generator
    zone = llm.generate(one_zone_spec(), seed=5)["zones"][0]
    assert zone["inputs"]["sensors"]["rainfall_mm"] == 96.0
    assert "42 people affected" in zone["inputs"]["text"]
    assert "42 people affected" in zone["inputs"]["incident_log"]
    # the gauge history has to end where the LLM put the river
    assert zone["inputs"]["water_levels"][-1] == pytest.approx(7.4, abs=1e-3)


def test_zone_falls_back_to_the_cvae_when_the_llm_keeps_failing(generator):
    llm = LLMGEN.LLMScenarioGenerator(ScriptedBackend([]), processed_dir=PROCESSED, max_retries=1)
    llm.base = generator
    zone = llm.generate(one_zone_spec(), seed=7)["zones"][0]
    assert zone["provenance"]["llm"]["status"] == "fallback"
    assert zone["inputs"]["sensors"] is not None      # the scenario is still complete
    assert llm.summary()["fallback_rate"] == 1.0


def test_a_blacked_out_zone_is_never_given_generated_text(generator):
    llm = LLMGEN.LLMScenarioGenerator(ScriptedBackend([llm_answer()] * 4), processed_dir=PROCESSED)
    llm.base = generator
    spec = one_zone_spec(modifiers=["total_blackout"])
    zone = llm.generate(spec, seed=9)["zones"][0]
    assert zone["provenance"]["llm"]["status"] == "skipped"
    assert zone["inputs"]["text"] is None and zone["inputs"]["sensors"] is None


def test_generated_scenario_declares_the_sequence_family_and_its_techniques(generator):
    llm = LLMGEN.LLMScenarioGenerator(ScriptedBackend([llm_answer()]), processed_dir=PROCESSED)
    llm.base = generator
    meta = llm.generate(one_zone_spec(), seed=11)["generator"]
    assert meta["family"] == "sequence_llm"
    for technique in ("autoregressive_llm", "few_shot", "retrieval_augmented",
                      "chain_of_thought", "constrained_decoding", "sampling_temperature"):
        assert technique in meta["techniques"]


def test_retrieval_returns_rows_of_the_requested_risk_class():
    index = LLMGEN.GroundingIndex(PROCESSED)
    shots = index.sensor_exemplars("Severe", {"rainfall_mm": 100.0, "river_level_m": 8.0}, 3)
    assert len(shots) == 3
    assert all(set(s) == set(GENAI.NUMERIC_FEATURES) for s in shots)
    calm = index.sensor_exemplars("Low", {"rainfall_mm": 2.0, "river_level_m": 1.0}, 3)
    assert np.mean([s["rainfall_mm"] for s in calm]) < np.mean([s["rainfall_mm"] for s in shots])


# ------------------------------------------------------------ scenario audit ---

AUDIT_BANDS = {"rain_calm": 20.0, "rain_extreme": 93.0, "calls_calm": 31.0, "calls_extreme": 57.0}


def audit_zone(severity="CRITICAL", rain=120.0, river=7.5, threshold=6.0, calls=70,
               change=0.4, population=5000.0, modifiers=(), headcount=40, history=None,
               beyond=()):
    return {
        "true_severity": severity, "modifiers": list(modifiers), "headcount_reported": headcount,
        "inputs": {"sensors": {"rainfall_mm": rain, "river_level_m": river,
                               "river_level_threshold_m": threshold, "emergency_calls": calls,
                               "water_level_change_m": change, "population_affected": population},
                   "water_levels": list(history) if history else None},
        "provenance": {"beyond_record_fields": list(beyond)},
    }


def test_a_coherent_severe_zone_scores_a_perfect_one():
    verdict = EVAL.zone_realism(audit_zone(), AUDIT_BANDS)
    assert verdict["realism_score"] == 1.0
    assert verdict["violations"] == [] and not verdict["overconfident"]


def test_the_lecture_hallucination_is_caught():
    # 5 mm of rain, a river far below its danger mark -- and a CRITICAL label.
    verdict = EVAL.zone_realism(audit_zone(rain=5.0, river=1.2, calls=4), AUDIT_BANDS)
    assert verdict["overconfident"]
    assert "severity_unsupported" in verdict["violations"]
    assert verdict["realism_score"] < 1.0


@pytest.mark.parametrize("modifier", ["dam_release", "silent_rise", "receding", "river_overflow"])
def test_a_declared_decoupling_modifier_is_not_a_hallucination(modifier):
    verdict = EVAL.zone_realism(audit_zone(rain=5.0, river=1.2, calls=4, modifiers=[modifier]),
                                AUDIT_BANDS)
    assert not verdict["overconfident"] and verdict["exempt"]


def test_a_deliberate_conflict_scenario_is_exempt_but_counted():
    verdict = EVAL.zone_realism(audit_zone(rain=5.0, river=1.2, calls=4), AUDIT_BANDS,
                                archetype="text_sensor_conflict")
    assert not verdict["overconfident"] and verdict["exempt"]


def test_a_calm_label_on_extreme_physics_is_caught():
    verdict = EVAL.zone_realism(audit_zone(severity="ROUTINE", rain=150.0, calls=80), AUDIT_BANDS)
    assert "calm_vs_extreme" in verdict["violations"]


def test_headcount_cannot_exceed_the_population_of_the_zone():
    verdict = EVAL.zone_realism(audit_zone(population=30.0, headcount=400), AUDIT_BANDS)
    assert "headcount_vs_population" in verdict["violations"]


def test_a_reported_rise_that_contradicts_the_gauge_is_caught():
    verdict = EVAL.zone_realism(audit_zone(change=1.5, history=[7.5, 6.0]), AUDIT_BANDS)
    assert "change_vs_history" in verdict["violations"]


def test_a_zone_with_no_sensors_is_not_scored():
    zone = audit_zone()
    zone["inputs"]["sensors"] = None
    verdict = EVAL.zone_realism(zone, AUDIT_BANDS)
    assert verdict["scored"] is False and verdict["realism_score"] is None


def test_diversity_coverage_counts_failure_modes_and_condition_cells():
    reference = pd.read_csv(PROCESSED / "sensor_reference.csv")
    scenarios = [
        {"scenario_id": "A", "archetype": "flash_flood", "blind_spots": ["BS08"],
         "global_modifiers": ["night"], "zones": [audit_zone() | {"hazards": ["Flood"]}]},
        {"scenario_id": "B", "archetype": "dam_release", "blind_spots": [],
         "global_modifiers": [], "zones": [audit_zone(severity="ROUTINE", rain=1.0, river=1.0,
                                                      threshold=6.0, calls=2)
                                           | {"hazards": ["Road Blockage"]}]},
    ]
    coverage = EVAL.diversity_coverage(scenarios, reference,
                                       [{"id": "BS08", "status": "absent"},
                                        {"id": "BS09", "status": "absent"}])
    assert coverage["distinct_archetypes"] == 2
    assert coverage["distinct_hazards"] == 2
    assert coverage["blind_spot_coverage"] == 0.5
    assert coverage["uncovered_blind_spots"] == ["BS09"]
    # two zones at opposite corners of the grid must occupy two distinct cells
    assert coverage["sensor_grid_cells_occupied"] == 2


def test_scenario_audit_separates_flagged_zones_from_exempt_ones():
    reference = pd.read_csv(PROCESSED / "sensor_reference.csv")
    scenarios = [
        {"scenario_id": "A", "name": "drifted", "archetype": "flash_flood",
         "zones": [audit_zone(rain=5.0, river=1.2, calls=4)
                   | {"zone_id": "Z1", "label": "z", "hazards": ["Flood"]}]},
        {"scenario_id": "B", "name": "by design", "archetype": "text_sensor_conflict",
         "zones": [audit_zone(rain=5.0, river=1.2, calls=4)
                   | {"zone_id": "Z1", "label": "z", "hazards": ["Flood"]}]},
    ]
    audit = EVAL.scenario_audit(scenarios, reference)
    assert audit["overconfidence"]["zones"] == 1
    assert audit["overconfidence"]["exempt_by_design"] == 1
    assert audit["overconfidence"]["detail"][0]["scenario_id"] == "A"
    assert audit["realism_score"]["mean"] < 1.0



def test_a_corrupted_checkpoint_is_refused_before_it_can_generate():
    """The gate that caught the damaged Qwen2.5-3B copy in Stage04_SLM.

    Both halves matter. Repairing that checkpoint's four blown-up embedding rows
    made the norm check pass while the model was still ruined (loss 19.6, one
    token repeated forever), so the behavioural probe is the real gate.
    """
    import torch

    class FakeModel:
        def __init__(self, norms, loss):
            weight = torch.zeros(len(norms), 4)
            weight[:, 0] = torch.tensor(norms, dtype=torch.float32)
            self._embeddings = torch.nn.Embedding.from_pretrained(weight)
            self._loss = loss

        def get_input_embeddings(self):
            return self._embeddings

    def backend_for(norms, loss):
        b = LLMGEN.QwenGenerator.__new__(LLMGEN.QwenGenerator)
        b.model_id = "test-model"
        b.model = FakeModel(norms, loss)
        b._probe_loss = lambda: loss
        return b

    healthy = [1.0, 1.1, 1.2, 2.0]

    backend_for(healthy, 2.5)._check_checkpoint()            # sane weights, sane loss

    with pytest.raises(RuntimeError, match="embedding rows exceed"):
        backend_for([1.0, 1.1, 1.2, 55.0], 2.5)._check_checkpoint()

    # The case that motivated the probe: statistics fine, model still broken.
    with pytest.raises(RuntimeError, match="loss on plain English"):
        backend_for(healthy, 19.6)._check_checkpoint()


# ------------------------------------------------- domain SLM sequence gen ---
#
# The SLM is trained here rather than downloaded, so these tests train a tiny
# one on a slice of the corpus (a few seconds) and check the parts that carry
# the correctness: the serialization round-trip, the constrained decoder, the
# severity-to-conditions pairing, and the zone plumbing.

SLMGEN = _load("stage05_slm_generator_under_test", "03c_slm_sequence_generator.py")


@pytest.fixture(scope="module")
def slm_bundle(tmp_path_factory):
    """A real but tiny SLM: 1 epoch over 3,000 paired messages."""
    path = tmp_path_factory.mktemp("slm") / "scenario_slm.pt"
    original = SLMGEN.SLM_PATH
    SLMGEN.SLM_PATH = path
    try:
        manifest = SLMGEN.train(seed=0, epochs=1, limit=3000, verbose=False)
        yield path, manifest
    finally:
        SLMGEN.SLM_PATH = original


def test_vocabulary_round_trips_a_sensor_row_through_bins():
    sensors = pd.read_csv(PROCESSED / "sensor_reference.csv")
    vocab = SLMGEN.ScenarioVocab(sensors, pd.Series(["water rising near the bridge"]),
                                 states=["Assam"], districts=["Nagaon"],
                                 hazards=["Flood"], severities=["CRITICAL"])
    rng = np.random.default_rng(0)
    for field in GENAI.NUMERIC_FEATURES:
        for value in sensors[field].sample(5, random_state=0):
            decoded = vocab.value_of(field, vocab.bin_of(field, float(value)), rng)
            low, high = GENAI.PHYSICAL_LIMITS[field]
            assert low <= decoded <= high
            # A quantile bin is narrow, so the round trip must stay near the
            # original -- within the spread of the field itself.
            assert abs(float(decoded) - float(value)) <= 3 * float(sensors[field].std())


def test_severity_pairing_gives_critical_zones_a_river_over_its_threshold():
    """The bug that made the first trained SLM hallucinate severe zones.

    Sampling uniformly inside the Severe class paired CRITICAL messages with
    rivers metres below their danger threshold, because only 58% of Severe rows
    are over it. Banding by river margin is what fixes that.
    """
    sensors = pd.read_csv(PROCESSED / "sensor_reference.csv")
    severe = sensors[sensors.zone_risk == "Severe"].reset_index(drop=True)
    rng = np.random.default_rng(0)

    def over_threshold_rate(severity):
        picks = [severe.iloc[SLMGEN._pick_by_margin(severe, severity, rng)] for _ in range(200)]
        return np.mean([p.river_level_m >= p.river_level_threshold_m for p in picks])

    assert over_threshold_rate("CRITICAL") > 0.95
    assert over_threshold_rate("CRITICAL") > over_threshold_rate("HIGH")


def test_training_learns_the_corpus(slm_bundle):
    _, manifest = slm_bundle
    # This fixture is deliberately tiny (1 epoch, 3k messages), so it is judged
    # against the only baseline that means anything at that size: a model that
    # has learned nothing scores a perplexity near the vocabulary size. An
    # order of magnitude below that is real learning. The production run -- 8
    # epochs over all 60k pairs -- reaches about 2.
    uniform = manifest["corpus"]["vocab_size"]
    assert manifest["final_val_perplexity"] < uniform / 10
    assert manifest["history"][0]["val_loss"] < manifest["history"][0]["train_loss"]
    assert manifest["corpus"]["paired"] > 2500


def test_constrained_decoding_can_only_emit_a_valid_scenario(slm_bundle):
    path, _ = slm_bundle
    sampler = SLMGEN.SLMSampler(path)
    sensors, text = sampler.sample("CRITICAL", "Flood", "Assam", "Nagaon", seed=1)

    assert set(sensors) == set(GENAI.NUMERIC_FEATURES)
    for field, value in sensors.items():
        low, high = GENAI.PHYSICAL_LIMITS[field]
        assert low <= value <= high, f"{field}={value} outside physical limits"
        if field in GENAI.INTEGER_FEATURES:
            assert isinstance(value, int)
    # Every generated word must come from the real corpus vocabulary.
    assert text and set(text.split()) <= set(sampler.vocab.word_tokens)


def test_sampling_is_deterministic_for_a_seed_and_varies_across_seeds(slm_bundle):
    path, _ = slm_bundle
    sampler = SLMGEN.SLMSampler(path)
    a = sampler.sample("HIGH", "Flood", "Assam", "Nagaon", seed=7)
    b = sampler.sample("HIGH", "Flood", "Assam", "Nagaon", seed=7)
    c = sampler.sample("HIGH", "Flood", "Assam", "Nagaon", seed=8)
    assert a == b
    assert a != c


def test_temperature_controls_diversity(slm_bundle):
    path, _ = slm_bundle
    sampler = SLMGEN.SLMSampler(path)
    def spread(temperature):
        texts = {sampler.sample("HIGH", "Flood", "Assam", "Nagaon",
                                temperature=temperature, seed=s)[1] for s in range(8)}
        return len(texts)
    assert spread(1.3) >= spread(0.3)


def test_slm_writes_sensors_and_narrative_into_the_same_zone(slm_bundle, generator):
    path, _ = slm_bundle
    gen = SLMGEN.SLMScenarioGenerator(SLMGEN.SLMSampler(path), processed_dir=PROCESSED)
    gen.base = generator
    zone = gen.generate(one_zone_spec(), seed=3)["zones"][0]

    assert zone["provenance"]["slm"]["status"] == "ok"
    assert zone["provenance"]["slm"]["constrained"] is True
    assert zone["inputs"]["sensors"] and zone["inputs"]["text"]
    assert zone["inputs"]["text"] in zone["inputs"]["incident_log"]
    # gauge history has to end where the model put the river
    assert zone["inputs"]["water_levels"][-1] == pytest.approx(
        zone["inputs"]["sensors"]["river_level_m"], abs=1e-3)


def test_blacked_out_zone_gets_no_generated_evidence(slm_bundle, generator):
    path, _ = slm_bundle
    gen = SLMGEN.SLMScenarioGenerator(SLMGEN.SLMSampler(path), processed_dir=PROCESSED)
    gen.base = generator
    zone = gen.generate(one_zone_spec(modifiers=["total_blackout"]), seed=4)["zones"][0]
    assert zone["provenance"]["slm"]["status"] == "skipped"
    assert zone["inputs"]["sensors"] is None and zone["inputs"]["text"] is None


def test_scenario_declares_the_sequence_family_and_is_honest_about_techniques(slm_bundle, generator):
    path, _ = slm_bundle
    gen = SLMGEN.SLMScenarioGenerator(SLMGEN.SLMSampler(path), processed_dir=PROCESSED)
    gen.base = generator
    meta = gen.generate(one_zone_spec(), seed=5)["generator"]
    assert meta["family"] == "sequence_slm"
    for claimed in ("autoregressive_slm", "prompt_engineering", "fine_tuned_domain_model",
                    "constrained_decoding", "sampling_temperature"):
        assert claimed in meta["techniques"]
    # The three that need a large pretrained model must NOT be claimed here.
    for withheld in ("few_shot", "retrieval_augmented", "chain_of_thought"):
        assert withheld not in meta["techniques"]
        assert withheld in meta["techniques_not_claimed"]


def test_generator_falls_back_when_no_slm_is_trained(generator):
    gen = SLMGEN.SLMScenarioGenerator(None, processed_dir=PROCESSED)
    gen.base = generator
    zone = gen.generate(one_zone_spec(), seed=6)["zones"][0]
    assert zone["provenance"]["slm"]["status"] == "fallback"
    assert zone["inputs"]["sensors"] is not None      # scenario still complete
    assert gen.summary()["fallback_rate"] == 1.0


def test_api_accepts_the_three_generator_families_and_rejects_others(generator):
    engine = INTEGRATION.GenAIIntegrationEngine()
    engine._generator = generator
    for family in ("cvae", "llm", "slm"):
        # Reaching the family branch is enough; loading a backend is not the point.
        try:
            engine.generate(prompt_id="S01", seed=1, generator=family)
        except Exception as exc:
            assert "generator must be" not in str(exc)
    with pytest.raises(ValueError, match="generator must be"):
        engine.generate(prompt_id="S01", seed=1, generator="gan")
