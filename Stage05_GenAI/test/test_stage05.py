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
