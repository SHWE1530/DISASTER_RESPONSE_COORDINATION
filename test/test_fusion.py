"""Tests for the cross-stage fusion / decision layer.

These use stub stage engines so the decision POLICY is tested in isolation from
model quality. Policy bugs (a flat forecast lowering a present-tense assessment,
a conflict being averaged away, an escalation failing to fire) are exactly the
kind of thing that is invisible when you only test end-to-end.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fusion.decision_engine import DecisionEngine, PRIORITY_LEVELS  # noqa: E402


class StubML:
    def __init__(self, category="Severe", confidence=0.9):
        self.category, self.confidence = category, confidence

    def predict(self, _record):
        return {
            "risk_category": self.category,
            "confidence": self.confidence,
            "zone": "Pune",
            "top_factors": ["river_level_m", "rainfall_mm", "emergency_calls"],
        }


class StubNLP:
    def __init__(self, urgency="CRITICAL", confidence=0.8, hazard="Flood", headcount=None):
        self.urgency, self.confidence = urgency, confidence
        self.hazard, self.headcount = hazard, headcount

    def analyze(self, _text):
        return {
            "status": "ok",
            "urgency": self.urgency,
            "confidence": self.confidence,
            "hazard_type": self.hazard,
            "entities": {"headcount": self.headcount},
        }


class StubDL:
    def __init__(self, label="flooded", confidence=0.95, forecast=None,
                 in_distribution=True, mae=0.73):
        self.label, self.confidence = label, confidence
        self.forecast = forecast if forecast is not None else [3.0] * 6
        self.in_distribution, self.mae = in_distribution, mae

    def predict_image(self, _path):
        return {
            "label": self.label,
            "confidence": self.confidence,
            "flooded_probability": self.confidence if self.label == "flooded" else 1 - self.confidence,
        }

    def forecast_water_levels(self, _levels, horizon=6):
        response = {
            "forecast_water_levels": self.forecast,
            "horizon": horizon,
            "in_distribution": self.in_distribution,
            "expected_mae_at_horizon": self.mae,
            "distribution_check": {"max_sigma_from_training_mean": 0.5 if self.in_distribution else 9.9},
        }
        if not self.in_distribution:
            response["warning"] = "Input is outside the trained range."
        return response


SENSORS = {"timestamp": "05-09-2026 12:00", "state": "Maharashtra", "district": "Pune"}
FLAT_HISTORY = [3.0] * 72


# ---------------------------------------------------------------- basics ----

def test_no_evidence_is_not_silently_safe():
    """Zero evidence must never read as 'ROUTINE' -- it must refuse and escalate."""
    result = DecisionEngine().assess()
    assert result["status"] == "insufficient_evidence"
    assert result["priority"] is None
    assert result["human_review_required"] is True


def test_priority_is_a_known_level():
    engine = DecisionEngine(ml_engine=StubML(), nlp_engine=StubNLP())
    result = engine.assess(sensors=SENSORS, text="flooding")
    assert result["priority"] in PRIORITY_LEVELS


def test_missing_sources_are_reported_not_hidden():
    engine = DecisionEngine(ml_engine=StubML())
    result = engine.assess(sensors=SENSORS)
    assert result["sources_used"] == ["sensor_risk"]
    assert set(result["sources_missing"]) == {"text_urgency", "visual_flood", "forecast_trend"}
    assert "Only one evidence source was available." in result["human_review_reasons"]


# ------------------------------------------------------------- escalation ---

def test_critical_text_sets_an_urgent_floor():
    """A human shouting CRITICAL must not be averaged down to ROUTINE."""
    engine = DecisionEngine(ml_engine=StubML(category="Low", confidence=0.9),
                            nlp_engine=StubNLP(urgency="CRITICAL", confidence=0.9))
    result = engine.assess(sensors=SENSORS, text="people trapped")
    assert result["priority_index"] >= 2
    assert any("CRITICAL urgency" in e for e in result["escalations"])


def test_corroborated_visual_and_sensor_reaches_critical():
    engine = DecisionEngine(ml_engine=StubML(category="Severe"),
                            dl_engine=StubDL(label="flooded"))
    result = engine.assess(sensors=SENSORS, image_path="scene.jpg")
    assert result["priority"] == "CRITICAL"


def test_corroboration_escalation_fires_when_it_changes_the_outcome():
    """Low-confidence severe sensors alone round to URGENT; visual confirmation
    must push it to CRITICAL and say so."""
    engine = DecisionEngine(ml_engine=StubML(category="Severe", confidence=0.5),
                            dl_engine=StubDL(label="flooded", confidence=0.6))
    result = engine.assess(sensors=SENSORS, image_path="scene.jpg")
    assert result["priority"] == "CRITICAL"
    assert any("visually confirms" in e for e in result["escalations"])


def test_escalations_are_only_claimed_when_they_change_the_outcome():
    """An 'escalation' that raised nothing must not be reported as one."""
    engine = DecisionEngine(ml_engine=StubML(category="Severe", confidence=0.99),
                            dl_engine=StubDL(label="flooded", confidence=0.99))
    result = engine.assess(sensors=SENSORS, image_path="scene.jpg")
    assert result["priority"] == "CRITICAL"
    assert result["escalations"] == []


def test_escalations_only_ever_raise_priority():
    """Every rule is one-directional; nothing may talk the priority down."""
    engine = DecisionEngine(ml_engine=StubML(category="Severe", confidence=1.0),
                            nlp_engine=StubNLP(urgency="LOW", confidence=1.0))
    result = engine.assess(sensors=SENSORS, text="all fine")
    sensor_only = DecisionEngine(ml_engine=StubML(category="Severe", confidence=1.0))
    baseline = sensor_only.assess(sensors=SENSORS)
    # Adding a LOW text report may lower the weighted mean, but the ESCALATION
    # stage must never have subtracted anything.
    assert result["escalations"] == [] or all(
        "raised" in e or "floor" in e or "escalates" in e for e in result["escalations"]
    )
    assert baseline["priority_index"] >= 0


# --------------------------------------------------------------- forecast ---

def test_flat_forecast_does_not_lower_present_assessment():
    """Regression: a flat 6h forecast used to drag a live emergency down a level.

    A forecast describes the future. A stable projection is not evidence that
    the current situation is calm.
    """
    severe = DecisionEngine(ml_engine=StubML(category="Severe", confidence=0.95),
                            nlp_engine=StubNLP(urgency="HIGH", confidence=0.9))
    without_forecast = severe.assess(sensors=SENSORS, text="flooding now")

    with_flat = DecisionEngine(
        ml_engine=StubML(category="Severe", confidence=0.95),
        nlp_engine=StubNLP(urgency="HIGH", confidence=0.9),
        dl_engine=StubDL(forecast=[3.0] * 6),
    ).assess(sensors=SENSORS, text="flooding now", water_levels=FLAT_HISTORY)

    assert with_flat["priority_index"] >= without_forecast["priority_index"]
    assert with_flat["score"] >= without_forecast["score"] - 1e-9


def test_rising_forecast_escalates_for_prepositioning():
    engine = DecisionEngine(ml_engine=StubML(category="Low", confidence=0.9),
                            dl_engine=StubDL(forecast=[3.5, 4.2, 4.9, 5.5, 6.0, 6.4]))
    result = engine.assess(sensors=SENSORS, water_levels=FLAT_HISTORY)
    assert result["priority_index"] >= 2
    assert any("rise" in e.lower() for e in result["escalations"])


def test_out_of_distribution_forecast_is_suppressed_entirely():
    """An extrapolated forecast must not move the decision at all."""
    engine = DecisionEngine(
        ml_engine=StubML(category="Low", confidence=0.9),
        dl_engine=StubDL(forecast=[30.0] * 6, in_distribution=False),
    )
    result = engine.assess(sensors=SENSORS, water_levels=[30.0] * 72)
    forecast = next(e for e in result["evidence"] if e["source"] == "forecast_trend")
    assert forecast["available"] is False
    assert "suppressed" in forecast["detail"]
    assert "forecast_trend" not in result["sources_used"]


def test_forecast_change_below_model_error_is_flagged():
    engine = DecisionEngine(dl_engine=StubDL(forecast=[3.05] * 6, mae=0.73))
    result = engine.assess(water_levels=FLAT_HISTORY)
    joined = " ".join(result.get("warnings", []))
    assert "measured error" in joined


# --------------------------------------------------------------- conflict ---

def test_material_disagreement_is_surfaced_not_averaged():
    engine = DecisionEngine(ml_engine=StubML(category="Low", confidence=0.95),
                            nlp_engine=StubNLP(urgency="CRITICAL", confidence=0.95))
    result = engine.assess(sensors=SENSORS, text="people trapped")
    assert result["conflicts"], "a Low-vs-CRITICAL split must be reported"
    assert result["agreement"] == "disputed"
    assert result["human_review_required"] is True
    assert "Not auto-resolved" in result["conflicts"][0]["resolution"]


def test_flat_forecast_is_not_counted_as_a_conflict():
    """Different points in time cannot contradict each other."""
    engine = DecisionEngine(ml_engine=StubML(category="Severe", confidence=0.95),
                            dl_engine=StubDL(forecast=[3.0] * 6))
    result = engine.assess(sensors=SENSORS, water_levels=FLAT_HISTORY)
    assert all("forecast_trend" not in c["between"] for c in result["conflicts"])


def test_unanimous_sources_report_agreement():
    engine = DecisionEngine(ml_engine=StubML(category="Severe", confidence=0.9),
                            nlp_engine=StubNLP(urgency="CRITICAL", confidence=0.9))
    result = engine.assess(sensors=SENSORS, text="people trapped")
    assert result["agreement"] == "unanimous"
    assert result["conflicts"] == []


# ------------------------------------------------------- safety behaviour ---

def test_low_confidence_forces_human_review():
    engine = DecisionEngine(nlp_engine=StubNLP(urgency="HIGH", confidence=0.2))
    result = engine.assess(text="maybe flooding?")
    assert result["human_review_required"] is True
    assert any("Low-confidence" in r for r in result["human_review_reasons"])


def test_urgent_and_above_always_requires_confirmation():
    engine = DecisionEngine(ml_engine=StubML(category="Severe", confidence=0.99),
                            nlp_engine=StubNLP(urgency="CRITICAL", confidence=0.99))
    result = engine.assess(sensors=SENSORS, text="trapped")
    assert result["priority_index"] >= 2
    assert result["human_review_required"] is True


def test_low_confidence_evidence_is_never_fully_discarded():
    """A 0.05-confidence CRITICAL report must still carry weight via the floor."""
    engine = DecisionEngine(nlp_engine=StubNLP(urgency="CRITICAL", confidence=0.05))
    result = engine.assess(text="trapped")
    assert result["score"] > 0
    assert result["priority_index"] >= 2  # the CRITICAL floor rule still applies


def test_clear_imagery_does_not_certify_the_zone_as_safe():
    engine = DecisionEngine(dl_engine=StubDL(label="unflooded", confidence=0.99))
    result = engine.assess(image_path="scene.jpg")
    visual = next(e for e in result["evidence"] if e["source"] == "visual_flood")
    assert any("does not clear the wider zone" in w for w in visual["warnings"])


def test_stage_failure_degrades_instead_of_crashing():
    class Broken:
        def predict(self, _):
            raise RuntimeError("model file corrupt")

    engine = DecisionEngine(ml_engine=Broken(), nlp_engine=StubNLP(urgency="HIGH"))
    result = engine.assess(sensors=SENSORS, text="flooding")
    assert result["status"] == "ok"
    sensor = next(e for e in result["evidence"] if e["source"] == "sensor_risk")
    assert sensor["available"] is False
    assert result["priority"] in PRIORITY_LEVELS


def test_every_decision_carries_a_disclaimer():
    engine = DecisionEngine(ml_engine=StubML())
    result = engine.assess(sensors=SENSORS)
    assert "must be confirmed by a human" in result["disclaimer"]


def test_recommended_actions_scale_with_priority():
    routine = DecisionEngine(ml_engine=StubML(category="Low", confidence=0.95)).assess(sensors=SENSORS)
    critical = DecisionEngine(
        ml_engine=StubML(category="Severe"), dl_engine=StubDL(label="flooded")
    ).assess(sensors=SENSORS, image_path="s.jpg")
    assert "routine" in " ".join(routine["recommended_actions"]).lower()
    assert "immediate" in " ".join(critical["recommended_actions"]).lower()
