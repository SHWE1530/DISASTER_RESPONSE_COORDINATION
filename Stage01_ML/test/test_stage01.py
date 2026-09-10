"""Unit and contract tests for the Stage 01 ML integration layer.

Stage 01 was previously the only stage with no tests at all. These cover the
serving contract the Flask dashboard depends on, plus the input-validation
guarantees added during the audit remediation (finite-number enforcement,
per-row explanations, and a health check that actually scores a record).
"""

import importlib.util
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BASE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


api = _load("stage01_integration_engineer", "05_integration_engineer.py")


VALID_RECORD = {
    "timestamp": "05-09-2026 12:00",
    "state": "Maharashtra",
    "district": "Pune",
    "rainfall_mm": 120.5,
    "river_level_m": 14.2,
    "river_level_threshold_m": 12.0,
    "emergency_calls": 150,
    "water_level_change_m": 0.8,
    "road_closures": 0,
    "bridge_closures": 0,
    "flood_history_count": 5,
    "population_affected": 50000,
}

CALM_RECORD = {
    **VALID_RECORD,
    "rainfall_mm": 0.0,
    "river_level_m": 2.0,
    "emergency_calls": 1,
    "population_affected": 500,
}


@pytest.fixture(scope="module")
def engine():
    instance = api.integration_engine
    if instance.model is None:
        pytest.skip(f"Stage 01 model artifact unavailable: {instance.load_error}")
    return instance


def test_health_check_verifies_inference_not_just_loading(engine):
    """health_check must score a record, not merely confirm the pickle opened."""
    health = engine.health_check()
    assert health["status"] == "healthy"
    assert health["model_loaded"] is True
    assert health["inference_ok"] is True
    assert health["error"] is None


def test_predict_returns_full_serving_contract(engine):
    result = engine.predict(dict(VALID_RECORD))
    for key in (
        "zone", "state", "timestamp", "risk_category",
        "risk_score", "confidence", "top_factors", "global_top_factors",
    ):
        assert key in result, f"missing contract key: {key}"

    assert result["risk_category"] in {"Low", "Moderate", "Severe"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["zone"] == "Pune"


def test_top_factors_are_per_row_not_a_constant(engine):
    """Regression: serving used to return the same global top-3 for every request."""
    severe = engine.predict(dict(VALID_RECORD))
    calm = engine.predict(dict(CALM_RECORD))

    assert len(severe["top_factors"]) == 3
    assert len(calm["top_factors"]) == 3
    # The global ranking is constant by definition...
    assert severe["global_top_factors"] == calm["global_top_factors"]
    # ...but the per-row explanation must respond to the row's actual values.
    assert severe["top_factors"] != calm["top_factors"]


@pytest.mark.parametrize("field", ["rainfall_mm", "river_level_m", "emergency_calls"])
def test_nan_is_rejected(engine, field):
    """Regression: NaN used to yield 'Severe' at 0.9999999 confidence."""
    payload = {**VALID_RECORD, field: float("nan")}
    with pytest.raises(ValueError, match="finite number"):
        engine.predict(payload)


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinity_is_rejected(engine, value):
    with pytest.raises(ValueError, match="finite number"):
        engine.predict({**VALID_RECORD, "rainfall_mm": value})


def test_non_numeric_string_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.predict({**VALID_RECORD, "rainfall_mm": "a lot"})


def test_missing_field_names_the_field(engine):
    payload = {key: value for key, value in VALID_RECORD.items() if key != "rainfall_mm"}
    with pytest.raises(ValueError, match="rainfall_mm"):
        engine.predict(payload)


def test_empty_payload_is_rejected(engine):
    with pytest.raises(ValueError, match="Missing required fields"):
        engine.predict({})


def test_invalid_timestamp_is_rejected(engine):
    with pytest.raises(ValueError, match="timestamp"):
        engine.predict({**VALID_RECORD, "timestamp": "not-a-date"})


def test_unseen_category_degrades_gracefully(engine):
    """OneHotEncoder(handle_unknown='ignore') must absorb unknown districts."""
    result = engine.predict({**VALID_RECORD, "district": "Atlantis", "state": "ZZ"})
    assert result["risk_category"] in {"Low", "Moderate", "Severe"}
    assert result["zone"] == "Atlantis"


def test_predict_batch_matches_single_predictions(engine):
    batch = engine.predict_batch([dict(VALID_RECORD), dict(CALM_RECORD)])
    assert batch["count"] == 2
    assert len(batch["predictions"]) == 2
    assert batch["predictions"][0]["risk_category"] == engine.predict(dict(VALID_RECORD))["risk_category"]


def test_model_info_exposes_feature_contract(engine):
    info = engine.get_model_info()
    assert info["target_map"] == {"Low": 0, "Moderate": 1, "Severe": 2}
    assert "river_level_margin_m" in info["numeric_features"]
    assert set(info["categorical_features"]) == {"state", "district"}


def test_engineered_features_are_derived_not_required(engine):
    """The caller supplies raw fields only; margin/ratio/time features are derived."""
    assert "river_level_margin_m" not in api.RAW_REQUIRED_COLUMNS
    assert "is_monsoon" not in api.RAW_REQUIRED_COLUMNS
    result = engine.predict(dict(VALID_RECORD))
    assert result["risk_category"] in {"Low", "Moderate", "Severe"}


def test_confidence_is_calibrated_and_never_claims_certainty(engine):
    """Isotonic regression can emit exactly 1.0; a flood warning must never
    display 100% certainty."""
    result = engine.predict(dict(VALID_RECORD))
    assert result["confidence_is_calibrated"] is True
    low, high = api.CONFIDENCE_CLIP
    assert low <= result["confidence"] <= high
    assert result["raw_confidence"] is not None


def test_decision_comes_from_the_raw_model_not_the_calibrator(engine):
    """Regression: taking argmax over calibrated probabilities moved the
    operating point and cost 9 additional missed Severe zones on the held-out
    test set. Calibration must change the confidence, not the decision."""
    import numpy as np

    _frame, features = engine._prepare_features(dict(VALID_RECORD))
    transformed = engine.preprocessor.transform(features)
    predictions, _calibrated, _raw = engine._score(transformed)
    assert predictions[0] == np.asarray(engine.model.predict(transformed))[0]
