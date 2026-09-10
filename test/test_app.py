"""End-to-end HTTP tests for the Flask dashboard.

app.py previously had no test coverage of any kind: no endpoint contract tests,
no status-code tests, no malformed-input tests. These exercise every route
through Flask's test client, which is the closest thing to what a capstone
evaluator will actually do during a demo.

Stage-dependent tests skip (rather than fail) when a stage's artifacts are not
present, so the suite is still useful on a fresh clone before training has run.
"""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def app_module():
    spec = importlib.util.spec_from_file_location("dashboard_app", REPO_ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def client(app_module):
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


VALID_ML_PAYLOAD = {
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


def _require(app_module, engine_name):
    engine = getattr(app_module, engine_name)
    if engine is None:
        pytest.skip(f"{engine_name} is not loaded")
    return engine


# ---------------------------------------------------------------- dashboard --

def test_dashboard_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Disaster Response AI Interface" in response.data


def test_health_endpoint_reports_every_stage(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert set(payload["stages"]) == {"stage01_ml", "stage02_dl", "stage03_nlp"}
    for stage, report in payload["stages"].items():
        assert "status" in report, stage


def test_status_is_not_reported_online_without_working_inference(app_module):
    """Regression: the dashboard printed "Online" for a stage whose every
    prediction failed, because status was derived from module import alone."""
    for status_name, engine_name in (
        ("stage01_status", "stage01_engine"),
        ("stage02_status", "stage02_engine"),
        ("stage03_status", "stage03_engine"),
    ):
        status = getattr(app_module, status_name)
        engine = getattr(app_module, engine_name)
        if status != "Online":
            continue
        assert engine is not None
        health = engine.health_check()
        assert health.get("inference_ok") is True, (
            f"{status_name} says Online but inference_ok is not True: {health}"
        )


# --------------------------------------------------------------------- ML ----

def test_ml_valid_request(client, app_module):
    _require(app_module, "stage01_engine")
    response = client.post("/api/predict/ml", json=VALID_ML_PAYLOAD)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["risk_category"] in {"Low", "Moderate", "Severe"}
    assert 0.0 <= payload["confidence"] <= 1.0
    assert len(payload["top_factors"]) == 3
    assert "global_top_factors" in payload


def test_ml_empty_body(client, app_module):
    _require(app_module, "stage01_engine")
    response = client.post("/api/predict/ml", json={})
    assert response.status_code == 400
    assert "Missing required fields" in response.get_json()["error"]


def test_ml_no_json_body(client, app_module):
    _require(app_module, "stage01_engine")
    response = client.post("/api/predict/ml", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_ml_nan_is_rejected(client, app_module):
    """Regression: NaN returned 200 with "Severe" at 0.9999999 confidence."""
    _require(app_module, "stage01_engine")
    payload = dict(VALID_ML_PAYLOAD)
    body = json.dumps(payload).replace("120.5", "NaN")
    response = client.post("/api/predict/ml", data=body, content_type="application/json")
    assert response.status_code == 400
    assert "finite" in response.get_json()["error"]


def test_ml_invalid_timestamp(client, app_module):
    _require(app_module, "stage01_engine")
    response = client.post(
        "/api/predict/ml", json={**VALID_ML_PAYLOAD, "timestamp": "not-a-date"}
    )
    assert response.status_code == 400


def test_ml_error_does_not_leak_filesystem_paths(client, app_module):
    """Errors must not expose absolute paths or internal class names."""
    _require(app_module, "stage01_engine")
    response = client.post("/api/predict/ml", json={**VALID_ML_PAYLOAD, "rainfall_mm": "lots"})
    assert response.status_code == 400
    message = response.get_json()["error"]
    assert "C:\\" not in message and "/Users/" not in message
    assert "Traceback" not in message


# -------------------------------------------------------------------- LSTM ---

def test_lstm_valid_sequence(client, app_module):
    engine = _require(app_module, "stage02_engine")
    if not engine.models_ready:
        pytest.skip("Stage 02 models unavailable")
    lookback = int(engine.metrics.get("lstm", {}).get("lookback", 72))
    mean_level = float(engine.module._load_lstm_bundle()["scaler"].mean_[0])
    response = client.post("/api/predict/lstm".replace("/lstm", "/dl/lstm"),
                           json={"sequence": [mean_level] * lookback})
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["forecast_water_levels"]) == 6
    assert payload["in_distribution"] is True


def test_lstm_short_sequence_is_rejected(client, app_module):
    engine = _require(app_module, "stage02_engine")
    if not engine.models_ready:
        pytest.skip("Stage 02 models unavailable")
    response = client.post("/api/predict/dl/lstm", json={"sequence": [1, 2, 3]})
    assert response.status_code == 400
    assert "At least" in response.get_json()["error"]


def test_lstm_out_of_distribution_carries_a_warning(client, app_module):
    engine = _require(app_module, "stage02_engine")
    if not engine.models_ready:
        pytest.skip("Stage 02 models unavailable")
    lookback = int(engine.metrics.get("lstm", {}).get("lookback", 72))
    response = client.post("/api/predict/dl/lstm", json={"sequence": [30.0] * lookback})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["in_distribution"] is False
    assert "warning" in payload


def test_lstm_no_body(client, app_module):
    _require(app_module, "stage02_engine")
    response = client.post("/api/predict/dl/lstm", data="", content_type="text/plain")
    assert response.status_code == 400


# --------------------------------------------------------------------- CNN ---

def test_cnn_no_file(client, app_module):
    _require(app_module, "stage02_engine")
    response = client.post("/api/predict/dl/image", data={})
    assert response.status_code == 400
    assert "No file uploaded" in response.get_json()["error"]


def test_cnn_rejects_disallowed_extension(client, app_module):
    _require(app_module, "stage02_engine")
    response = client.post(
        "/api/predict/dl/image",
        data={"file": (io.BytesIO(b"not an image"), "payload.exe")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unsupported image type" in response.get_json()["error"]


def test_cnn_valid_image(client, app_module):
    engine = _require(app_module, "stage02_engine")
    if not engine.models_ready:
        pytest.skip("Stage 02 models unavailable")
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (128, 128), color="blue").save(buffer, format="JPEG")
    buffer.seek(0)
    response = client.post(
        "/api/predict/dl/image",
        data={"file": (buffer, "scene.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["label"] in {"flooded", "unflooded"}
    assert 0.0 <= payload["confidence"] <= 1.0


def test_cnn_upload_does_not_persist_on_disk(client, app_module):
    """Uploads are transient and must not accumulate in the raw data folder."""
    engine = _require(app_module, "stage02_engine")
    if not engine.models_ready:
        pytest.skip("Stage 02 models unavailable")
    from PIL import Image

    upload_dir = Path(app_module.app.config["UPLOAD_FOLDER"])
    before = set(upload_dir.glob("upload_*"))
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color="green").save(buffer, format="PNG")
    buffer.seek(0)
    client.post(
        "/api/predict/dl/image",
        data={"file": (buffer, "scene.png")},
        content_type="multipart/form-data",
    )
    assert set(upload_dir.glob("upload_*")) == before


# --------------------------------------------------------------------- NLP ---

def test_nlp_valid_message(client, app_module):
    engine = _require(app_module, "stage03_engine")
    if not engine.health_check().get("inference_ok"):
        pytest.skip("Stage 03 NLP inference unavailable")
    response = client.post(
        "/api/predict/nlp",
        json={"text": "Three people are trapped in a flooded building near the railway bridge."},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["urgency"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert payload["hazard_type"]
    assert 0.0 <= payload["confidence"] <= 1.0


@pytest.mark.parametrize("body", [{"text": ""}, {"text": "   "}, {}])
def test_nlp_empty_input_is_rejected(client, app_module, body):
    _require(app_module, "stage03_engine")
    response = client.post("/api/predict/nlp", json=body)
    assert response.status_code == 400
    assert "emergency message" in response.get_json()["error"]


def test_nlp_handles_noisy_and_long_input(client, app_module):
    engine = _require(app_module, "stage03_engine")
    if not engine.health_check().get("inference_ok"):
        pytest.skip("Stage 03 NLP inference unavailable")
    for text in (
        "HELP!!! FLOOD HERE NOW",
        "\U0001F525\U0001F525 FLOOD \U0001F525\U0001F525",
        "pls hlp!!! flood water everywhere near main rd",
        "asdfghjkl",
        "flooding " * 800,
    ):
        response = client.post("/api/predict/nlp", json={"text": text})
        assert response.status_code == 200, f"failed on {text[:40]!r}: {response.get_json()}"
        assert response.get_json()["urgency"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_nlp_does_not_trigger_training(client, app_module, monkeypatch):
    """Regression: one HTTP request used to start a full training run.

    A single prediction kicked off a 63k-record retrain plus a DistilBERT
    fine-tune inside the request handler, and the failed run truncated the
    deployed NER model to 2 bytes.
    """
    engine = _require(app_module, "stage03_engine")
    if not engine.health_check().get("inference_ok"):
        pytest.skip("Stage 03 NLP inference unavailable")

    def explode(*args, **kwargs):
        raise AssertionError("train_models() must never be called from an inference path")

    monkeypatch.setattr(engine.module, "train_models", explode)
    response = client.post("/api/predict/nlp", json={"text": "Flooding near the bus stand."})
    assert response.status_code == 200


# ------------------------------------------------------------------ FUSION ---

def test_assess_requires_json(client, app_module):
    if app_module.decision_engine is None:
        pytest.skip("fusion engine unavailable")
    response = client.post("/api/assess", data="nope", content_type="text/plain")
    assert response.status_code == 400


def test_assess_with_no_evidence_refuses_rather_than_saying_routine(client, app_module):
    if app_module.decision_engine is None:
        pytest.skip("fusion engine unavailable")
    response = client.post("/api/assess", json={})
    assert response.status_code == 422
    payload = response.get_json()
    assert payload["status"] == "insufficient_evidence"
    assert payload["priority"] is None
    assert payload["human_review_required"] is True


def test_assess_combines_sensors_and_text(client, app_module):
    if app_module.decision_engine is None or app_module.stage01_engine is None:
        pytest.skip("fusion engine or Stage 01 unavailable")
    response = client.post("/api/assess", json={
        "sensors": VALID_ML_PAYLOAD,
        "text": "Water is rising fast, three people are trapped near the bridge.",
    })
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["priority"] in ["ROUTINE", "ELEVATED", "URGENT", "CRITICAL"]
    assert "sensor_risk" in payload["sources_used"]
    assert payload["recommended_actions"]
    assert "must be confirmed by a human" in payload["disclaimer"]


def test_assess_rejects_malformed_field_types(client, app_module):
    if app_module.decision_engine is None:
        pytest.skip("fusion engine unavailable")
    assert client.post("/api/assess", json={"sensors": "not-an-object"}).status_code == 400
    assert client.post("/api/assess", json={"water_levels": "not-a-list"}).status_code == 400


def test_assess_reports_missing_modalities(client, app_module):
    if app_module.decision_engine is None or app_module.stage01_engine is None:
        pytest.skip("fusion engine or Stage 01 unavailable")
    payload = client.post("/api/assess", json={"sensors": VALID_ML_PAYLOAD}).get_json()
    assert "visual_flood" in payload["sources_missing"]
    assert "text_urgency" in payload["sources_missing"]
