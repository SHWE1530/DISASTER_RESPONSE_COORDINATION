import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_dir = Path(__file__).resolve().parent.parent
dl = load_module("stage02_dl_engineer", base_dir / "03_dl_engineer.py")
api = load_module("stage02_api", base_dir / "05_integration_engineer.py")

# Use mock data if real data is missing, to ensure tests always pass when files are absent
@pytest.fixture
def dummy_image(tmp_path):
    img = Image.new('RGB', (128, 128), color = 'blue')
    path = tmp_path / "dummy.jpg"
    img.save(path)
    return path

def test_sequence_preprocessing():
    # Phase 11: sequence preprocessing test
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    lookback = 3
    x, y = dl.make_sequences(values, lookback)
    assert x.shape == (2, 3, 1), "Sequence shape mismatch"
    assert y.shape == (2,), "Target shape mismatch"
    assert x[0].flatten().tolist() == [1.0, 2.0, 3.0]
    assert y[0] == 4.0

def test_cnn_inference_shape(dummy_image):
    # Phase 11: CNN inference shape test
    model = dl.DisasterCNN(classes=2)
    model.eval()
    dummy_tensor = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_tensor)
    assert output.shape == (1, 2), "CNN output shape mismatch"

def test_lstm_inference_shape():
    # Phase 11: LSTM inference shape test
    model = dl.WaterLevelLSTM(input_size=4)
    model.eval()
    dummy_tensor = torch.randn(1, 24, 4) # batch, lookback, features
    with torch.no_grad():
        output = model(dummy_tensor)
    assert output.shape == (1,), "LSTM output shape mismatch"


def test_lstm_rejects_wrong_feature_count():
    """Regression: forward() used to silently rebuild the LSTM on a shape
    mismatch, discarding trained weights and returning random-network output."""
    model = dl.WaterLevelLSTM(input_size=4)
    model.eval()
    with pytest.raises(ValueError, match="expects 4 features"):
        with torch.no_grad():
            model(torch.randn(1, 24, 1))

@pytest.fixture(scope="module")
def engine():
    instance = api.DLIntegrationEngine()
    if not instance.models_ready:
        pytest.skip("Stage 02 model artifacts unavailable")
    return instance


def test_api_health_schema(engine):
    # Phase 11: API response schema test
    health = engine.health_check()
    assert "status" in health
    assert "models_ready" in health
    assert "evaluation_available" in health
    assert "error" in health
    # health_check must verify inference, not just that files exist.
    assert "inference_ok" in health
    assert health["inference_ok"] is True, health.get("error")
    assert health["status"] == "healthy"


def test_api_invalid_forecast_input(engine):
    """Invalid forecast input must raise.

    This previously wrapped its own assertion in `try/except Exception: pass`,
    so the test passed unconditionally -- including when the code under test
    accepted the invalid input.
    """
    with pytest.raises(ValueError):
        engine.forecast_water_levels(["invalid_string", 2.0])


def test_forecast_rejects_short_sequence(engine):
    with pytest.raises(ValueError, match="At least"):
        engine.forecast_water_levels([1.0, 2.0, 3.0])


def test_forecast_rejects_nan(engine):
    """Regression: a NaN window used to return a list of NaN forecasts."""
    lookback = int(engine.metrics.get("lstm", {}).get("lookback", 72))
    with pytest.raises(ValueError, match="finite"):
        engine.forecast_water_levels([float("nan")] * lookback)


def test_forecast_rejects_out_of_range_horizon(engine):
    lookback = int(engine.metrics.get("lstm", {}).get("lookback", 72))
    with pytest.raises(ValueError, match="horizon"):
        engine.forecast_water_levels([3.0] * lookback, horizon=1000)


def test_forecast_flags_out_of_distribution_input(engine):
    """A window far outside the trained range must be labelled as extrapolation."""
    lookback = int(engine.metrics.get("lstm", {}).get("lookback", 72))
    scaler = dl._load_lstm_bundle()["scaler"]
    mean_level = float(scaler.mean_[0])
    std_level = float(scaler.scale_[0])

    in_range = engine.forecast_water_levels([mean_level] * lookback, horizon=3)
    assert in_range["in_distribution"] is True
    assert "warning" not in in_range

    far_out = engine.forecast_water_levels(
        [mean_level + 10 * std_level] * lookback, horizon=3
    )
    assert far_out["in_distribution"] is False
    assert "warning" in far_out
    assert far_out["distribution_check"]["max_sigma_from_training_mean"] > 4.0


def test_forecast_is_stable_for_a_flat_in_range_series(engine):
    """A flat input inside the trained range must not produce wild swings.

    Guards the failure mode found in the audit: fed a flat series the model
    returned an oscillating forecast spanning several metres.
    """
    lookback = int(engine.metrics.get("lstm", {}).get("lookback", 72))
    scaler = dl._load_lstm_bundle()["scaler"]
    mean_level = float(scaler.mean_[0])

    result = engine.forecast_water_levels([mean_level] * lookback, horizon=6)
    forecasts = result["forecast_water_levels"]
    assert result["in_distribution"] is True
    spread = max(forecasts) - min(forecasts)
    assert spread < 1.0, f"flat input produced a {spread:.2f} m swing: {forecasts}"


def test_model_is_cached_between_calls(engine):
    """Regression: the 44 MB checkpoint was reloaded on every request."""
    lookback = int(engine.metrics.get("lstm", {}).get("lookback", 72))
    engine.forecast_water_levels([3.0] * lookback, horizon=1)
    first_model = dl._load_lstm_bundle()["model"]
    engine.forecast_water_levels([3.0] * lookback, horizon=1)
    assert dl._load_lstm_bundle()["model"] is first_model

