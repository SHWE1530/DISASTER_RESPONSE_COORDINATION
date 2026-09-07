"""Integration layer for the trained Stage02 deep-learning artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "data" / "models"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
RAW_DIR = BASE_DIR / "data" / "raw"


class DLIntegrationEngine:
	"""Serve Stage02 model status and inference without retraining."""

	def __init__(self) -> None:
		self.module = self._load_training_module()
		self.metrics = self._read_json(OUTPUT_DIR / "evaluation_metrics.json")
		self.training_metrics = self._read_json(OUTPUT_DIR / "dl_training_metrics.json")
		self.load_error: str | None = None

	@staticmethod
	def _read_json(path: Path) -> dict[str, Any]:
		try:
			return json.loads(path.read_text(encoding="utf-8"))
		except (OSError, ValueError):
			return {}

	@staticmethod
	def _load_training_module():
		try:
			spec = importlib.util.spec_from_file_location(
				"stage02_dl_engineer", BASE_DIR / "03_dl_engineer.py"
			)
			module = importlib.util.module_from_spec(spec)
			assert spec.loader is not None
			spec.loader.exec_module(module)
			return module
		except Exception:
			return None

	@property
	def models_ready(self) -> bool:
		return all((MODEL_DIR / filename).exists() for filename in (
			"disaster_cnn.pt",
			"water_level_lstm.pt",
			"water_level_scaler.joblib",
			"severity_model.joblib",
		))

	def health_check(self) -> dict[str, Any]:
		return {
			"status": "healthy" if self.module and self.models_ready else "unavailable",
			"models_ready": self.models_ready,
			"evaluation_available": bool(self.metrics),
			"error": self.load_error,
		}

	def summary(self) -> dict[str, Any]:
		"""Return dashboard-safe Stage02 metrics and artifact status."""
		cnn = self.metrics.get("cnn", {})
		lstm = self.metrics.get("lstm", {})
		return {
			"status": self.health_check()["status"],
			"models": {
				"cnn": (MODEL_DIR / "disaster_cnn.pt").exists(),
				"lstm": (MODEL_DIR / "water_level_lstm.pt").exists(),
				"severity": (MODEL_DIR / "severity_model.joblib").exists(),
			},
			"cnn": {
				"accuracy": cnn.get("accuracy"),
				"f1": cnn.get("f1_macro"),
				"flooded_recall": cnn.get("classification_report", {}).get("0", {}).get("recall"),
				"test_samples": cnn.get("test_samples"),
			},
			"lstm": {
				"mae": lstm.get("mae"),
				"rmse": lstm.get("rmse"),
				"test_samples": lstm.get("test_samples"),
			},
			"image_samples": sum(
				len(list((RAW_DIR / "Flood_Image_Dataset" / label).glob("*")))
				for label in ("flooded", "unflooded")
				if (RAW_DIR / "Flood_Image_Dataset" / label).exists()
			),
			"water_level_lookback": lstm.get("lookback"),
		}

	def predict_image(self, image_path: str) -> dict[str, Any]:
		if not self.module or not self.models_ready:
			raise RuntimeError("Stage02 models are unavailable")
		path = Path(image_path)
		if not path.is_absolute():
			path = BASE_DIR / path
		path = path.resolve()
		try:
			path.relative_to(RAW_DIR.resolve())
		except ValueError as error:
			raise ValueError("image_path must point inside Stage02_DL/data/raw") from error
		if not path.is_file():
			raise ValueError(f"Image file not found: {path}")
		result = self.module.predict_image(path)
		result["image"] = str(path.relative_to(BASE_DIR)).replace("\\", "/")
		return result

	def forecast_water_level(self, water_levels: list[float]) -> dict[str, Any]:
		return self.forecast_water_levels(water_levels, 1)

	def forecast_water_levels(self, water_levels: list[float], horizon: int = 6) -> dict[str, Any]:
		if not self.module or not self.models_ready:
			raise RuntimeError("Stage02 models are unavailable")
		try:
			values = [float(value) for value in water_levels]
		except (TypeError, ValueError) as error:
			raise ValueError("water_levels must contain numeric values") from error
		try:
			steps = int(horizon)
		except (TypeError, ValueError) as error:
			raise ValueError("horizon must be an integer") from error
		forecasts = self.module.forecast_water_levels(values, steps)
		return {
			"forecast_water_level": forecasts[0],
			"forecast_water_levels": forecasts,
			"horizon": steps,
			"lookback": self.metrics.get("lstm", {}).get("lookback", 24),
		}


dl_integration_engine = DLIntegrationEngine()
