"""Stage 01 ML integration layer.

Loads the serialized model contract and exposes a small API for the Flask
application without retraining or duplicating model-selection logic.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "data" / "models" / "ml_pipeline.joblib"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "data" / "outputs" / "feature_importance.csv"

RAW_REQUIRED_COLUMNS = (
	"timestamp",
	"state",
	"district",
	"rainfall_mm",
	"river_level_m",
	"river_level_threshold_m",
	"emergency_calls",
	"road_closures",
	"bridge_closures",
	"flood_history_count",
	"population_affected",
	"water_level_change_m",
)


class IntegrationEngine:
	"""Serve predictions from the persisted Stage 01 ML pipeline."""

	def __init__(self, model_path: Path = MODEL_PATH) -> None:
		self.model_path = Path(model_path)
		self.pipeline: dict[str, Any] | None = None
		self.preprocessor: Any = None
		self.model: Any = None
		self.metrics: dict[str, Any] = {}
		self.feature_importance = self._load_feature_importance()
		self.load_error: str | None = None
		self._load_pipeline()

	def _load_pipeline(self) -> None:
		if not self.model_path.exists():
			self.load_error = f"Model pipeline not found: {self.model_path}"
			return

		try:
			pipeline = joblib.load(self.model_path)
			required_keys = {
				"preprocessor",
				"model",
				"feature_cols_num",
				"feature_cols_cat",
				"target_map",
				"inv_target_map",
			}
			missing_keys = sorted(required_keys.difference(pipeline))
			if missing_keys:
				raise ValueError(f"Model pipeline is missing keys: {missing_keys}")

			self.pipeline = pipeline
			self.preprocessor = pipeline["preprocessor"]
			self.model = pipeline["model"]
			self._restore_column_transformer_compatibility(self.preprocessor)
		except Exception as exc:
			self.load_error = f"Unable to load model pipeline: {exc}"

	@staticmethod
	def _restore_column_transformer_compatibility(preprocessor: Any) -> None:
		"""Repair pickled ColumnTransformer metadata required by newer scikit-learn."""
		if preprocessor is None or not hasattr(preprocessor, "transformers_"):
			return
		if not hasattr(preprocessor, "_name_to_fitted_passthrough"):
			preprocessor._name_to_fitted_passthrough = {}
		for name, transformer, _ in preprocessor.transformers_:
			if transformer is not None:
				preprocessor._name_to_fitted_passthrough.setdefault(name, True)

	@staticmethod
	def _load_feature_importance() -> list[str]:
		if not FEATURE_IMPORTANCE_PATH.exists():
			return []
		importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
		if "feature" not in importance.columns:
			return []
		return importance["feature"].dropna().astype(str).head(3).tolist()

	def _require_pipeline(self) -> dict[str, Any]:
		if self.pipeline is None or self.preprocessor is None or self.model is None:
			raise RuntimeError(self.load_error or "Model pipeline is unavailable")
		return self.pipeline

	@staticmethod
	def _validate_records(records: Any) -> pd.DataFrame:
		if not isinstance(records, dict) and not isinstance(records, list):
			raise ValueError("Input must be a JSON object or a list of objects")

		rows = [records] if isinstance(records, dict) else records
		if not rows or not all(isinstance(row, dict) for row in rows):
			raise ValueError("Input records must be non-empty JSON objects")

		missing = sorted(set(RAW_REQUIRED_COLUMNS).difference(rows[0]))
		if missing:
			raise ValueError(f"Missing required fields: {missing}")

		frame = pd.DataFrame(rows)
		timestamp = pd.to_datetime(
			frame["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce"
		)
		if timestamp.isna().any():
			timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
		if timestamp.isna().any():
			raise ValueError("timestamp must be a valid date/time")
		frame["timestamp"] = timestamp
		return frame

	def _prepare_features(self, records: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
		pipeline = self._require_pipeline()
		frame = self._validate_records(records)
		frame["river_level_margin_m"] = (
			frame["river_level_m"] - frame["river_level_threshold_m"]
		)
		frame["river_level_ratio"] = frame["river_level_m"] / (
			frame["river_level_threshold_m"] + 1e-5
		)
		frame["hour"] = frame["timestamp"].dt.hour
		frame["month"] = frame["timestamp"].dt.month
		frame["dayofweek"] = frame["timestamp"].dt.dayofweek
		frame["is_monsoon"] = frame["month"].isin([6, 7, 8, 9]).astype(int)

		feature_columns = pipeline["feature_cols_num"] + pipeline["feature_cols_cat"]
		features = frame[feature_columns].copy()
		return frame, features

	def predict(self, records: dict[str, Any]) -> dict[str, Any]:
		frame, features = self._prepare_features(records)
		transformed = self.preprocessor.transform(features)
		predictions = np.asarray(self.model.predict(transformed))
		probabilities = (
			np.asarray(self.model.predict_proba(transformed))
			if hasattr(self.model, "predict_proba")
			else None
		)
		pipeline = self._require_pipeline()
		inverse_map = pipeline["inv_target_map"]
		results = self._format_predictions(frame, predictions, probabilities, inverse_map)
		return results[0]

	def predict_batch(self, records: list[dict[str, Any]]) -> dict[str, Any]:
		frame, features = self._prepare_features(records)
		transformed = self.preprocessor.transform(features)
		predictions = np.asarray(self.model.predict(transformed))
		probabilities = (
			np.asarray(self.model.predict_proba(transformed))
			if hasattr(self.model, "predict_proba")
			else None
		)
		pipeline = self._require_pipeline()
		results = self._format_predictions(frame, predictions, probabilities, pipeline["inv_target_map"])
		return {"count": len(results), "predictions": results}

	def _format_predictions(
		self,
		frame: pd.DataFrame,
		predictions: np.ndarray,
		probabilities: np.ndarray | None,
		inverse_map: dict[int, str],
	) -> list[dict[str, Any]]:
		results = []
		for index, prediction in enumerate(predictions):
			risk_category = inverse_map[int(prediction)]
			probability = probabilities[index] if probabilities is not None else None
			risk_score = float(probability[2]) if probability is not None and len(probability) > 2 else None
			confidence = float(np.max(probability)) if probability is not None else None
			row = frame.iloc[index]
			results.append(
				{
					"zone": str(row["district"]),
					"state": str(row["state"]),
					"timestamp": row["timestamp"].strftime("%d-%m-%Y %H:%M"),
					"risk_category": risk_category,
					"risk_score": risk_score,
					"confidence": confidence,
					"top_factors": self.feature_importance,
				}
			)
		return results

	def health_check(self) -> dict[str, Any]:
		return {
			"status": "healthy" if self.model is not None else "unavailable",
			"model_loaded": self.model is not None,
			"model_path": str(self.model_path),
			"error": self.load_error,
		}

	def get_model_info(self) -> dict[str, Any]:
		pipeline = self._require_pipeline()
		return {
			"model_name": pipeline.get("model_name", "unknown"),
			"model_path": str(self.model_path),
			"numeric_features": pipeline["feature_cols_num"],
			"categorical_features": pipeline["feature_cols_cat"],
			"target_map": pipeline["target_map"],
		}


integration_engine = IntegrationEngine()
