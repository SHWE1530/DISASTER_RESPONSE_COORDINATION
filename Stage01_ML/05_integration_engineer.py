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

# Fields that must be finite numbers. NaN/inf used to pass straight through the
# preprocessor and produce a confident "Severe" prediction from missing data.
NUMERIC_REQUIRED_COLUMNS = (
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

# Reference payload used by health_check() to prove the model can actually
# score a request, rather than only that a file was unpickled.
SMOKE_TEST_RECORD = {
	"timestamp": "01-07-2025 12:00",
	"state": "Maharashtra",
	"district": "Pune",
	"rainfall_mm": 25.0,
	"river_level_m": 4.5,
	"river_level_threshold_m": 6.0,
	"emergency_calls": 20,
	"road_closures": 0,
	"bridge_closures": 0,
	"flood_history_count": 2,
	"population_affected": 1000,
	"water_level_change_m": 0.1,
}


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
		self.feature_names: list[str] = []
		self.global_weights: np.ndarray | None = None
		self._load_pipeline()
		self._build_explainer()

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

	def _build_explainer(self) -> None:
		"""Recover transformed feature names and global weights for per-row explanations.

		03_ml_engineer.py computes genuine per-instance top factors but only writes
		them to CSV; serving used to return the same three global feature names for
		every request while the dashboard labelled them "top factors" for that
		prediction. This reconstructs the machinery needed to explain each row.
		"""
		if self.pipeline is None or self.preprocessor is None or self.model is None:
			return
		try:
			numeric = list(self.pipeline["feature_cols_num"])
			categorical = list(self.pipeline["feature_cols_cat"])
			encoder = self.preprocessor.named_transformers_.get("cat")
			categorical_names = (
				list(encoder.get_feature_names_out(categorical))
				if encoder is not None and hasattr(encoder, "get_feature_names_out")
				else list(categorical)
			)
			self.feature_names = numeric + categorical_names

			model = self.model
			if hasattr(model, "feature_importances_"):
				weights = np.asarray(model.feature_importances_, dtype=float)
			elif hasattr(model, "coef_"):
				weights = np.mean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
			elif hasattr(model, "fitted_base_models"):
				stacked = [
					np.asarray(base.feature_importances_, dtype=float)
					for base in model.fitted_base_models.values()
					if hasattr(base, "feature_importances_")
				]
				weights = np.mean(stacked, axis=0) if stacked else None
			else:
				weights = None

			if weights is not None and len(weights) == len(self.feature_names):
				self.global_weights = weights
		except Exception:
			# Explanations are a presentation concern; never let them break scoring.
			self.feature_names = []
			self.global_weights = None

	def _instance_top_factors(self, transformed_row: np.ndarray, top_n: int = 3) -> list[str]:
		"""Rank this row's features by |value x global weight| contribution."""
		if self.global_weights is None or not self.feature_names:
			return self.feature_importance
		contributions = np.asarray(transformed_row, dtype=float) * self.global_weights
		order = np.argsort(np.abs(contributions))[::-1][:top_n]
		return [self.feature_names[index] for index in order]

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

		# Reject non-finite numerics. NaN previously flowed through StandardScaler
		# and XGBoost untouched and came back as "Severe" with 0.9999999
		# confidence -- a confident answer derived from missing data.
		for column in NUMERIC_REQUIRED_COLUMNS:
			values = pd.to_numeric(frame[column], errors="coerce")
			if not np.isfinite(values.to_numpy(dtype="float64")).all():
				raise ValueError(
					f"Field '{column}' must be a finite number "
					"(received NaN, infinity, or a non-numeric value)"
				)
			frame[column] = values

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
		results = self._format_predictions(frame, predictions, probabilities, inverse_map, transformed)
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
		results = self._format_predictions(
			frame, predictions, probabilities, pipeline["inv_target_map"], transformed
		)
		return {"count": len(results), "predictions": results}

	def _format_predictions(
		self,
		frame: pd.DataFrame,
		predictions: np.ndarray,
		probabilities: np.ndarray | None,
		inverse_map: dict[int, str],
		transformed: np.ndarray | None = None,
	) -> list[dict[str, Any]]:
		results = []
		transformed_matrix = np.asarray(transformed) if transformed is not None else None
		for index, prediction in enumerate(predictions):
			risk_category = inverse_map[int(prediction)]
			probability = probabilities[index] if probabilities is not None else None
			risk_score = float(probability[2]) if probability is not None and len(probability) > 2 else None
			confidence = float(np.max(probability)) if probability is not None else None
			row = frame.iloc[index]

			if transformed_matrix is not None and index < len(transformed_matrix):
				top_factors = self._instance_top_factors(transformed_matrix[index])
			else:
				top_factors = self.feature_importance

			results.append(
				{
					"zone": str(row["district"]),
					"state": str(row["state"]),
					"timestamp": row["timestamp"].strftime("%d-%m-%Y %H:%M"),
					"risk_category": risk_category,
					"risk_score": risk_score,
					"confidence": confidence,
					# Ranked for THIS row (|feature value x model weight|).
					"top_factors": top_factors,
					# Dataset-wide ranking, constant across requests.
					"global_top_factors": self.feature_importance,
				}
			)
		return results

	def health_check(self) -> dict[str, Any]:
		"""Report readiness by actually scoring a reference record.

		Checking only that the pickle loaded is not enough: a model can unpickle
		cleanly under a mismatched library version and then fail at predict time.
		"""
		if self.model is None:
			return {
				"status": "unavailable",
				"model_loaded": False,
				"inference_ok": False,
				"model_path": str(self.model_path),
				"error": self.load_error,
			}

		inference_ok = True
		inference_error: str | None = None
		try:
			self.predict(dict(SMOKE_TEST_RECORD))
		except Exception as exc:
			inference_ok = False
			inference_error = f"{type(exc).__name__}: {exc}"

		return {
			"status": "healthy" if inference_ok else "degraded",
			"model_loaded": True,
			"inference_ok": inference_ok,
			"model_path": str(self.model_path),
			"error": self.load_error or inference_error,
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
