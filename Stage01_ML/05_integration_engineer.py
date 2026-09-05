from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field

import uvicorn


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "data"
    / "models"
    / "ml_pipeline.joblib"
)

FRONTEND_DIR = (
    BASE_DIR
    / "frontend"
)


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

model_bundle = joblib.load(MODEL_PATH)

if isinstance(model_bundle, dict):
    preprocessor = model_bundle["preprocessor"]
    model = model_bundle["model"]
    feature_columns = (
        model_bundle["feature_cols_num"]
        + model_bundle["feature_cols_cat"]
    )
    inverse_target_map = model_bundle.get("inv_target_map", {})
else:
    preprocessor = None
    model = model_bundle
    feature_columns = None
    inverse_target_map = {}

if not hasattr(model, "multi_class"):
    model.multi_class = "auto"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Disaster Response AI",
    description="Real-time Zone Risk Prediction API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class RiskPredictionRequest(BaseModel):

    timestamp: str

    state: str

    district: str

    rainfall_mm: float = Field(..., ge=0)

    river_level_m: float = Field(..., ge=0)

    river_level_threshold_m: float = Field(..., ge=0)

    emergency_calls: int = Field(..., ge=0)

    road_closures: int = Field(..., ge=0)

    bridge_closures: int = Field(..., ge=0)

    flood_history_count: int = Field(..., ge=0)

    population_affected: int = Field(..., ge=0)

    water_level_change_m: float


# ============================================================
# FRONTEND
# ============================================================

@app.get("/", include_in_schema=False)
def home():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


@app.get("/style.css", include_in_schema=False)
def css():

    return FileResponse(
        FRONTEND_DIR / "style.css",
        media_type="text/css"
    )


@app.get("/script.js", include_in_schema=False)
def javascript():

    return FileResponse(
        FRONTEND_DIR / "script.js",
        media_type="application/javascript"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# PREDICT RISK
# ============================================================

@app.post("/predict-risk")
def predict_risk(
    request: RiskPredictionRequest
):

    try:

        data = {
            "timestamp": [request.timestamp],
            "state": [request.state],
            "district": [request.district],
            "rainfall_mm": [request.rainfall_mm],
            "river_level_m": [request.river_level_m],
            "river_level_threshold_m": [
                request.river_level_threshold_m
            ],
            "emergency_calls": [
                request.emergency_calls
            ],
            "road_closures": [
                request.road_closures
            ],
            "bridge_closures": [
                request.bridge_closures
            ],
            "flood_history_count": [
                request.flood_history_count
            ],
            "population_affected": [
                request.population_affected
            ],
            "water_level_change_m": [
                request.water_level_change_m
            ]
        }

        df = pd.DataFrame(data)

        # Timestamp processing
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            dayfirst=True
        )

        if df["timestamp"].isna().any():

            raise HTTPException(
                status_code=400,
                detail="Invalid timestamp."
            )

        # Time features
        df["hour"] = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month
        df["dayofweek"] = (
            df["timestamp"].dt.dayofweek
        )
        df["is_monsoon"] = df["month"].isin(
            [6, 7, 8, 9]
        ).astype(int)
        df["river_level_margin_m"] = (
            df["river_level_m"]
            - df["river_level_threshold_m"]
        )
        df["river_level_ratio"] = (
            df["river_level_m"]
            / (df["river_level_threshold_m"] + 1e-5)
        )

        df.drop(
            columns=["timestamp"],
            inplace=True
        )

        # Apply the same feature transformation used during training.
        model_input = (
            df[feature_columns]
            if feature_columns is not None
            else df
        )
        transformed_input = (
            preprocessor.transform(model_input)
            if preprocessor is not None
            else model_input
        )

        # Prediction
        prediction = model.predict(transformed_input)[0]
        prediction = inverse_target_map.get(prediction, prediction)

        prediction = str(prediction)

        # Confidence
        confidence: Optional[float] = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(transformed_input)[0]

            confidence = float(
                max(probabilities)
            )

        return {
            "district": request.district,
            "risk_level": prediction,
            "confidence": (
                round(confidence, 4)
                if confidence is not None
                else None
            ),
            "status": "prediction_successful"
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("DISASTER RESPONSE AI")
    print("Integration Engineer")
    print("=" * 60)

    print(
        "Dashboard: http://127.0.0.1:8000"
    )

    print(
        "API Docs: http://127.0.0.1:8000/docs"
    )

    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )