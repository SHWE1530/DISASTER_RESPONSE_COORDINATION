from fastapi import APIRouter, HTTPException, status

from ..core.config import get_settings
from ..schemas.common import HealthResponse, PredictionRequest, PredictionResponse
from ..services.integration_service import integration_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", service=settings.app_name, version=settings.app_version)


@router.get("/status", tags=["system"])
def integration_status() -> dict:
    return integration_service.status().model_dump()


@router.get("/integration-readiness", tags=["system"])
def integration_readiness() -> dict:
    return integration_service.status().model_dump()


@router.post("/model/predict", response_model=PredictionResponse, tags=["model"])
def predict(request: PredictionRequest) -> PredictionResponse:
    del request
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={"code": "MODEL_NOT_CONNECTED", "message": "PENDING TEAM INPUT: ML model is not connected."},
    )