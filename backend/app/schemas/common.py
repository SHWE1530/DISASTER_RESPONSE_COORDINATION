from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ComponentStatus(BaseModel):
    name: str
    status: str
    detail: str
    required_inputs: list[str] = Field(default_factory=list)


class IntegrationStatusResponse(BaseModel):
    status: str
    message: str
    components: list[ComponentStatus]


class PredictionRequest(BaseModel):
    """Opaque payload until the ML team supplies the finalized input schema."""

    features: dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    status: str
    message: str
    prediction: Any | None = None