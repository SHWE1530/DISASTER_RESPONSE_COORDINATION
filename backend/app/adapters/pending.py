from typing import Any

from .interfaces import DataAdapter, EDAAdapter, EvaluationAdapter, MLAdapter


class PendingDataAdapter(DataAdapter):
    def status(self) -> dict[str, Any]:
        return {"status": "pending", "detail": "PENDING TEAM INPUT", "required_inputs": ["final data artifact and schema"]}


class PendingEDAAdapter(EDAAdapter):
    def status(self) -> dict[str, Any]:
        return {"status": "pending", "detail": "PENDING TEAM INPUT", "required_inputs": ["validated EDA findings and data-quality notes"]}


class PendingMLAdapter(MLAdapter):
    def status(self) -> dict[str, Any]:
        return {"status": "pending", "detail": "PENDING TEAM INPUT", "required_inputs": ["model artifact, feature contract, and inference contract"]}

    def predict(self, features: dict[str, Any]) -> Any:
        raise NotImplementedError("PENDING TEAM INPUT: ML model is not connected")


class PendingEvaluationAdapter(EvaluationAdapter):
    def status(self) -> dict[str, Any]:
        return {"status": "pending", "detail": "PENDING TEAM INPUT", "required_inputs": ["evaluation report, metrics, and acceptance criteria"]}