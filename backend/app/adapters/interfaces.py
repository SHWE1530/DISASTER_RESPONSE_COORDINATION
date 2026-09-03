from abc import ABC, abstractmethod
from typing import Any


class TeamAdapter(ABC):
    component_name: str

    @abstractmethod
    def status(self) -> dict[str, Any]:
        """Return readiness information without assuming an upstream schema."""


class DataAdapter(TeamAdapter):
    component_name = "Data Engineer"


class EDAAdapter(TeamAdapter):
    component_name = "EDA Engineer"


class MLAdapter(TeamAdapter):
    component_name = "ML Engineer"

    @abstractmethod
    def predict(self, features: dict[str, Any]) -> Any:
        """Run a prediction once the finalized ML contract is connected."""


class EvaluationAdapter(TeamAdapter):
    component_name = "Evaluation Engineer"