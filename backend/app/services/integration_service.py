from . import __name__ as _services_package
from ..adapters.pending import (
    PendingDataAdapter,
    PendingEDAAdapter,
    PendingEvaluationAdapter,
    PendingMLAdapter,
)
from ..schemas.common import ComponentStatus, IntegrationStatusResponse


class IntegrationService:
    def __init__(self) -> None:
        self.data = PendingDataAdapter()
        self.eda = PendingEDAAdapter()
        self.ml = PendingMLAdapter()
        self.evaluation = PendingEvaluationAdapter()

    def status(self) -> IntegrationStatusResponse:
        adapters = [self.data, self.eda, self.ml, self.evaluation]
        components = [
            ComponentStatus(name=adapter.component_name, **adapter.status()) for adapter in adapters
        ]
        return IntegrationStatusResponse(
            status="pending",
            message="PENDING TEAM INPUT: integration is waiting for all four upstream outputs.",
            components=components,
        )


integration_service = IntegrationService()