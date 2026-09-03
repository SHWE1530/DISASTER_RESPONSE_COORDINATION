from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .core.config import get_settings
from .core.logging import configure_logging
from .middleware.request_logging import RequestLoggingMiddleware

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Integration foundation for the Disaster Response Coordination Stage-01 ML system.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
app.include_router(router, prefix=settings.api_prefix)