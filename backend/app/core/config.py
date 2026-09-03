from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Crisis Command Integration API"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    frontend_origin: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()