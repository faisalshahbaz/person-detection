from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    jwt_secret: str = "dev-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480
    ml_internal_token: str = "dev-ml-token"
    ml_service_url: str = "http://ml-service:9000"
    redis_url: str = "redis://redis:6379/0"
    max_global_sessions: int = 10
    max_user_sessions: int = 2
    max_inflight_infer: int = 8
    ml_request_timeout_sec: float = 15.0
    demo_users: str = "admin:admin123,demo:demo123"
    upload_dir: str = "/tmp/uploads"
    ml_circuit_fail_threshold: int = 5
    ml_circuit_open_sec: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
