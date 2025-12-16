from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Redis Queue"
    REDIS_URL: str = "redis://host.docker.internal:6379/0"

    # Redis Keys
    QUEUE_NAME: str = "task_queue"
    DATA_STORE_NAME: str = "task_data"
    STATS_KEY: str = "server_stats"  # New key for statistics

    # Logic Settings
    BATCH_SIZE: int = 5
    BATCH_TIMEOUT: int = 30

    # TTL in seconds (default 1 day)
    DOCUMENT_TTL: int = 86400

    # Safety Limits
    MAX_QUEUE_SIZE: int = 1000
    MAX_TEXT_LENGTH: int = 10000

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 20
    RATE_LIMIT_WINDOW: int = 60

    class Config:
        env_file = ".env"


settings = Settings()