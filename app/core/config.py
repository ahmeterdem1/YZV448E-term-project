from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Redis Queue"
    REDIS_URL: str = "redis://host.docker.internal:6379/0"

    # Redis Keys
    QUEUE_NAME: str = "task_queue"
    DATA_STORE_NAME: str = "task_data"  # Now used as a key prefix (e.g., task_data:uuid)

    # Logic Settings
    BATCH_SIZE: int = 5
    BATCH_TIMEOUT: int = 30

    # New Config: TTL in seconds (default 1 day)
    DOCUMENT_TTL: int = 86400

    class Config:
        env_file = ".env"


settings = Settings()