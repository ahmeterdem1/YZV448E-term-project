from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Redis Queue"
    REDIS_URL: str = "redis://host.docker.internal:6379/0"

    # Redis Keys
    QUEUE_NAME: str = "task_queue"
    DATA_STORE_NAME: str = "task_data"

    # Logic Settings
    BATCH_SIZE: int = 5
    BATCH_TIMEOUT: int = 30

    # TTL in seconds (default 1 day)
    DOCUMENT_TTL: int = 86400

    # Safety Limits
    MAX_QUEUE_SIZE: int = 1000  # Max items pending in queue
    MAX_TEXT_LENGTH: int = 10000  # Max characters per request

    class Config:
        env_file = ".env"


settings = Settings()