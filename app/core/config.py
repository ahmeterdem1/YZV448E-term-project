from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Redis Queue"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Redis Keys
    QUEUE_NAME: str = "task_queue"  # The List (for ordering)
    DATA_STORE_NAME: str = "task_data"  # The Hash (for storage)

    # Logic Settings
    BATCH_SIZE: int = 5  # Flush triggers when queue hits this size
    BATCH_TIMEOUT: int = 30  # Seconds before timeout flush

    class Config:
        env_file = ".env"


settings = Settings()