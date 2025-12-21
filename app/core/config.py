from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Redis Queue"

    # Dynamic Redis URL - localhost for local development, host.docker.internal for Docker
    
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_URL: str = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}/0"

    # Redis Keys
    QUEUE_NAME: str = "task_queue"
    DATA_STORE_NAME: str = "task_data"
    STATS_KEY: str = "server_stats"

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