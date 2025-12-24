from pydantic_settings import BaseSettings
import os
from pathlib import Path


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Redis Queue"

    # Dynamic Redis URL
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

    # --- Model Training & Registry Settings ---
    MODEL_REGISTRY_DIR: str = "models_registry"
    TRAIN_DATASET_PATH: str = "data/train.json"
    TEST_DATASET_PATH: str = "data/test.json"

    # Thresholds
    MIN_F5_SCORE: float = 0.85
    AUTO_TRAIN_ON_PERFORMANCE_DROP: bool = True
    CHECK_INTERVAL_SECONDS: int = 86400  # 24 hours

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure directories exist
Path(settings.MODEL_REGISTRY_DIR).mkdir(parents=True, exist_ok=True)
Path("data").mkdir(exist_ok=True)
