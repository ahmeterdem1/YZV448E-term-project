import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from redis.asyncio import Redis
from loguru import logger

from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging
from app.utils.model import load_model
from app.services.queue import QueueService

# Setup logging first
setup_logging()

# Global dictionary to hold models
ml_models = {}


async def process_queue_periodically():
    """
    Background task that checks the queue every 5 seconds
    to see if items have timed out.
    """
    logger.info("Starting background queue processor")
    redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)

    while True:
        try:
            await asyncio.sleep(5)

            if "bert" in ml_models:
                service = QueueService(redis_client, ml_models["bert"])
                await service.check_time_limit()

        except asyncio.CancelledError:
            logger.info("Background task cancelled, shutting down gracefully")
            break
        except Exception as e:
            logger.error(f"Error in background task: {e}")
            await asyncio.sleep(5)

    await redis_client.close()
    logger.info("Background queue processor stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting FastAPI application")
    
    # 1. Load the BERT model
    logger.info("Loading ML models...")
    try:
        ml_models["bert"] = load_model()
        logger.success("✅ BERT model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load BERT model: {e}")
        raise

    # 2. Start the background task
    logger.info("Starting background tasks...")
    task = asyncio.create_task(process_queue_periodically())
    logger.success("✅ Background tasks started")

    yield

    # 3. Clean up
    logger.info("🛑 Shutting down application...")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    ml_models.clear()
    logger.info("✅ Application shutdown complete")


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health_check():
    model_loaded = "bert" in ml_models
    logger.info(f"Health check - Model loaded: {model_loaded}")
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)