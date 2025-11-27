import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from redis.asyncio import Redis

from app.api.routes import router
from app.core.config import settings
from app.utils.model import load_model
from app.services.queue import QueueService

# Global dictionary to hold models
ml_models = {}


async def process_queue_periodically():
    """
    Background task that checks the queue every 5 seconds
    to see if items have timed out.
    """
    # Create a dedicated Redis connection for the background task
    redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)

    while True:
        try:
            # Sleep first to avoid hammering Redis immediately on startup
            await asyncio.sleep(5)

            # Check queue
            if "bert" in ml_models:
                service = QueueService(redis_client, ml_models["bert"])
                await service.check_time_limit()

        except asyncio.CancelledError:
            # Handle graceful shutdown
            break
        except Exception as e:
            print(f"Error in background task: {e}")
            await asyncio.sleep(5)  # Wait before retrying

    await redis_client.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load the BERT model
    ml_models["bert"] = load_model()

    # 2. Start the background task
    task = asyncio.create_task(process_queue_periodically())

    yield

    # 3. Clean up
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    ml_models.clear()


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": "bert" in ml_models
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)