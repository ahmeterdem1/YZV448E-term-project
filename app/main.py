import asyncio
import psutil
from fastapi import FastAPI
from contextlib import asynccontextmanager
from redis.asyncio import Redis
from loguru import logger

from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging
from app.utils.model import load_model
from app.services.queue import QueueService
from app.services.training import TrainingService  # <--- Import this

setup_logging()

ml_models = {}


async def process_queue_periodically():
    """Checks queue for timeouts."""
    logger.info("Starting background queue processor")
    redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    while True:
        try:
            await asyncio.sleep(5)
            if "bert" in ml_models:
                service = QueueService(redis_client, ml_models["bert"])
                await service.check_time_limit()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in background task: {e}")
            await asyncio.sleep(5)
    await redis_client.close()
    logger.info("Background queue processor stopped")


async def collect_system_stats():
    """Collects CPU and RAM usage periodically."""
    logger.info("Starting system monitoring task")
    redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    await redis_client.hsetnx(settings.STATS_KEY, "startup_time", str(asyncio.get_running_loop().time()))
    while True:
        try:
            await asyncio.sleep(10)
            cpu_percent = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            async with redis_client.pipeline() as pipe:
                pipe.hset(settings.STATS_KEY, "system_cpu_percent", cpu_percent)
                pipe.hset(settings.STATS_KEY, "system_ram_percent", ram.percent)
                pipe.hset(settings.STATS_KEY, "system_ram_used_mb", ram.used // (1024 * 1024))
                pipe.hset(settings.STATS_KEY, "last_stats_update", str(asyncio.get_running_loop().time()))
                await pipe.execute()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in monitoring task: {e}")
            await asyncio.sleep(10)
    await redis_client.close()
    logger.info("System monitoring stopped")


# --- NEW FUNCTION ---
async def monitor_model_performance():
    """Periodically evaluate F5 score on the test set."""
    logger.info("Starting F5 score monitor")
    redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    trainer = TrainingService()

    # Check every 60 seconds (set low for debugging, increase for production)
    CHECK_INTERVAL = 60

    while True:
        try:
            # Wait first so we don't block startup
            await asyncio.sleep(CHECK_INTERVAL)

            if "bert" in ml_models:
                # Access the underlying HuggingFace model
                model_wrapper = ml_models["bert"]
                # Handle Hybrid wrapper
                hf_cleaner = model_wrapper.bert_model if hasattr(model_wrapper, 'bert_model') else model_wrapper

                # Run evaluation (in thread to avoid blocking loop)
                metrics = await asyncio.to_thread(trainer.evaluate_model, hf_cleaner, settings.TEST_DATASET_PATH)

                if metrics:
                    f5_score = metrics.get("f5", 0.0)
                    # Save to Redis so Dashboard can see it
                    await redis_client.hset(settings.STATS_KEY, "model_f5_score", f"{f5_score:.4f}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in F5 monitor: {e}")
            await asyncio.sleep(60)

    await redis_client.close()
    logger.info("Model monitor stopped")


# --------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting FastAPI application")

    logger.info("Loading ML models...")
    try:
        ml_models["bert"] = load_model(use_hybrid=True)
        logger.success("✅ Hybrid PII Cleaner (BERT + Regex) loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load PII cleaner: {e}")

    logger.info("Starting background tasks...")
    queue_task = asyncio.create_task(process_queue_periodically())
    monitor_task = asyncio.create_task(collect_system_stats())

    # --- ADD THIS LINE ---
    f5_task = asyncio.create_task(monitor_model_performance())
    # ---------------------

    logger.success("✅ Background tasks started")

    yield

    logger.info("🛑 Shutting down application...")
    queue_task.cancel()
    monitor_task.cancel()
    f5_task.cancel()  # Cancel on shutdown
    try:
        await queue_task
        await monitor_task
        await f5_task
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
    return {"status": "ok", "model_loaded": model_loaded}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)