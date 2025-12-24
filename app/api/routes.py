from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any
from redis.asyncio import Redis
from loguru import logger
from datetime import datetime
import json
import shutil
from pathlib import Path
import asyncio

from app.core.config import settings
from app.schemas.task import TaskCreate, TaskResponse
from app.services.queue import QueueService
from app.services.training import TrainingService

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ... [Keep existing dependencies: get_redis, check_rate_limit] ...
async def get_redis() -> Redis:
    client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        yield client
    finally:
        await client.close()


async def check_rate_limit(request: Request, redis: Redis = Depends(get_redis)):
    # ... [Keep existing code] ...
    client_ip = request.client.host or "127.0.0.1"
    key = f"ratelimit:{client_ip}"
    try:
        current_count = await redis.incr(key)
        if current_count == 1:
            await redis.expire(key, settings.RATE_LIMIT_WINDOW)
        if current_count > settings.RATE_LIMIT_REQUESTS:
            logger.warning(f"⛔ Rate limit exceeded for {client_ip}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"⚠️ Rate limit check failed: {e}")


# ... [Keep existing endpoints: /stats, /dashboard, /processor, /process-text, /text, /queue] ...

@router.get("/stats", status_code=200)
async def get_server_stats(redis: Redis = Depends(get_redis)):
    try:
        stats = await redis.hgetall(settings.STATS_KEY)
        if not stats:
            return {"status": "waiting_for_data"}

        formatted_stats = {}
        for k, v in stats.items():
            try:
                formatted_stats[k] = float(v) if "." in v else int(v)
            except ValueError:
                formatted_stats[k] = v

        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": formatted_stats
        }
    except Exception as e:
        logger.error(f"❌ Error retrieving stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/processor", response_class=HTMLResponse)
async def get_processor(request: Request):
    return templates.TemplateResponse("processor.html", {"request": request})


@router.post("/process-text", response_model=TaskResponse, status_code=201, dependencies=[Depends(check_rate_limit)])
async def upload_text(request: Request, task: TaskCreate, redis: Redis = Depends(get_redis)):
    # ... [Same as original] ...
    if len(task.text_content) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(status_code=413, detail="Text too long")
    current_queue_size = await redis.llen(settings.QUEUE_NAME)
    if current_queue_size >= settings.MAX_QUEUE_SIZE:
        raise HTTPException(status_code=503, detail="Queue full")
    try:
        from app.main import ml_models
        model = ml_models.get("bert")
        if not model:
            raise HTTPException(status_code=503, detail="Model not loaded")
        service = QueueService(redis, model)
        result = await service.enqueue_and_process(task)
        return result
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/text/{item_id}", response_model=TaskResponse)
async def get_text_by_id(item_id: str, redis: Redis = Depends(get_redis)):
    # ... [Same as original] ...
    try:
        service = QueueService(redis)
        item = await service.get_item_by_id(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return item
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue", response_model=List[TaskResponse])
async def peek_queue(redis: Redis = Depends(get_redis)):
    # ... [Same as original] ...
    try:
        service = QueueService(redis)
        return await service.get_all_queue_items()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- NEW ENDPOINTS ---

@router.post("/dataset/train", status_code=201)
async def upload_training_data(file: UploadFile = File(...)):
    """Upload a JSON file to be used as training data."""
    try:
        with open(settings.TRAIN_DATASET_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"📁 Training dataset uploaded to {settings.TRAIN_DATASET_PATH}")
        return {"status": "success", "message": "Training dataset uploaded"}
    except Exception as e:
        logger.error(f"Error uploading training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dataset/test", status_code=201)
async def upload_test_data(file: UploadFile = File(...)):
    """Upload a JSON file to be used as test/validation data."""
    try:
        with open(settings.TEST_DATASET_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"📁 Test dataset uploaded to {settings.TEST_DATASET_PATH}")
        return {"status": "success", "message": "Test dataset uploaded"}
    except Exception as e:
        logger.error(f"Error uploading test data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", status_code=202)
async def trigger_training(background_tasks: BackgroundTasks):
    """Manually trigger model training."""
    if not Path(settings.TRAIN_DATASET_PATH).exists():
        raise HTTPException(status_code=400, detail="Training dataset not found. Upload it first.")

    background_tasks.add_task(run_training_job)
    return {"status": "accepted", "message": "Training job started in background"}


@router.get("/models")
async def list_models():
    """List all models in the registry."""
    p = Path(settings.MODEL_REGISTRY_DIR)
    if not p.exists():
        return []
    models = [x.name for x in p.iterdir() if x.is_dir() and x.name.startswith("model_")]
    models.sort(reverse=True)
    return {"models": models}


async def run_training_job():
    """Wrapper to run training service in background task."""
    try:
        logger.info("🏋️‍♂️ Background training started...")
        trainer = TrainingService()
        # Note: In a production app, we should use a process pool or celery
        # to avoid blocking the event loop with heavy CPU operations.
        # For this example, we assume it's acceptable or user is aware.
        new_model_path = await asyncio.to_thread(trainer.train_model)

        logger.success(f"🏋️‍♂️ Training complete. New model at: {new_model_path}")

        # Reload model logic (Updating the global ml_models)
        from app.main import ml_models
        from app.utils.model import load_model

        # We need to reload the model in the main application
        ml_models["bert"] = await asyncio.to_thread(load_model, use_hybrid=True)
        logger.info("🔄 Application model reloaded with latest version.")

    except Exception as e:
        logger.error(f"❌ Training job failed: {e}")