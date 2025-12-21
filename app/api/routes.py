from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates  # Import Jinja2Templates
from typing import List, Dict, Any
from redis.asyncio import Redis
from loguru import logger
from datetime import datetime
import json
from pathlib import Path

from app.core.config import settings
from app.schemas.task import TaskCreate, TaskResponse
from app.services.queue import QueueService

router = APIRouter()

# Setup templates directory
# Using Path logic to ensure it works regardless of where the app is run from
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


async def get_redis() -> Redis:
    client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        yield client
    finally:
        await client.close()


async def check_rate_limit(request: Request, redis: Redis = Depends(get_redis)):
    """Rate limiter dependency: Fixed window counter per IP."""
    client_ip = request.client.host or "127.0.0.1"
    key = f"ratelimit:{client_ip}"

    try:
        current_count = await redis.incr(key)
        if current_count == 1:
            await redis.expire(key, settings.RATE_LIMIT_WINDOW)

        if current_count > settings.RATE_LIMIT_REQUESTS:
            logger.warning(f"⛔ Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please slow down."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"⚠️ Rate limit check failed: {e}")


@router.get("/stats", status_code=200)
async def get_server_stats(redis: Redis = Depends(get_redis)):
    """JSON endpoint for raw statistics."""
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
    """
    Serves the dashboard template.
    The actual data fetching is done by JS in the template hitting /stats.
    """
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/processor", response_class=HTMLResponse)
async def get_processor(request: Request):
    """
    Serves the text processor UI with file upload and drag-drop support.
    """
    return templates.TemplateResponse("processor.html", {"request": request})


@router.post("/process-text", response_model=TaskResponse, status_code=201, dependencies=[Depends(check_rate_limit)])
async def upload_text(
        request: Request,
        task: TaskCreate,
        background_tasks: BackgroundTasks,
        redis: Redis = Depends(get_redis)
):
    if len(task.text_content) > settings.MAX_TEXT_LENGTH:
        raise HTTPException(status_code=413, detail="Text too long")

    current_queue_size = await redis.llen(settings.QUEUE_NAME)
    if current_queue_size >= settings.MAX_QUEUE_SIZE:
        raise HTTPException(status_code=503, detail="Queue full")

    logger.info(f"📝 New request - Length: {len(task.text_content)}")

    try:
        from app.main import ml_models
        model = ml_models.get("bert")
        if not model:
            raise HTTPException(status_code=503, detail="Model not loaded")

        service = QueueService(redis, model)
        result = await service.enqueue_and_process(task)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/text/{item_id}", response_model=TaskResponse)
async def get_text_by_id(item_id: str, redis: Redis = Depends(get_redis)):
    try:
        service = QueueService(redis)
        item = await service.get_item_by_id(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return item
    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue", response_model=List[TaskResponse])
async def peek_queue(redis: Redis = Depends(get_redis)):
    try:
        service = QueueService(redis)
        return await service.get_all_queue_items()
    except Exception as e:
        logger.error(f"❌ Queue error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
