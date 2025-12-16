from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from typing import List
from redis.asyncio import Redis
from loguru import logger

from app.core.config import settings
from app.schemas.task import TaskCreate, TaskResponse
from app.services.queue import QueueService

router = APIRouter()


async def get_redis() -> Redis:
    client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        yield client
    finally:
        await client.close()


@router.post("/process-text", response_model=TaskResponse, status_code=201)
async def upload_text(
        request: Request,
        task: TaskCreate,
        background_tasks: BackgroundTasks,
        redis: Redis = Depends(get_redis)
):
    """
    Uploads a text string.
    Checks for text length and queue capacity limits before accepting.
    If Queue size >= BATCH_SIZE, triggers a flush automatically.
    """
    # 1. Validation: Check Text Length
    if len(task.text_content) > settings.MAX_TEXT_LENGTH:
        logger.warning(f"❌ Request rejected: Text too long ({len(task.text_content)} chars)")
        raise HTTPException(
            status_code=413,  # Payload Too Large
            detail=f"Text content exceeds maximum allowed length of {settings.MAX_TEXT_LENGTH} characters."
        )

    # 2. Validation: Check Queue Capacity
    # We check directly against Redis before initializing the service
    current_queue_size = await redis.llen(settings.QUEUE_NAME)
    if current_queue_size >= settings.MAX_QUEUE_SIZE:
        logger.warning(f"❌ Request rejected: Queue full ({current_queue_size}/{settings.MAX_QUEUE_SIZE})")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="System is under heavy load. Please try again later."
        )

    logger.info(f"📝 New text processing request - Length: {len(task.text_content)} chars")

    try:
        from app.main import ml_models
        model = ml_models.get("bert")

        if not model:
            logger.error("❌ BERT model not loaded")
            raise HTTPException(status_code=503, detail="Model not available")

        service = QueueService(redis, model)
        result = await service.enqueue_and_process(task)

        logger.success(f"✅ Task created successfully - ID: {result.id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/text/{item_id}", response_model=TaskResponse)
async def get_text_by_id(
        item_id: str,
        redis: Redis = Depends(get_redis)
):
    logger.info(f"🔍 Retrieving task: {item_id}")

    try:
        service = QueueService(redis)
        item = await service.get_item_by_id(item_id)

        if not item:
            logger.warning(f"⚠️ Task not found: {item_id}")
            raise HTTPException(status_code=404, detail="Item not found")

        logger.success(f"✅ Task retrieved: {item_id} - Status: {item.status}")
        return item

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving task {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue", response_model=List[TaskResponse])
async def peek_queue(redis: Redis = Depends(get_redis)):
    logger.info("🔍 Checking queue status")

    try:
        service = QueueService(redis)
        items = await service.get_all_queue_items()

        logger.info(f"📋 Queue status - Items: {len(items)}")
        return items

    except Exception as e:
        logger.error(f"❌ Error checking queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))