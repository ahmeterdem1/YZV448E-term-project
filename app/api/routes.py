from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from typing import List
from redis.asyncio import Redis

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
    If Queue size >= BATCH_SIZE, triggers a flush automatically using the loaded BERT model.
    """
    # Access the model loaded in lifespan via app.state (if using state)
    # or via the global ml_models dict from main (common pattern)

    # Importing here to avoid circular imports at top level if structure is complex,
    # but accessing via request.app is cleaner in this setup if we attached it to state.
    # However, since we used a global dict in main, let's grab it from request.state
    # OR simpler: use the global if available.

    # Best Practice: Pass it via Request context if attached to state, or import the global.
    # Let's rely on the main.py `ml_models` by importing it dynamically or accessing request.state
    # if we had attached it there. Since we didn't attach to state in the provided main.py,
    # let's update how we retrieve it.

    from app.main import ml_models
    model = ml_models.get("bert")

    service = QueueService(redis, model)
    result = await service.enqueue_and_process(task)
    return result


@router.get("/text/{item_id}", response_model=TaskResponse)
async def get_text_by_id(
        item_id: str,
        redis: Redis = Depends(get_redis)
):
    service = QueueService(redis)
    item = await service.get_item_by_id(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.get("/queue", response_model=List[TaskResponse])
async def peek_queue(redis: Redis = Depends(get_redis)):
    service = QueueService(redis)
    return await service.get_all_queue_items()