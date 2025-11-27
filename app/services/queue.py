import json
import uuid
from datetime import datetime
from typing import List, Optional, Any
from redis.asyncio import Redis
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.task import TaskCreate, TaskResponse
from app.utils.model import get_batch_outputs


class QueueService:
    def __init__(self, redis: Redis, model: Any = None):
        self.redis = redis
        self.model = model
        # Key to track when the current batch started
        self.timestamp_key = f"{settings.QUEUE_NAME}:start_time"

    async def enqueue_and_process(self, task_in: TaskCreate) -> TaskResponse:
        # (This part stays mostly the same, handling the 'active' push logic)
        task_id = str(uuid.uuid4())
        current_time = datetime.utcnow()
        timestamp_str = current_time.isoformat()

        task_data = {
            "id": task_id,
            "status": "queued",
            "created_at": timestamp_str,
            "text_content": task_in.text_content
        }

        async with self.redis.pipeline() as pipe:
            pipe.hset(settings.DATA_STORE_NAME, task_id, json.dumps(task_data))
            pipe.rpush(settings.QUEUE_NAME, task_id)
            pipe.llen(settings.QUEUE_NAME)
            pipe.get(self.timestamp_key)
            results = await pipe.execute()

        queue_len = results[2]
        start_time_bytes = results[3]

        # Initialize timer if this is the first item
        if queue_len == 1 or not start_time_bytes:
            await self.redis.set(self.timestamp_key, timestamp_str)

        # Check if we should flush immediately (Size or Time)
        await self._check_and_flush_logic(queue_len, start_time_bytes)

        return TaskResponse(**task_data)

    async def check_time_limit(self):
        """
        Public method to be called by the background task.
        Checks current queue state and flushes if timeout is reached.
        """
        async with self.redis.pipeline() as pipe:
            pipe.llen(settings.QUEUE_NAME)
            pipe.get(self.timestamp_key)
            results = await pipe.execute()

        queue_len = results[0]
        start_time_bytes = results[1]

        if queue_len > 0:
            await self._check_and_flush_logic(queue_len, start_time_bytes)

    async def _check_and_flush_logic(self, queue_len: int, start_time_bytes: Any):
        """
        Shared logic to determine if a flush is needed based on size or time.
        """
        should_flush = False
        current_time = datetime.utcnow()

        # 1. Logic for Batch Size Check
        if queue_len >= settings.BATCH_SIZE:
            should_flush = True

        # 2. Logic for Timeout Check (only if we have a start time)
        elif start_time_bytes:
            try:
                start_time = datetime.fromisoformat(start_time_bytes.decode('utf-8'))
                timeout_seconds = getattr(settings, "BATCH_TIMEOUT", 30)

                if (current_time - start_time).total_seconds() > timeout_seconds:
                    should_flush = True
            except Exception:
                # If date parsing fails, ignore validation to prevent crash
                pass

        if should_flush:
            if not self.model:
                # Should probably log this in real app
                print("Cannot flush: Model not loaded")
                return
            await self._flush_queue(queue_len)

    async def _flush_queue(self, count: int):
        task_ids = await self.redis.lpop(settings.QUEUE_NAME, count)
        if not task_ids:
            return

        # Reset the timer immediately as the batch is being processed
        await self.redis.delete(self.timestamp_key)

        raw_data_list = await self.redis.hmget(settings.DATA_STORE_NAME, task_ids)

        # Prepare data for your function: Mapping[id, text]
        documents_map = {}

        for raw in raw_data_list:
            if raw:
                item = json.loads(raw)
                documents_map[item['id']] = item['text_content']

        # Run your synchronous BERT function in a separate thread
        processed_results = await run_in_threadpool(
            get_batch_outputs,
            self.model,
            documents_map
        )

        print(f"--- FLUSHED & PROCESSED {len(processed_results)} ITEMS ---")
        if processed_results:
            print(f"Sample Output: {list(processed_results.values())[0]}")

    async def get_item_by_id(self, item_id: str) -> Optional[TaskResponse]:
        raw_data = await self.redis.hget(settings.DATA_STORE_NAME, item_id)
        if not raw_data:
            return None
        return TaskResponse(**json.loads(raw_data))

    async def get_all_queue_items(self) -> List[TaskResponse]:
        task_ids = await self.redis.lrange(settings.QUEUE_NAME, 0, -1)
        if not task_ids:
            return []
        raw_data_list = await self.redis.hmget(settings.DATA_STORE_NAME, task_ids)
        return [TaskResponse(**json.loads(x)) for x in raw_data_list if x]