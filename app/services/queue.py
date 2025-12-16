import json
import uuid
from datetime import datetime
from typing import List, Optional, Any
from redis.asyncio import Redis
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from app.core.config import settings
from app.schemas.task import TaskCreate, TaskResponse
from app.utils.model import get_batch_outputs


class QueueService:
    def __init__(self, redis: Redis, model: Any = None):
        self.redis = redis
        self.model = model
        self.timestamp_key = f"{settings.QUEUE_NAME}:start_time"
        # New: Key for the distributed lock
        self.lock_key = f"{settings.QUEUE_NAME}:processing_lock"

    # ... [Previous methods: _get_task_key, enqueue_and_process, check_time_limit remain unchanged] ...

    # We only need to copy the helper methods if they aren't already in your context,
    # but strictly we only need to change _flush_queue.

    # ... [Include previous enqueue_and_process and check_time_limit code here if creating a fresh file] ...

    def _get_task_key(self, task_id: str) -> str:
        """Helper to construct the redis key for a task"""
        return f"{settings.DATA_STORE_NAME}:{task_id}"

    async def enqueue_and_process(self, task_in: TaskCreate) -> TaskResponse:
        # ... [Same code as previous step] ...
        task_id = str(uuid.uuid4())
        current_time = datetime.utcnow()
        timestamp_str = current_time.isoformat()
        task_key = self._get_task_key(task_id)

        task_data = {
            "id": task_id,
            "status": "queued",
            "created_at": timestamp_str,
            "text_content": task_in.text_content
        }

        try:
            async with self.redis.pipeline() as pipe:
                pipe.set(task_key, json.dumps(task_data), ex=settings.DOCUMENT_TTL)
                pipe.rpush(settings.QUEUE_NAME, task_id)
                pipe.llen(settings.QUEUE_NAME)
                pipe.get(self.timestamp_key)
                results = await pipe.execute()

            queue_len = results[2]
            start_time_bytes = results[3]

            if queue_len == 1 or not start_time_bytes:
                await self.redis.set(self.timestamp_key, timestamp_str)

            await self._check_and_flush_logic(queue_len, start_time_bytes)
            return TaskResponse(**task_data)
        except Exception as e:
            logger.error(f"❌ Redis error during enqueue: {e}")
            raise

    async def check_time_limit(self):
        # ... [Same code as previous step] ...
        async with self.redis.pipeline() as pipe:
            pipe.llen(settings.QUEUE_NAME)
            pipe.get(self.timestamp_key)
            results = await pipe.execute()

        queue_len = results[0]
        start_time_bytes = results[1]

        if queue_len > 0:
            await self._check_and_flush_logic(queue_len, start_time_bytes)

    async def _check_and_flush_logic(self, queue_len: int, start_time_bytes: Any):
        # ... [Same logic code, but we will make it smarter about the lock] ...
        should_flush = False
        flush_reason = ""
        current_time = datetime.utcnow()

        if queue_len >= settings.BATCH_SIZE:
            should_flush = True
            flush_reason = f"batch_size ({queue_len} >= {settings.BATCH_SIZE})"
        elif start_time_bytes:
            # ... [Timeout logic same as before] ...
            try:
                if isinstance(start_time_bytes, bytes):
                    start_time_str = start_time_bytes.decode('utf-8')
                else:
                    start_time_str = start_time_bytes
                start_time = datetime.fromisoformat(start_time_str)
                timeout_seconds = getattr(settings, "BATCH_TIMEOUT", 30)
                elapsed = (current_time - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    should_flush = True
                    flush_reason = f"timeout ({elapsed:.1f}s > {timeout_seconds}s)"
            except Exception:
                pass

        if should_flush:
            if not self.model:
                logger.error("❌ Cannot flush: Model not loaded")
                return
            await self._flush_queue(queue_len)

    async def _flush_queue(self, count: int):
        """
        Flushes the queue with a distributed lock to ensure single-threaded model inference.
        """
        # 1. Try to acquire the lock (Set key if not exists, expire in 60s safety)
        # We use a unique ID for the lock value so we only delete OUR lock
        lock_id = str(uuid.uuid4())
        is_locked = await self.redis.set(
            self.lock_key,
            lock_id,
            nx=True,
            ex=60
        )

        if not is_locked:
            logger.info("🔒 Flush skipped - Another worker is processing the batch")
            return

        try:
            # 2. Re-check queue length inside the lock!
            # It's possible someone drained it just before we got the lock
            # or we are competing with another thread.

            # Note: We stick to lpop logic. If lpop returns None, we just exit.
            logger.info(f"🚀 Flushing queue - Processing up to {count} items")

            task_ids = await self.redis.lpop(settings.QUEUE_NAME, count)
            if not task_ids:
                logger.warning("⚠️ No items to flush (queue was empty)")
                return

            # Reset the timer
            await self.redis.delete(self.timestamp_key)

            # 3. Retrieve Data
            task_keys = [self._get_task_key(tid) for tid in task_ids]
            raw_data_list = await self.redis.mget(task_keys)

            documents_map = {}
            for raw in raw_data_list:
                if raw:
                    item = json.loads(raw)
                    documents_map[item['id']] = item['text_content']

            if not documents_map:
                logger.warning("⚠️ Tasks popped but data not found")
                return

            # 4. Inference (Protected by Lock)
            logger.info(f"🤖 Running BERT inference on {len(documents_map)} documents")
            start_time = datetime.utcnow()

            processed_results = await run_in_threadpool(
                get_batch_outputs,
                self.model,
                documents_map
            )

            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.success(f"✅ Batch processed in {processing_time:.2f}s")

            # 5. Update Results
            updated_count = 0
            for task_id, processed_text in processed_results.items():
                task_key = self._get_task_key(task_id)
                raw_data = await self.redis.get(task_key)
                if raw_data:
                    task_data = json.loads(raw_data)
                    task_data['text_content'] = processed_text
                    task_data['status'] = 'processed'
                    task_data['processed_at'] = datetime.utcnow().isoformat()

                    await self.redis.set(
                        task_key,
                        json.dumps(task_data),
                        ex=settings.DOCUMENT_TTL
                    )
                    updated_count += 1

            logger.success(f"💾 Updated {updated_count}/{len(processed_results)} tasks")

        except Exception as e:
            logger.error(f"❌ Error during batch processing: {e}")
            raise

        finally:
            # 6. Release Lock (Lua script to ensure we only delete OUR lock)
            # Simple version: Check if value matches lock_id, then delete
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            await self.redis.eval(script, 1, self.lock_key, lock_id)
            logger.debug("🔓 Lock released")

    # ... [Previous get_item_by_id and get_all_queue_items helper methods] ...
    async def get_item_by_id(self, item_id: str) -> Optional[TaskResponse]:
        task_key = self._get_task_key(item_id)
        raw_data = await self.redis.get(task_key)
        if not raw_data:
            return None
        return TaskResponse(**json.loads(raw_data))

    async def get_all_queue_items(self) -> List[TaskResponse]:
        task_ids = await self.redis.lrange(settings.QUEUE_NAME, 0, -1)
        if not task_ids:
            return []
        task_keys = [self._get_task_key(tid) for tid in task_ids]
        raw_data_list = await self.redis.mget(task_keys)
        return [TaskResponse(**json.loads(x)) for x in raw_data_list if x]