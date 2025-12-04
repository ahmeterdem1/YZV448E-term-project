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
        # Key to track when the current batch started
        self.timestamp_key = f"{settings.QUEUE_NAME}:start_time"

    async def enqueue_and_process(self, task_in: TaskCreate) -> TaskResponse:
        task_id = str(uuid.uuid4())
        current_time = datetime.utcnow()
        timestamp_str = current_time.isoformat()
        
        logger.info(f"📥 Enqueueing task: {task_id}")

        task_data = {
            "id": task_id,
            "status": "queued",
            "created_at": timestamp_str,
            "text_content": task_in.text_content
        }

        try:
            async with self.redis.pipeline() as pipe:
                pipe.hset(settings.DATA_STORE_NAME, task_id, json.dumps(task_data))
                pipe.rpush(settings.QUEUE_NAME, task_id)
                pipe.llen(settings.QUEUE_NAME)
                pipe.get(self.timestamp_key)
                results = await pipe.execute()

            queue_len = results[2]
            start_time_bytes = results[3]
            
            logger.info(f"📋 Queue length after enqueue: {queue_len}")

            # Initialize timer if this is the first item
            if queue_len == 1 or not start_time_bytes:
                await self.redis.set(self.timestamp_key, timestamp_str)
                logger.info("⏱️ Started new batch timer")

            # Check if we should flush immediately (Size or Time)
            await self._check_and_flush_logic(queue_len, start_time_bytes)

            return TaskResponse(**task_data)
            
        except Exception as e:
            logger.error(f"❌ Redis error during enqueue: {e}")
            raise

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

        logger.debug(f"🔍 Background check - Queue length: {queue_len}")
        
        if queue_len > 0:
            await self._check_and_flush_logic(queue_len, start_time_bytes)
        else:
            logger.debug("📭 Queue is empty, nothing to process")

    async def _check_and_flush_logic(self, queue_len: int, start_time_bytes: Any):
        """
        Shared logic to determine if a flush is needed based on size or time.
        """
        should_flush = False
        flush_reason = ""
        current_time = datetime.utcnow()

        logger.debug(f"🔍 Checking flush logic - Queue: {queue_len}, Batch size: {settings.BATCH_SIZE}")

        # 1. Logic for Batch Size Check
        if queue_len >= settings.BATCH_SIZE:
            should_flush = True
            flush_reason = f"batch_size ({queue_len} >= {settings.BATCH_SIZE})"

        # 2. Logic for Timeout Check (only if we have a start time)
        elif start_time_bytes:
            try:
                start_time = datetime.fromisoformat(start_time_bytes.decode('utf-8'))
                timeout_seconds = getattr(settings, "BATCH_TIMEOUT", 30)
                elapsed = (current_time - start_time).total_seconds()

                logger.debug(f"⏱️ Time check - Elapsed: {elapsed:.1f}s, Timeout: {timeout_seconds}s")
                
                if elapsed > timeout_seconds:
                    should_flush = True
                    flush_reason = f"timeout ({elapsed:.1f}s > {timeout_seconds}s)"
            except Exception as e:
                logger.warning(f"⚠️ Date parsing failed: {e}")
                pass

        if should_flush:
            if not self.model:
                logger.error("❌ Cannot flush: Model not loaded")
                return
            
            logger.info(f"🚨 Triggering flush - Reason: {flush_reason}")
            await self._flush_queue(queue_len)
        else:
            logger.debug(f"⏸️ No flush needed - Queue: {queue_len}, Model loaded: {self.model is not None}")

    async def _flush_queue(self, count: int):
        logger.info(f"🚀 Flushing queue - Processing {count} items")
        
        try:
            task_ids = await self.redis.lpop(settings.QUEUE_NAME, count)
            if not task_ids:
                logger.warning("⚠️ No items to flush")
                return

            # Reset the timer immediately as the batch is being processed
            await self.redis.delete(self.timestamp_key)
            logger.info("⏱️ Batch timer reset")

            raw_data_list = await self.redis.hmget(settings.DATA_STORE_NAME, task_ids)

            # Prepare data for your function: Mapping[id, text]
            documents_map = {}

            for raw in raw_data_list:
                if raw:
                    item = json.loads(raw)
                    documents_map[item['id']] = item['text_content']

            logger.info(f"🤖 Running BERT inference on {len(documents_map)} documents")
            
            # Run your synchronous BERT function in a separate thread
            start_time = datetime.utcnow()
            processed_results = await run_in_threadpool(
                get_batch_outputs,
                self.model,
                documents_map
            )
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            logger.success(f"✅ Batch processed successfully - {len(processed_results)} items in {processing_time:.2f}s")
            
            # Log sample results for debugging
            if processed_results:
                sample_id = list(processed_results.keys())[0]
                sample_original = documents_map[sample_id][:100]
                sample_processed = processed_results[sample_id][:100]
                logger.debug(f"📝 Sample processing - Original: '{sample_original}...' -> Processed: '{sample_processed}...'")
            
            # Update Redis with processed results
            updated_count = 0
            for task_id, processed_text in processed_results.items():
                # Get original task data
                raw_data = await self.redis.hget(settings.DATA_STORE_NAME, task_id)
                if raw_data:
                    task_data = json.loads(raw_data)
                    task_data['text_content'] = processed_text
                    task_data['status'] = 'processed'
                    task_data['processed_at'] = datetime.utcnow().isoformat()
                    # Save back to Redis
                    await self.redis.hset(settings.DATA_STORE_NAME, task_id, json.dumps(task_data))
                    updated_count += 1
                    logger.debug(f"✅ Updated task {task_id} status to 'processed'")
                else:
                    logger.warning(f"⚠️ Task {task_id} not found in data store")
            
            logger.success(f"💾 Updated {updated_count}/{len(processed_results)} tasks in Redis")
            
            # Final verification
            logger.info(f"🎯 Flush complete - Processed {len(processed_results)} documents, Updated {updated_count} tasks")
                
        except Exception as e:
            logger.error(f"❌ Error during batch processing: {e}")
            raise

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