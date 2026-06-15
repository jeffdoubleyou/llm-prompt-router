from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime

import redis.asyncio as aioredis
import redis.exceptions

from app.core.config import settings

logger = logging.getLogger(__name__)

QUEUE_KEY = "router:unclassified_queue"
METRICS_KEY_PREFIX = "router:metrics:"
PROCESSING_SET_KEY = "router:in_flight"
PROMPT_DEBUG_KEY = "router:prompt_debug"


@dataclass
class QueueItem:
    id: str
    request_id: str
    prompt_features: dict
    model_id: str | None
    created_at: str


class RedisQueue:
    def __init__(self) -> None:
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._client = await aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=10.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        logger.info("Connected to Redis")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("Disconnected from Redis")

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("Redis not connected — call connect() first")
        return self._client

    async def enqueue(self, item: QueueItem) -> None:
        await self.client.rpush(QUEUE_KEY, json.dumps(asdict(item)))
        logger.debug("Enqueued request %s", item.request_id)

    async def dequeue(self, timeout: int = 5) -> QueueItem | None:
        result = await self.client.blpop(QUEUE_KEY, timeout=timeout)
        if result is None:
            return None
        _, data = result
        parsed = json.loads(data)
        return QueueItem(**parsed)

    async def queue_depth(self) -> int:
        return await self.client.llen(QUEUE_KEY)

    async def mark_in_flight(self, item_id: str) -> None:
        await self.client.sadd(PROCESSING_SET_KEY, item_id)
        await self.client.expire(PROCESSING_SET_KEY, 300)

    async def remove_in_flight(self, item_id: str) -> None:
        await self.client.srem(PROCESSING_SET_KEY, item_id)

    async def in_flight_count(self) -> int:
        return await self.client.scard(PROCESSING_SET_KEY)

    async def store_metrics(self, model_id: str, snapshot: dict) -> None:
        key = f"{METRICS_KEY_PREFIX}{model_id}"
        await self.client.hset(key, mapping=snapshot)
        await self.client.expire(key, 7200)

    async def get_metrics(self, model_id: str) -> dict | None:
        key = f"{METRICS_KEY_PREFIX}{model_id}"
        data = await self.client.hgetall(key)
        return data if data else None

    async def get_all_metrics_keys(self) -> list[str]:
        keys = await self.client.keys(f"{METRICS_KEY_PREFIX}*")
        return [k.replace(METRICS_KEY_PREFIX, "") for k in keys]

    async def store_prompt_debug(self, entry: dict) -> None:
        max_stored = settings.prompt_debug_max_stored
        if max_stored <= 0:
            return
        await self.client.lpush(PROMPT_DEBUG_KEY, json.dumps(entry))
        await self.client.ltrim(PROMPT_DEBUG_KEY, 0, max_stored - 1)
        if settings.prompt_debug_ttl_seconds > 0:
            await self.client.expire(PROMPT_DEBUG_KEY, settings.prompt_debug_ttl_seconds)

    async def get_prompt_debug(self, limit: int = 20) -> list[dict]:
        max_stored = settings.prompt_debug_max_stored
        if max_stored <= 0:
            return []
        limit = max(1, min(limit, max_stored))
        items = await self.client.lrange(PROMPT_DEBUG_KEY, 0, limit - 1)
        return [json.loads(item) for item in items]


redis_queue = RedisQueue()
