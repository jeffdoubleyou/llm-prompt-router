"""Per-base-URL FIFO queue for upstream LLM requests.

When enabled, only one request at a time is sent to each distinct base_url.
Additional requests wait in arrival order — useful for llama.cpp servers that
cancel in-flight work when the loaded model changes.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class UpstreamQueueEntry:
    request_id: str
    model_id: str
    base_url: str
    status: str  # waiting | processing
    position: int
    created_at: str
    queued_at_monotonic: float

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "status": self.status,
            "position": self.position,
            "created_at": self.created_at,
        }


class UpstreamQueueManager:
    """FIFO gate: one in-flight upstream request per normalized base URL."""

    def __init__(self) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._waiting: dict[str, list[UpstreamQueueEntry]] = defaultdict(list)
        self._processing: dict[str, UpstreamQueueEntry | None] = defaultdict(lambda: None)
        self._registry_lock = asyncio.Lock()

    @staticmethod
    def normalize_base_url(base_url: str) -> str:
        return base_url.rstrip("/").lower()

    async def _ensure_semaphore(self, key: str) -> asyncio.Semaphore:
        async with self._registry_lock:
            if key not in self._semaphores:
                self._semaphores[key] = asyncio.Semaphore(1)
            return self._semaphores[key]

    def _refresh_positions(self, key: str) -> None:
        for idx, entry in enumerate(self._waiting[key], start=1):
            entry.position = idx

    @asynccontextmanager
    async def acquire(self, base_url: str, request_id: str, model_id: str):
        key = self.normalize_base_url(base_url)
        sem = await self._ensure_semaphore(key)

        entry = UpstreamQueueEntry(
            request_id=request_id,
            model_id=model_id,
            base_url=base_url,
            status="waiting",
            position=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            queued_at_monotonic=asyncio.get_event_loop().time(),
        )

        async with self._registry_lock:
            self._waiting[key].append(entry)
            self._refresh_positions(key)
            queue_depth = len(self._waiting[key])

        if queue_depth > 1:
            logger.info(
                "Request %s queued for %s (position %d)",
                request_id,
                base_url,
                entry.position,
            )

        await sem.acquire()
        try:
            async with self._registry_lock:
                self._waiting[key] = [
                    e for e in self._waiting[key] if e.request_id != request_id
                ]
                self._refresh_positions(key)
                entry.status = "processing"
                entry.position = 0
                self._processing[key] = entry
            yield
        finally:
            async with self._registry_lock:
                self._processing[key] = None
            sem.release()

    def snapshot(self) -> dict:
        """Current waiting and processing requests grouped by base URL."""
        groups: list[dict] = []
        keys = set(self._waiting.keys()) | set(self._processing.keys())
        for key in sorted(keys):
            waiting = [e.to_dict() for e in self._waiting.get(key, [])]
            active = self._processing.get(key)
            if not waiting and active is None:
                continue
            groups.append({
                "base_url": active.base_url if active else waiting[0]["base_url"],
                "base_url_key": key,
                "waiting_count": len(waiting),
                "processing": active.to_dict() if active else None,
                "waiting": waiting,
            })
        total_waiting = sum(g["waiting_count"] for g in groups)
        total_processing = sum(1 for g in groups if g["processing"])
        return {
            "enabled": True,
            "base_urls": groups,
            "total_waiting": total_waiting,
            "total_processing": total_processing,
        }


upstream_queue_manager = UpstreamQueueManager()
