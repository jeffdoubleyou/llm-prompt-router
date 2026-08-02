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
    client_ip: str | None = None
    user_agent: str | None = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "status": self.status,
            "position": self.position,
            "created_at": self.created_at,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
        }


def client_info_from_request(request) -> tuple[str | None, str | None]:
    """Extract client IP and User-Agent from a Starlette/FastAPI Request."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip() or None
    else:
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            client_ip = real_ip.strip() or None
        else:
            client_ip = request.client.host if request.client else None

    user_agent = request.headers.get("user-agent")
    if user_agent:
        user_agent = user_agent.strip() or None
    return client_ip, user_agent


class UpstreamQueueManager:
    """FIFO gate: one in-flight upstream request per normalized base URL."""

    def __init__(self) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._waiting: dict[str, list[UpstreamQueueEntry]] = defaultdict(list)
        self._processing: dict[str, UpstreamQueueEntry | None] = defaultdict(lambda: None)
        self._client_totals: dict[str, int] = defaultdict(int)
        self._client_last_ua: dict[str, str] = {}
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

    def _record_client(self, client_ip: str | None, user_agent: str | None) -> None:
        key = client_ip or "unknown"
        self._client_totals[key] += 1
        if user_agent:
            self._client_last_ua[key] = user_agent

    @asynccontextmanager
    async def acquire(
        self,
        base_url: str,
        request_id: str,
        model_id: str,
        *,
        client_ip: str | None = None,
        user_agent: str | None = None,
    ):
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
            client_ip=client_ip,
            user_agent=user_agent,
        )

        async with self._registry_lock:
            self._waiting[key].append(entry)
            self._refresh_positions(key)
            self._record_client(client_ip, user_agent)
            queue_depth = len(self._waiting[key])

        if queue_depth > 1:
            logger.info(
                "Request %s queued for %s (position %d, client=%s)",
                request_id,
                base_url,
                entry.position,
                client_ip or "unknown",
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

    def _client_summary(self) -> list[dict]:
        in_queue: dict[str, dict[str, int]] = defaultdict(
            lambda: {"waiting": 0, "processing": 0}
        )
        last_ua: dict[str, str] = {}

        for entries in self._waiting.values():
            for entry in entries:
                ip = entry.client_ip or "unknown"
                in_queue[ip]["waiting"] += 1
                if entry.user_agent:
                    last_ua[ip] = entry.user_agent

        for entry in self._processing.values():
            if entry is None:
                continue
            ip = entry.client_ip or "unknown"
            in_queue[ip]["processing"] += 1
            if entry.user_agent:
                last_ua[ip] = entry.user_agent

        ips = set(self._client_totals.keys()) | set(in_queue.keys())
        clients: list[dict] = []
        for ip in ips:
            waiting = in_queue[ip]["waiting"]
            processing = in_queue[ip]["processing"]
            clients.append({
                "client_ip": ip,
                "waiting": waiting,
                "processing": processing,
                "in_queue": waiting + processing,
                "total_requests": self._client_totals.get(ip, 0),
                "user_agent": last_ua.get(ip) or self._client_last_ua.get(ip),
            })

        clients.sort(
            key=lambda c: (-c["in_queue"], -c["total_requests"], c["client_ip"])
        )
        return clients

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
            "clients": self._client_summary(),
        }


upstream_queue_manager = UpstreamQueueManager()
