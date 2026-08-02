"""Tests for complexity debug and upstream queue."""

from __future__ import annotations

import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.modules["asyncpg"] = MagicMock()

from httpx import ASGITransport, AsyncClient

from app.main import app
from app.core.database import get_db
from app.core.config import settings
from app.services.upstream_queue import upstream_queue_manager


def _make_model(model_id: str, max_cx: float, cost_in: float = 0.001, cost_out: float = 0.005):
    from app.models.db import Model
    return Model(
        id=model_id,
        display_name=model_id,
        provider="openai",
        capabilities=["text", "streaming"],
        tags=[],
        cost_per_1k_input=cost_in,
        cost_per_1k_output=cost_out,
        max_tokens=4096,
        context_window=128000,
        rpm_limit=60,
        tpm_limit=100000,
        is_active=True,
        priority=10,
        max_complexity_score=max_cx,
    )


@pytest.fixture
def mock_db_with_models():
    session = AsyncMock()
    session.commit = AsyncMock()
    session.add = MagicMock()
    session.refresh = MagicMock()
    models = [
        _make_model("cheap", 0.5, 0.0001, 0.0003),
        _make_model("capable", 0.9, 0.01, 0.03),
    ]
    result = MagicMock()
    result.scalars.return_value.all.return_value = models
    session.execute = AsyncMock(return_value=result)
    return session


@pytest.mark.asyncio
async def test_debug_complexity_returns_breakdown(mock_db_with_models):
    app.dependency_overrides[get_db] = lambda: mock_db_with_models
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/debug/complexity",
                json={"messages": [{"role": "user", "content": "Debug this race condition"}]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["routing_method"] in ("complexity", "rules", "default")
        assert "complexity_explanation" in data
        assert "task_difficulty_breakdown" in data["complexity_explanation"]
        assert len(data["model_evaluations"]) == 2
    finally:
        app.dependency_overrides.clear()


def _reset_upstream_queue():
    upstream_queue_manager._semaphores.clear()
    upstream_queue_manager._waiting.clear()
    upstream_queue_manager._processing.clear()
    upstream_queue_manager._client_totals.clear()
    upstream_queue_manager._client_last_ua.clear()


@pytest.mark.asyncio
async def test_upstream_queue_serializes_per_base_url():
    _reset_upstream_queue()

    order: list[str] = []

    async def worker(name: str, delay: float):
        async with upstream_queue_manager.acquire(
            "http://localhost:8080/v1", f"req-{name}", f"model-{name}",
        ):
            order.append(f"start-{name}")
            await asyncio.sleep(delay)
            order.append(f"end-{name}")

    await asyncio.gather(
        worker("a", 0.05),
        worker("b", 0.01),
    )
    assert order.index("start-a") < order.index("end-a")
    assert order.index("end-a") < order.index("start-b")


@pytest.mark.asyncio
async def test_upstream_queue_status_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/upstream-queue")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False
    assert resp.json()["clients"] == []


@pytest.mark.asyncio
async def test_upstream_queue_snapshot_omits_idle_base_urls():
    _reset_upstream_queue()

    async with upstream_queue_manager.acquire(
        "http://localhost:8080/v1",
        "req-a",
        "model-a",
        client_ip="203.0.113.10",
        user_agent="test-agent/1.0",
    ):
        snap = upstream_queue_manager.snapshot()
        assert snap["total_processing"] == 1
        assert len(snap["base_urls"]) == 1
        processing = snap["base_urls"][0]["processing"]
        assert processing["client_ip"] == "203.0.113.10"
        assert processing["user_agent"] == "test-agent/1.0"
        assert snap["clients"] == [
            {
                "client_ip": "203.0.113.10",
                "waiting": 0,
                "processing": 1,
                "in_queue": 1,
                "total_requests": 1,
                "user_agent": "test-agent/1.0",
            }
        ]

    idle = upstream_queue_manager.snapshot()
    assert idle["total_processing"] == 0
    assert idle["total_waiting"] == 0
    assert idle["base_urls"] == []
    assert idle["clients"] == [
        {
            "client_ip": "203.0.113.10",
            "waiting": 0,
            "processing": 0,
            "in_queue": 0,
            "total_requests": 1,
            "user_agent": "test-agent/1.0",
        }
    ]


@pytest.mark.asyncio
async def test_client_info_from_request_prefers_forwarded_for():
    from app.services.upstream_queue import client_info_from_request

    class _Client:
        host = "10.0.0.1"

    class _Request:
        headers = {
            "x-forwarded-for": "198.51.100.7, 10.0.0.1",
            "user-agent": "curl/8.0",
        }
        client = _Client()

    ip, ua = client_info_from_request(_Request())
    assert ip == "198.51.100.7"
    assert ua == "curl/8.0"
