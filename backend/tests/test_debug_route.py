"""Tests for the dry-run routing debug endpoint."""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.modules["asyncpg"] = MagicMock()

from httpx import ASGITransport, AsyncClient

from app.main import app
from app.core.database import get_db


@pytest.fixture
def mock_db():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.add = MagicMock()
    session.refresh = MagicMock()

    # Return empty model list by default — classify_and_route uses default_model
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    session.execute.return_value = result
    return session


@pytest.mark.asyncio
async def test_debug_route_returns_features(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/debug/route",
                json={"messages": [{"role": "user", "content": "Hello!"}]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "model_id" in data
        assert "features" in data
        assert data["features"]["task_type"] == "chitchat"
        assert "routing_difficulty" in data
    finally:
        app.dependency_overrides.clear()
