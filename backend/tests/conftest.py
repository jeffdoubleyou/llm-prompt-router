"""Shared fixtures for all tests.

The real ``app.core.database`` module eagerly creates a PostgreSQL engine.
We work around this by patching ``asyncpg`` out of ``sys.modules`` before
any app module is imported, then overriding the database URL to use SQLite
in-memory (which needs no server). The ``pool_size`` / ``max_overflow``
parameters that are valid for PostgreSQL are stripped for SQLite.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import MagicMock

# 1. Mock asyncpg BEFORE any app module is imported
sys.modules["asyncpg"] = MagicMock()

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base, get_db
from app.main import app
from app.models.db import ClassifierSample

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-level event loop for pytest-asyncio."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def engine():
    """Create a session-scoped database engine (shared across tests)."""
    eng = create_async_engine(TEST_DATABASE_URL, echo=False)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(engine, event_loop):
    """Fresh database session with dropped/created tables for each test."""
    # Recreate all tables for a clean state
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
        await session.rollback()
        await session.close()


@pytest_asyncio.fixture(scope="function")
async def client(db_session):
    """Async HTTP client using a test database session."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
def sample_data():
    """Factory to create ClassifierSample records for testing."""

    async def _make(
        prompt_text: str,
        selected_model: str = "gpt-4",
        features: dict | None = None,
        confidence: float | None = None,
        is_correct: bool | None = None,
    ) -> ClassifierSample:
        return ClassifierSample(
            id=f"sample-{prompt_text[:10]}",
            prompt_text=prompt_text,
            selected_model=selected_model,
            features=features or {},
            confidence=confidence,
            is_correct=is_correct,
        )

    return _make


@pytest_asyncio.fixture
async def populated_db(db_session, sample_data):
    """Populate the database with known test data and return the count."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    samples = [
        ClassifierSample(
            id="s1",
            prompt_text="What is the capital of France?",
            selected_model="gpt-4",
            features={"token_count": 8, "char_length": 35},
            confidence=0.95,
            is_correct=True,
            created_at=now - timedelta(hours=3),
        ),
        ClassifierSample(
            id="s2",
            prompt_text="Write a Python function to sort a list",
            selected_model="claude-3",
            features={"token_count": 12, "has_code_blocks": True},
            confidence=0.88,
            is_correct=True,
            created_at=now - timedelta(hours=2),
        ),
        ClassifierSample(
            id="s3",
            prompt_text="Explain quantum computing in simple terms",
            selected_model="gpt-4",
            features={"token_count": 15},
            confidence=0.72,
            is_correct=False,
            created_at=now - timedelta(hours=1),
        ),
        ClassifierSample(
            id="s4",
            prompt_text="How to implement binary search in JavaScript?",
            selected_model="claude-3",
            features={"token_count": 14, "has_code_blocks": True},
            confidence=0.91,
            is_correct=True,
            created_at=now - timedelta(minutes=30),
        ),
        ClassifierSample(
            id="s5",
            prompt_text="Translate this to French: Hello world",
            selected_model="gpt-4",
            features={"token_count": 10},
            confidence=0.85,
            is_correct=False,
            created_at=now - timedelta(minutes=15),
        ),
        ClassifierSample(
            id="s6",
            prompt_text="What are the best practices for REST API design?",
            selected_model="gemini-pro",
            features={"token_count": 20, "has_urls": True},
            confidence=0.93,
            is_correct=True,
            created_at=now - timedelta(minutes=5),
        ),
        ClassifierSample(
            id="s7",
            prompt_text="Summarize the key points of this article about AI",
            selected_model="gpt-4",
            features={"token_count": 18},
            confidence=0.67,
            is_correct=None,
            created_at=now,
        ),
    ]
    for s in samples:
        db_session.add(s)
    await db_session.commit()
    return len(samples)
