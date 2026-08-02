from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def _apply_additive_migrations(conn) -> None:
    """Add columns/indexes that create_all will not alter on existing tables."""
    await conn.execute(
        text(
            "ALTER TABLE request_logs "
            "ADD COLUMN IF NOT EXISTS client_ip VARCHAR(64)"
        )
    )
    await conn.execute(
        text(
            "ALTER TABLE request_logs "
            "ADD COLUMN IF NOT EXISTS user_agent VARCHAR(512)"
        )
    )
    await conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_request_logs_client_ip "
            "ON request_logs (client_ip)"
        )
    )


async def init_db() -> None:
    async with engine.begin() as conn:
        from app.models.db import Base  # noqa: F811
        await conn.run_sync(Base.metadata.create_all)
        await _apply_additive_migrations(conn)
    logger.info("Database tables created / verified")


async def close_db() -> None:
    await engine.dispose()
    logger.info("Database engine disposed")
