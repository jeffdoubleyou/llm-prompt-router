#!/usr/bin/env python3
"""Migration: add client identity fields to request_logs.

Adds:
  - client_ip   (varchar 64, indexed)
  - user_agent  (varchar 512)

Usage:
    python scripts/migrate_add_client_fields.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from sqlalchemy import text

from app.core.database import engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLUMNS = [
    ("client_ip", "VARCHAR(64)"),
    ("user_agent", "VARCHAR(512)"),
]


async def migrate():
    async with engine.begin() as conn:
        for col_name, col_type in COLUMNS:
            result = await conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'request_logs' AND column_name = :col"
                ),
                {"col": col_name},
            )
            if result.fetchone():
                logger.info("Column %s already exists — skipping", col_name)
                continue
            await conn.execute(
                text(
                    f"ALTER TABLE request_logs "
                    f"ADD COLUMN {col_name} {col_type} DEFAULT NULL"
                )
            )
            logger.info("Added column %s", col_name)

        idx_result = await conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'request_logs' AND indexname = 'ix_request_logs_client_ip'"
            )
        )
        if idx_result.fetchone():
            logger.info("Index ix_request_logs_client_ip already exists — skipping")
        else:
            await conn.execute(
                text(
                    "CREATE INDEX ix_request_logs_client_ip "
                    "ON request_logs (client_ip)"
                )
            )
            logger.info("Created index ix_request_logs_client_ip")


if __name__ == "__main__":
    asyncio.run(migrate())
