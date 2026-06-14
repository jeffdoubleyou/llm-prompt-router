#!/usr/bin/env python3
"""Migration: add complexity and speed fields to the models table.

Adds three nullable columns:
  - estimated_parameters_billions  (float)
  - estimated_tokens_per_second    (float)
  - max_complexity_score           (float)

Usage:
    python scripts/migrate_add_complexity_fields.py
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
    ("estimated_parameters_billions", "FLOAT"),
    ("estimated_tokens_per_second", "FLOAT"),
    ("max_complexity_score", "FLOAT"),
]


async def migrate():
    async with engine.begin() as conn:
        for col_name, col_type in COLUMNS:
            # Check if column already exists
            result = await conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'models' AND column_name = :col"
                ),
                {"col": col_name},
            )
            if result.fetchone():
                logger.info("Column %s already exists — skipping", col_name)
                continue
            await conn.execute(
                text(f"ALTER TABLE models ADD COLUMN {col_name} {col_type} DEFAULT NULL")
            )
            logger.info("Added column %s", col_name)


if __name__ == "__main__":
    asyncio.run(migrate())
