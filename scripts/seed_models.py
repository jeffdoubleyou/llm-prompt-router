#!/usr/bin/env python3
"""Seed the models table with default models if the database is empty.

Usage:
    python scripts/seed_models.py

This script is idempotent — it only inserts models that don't already exist.
It is designed to run as part of a Docker entrypoint or startup script.
"""

import asyncio
import logging
import os
import sys

from sqlalchemy import select

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.database import AsyncSession, async_session_factory
from app.models.db import Model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default models to seed when the database is empty.
# Adjust or extend this list for your deployment.
DEFAULT_MODELS = [
    {
        "id": "gpt-4o-mini",
        "display_name": "GPT-4o Mini",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "capabilities": ["text", "streaming", "json_mode"],
        "tags": ["general", "fast", "cheap"],
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "max_tokens": 16384,
        "context_window": 128000,
        "rpm_limit": 3000,
        "tpm_limit": 200000,
        "is_active": True,
        "priority": 10,
        "estimated_parameters_billions": 13.0,
        "estimated_tokens_per_second": 60.0,
        "max_complexity_score": 0.5,
    },
    {
        "id": "gpt-4o",
        "display_name": "GPT-4o",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "capabilities": ["text", "vision", "streaming", "json_mode", "tool_calling"],
        "tags": ["general", "vision", "capable"],
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
        "max_tokens": 4096,
        "context_window": 128000,
        "rpm_limit": 500,
        "tpm_limit": 200000,
        "is_active": True,
        "priority": 20,
        "estimated_parameters_billions": 50.0,
        "estimated_tokens_per_second": 40.0,
        "max_complexity_score": 0.7,
    },
    {
        "id": "claude-sonnet-4-20250514",
        "display_name": "Claude Sonnet 4",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "capabilities": ["text", "streaming", "json_mode", "tool_calling", "reasoning", "code"],
        "tags": ["reasoning", "code", "capable"],
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "max_tokens": 8192,
        "context_window": 200000,
        "rpm_limit": 100,
        "tpm_limit": 500000,
        "is_active": True,
        "priority": 15,
        "estimated_parameters_billions": 30.0,
        "estimated_tokens_per_second": 35.0,
        "max_complexity_score": 0.75,
    },
    {
        "id": "claude-opus-4-20250514",
        "display_name": "Claude Opus 4",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "capabilities": ["text", "streaming", "json_mode", "tool_calling", "reasoning", "code"],
        "tags": ["reasoning", "code", "most-capable"],
        "cost_per_1k_input": 0.015,
        "cost_per_1k_output": 0.075,
        "max_tokens": 8192,
        "context_window": 200000,
        "rpm_limit": 50,
        "tpm_limit": 500000,
        "is_active": True,
        "priority": 25,
        "estimated_parameters_billions": 100.0,
        "estimated_tokens_per_second": 15.0,
        "max_complexity_score": 0.95,
    },
    {
        "id": "gemini-2.5-pro",
        "display_name": "Gemini 2.5 Pro",
        "provider": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "capabilities": ["text", "vision", "streaming", "json_mode", "tool_calling", "long_context"],
        "tags": ["vision", "long-context", "capable"],
        "cost_per_1k_input": 0.00125,
        "cost_per_1k_output": 0.0075,
        "max_tokens": 8192,
        "context_window": 1048576,
        "rpm_limit": 200,
        "tpm_limit": 4000000,
        "is_active": True,
        "priority": 18,
        "estimated_parameters_billions": 70.0,
        "estimated_tokens_per_second": 45.0,
        "max_complexity_score": 0.85,
    },
]


async def seed_models():
    async with async_session_factory() as session:
        models = await session.execute(select(Model))
        existing = models.scalars().all()

        if existing:
            logger.info("Database already has %d model(s) — skipping seed.", len(existing))
            return

        for model_data in DEFAULT_MODELS:
            model = Model(**model_data)
            session.add(model)

        await session.commit()
        logger.info("Seeded %d default models into the database.", len(DEFAULT_MODELS))


if __name__ == "__main__":
    asyncio.run(seed_models())
