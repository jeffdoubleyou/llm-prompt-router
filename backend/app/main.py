from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import close_db, init_db
from app.services.redis_queue import redis_queue
from app.workers.classifier_worker import start_workers

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)

worker_tasks: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LLM Prompt Router...")
    await init_db()
    await redis_queue.connect()

    if settings.embedding_routing_enabled:
        from app.services.embedding_complexity import warm_up_embedding_scorer

        logger.info("Embedding routing enabled — warming up model...")
        loaded = await asyncio.to_thread(warm_up_embedding_scorer)
        if loaded:
            logger.info("Embedding complexity scorer loaded")
        else:
            logger.warning(
                "Embedding routing enabled but scorer failed to load; "
                "falling back to heuristics only"
            )

    try:
        tasks = await start_workers()
        worker_tasks.extend(tasks)
    except Exception:
        logger.warning("Classifier workers not started (ML may be unavailable)")

    yield

    logger.info("Shutting down LLM Prompt Router...")
    for task in worker_tasks:
        task.cancel()
    await redis_queue.disconnect()
    await close_db()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.v1.chat import router as chat_router
from app.api.v1.router import router as models_router

app.include_router(chat_router)
app.include_router(models_router)


@app.get("/")
async def root():
    return {
        "app": settings.app_name,
        "version": "1.0.0",
        "status": "running",
    }
