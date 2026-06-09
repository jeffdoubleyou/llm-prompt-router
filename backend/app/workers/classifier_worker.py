from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime

from sqlalchemy import select

from app.core.config import settings
from app.core.database import async_session_factory
from app.models.db import ClassifierSample, Model
from app.services.redis_queue import redis_queue
from app.services.router_service import extract_features

logger = logging.getLogger(__name__)

try:
    from ml.classifier import LLMClassifier

    classifier = LLMClassifier()
    classifier_loaded = True
except Exception:
    classifier = None
    classifier_loaded = False
    logger.warning("ML classifier not available — running with fallback scoring")

try:
    from ml.feature_extraction import extract_features as ml_extract_features

    ml_features_available = True
except Exception:
    ml_features_available = False


async def process_item(item_data: dict) -> None:
    start_time = time.monotonic()
    item_id = item_data.get("id", "unknown")
    request_id = item_data.get("request_id", "unknown")
    prompt_features = item_data.get("prompt_features", {})
    fallback_model = item_data.get("model_id", settings.default_model)

    async with async_session_factory() as db:
        try:
            result = await db.execute(
                select(Model).where(Model.is_active == True).order_by(Model.priority.desc())
            )
            models = list(result.scalars().all())

            if not models:
                logger.warning("No active models found for classification")
                return

            predicted = fallback_model
            confidence = 0.0

            if classifier_loaded:
                try:
                    prediction = classifier.predict(prompt_features)
                    predicted = prediction.model_id
                    confidence = prediction.confidence
                    logger.info(
                        "Classifier predicted %s (confidence=%.3f) for request %s",
                        predicted,
                        confidence,
                        request_id,
                    )
                except Exception:
                    logger.exception("Classifier prediction failed, using fallback")

            if confidence < settings.classifier_min_confidence:
                logger.info(
                    "Confidence %.3f below threshold %.3f for request %s",
                    confidence,
                    settings.classifier_min_confidence,
                    request_id,
                )

            sample = ClassifierSample(
                id=str(uuid.uuid4()),
                prompt_text=json.dumps(prompt_features),
                selected_model=predicted,
                features=prompt_features,
                confidence=float(confidence),
                is_correct=None,
            )
            db.add(sample)
            await db.commit()
            await redis_queue.remove_in_flight(item_id)

            processing_time = (time.monotonic() - start_time) * 1000
            logger.debug(
                "Processed item %s in %.1fms -> model %s",
                item_id,
                processing_time,
                predicted,
            )
        except Exception:
            logger.exception("Error processing queue item %s", item_id)
            await redis_queue.remove_in_flight(item_id)


async def worker_loop(worker_id: int) -> None:
    logger.info("Classifier worker %d started", worker_id)
    while True:
        try:
            item = await redis_queue.dequeue(timeout=5)
            if item is None:
                continue
            await redis_queue.mark_in_flight(item.id)
            await process_item({
                "id": item.id,
                "request_id": item.request_id,
                "prompt_features": item.prompt_features,
                "model_id": item.model_id,
            })
        except asyncio.CancelledError:
            logger.info("Worker %d shutting down", worker_id)
            break
        except Exception:
            logger.exception("Worker %d unexpected error", worker_id)


async def start_workers() -> list[asyncio.Task]:
    tasks = []
    for i in range(settings.worker_concurrency):
        task = asyncio.create_task(worker_loop(i))
        tasks.append(task)
    logger.info("Started %d classifier workers", settings.worker_concurrency)
    return tasks
