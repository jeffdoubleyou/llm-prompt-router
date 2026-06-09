from __future__ import annotations

import logging
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.models import (
    ChatCompletionRequest,
    ClassifierPrediction,
    ModelCapability,
    PromptFeatures,
)
from app.models.db import ClassifierSample, Model, RequestLog
from app.services.redis_queue import QueueItem, RedisQueue, redis_queue

logger = logging.getLogger(__name__)


def extract_features(messages: list[dict]) -> PromptFeatures:
    full_text = " ".join(
        m.get("content") or "" for m in messages if isinstance(m.get("content"), str)
    )
    text_lower = full_text.lower()

    has_code = "```" in full_text or "`" in full_text or "def " in full_text or "class " in full_text
    has_urls = "http://" in text_lower or "https://" in text_lower
    has_images = any(
        isinstance(m.get("content"), list)
        and any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in m["content"]
        )
        for m in messages
    )
    has_tool_calls = any(m.get("tool_calls") for m in messages) or any(
        m.get("tool_call_id") for m in messages
    )
    has_tools_in_request = False

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(full_text)) if full_text else 0
    except Exception:
        token_count = len(full_text) // 4

    dominant_lang = _detect_dominant_language(text_lower)
    reasoning_complexity = _compute_reasoning_complexity(full_text, text_lower)

    import datetime as dt
    hour = dt.datetime.utcnow().hour

    return PromptFeatures(
        token_count=token_count,
        char_length=len(full_text),
        has_code_blocks=has_code,
        has_urls=has_urls,
        has_images=has_images,
        has_tool_calls=has_tool_calls,
        dominant_language=dominant_lang,
        reasoning_complexity=reasoning_complexity,
        hour_of_day=hour,
    )


def _detect_dominant_language(text_lower: str) -> str:
    code_keywords = {"def ", "class ", "import ", "fn ", "func", "function", "const ", "let ", "var "}
    code_score = sum(1 for kw in code_keywords if kw in text_lower)
    if code_score >= 3:
        return "code"
    math_keywords = {"solve", "equation", "derivative", "integral", "calculate", "compute", "∑", "∫"}
    if any(kw in text_lower for kw in math_keywords):
        return "math"
    if any(text_lower.startswith(p) for p in ("translate", "translation", "translate to")):
        return "translation"
    return "natural_language"


def _compute_reasoning_complexity(text: str, text_lower: str) -> float:
    score = 0.0
    reasoning_triggers = [
        "explain", "reason", "think step by step", "analyze", "compare",
        "contrast", "why", "how does", "what if", "derive", "prove",
        "evaluate", "synthesize", "critique",
    ]
    for trigger in reasoning_triggers:
        if trigger in text_lower:
            score += 0.15
    code_density = text.count("\n") / max(len(text), 1) * 100
    if code_density > 10:
        score += 0.2
    question_marks = text.count("?")
    score += min(question_marks * 0.05, 0.3)
    complexity_keywords = [
        "comprehensive", "detailed", "thorough", "in-depth", "complex",
        "sophisticated", "multi-step", "multistep", "hierarchical",
    ]
    for kw in complexity_keywords:
        if kw in text_lower:
            score += 0.1
    return min(round(score, 2), 1.0)


def match_model_by_rules(
    features: PromptFeatures,
    models: list[Model],
) -> tuple[Model | None, float]:
    scored: list[tuple[Model, float]] = []
    for model in models:
        if not model.is_active:
            continue
        caps = set(model.capabilities or [])
        score = 0.0
        if features.has_images and ModelCapability.vision.value in caps:
            score += 3.0
        if features.has_tool_calls and ModelCapability.tool_calling.value in caps:
            score += 2.0
        if features.token_count > 4000 and ModelCapability.long_context.value in caps:
            score += 2.0
        if features.has_code_blocks and ModelCapability.code.value in caps:
            score += 1.5
        if features.reasoning_complexity > 0.5 and ModelCapability.reasoning.value in caps:
            score += 2.0
        if features.reasoning_complexity > 0.5 and "reasoning" in (model.tags or []):
            score += 1.0
        score += model.priority * 0.1
        scored.append((model, score))

    if not scored:
        return None, 0.0

    scored.sort(key=lambda x: x[1], reverse=True)
    best_model, best_score = scored[0]
    confidence = min(best_score / 5.0, 1.0)
    return best_model, confidence


async def classify_and_route(
    request: ChatCompletionRequest,
    db: AsyncSession,
    queue: RedisQueue = redis_queue,
) -> str | None:
    messages_dicts = [m.model_dump(exclude_none=True) for m in request.messages]
    features = extract_features(messages_dicts)

    result = await db.execute(select(Model).where(Model.is_active == True))
    models = list(result.scalars().all())

    if not models:
        logger.warning("No active models found, using default")
        return settings.default_model

    matched_model, confidence = match_model_by_rules(features, models)

    if matched_model and confidence >= settings.classifier_min_confidence:
        logger.info("Rule-matched model %s with confidence %.2f", matched_model.id, confidence)
        return matched_model.id

    if matched_model and matched_model.id:
        try:
            queue_item = QueueItem(
                id=str(uuid.uuid4()),
                request_id=str(uuid.uuid4()),
                prompt_features=features.model_dump(),
                model_id=matched_model.id,
                created_at=datetime.utcnow().isoformat(),
            )
            await queue.enqueue(queue_item)
            logger.info(
                "Enqueued request for classifier (confidence %.2f < %.2f)",
                confidence,
                settings.classifier_min_confidence,
            )
        except Exception:
            logger.exception("Failed to enqueue request")
        return matched_model.id

    return settings.default_model


async def log_request(
    db: AsyncSession,
    request_id: str,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    cost: float,
    is_error: bool = False,
    error_message: str | None = None,
    model_used: str | None = None,
) -> None:
    log_entry = RequestLog(
        id=str(uuid.uuid4()),
        request_id=request_id,
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
        cost=cost,
        is_error=is_error,
        error_message=error_message,
        model_used=model_used,
    )
    db.add(log_entry)
    await db.commit()


async def get_request_logs(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
    model_id: str | None = None,
    is_error: bool | None = None,
) -> list[RequestLog]:
    query = select(RequestLog).order_by(RequestLog.created_at.desc())
    if model_id:
        query = query.where(RequestLog.model_id == model_id)
    if is_error is not None:
        query = query.where(RequestLog.is_error == is_error)
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


async def get_models(db: AsyncSession) -> list[Model]:
    result = await db.execute(select(Model).order_by(Model.priority.desc(), Model.id))
    return list(result.scalars().all())


async def get_model_by_id(db: AsyncSession, model_id: str) -> Model | None:
    result = await db.execute(select(Model).where(Model.id == model_id))
    return result.scalar_one_or_none()
