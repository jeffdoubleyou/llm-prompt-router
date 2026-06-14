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


def _detect_images(messages: list[dict]) -> bool:
    """Detect whether any message contains image data.

    Handles multiple formats:
    - OpenAI array-of-parts: [{"type": "image_url", "image_url": {...}}, ...]
    - OpenAI inline image: [{"type": "image", "image_url": {...}}, ...]
    - Anthropic image: [{"type": "image", "source": {...}}, ...]
    - Base64 data URIs in string content: "data:image/...;base64,..."
    - Image URLs in string content: "https://example.com/image.png"
    """
    for m in messages:
        content = m.get("content")
        if not content:
            continue

        # Case 1: content is a list of parts (OpenAI/Anthropic multimodal format)
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type", "")

                # OpenAI array-of-parts: {"type": "image_url", "image_url": {...}}
                if part_type == "image_url":
                    return True

                # OpenAI inline image: {"type": "image", "image_url": {...}}
                if part_type == "image" and "image_url" in part:
                    return True

                # Anthropic image: {"type": "image", "source": {...}}
                if part_type == "image" and "source" in part:
                    return True

                # OpenAI inline image with base64 source:
                # {"type": "image", "source": {"type": "base64", "data": "..."}}
                if part_type == "image" and "source" in part:
                    source = part["source"]
                    if isinstance(source, dict):
                        data = source.get("data", "")
                        if isinstance(data, str) and data.startswith("data:image/"):
                            return True
                        if isinstance(data, str) and ";base64," in data:
                            return True

                # Check for base64 data URIs nested in image_url
                image_url = part.get("image_url", {})
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                    if isinstance(url, str) and _is_image_uri(url):
                        return True

        # Case 2: content is a string - check for base64 data URIs or image URLs
        if isinstance(content, str):
            if _is_image_uri(content):
                return True

    return False


def _is_image_uri(text: str) -> bool:
    """Check if a string contains a base64 data URI or image URL."""
    if text.startswith("data:image/"):
        return True
    if ";base64," in text:
        return True
    # Check for common image URL patterns (with query params, anchors, or end of string)
    image_url_patterns = [
        ".png?", ".png&", ".png#", ".png",
        ".jpg?", ".jpg&", ".jpg#", ".jpg",
        ".jpeg?", ".jpeg&", ".jpeg#", ".jpeg",
        ".gif?", ".gif&", ".gif#", ".gif",
        ".webp?", ".webp&", ".webp#", ".webp",
        ".svg?", ".svg&", ".svg#", ".svg",
        ".bmp?", ".bmp&", ".bmp#", ".bmp",
    ]
    lower = text.lower()
    for pattern in image_url_patterns:
        if pattern in lower:
            return True
    return False


def _compute_complexity_score(features: PromptFeatures) -> float:
    """Compute an overall complexity score (0.0–1.0) for the given prompt features.

    Combines token count, reasoning complexity, domain, code density, and
    multimodal requirements into a single score that represents how
    cognitively demanding the prompt is.
    """
    score = 0.0

    # Token count contribution — longer prompts tend to be more complex
    tokens = features.token_count
    if tokens > 8000:
        score += 0.30
    elif tokens > 2000:
        score += 0.20
    elif tokens > 500:
        score += 0.10
    elif tokens > 100:
        score += 0.05

    # Reasoning complexity (0.0–1.0 from existing signal)
    score += features.reasoning_complexity * 0.30

    # Domain complexity — math and translation require structured reasoning
    domain_scores = {
        "code": 0.10,
        "math": 0.15,
        "translation": 0.08,
    }
    score += domain_scores.get(features.dominant_language, 0.0)

    # Code density bonus — more code lines means more complex prompt
    if features.has_code_blocks:
        score += 0.08

    # Multimodal requirements add complexity
    if features.has_images:
        score += 0.05
    if features.has_tool_calls:
        score += 0.05
    if features.has_urls:
        score += 0.02

    return min(round(score, 3), 1.0)


def extract_features(messages: list[dict]) -> PromptFeatures:
    full_text = " ".join(
        m.get("content") or "" for m in messages if isinstance(m.get("content"), str)
    )
    text_lower = full_text.lower()

    has_code = "```" in full_text or "`" in full_text or "def " in full_text or "class " in full_text
    has_urls = "http://" in text_lower or "https://" in text_lower
    has_images = _detect_images(messages)

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

    partial = PromptFeatures(
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
    complexity = _compute_complexity_score(partial)

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
        complexity_score=complexity,
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


def match_model_by_complexity(
    features: PromptFeatures,
    models: list[Model],
) -> tuple[Model | None, float]:
    """Select the cheapest capable model that can handle the prompt's complexity.

    Strategy:
    1. Filter models whose max_complexity_score >= prompt complexity_score
    2. Among capable models, prefer the one with lowest total cost
    3. Use estimated_tokens_per_second as a tiebreaker (faster is better)
    4. Fall back to rule-based matching for models without complexity metadata
    """
    # Compute complexity score if not already set (e.g., when called directly with PromptFeatures)
    if features.complexity_score == 0.0:
        features.complexity_score = _compute_complexity_score(features)
    complexity = features.complexity_score
    capable: list[tuple[Model, float, float]] = []  # (model, cost_score, speed_score)

    for model in models:
        if not model.is_active:
            continue

        caps = set(model.capabilities or [])

        # Hard capability checks — model must support required features
        if features.has_images and ModelCapability.vision.value not in caps:
            continue
        if features.has_tool_calls and ModelCapability.tool_calling.value not in caps:
            continue
        if features.token_count > model.context_window:
            continue

        # Check if model has complexity metadata
        max_cx = model.max_complexity_score
        if max_cx is not None:
            # Model can handle this complexity?
            if complexity > max_cx:
                continue
            # Score: prefer models whose capacity is just enough (not wasteful)
            capacity_headroom = max_cx - complexity
            # Cost score: lower cost = better (0.0 = cheapest, 1.0 = most expensive)
            total_cost = model.cost_per_1k_input + model.cost_per_1k_output
            cost_score = total_cost
            # Speed score: higher tokens/sec = better
            speed = model.estimated_tokens_per_second or 0.0
            capable.append((model, cost_score, speed))
        else:
            # No complexity metadata — treat as capable with neutral cost
            total_cost = model.cost_per_1k_input + model.cost_per_1k_output
            speed = model.estimated_tokens_per_second or 0.0
            capable.append((model, total_cost, speed))

    if not capable:
        return None, 0.0

    # Sort: lowest cost first, then highest speed as tiebreaker
    capable.sort(key=lambda x: (x[1], -x[2]))
    best_model, best_cost, best_speed = capable[0]

    # Confidence reflects how well the model's capacity matches the prompt
    max_cx = best_model.max_complexity_score
    if max_cx is not None:
        capacity_ratio = (max_cx - complexity) / max(0.01, max_cx)
        confidence = min(0.5 + capacity_ratio * 0.5, 1.0)
    else:
        confidence = 0.5

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

    # Try complexity-based routing first (if models have complexity metadata)
    has_complexity_metadata = any(m.max_complexity_score is not None for m in models)
    if has_complexity_metadata:
        cx_model, cx_confidence = match_model_by_complexity(features, models)
        if cx_model:
            logger.info(
                "Complexity-matched model %s with confidence %.2f (complexity %.2f)",
                cx_model.id, cx_confidence, features.complexity_score,
            )
            matched_model = cx_model
            confidence = cx_confidence

    # Fall back to rule-based matching if complexity routing didn't produce a match
    if not matched_model:
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
