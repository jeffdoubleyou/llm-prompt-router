from __future__ import annotations

import logging
import re
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
from app.services.prompt_complexity import analyze_prompt_complexity, get_routing_difficulty
from app.services.redis_queue import QueueItem, RedisQueue, redis_queue

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://[^\s,)'\"]+")
IMAGE_DATA_URI_PATTERN = re.compile(r"data:image/[^;]+;base64,")
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")
HTML_IMG_PATTERN = re.compile(r"<img[^>]*src=", re.IGNORECASE)

# System prompts from clients (e.g. IDE rules) often contain documentation URLs
# that are not part of the user's actual request.
_URL_SCAN_ROLES = frozenset({"user", "assistant", "tool"})


def _text_from_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                parts.append(part.get("text") or "")
        return " ".join(parts)
    return ""


def _messages_to_text(messages: list[dict], roles: frozenset[str] | None = None) -> str:
    texts: list[str] = []
    for m in messages:
        if roles is not None and m.get("role") not in roles:
            continue
        text = _text_from_content(m.get("content"))
        if text:
            texts.append(text)
    return " ".join(texts)


def _has_urls_in_messages(messages: list[dict]) -> bool:
    """Detect URLs in conversation content, excluding system prompts."""
    return bool(URL_PATTERN.search(_messages_to_text(messages, _URL_SCAN_ROLES)))


def _summarize_image_ref(ref: str, max_len: int = 100) -> str:
    if ref.startswith("data:image/"):
        return f"data:image/…;base64, [{len(ref)} chars]"
    if len(ref) > max_len:
        return f"{ref[:max_len]}…"
    return ref


def analyze_images(messages: list[dict]) -> dict:
    """Return detailed image detection results for debugging and routing.

    See docs/image-detection.md for the full ruleset.
    """
    detections: list[dict] = []

    for msg_index, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = message.get("content")
        if not content:
            continue

        if isinstance(content, list):
            for part_index, part in enumerate(content):
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type", "")

                if part_type == "image_url":
                    image_url = part.get("image_url", {})
                    url = ""
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "") or ""
                    detections.append({
                        "message_index": msg_index,
                        "role": role,
                        "part_index": part_index,
                        "match_type": "openai_image_url",
                        "summary": f"content[{part_index}] type=image_url",
                        "detail": _summarize_image_ref(url) if url else None,
                    })
                    continue

                if part_type == "image" and "image_url" in part:
                    image_url = part.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                    detections.append({
                        "message_index": msg_index,
                        "role": role,
                        "part_index": part_index,
                        "match_type": "openai_image_part",
                        "summary": f"content[{part_index}] type=image (image_url)",
                        "detail": _summarize_image_ref(url) if url else None,
                    })
                    continue

                if part_type == "image" and "source" in part:
                    source = part.get("source", {})
                    source_type = source.get("type", "unknown") if isinstance(source, dict) else "unknown"
                    detail = None
                    if isinstance(source, dict):
                        if source_type == "url":
                            detail = _summarize_image_ref(source.get("url", "") or "")
                        elif source_type == "base64":
                            media = source.get("media_type", "image/*")
                            data_len = len(source.get("data", "") or "")
                            detail = f"base64 {media} [{data_len} chars]"
                    detections.append({
                        "message_index": msg_index,
                        "role": role,
                        "part_index": part_index,
                        "match_type": "anthropic_image",
                        "summary": f"content[{part_index}] type=image (source={source_type})",
                        "detail": detail,
                    })
                    continue

                image_url = part.get("image_url", {})
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                    if isinstance(url, str) and _is_image_data_uri(url):
                        detections.append({
                            "message_index": msg_index,
                            "role": role,
                            "part_index": part_index,
                            "match_type": "nested_data_uri",
                            "summary": f"content[{part_index}] image_url.url is embedded data URI",
                            "detail": _summarize_image_ref(url),
                        })

        if isinstance(content, str):
            if IMAGE_DATA_URI_PATTERN.search(content):
                detections.append({
                    "message_index": msg_index,
                    "role": role,
                    "part_index": None,
                    "match_type": "string_data_uri",
                    "summary": "string content contains data:image/…;base64,",
                    "detail": _summarize_image_ref(content),
                })
                continue

            md_match = MARKDOWN_IMAGE_PATTERN.search(content)
            if md_match:
                detections.append({
                    "message_index": msg_index,
                    "role": role,
                    "part_index": None,
                    "match_type": "markdown_image",
                    "summary": "string content contains Markdown image syntax",
                    "detail": md_match.group(0)[:120],
                })
                continue

            html_match = HTML_IMG_PATTERN.search(content)
            if html_match:
                detections.append({
                    "message_index": msg_index,
                    "role": role,
                    "part_index": None,
                    "match_type": "html_img",
                    "summary": "string content contains <img src=…>",
                    "detail": html_match.group(0)[:120],
                })

    return {
        "has_images": bool(detections),
        "detection_count": len(detections),
        "detections": detections,
        "ignored": [
            "Plain-text mentions of image filenames (e.g. chart.png)",
            "Plain-text https:// URLs to images without multimodal parts",
            "Loose ;base64, markers without a data:image/ prefix",
            "URLs in system-role messages are ignored for has_urls only (not image detection)",
        ],
    }


def _detect_images(messages: list[dict]) -> bool:
    """Detect whether any message contains actual image payload.

    Only matches structured multimodal parts and embedded image data — not
    casual mentions of filenames or image URLs in plain text.
    """
    return analyze_images(messages)["has_images"]


def _is_image_data_uri(text: str) -> bool:
    """Check if a string is an embedded image data URI (not a plain https URL)."""
    return text.startswith("data:image/") or bool(IMAGE_DATA_URI_PATTERN.search(text))


def _compute_legacy_complexity_score(features: PromptFeatures) -> float:
    """Backward-compatible fallback when task_difficulty was not computed."""
    score = 0.0
    tokens = features.token_count
    if tokens > 8000:
        score += 0.30
    elif tokens > 2000:
        score += 0.20
    elif tokens > 500:
        score += 0.10
    elif tokens > 100:
        score += 0.05
    score += features.reasoning_complexity * 0.30
    domain_scores = {"code": 0.10, "math": 0.15, "translation": 0.08}
    score += domain_scores.get(features.dominant_language, 0.0)
    if features.has_code_blocks:
        score += 0.08
    if features.has_images:
        score += 0.05
    if features.has_tool_calls:
        score += 0.05
    if features.has_urls:
        score += 0.02
    return min(round(score, 3), 1.0)


def _compute_complexity_score(features: PromptFeatures) -> float:
    """Return routing difficulty, preferring content-aware task_difficulty."""
    legacy = _compute_legacy_complexity_score(features)
    return get_routing_difficulty(
        features.task_difficulty,
        features.requirement_load,
        legacy_complexity_score=legacy,
    )


def extract_features(messages: list[dict]) -> PromptFeatures:
    full_text = _messages_to_text(messages)
    text_lower = full_text.lower()

    has_code = "```" in full_text or "`" in full_text or "def " in full_text or "class " in full_text
    has_urls = _has_urls_in_messages(messages)
    has_images = _detect_images(messages)

    has_tool_calls = any(m.get("tool_calls") for m in messages) or any(
        m.get("tool_call_id") for m in messages
    )

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(full_text)) if full_text else 0
    except Exception:
        token_count = len(full_text) // 4

    dominant_lang = _detect_dominant_language(text_lower)

    import datetime as dt
    hour = dt.datetime.utcnow().hour

    analysis = analyze_prompt_complexity(
        messages,
        token_count=token_count,
        dominant_language=dominant_lang,
        has_code_blocks=has_code,
        has_images=has_images,
        has_tool_calls=has_tool_calls,
        full_text=full_text,
    )

    return PromptFeatures(
        token_count=token_count,
        char_length=len(full_text),
        has_code_blocks=has_code,
        has_urls=has_urls,
        has_images=has_images,
        has_tool_calls=has_tool_calls,
        dominant_language=dominant_lang,
        reasoning_complexity=analysis.reasoning_complexity,
        hour_of_day=hour,
        context_load=analysis.context_load,
        task_difficulty=analysis.task_difficulty,
        requirement_load=analysis.requirement_load,
        task_type=analysis.task_type,
        complexity_score=analysis.complexity_score,
        sub_task_count=analysis.sub_task_count,
        constraint_count=analysis.constraint_count,
        reference_count=analysis.reference_count,
        heuristic_task_difficulty=analysis.heuristic_task_difficulty,
        embedding_difficulty=analysis.embedding_difficulty,
        embedding_routing_applied=analysis.embedding_routing_applied,
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
    """Deprecated: use prompt_complexity.compute_reasoning_complexity."""
    from app.services.prompt_complexity import compute_reasoning_complexity
    return compute_reasoning_complexity(text)


def _model_caps(model: Model) -> set[str]:
    return set(model.capabilities or [])


def _needs_tool_support(features: PromptFeatures, *, request_has_tools: bool = False) -> bool:
    return features.has_tool_calls or request_has_tools


def _compute_rule_score(
    features: PromptFeatures,
    model: Model,
    *,
    request_has_tools: bool = False,
) -> float:
    """Score how well a model matches prompt capability signals."""
    caps = _model_caps(model)
    score = 0.0
    if features.has_images and ModelCapability.vision.value in caps:
        score += 3.0
    if (features.has_tool_calls or request_has_tools) and (
        ModelCapability.tool_calling.value in caps
        or ModelCapability.function_calling.value in caps
    ):
        score += 2.0
    if features.token_count > 4000 and ModelCapability.long_context.value in caps:
        score += 2.0
    if features.has_code_blocks and ModelCapability.code.value in caps:
        score += 1.5
    reasoning_signal = (
        features.task_difficulty if features.task_difficulty > 0 else features.reasoning_complexity
    )
    if reasoning_signal > 0.5 and ModelCapability.reasoning.value in caps:
        score += 2.0
    if reasoning_signal > 0.5 and "reasoning" in (model.tags or []):
        score += 1.0
    score += model.priority * 0.1
    return score


def _filter_eligible_models(
    features: PromptFeatures,
    models: list[Model],
    *,
    request_has_tools: bool = False,
) -> list[Model]:
    """Hard capability/context filter applied before rule or complexity ranking."""
    needs_tools = _needs_tool_support(features, request_has_tools=request_has_tools)
    eligible: list[Model] = []

    for model in models:
        if not model.is_active:
            continue
        caps = _model_caps(model)

        if features.has_images and ModelCapability.vision.value not in caps:
            continue
        if needs_tools and (
            ModelCapability.tool_calling.value not in caps
            and ModelCapability.function_calling.value not in caps
        ):
            continue
        if features.token_count > model.context_window:
            continue

        eligible.append(model)

    return eligible


def _model_total_cost(model: Model) -> float:
    return model.cost_per_1k_input + model.cost_per_1k_output


def _pick_fastest_cheapest(models: list[Model]) -> Model | None:
    """Among pre-filtered candidates, prefer speed then cost."""
    if not models:
        return None
    return min(
        models,
        key=lambda model: (
            -(model.estimated_tokens_per_second or 0.0),
            _model_total_cost(model),
            -(model.priority or 0),
            model.id,
        ),
    )


def _pick_best_by_rules(
    features: PromptFeatures,
    models: list[Model],
    *,
    request_has_tools: bool = False,
) -> tuple[Model | None, float]:
    if not models:
        return None, 0.0

    rule_scores = {
        model.id: _compute_rule_score(
            features, model, request_has_tools=request_has_tools,
        )
        for model in models
    }
    best_rule = max(rule_scores.values())
    top_tier = [
        model for model in models
        if rule_scores[model.id] >= best_rule - 1e-9
    ]
    best_model = _pick_fastest_cheapest(top_tier)
    if not best_model:
        return None, 0.0

    best_score = rule_scores[best_model.id]
    confidence = min(best_score / 5.0, 1.0) if best_score > 0 else 0.3
    return best_model, confidence


def select_routing_model(
    features: PromptFeatures,
    models: list[Model],
    *,
    request_has_tools: bool = False,
) -> tuple[Model | None, float, str]:
    """Route to the fastest, then cheapest model that meets minimum requirements.

    Minimum requirements are enforced by the capability/context filter and the
    highest rule-score tier (vision, tools, code, reasoning, etc.). When
    ``settings.complexity_routing_enabled`` is true, models must also satisfy
    ``max_complexity_score`` before speed/cost ranking is applied.
    """
    eligible = _filter_eligible_models(
        features, models, request_has_tools=request_has_tools,
    )
    if not eligible:
        return None, 0.0, "default"

    rule_scores = {
        model.id: _compute_rule_score(features, model, request_has_tools=request_has_tools)
        for model in eligible
    }
    best_rule = max(rule_scores.values())
    top_tier = [model for model in eligible if rule_scores[model.id] >= best_rule - 1e-9]

    use_complexity = (
        settings.complexity_routing_enabled
        and any(model.max_complexity_score is not None for model in top_tier)
    )
    if use_complexity:
        model, confidence = match_model_by_complexity(features, top_tier)
        if model:
            return model, confidence, "complexity"

    best_model = _pick_fastest_cheapest(top_tier)
    if not best_model:
        return None, 0.0, "default"

    best_score = rule_scores[best_model.id]
    confidence = min(best_score / 5.0, 1.0) if best_score > 0 else 0.3
    return best_model, confidence, "rules"


def match_model_by_rules(
    features: PromptFeatures,
    models: list[Model],
    *,
    request_has_tools: bool = False,
) -> tuple[Model | None, float]:
    eligible = _filter_eligible_models(
        features, models, request_has_tools=request_has_tools,
    )
    return _pick_best_by_rules(features, eligible, request_has_tools=request_has_tools)


def match_model_by_complexity(
    features: PromptFeatures,
    models: list[Model],
) -> tuple[Model | None, float]:
    """Pick the fastest/cheapest model that can handle prompt complexity.

    Only used when ``settings.complexity_routing_enabled`` is true.
    Callers should pass models already filtered by ``_filter_eligible_models``.
    """
    complexity = _compute_complexity_score(features)
    capable: list[Model] = []

    for model in models:
        if not model.is_active:
            continue

        max_cx = model.max_complexity_score
        if max_cx is not None and complexity > max_cx:
            continue

        capable.append(model)

    if not capable:
        return None, 0.0

    best_model = _pick_fastest_cheapest(capable)
    if not best_model:
        return None, 0.0

    max_cx = best_model.max_complexity_score
    if max_cx is not None:
        capacity_ratio = (max_cx - complexity) / max(0.01, max_cx)
        confidence = min(0.5 + capacity_ratio * 0.5, 1.0)
    else:
        confidence = 0.5

    return best_model, confidence


def _sanitize_for_debug_storage(value: object, max_str_len: int = 2000) -> object:
    """Redact large base64 blobs while keeping prompts readable for debugging."""
    if isinstance(value, str):
        if value.startswith("data:image/") or (
            len(value) > 200 and ";base64," in value
        ):
            return f"[base64 image data, {len(value)} chars]"
        if len(value) > max_str_len:
            return f"{value[:max_str_len]}... [{len(value)} chars total]"
        return value
    if isinstance(value, dict):
        return {k: _sanitize_for_debug_storage(v, max_str_len) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_debug_storage(item, max_str_len) for item in value]
    return value


async def store_prompt_debug(
    request_id: str,
    model_id: str | None,
    messages: list[dict],
    features: PromptFeatures,
    queue: RedisQueue = redis_queue,
) -> None:
    try:
        entry = {
            "request_id": request_id,
            "model_id": model_id,
            "messages": _sanitize_for_debug_storage(messages),
            "features": features.model_dump(),
            "image_detection": analyze_images(messages),
            "created_at": datetime.utcnow().isoformat(),
        }
        await queue.store_prompt_debug(entry)
    except Exception:
        logger.exception("Failed to store prompt debug entry for request %s", request_id)


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

    matched_model = None
    confidence = 0.0
    routing_method = "default"

    request_has_tools = bool(request.tools)
    matched_model, confidence, routing_method = select_routing_model(
        features, models, request_has_tools=request_has_tools,
    )

    if matched_model:
        logger.info(
            "Routed to %s via %s (confidence %.2f, task_type %s)",
            matched_model.id, routing_method, confidence, features.task_type,
        )

    if matched_model and confidence >= settings.classifier_min_confidence:
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


def _evaluate_models_for_routing(
    features: PromptFeatures,
    models: list[Model],
    routing_difficulty: float,
    *,
    request_has_tools: bool = False,
) -> list[dict]:
    """Document which models pass/fail capability and complexity checks."""
    eligible_ids = {
        m.id for m in _filter_eligible_models(
            features, models, request_has_tools=request_has_tools,
        )
    }
    needs_tools = _needs_tool_support(features, request_has_tools=request_has_tools)

    evaluations: list[dict] = []
    for model in models:
        if not model.is_active:
            evaluations.append({
                "model_id": model.id,
                "eligible": False,
                "exclusion_reason": "inactive",
                "max_complexity_score": model.max_complexity_score,
                "rule_score": None,
                "selected": False,
            })
            continue

        caps = _model_caps(model)
        reasons: list[str] = []
        if features.has_images and ModelCapability.vision.value not in caps:
            reasons.append("missing vision capability")
        if needs_tools and (
            ModelCapability.tool_calling.value not in caps
            and ModelCapability.function_calling.value not in caps
        ):
            reasons.append("missing tool_calling capability")
        if features.token_count > model.context_window:
            reasons.append(
                f"token_count {features.token_count} > context_window {model.context_window}"
            )

        max_cx = model.max_complexity_score
        rule_score = _compute_rule_score(
            features, model, request_has_tools=request_has_tools,
        )

        eligible = model.id in eligible_ids
        if (
            eligible
            and settings.complexity_routing_enabled
            and max_cx is not None
            and routing_difficulty > max_cx
        ):
            reasons.append(
                f"routing_difficulty {routing_difficulty:.3f} > max_complexity_score {max_cx:.3f}"
            )
            eligible = False
        evaluations.append({
            "model_id": model.id,
            "eligible": eligible,
            "exclusion_reason": "; ".join(reasons) if reasons else None,
            "max_complexity_score": model.max_complexity_score,
            "rule_score": round(rule_score, 2),
            "selected": False,
        })

    return evaluations


async def explain_routing(
    request: ChatCompletionRequest,
    db: AsyncSession,
) -> dict:
    """Dry-run routing with full complexity breakdown (no upstream, no classifier enqueue)."""
    from app.services.prompt_complexity import build_complexity_explanation, get_routing_difficulty

    messages_dicts = [m.model_dump(exclude_none=True) for m in request.messages]
    features = extract_features(messages_dicts)
    full_text = _messages_to_text(messages_dicts)

    routing_difficulty = get_routing_difficulty(
        features.task_difficulty,
        features.requirement_load,
        legacy_complexity_score=features.complexity_score,
    )

    complexity_explanation = build_complexity_explanation(
        messages_dicts,
        features,
        token_count=features.token_count,
        dominant_language=features.dominant_language,
        has_code_blocks=features.has_code_blocks,
        has_images=features.has_images,
        has_tool_calls=features.has_tool_calls,
        full_text=full_text,
    )

    result = await db.execute(select(Model).where(Model.is_active == True))
    models = list(result.scalars().all())

    if not models:
        return {
            "model_id": settings.default_model,
            "routing_method": "default",
            "routing_confidence": 0.0,
            "routing_difficulty": routing_difficulty,
            "would_enqueue_classifier": False,
            "features": features,
            "complexity_explanation": complexity_explanation,
            "model_evaluations": [],
            "complexity_candidate": None,
            "rule_candidate": None,
        }

    evaluations = _evaluate_models_for_routing(
        features, models, routing_difficulty,
        request_has_tools=bool(request.tools),
    )

    matched_model = None
    confidence = 0.0
    routing_method = "default"
    complexity_candidate = None
    rule_candidate = None

    selected, confidence, routing_method = select_routing_model(
        features, models, request_has_tools=bool(request.tools),
    )
    matched_model = selected

    rule_model, rule_confidence = match_model_by_rules(
        features, models, request_has_tools=bool(request.tools),
    )
    if rule_model:
        rule_candidate = rule_model.id

    if settings.complexity_routing_enabled:
        eligible = _filter_eligible_models(
            features, models, request_has_tools=bool(request.tools),
        )
        rule_scores = {
            m.id: _compute_rule_score(
                features, m, request_has_tools=bool(request.tools),
            )
            for m in eligible
        }
        if rule_scores:
            best_rule = max(rule_scores.values())
            top_tier = [m for m in eligible if rule_scores[m.id] >= best_rule - 1e-9]
            cx_model, _ = match_model_by_complexity(features, top_tier)
            if cx_model:
                complexity_candidate = cx_model.id

    model_id = matched_model.id if matched_model else settings.default_model
    if not matched_model:
        routing_method = "default"
        model_id = settings.default_model

    for ev in evaluations:
        ev["selected"] = ev["model_id"] == model_id

    would_enqueue = (
        matched_model is not None
        and confidence < settings.classifier_min_confidence
        and routing_method != "default"
    )

    return {
        "model_id": model_id,
        "routing_method": routing_method,
        "routing_confidence": round(confidence, 3),
        "routing_difficulty": routing_difficulty,
        "would_enqueue_classifier": would_enqueue,
        "features": features,
        "complexity_explanation": complexity_explanation,
        "model_evaluations": evaluations,
        "complexity_candidate": complexity_candidate,
        "rule_candidate": rule_candidate,
    }
