from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.core.database import get_db
from app.core.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatChoice,
    Usage,
)
from app.models.db import Model
from app.services.router_service import classify_and_route, extract_features, get_model_by_id, log_request, store_prompt_debug
from app.services.upstream_queue import upstream_queue_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _get_timeout(model_obj: Model) -> float:
    """Return the effective timeout for a model, falling back to the global setting."""
    if model_obj.timeout is not None:
        return model_obj.timeout
    return settings.upstream_timeout


@router.post("/v1/chat/completions")
async def chat_completions(
    chat_req: ChatCompletionRequest,
    fastapi_request: Request,
    db=Depends(get_db),
):
    request_id = str(uuid.uuid4())
    messages_dicts = [m.model_dump(exclude_none=True) for m in chat_req.messages]
    features = extract_features(messages_dicts)

    model_id = await classify_and_route(chat_req, db)

    await store_prompt_debug(request_id, model_id, messages_dicts, features)

    model_obj = await get_model_by_id(db, model_id)
    if not model_obj:
        model_obj = await get_model_by_id(db, settings.default_model)
    if not model_obj:
        raise HTTPException(status_code=503, detail="No models configured")

    base_url = model_obj.base_url or f"https://api.openai.com/v1"
    api_key = model_obj.api_key_encrypted or ""

    if model_obj.provider == "openai" and not model_obj.base_url:
        base_url = "https://api.openai.com/v1"

    upstream_url = f"{base_url.rstrip('/')}/chat/completions"
    logger.info("Routing request %s to model %s at %s", request_id, model_id, upstream_url)

    messages_dicts = []
    for m in chat_req.messages:
        md = {"role": m.role.value}
        if m.content is not None:
            md["content"] = m.content
        if m.tool_calls:
            md["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            md["tool_call_id"] = m.tool_call_id
        if m.name:
            md["name"] = m.name
        messages_dicts.append(md)

    payload = {
        "model": model_id,
        "messages": messages_dicts,
        "temperature": chat_req.temperature,
        "top_p": chat_req.top_p,
        "n": chat_req.n,
        "stream": chat_req.stream,
        "max_tokens": chat_req.max_tokens,
        "presence_penalty": chat_req.presence_penalty,
        "frequency_penalty": chat_req.frequency_penalty,
        "stop": chat_req.stop,
        "user": chat_req.user,
    }
    if chat_req.tools:
        payload["tools"] = chat_req.tools
    if chat_req.tool_choice:
        payload["tool_choice"] = chat_req.tool_choice

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if chat_req.stream:
        return await _handle_stream(
            request_id, model_id, model_obj, base_url, upstream_url, payload, headers, db,
        )

    return await _handle_non_stream(
        request_id, model_id, model_obj, base_url, upstream_url, payload, headers, db,
    )


async def _with_upstream_queue(base_url: str, request_id: str, model_id: str, coro):
    if settings.upstream_queue_enabled:
        async with upstream_queue_manager.acquire(base_url, request_id, model_id):
            return await coro()
    return await coro()


async def _handle_non_stream(
    request_id: str,
    model_id: str,
    model_obj: Model,
    base_url: str,
    upstream_url: str,
    payload: dict,
    headers: dict,
    db,
):
    async def _execute():
        return await _handle_non_stream_inner(
            request_id, model_id, model_obj, upstream_url, payload, headers, db,
        )

    return await _with_upstream_queue(base_url, request_id, model_id, _execute)


async def _handle_non_stream_inner(
    request_id: str,
    model_id: str,
    model_obj: Model,
    upstream_url: str,
    payload: dict,
    headers: dict,
    db,
):
    start_time = time.monotonic()
    timeout = _get_timeout(model_obj)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(upstream_url, json=payload, headers=headers)
            elapsed = (time.monotonic() - start_time) * 1000

            if resp.status_code != 200:
                error_detail = f"Upstream returned {resp.status_code}: {resp.text[:500]}"
                logger.error("Request %s: %s", request_id, error_detail)
                prompt_tokens = payload.get("messages", [])
                await log_request(
                    db, request_id, model_id,
                    prompt_tokens=0, completion_tokens=0,
                    latency_ms=elapsed, cost=0.0,
                    is_error=True, error_message=error_detail,
                    model_used=model_id,
                )
                raise HTTPException(status_code=resp.status_code, detail=error_detail)

            upstream_data = resp.json()

            prompt_tokens = upstream_data.get("usage", {}).get("prompt_tokens", 0)
            completion_tokens = upstream_data.get("usage", {}).get("completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens

            input_cost = (prompt_tokens / 1000) * model_obj.cost_per_1k_input
            output_cost = (completion_tokens / 1000) * model_obj.cost_per_1k_output
            total_cost = input_cost + output_cost

            await log_request(
                db, request_id, model_id,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                latency_ms=elapsed, cost=total_cost,
                model_used=model_id,
            )

            choices = []
            for choice_data in upstream_data.get("choices", []):
                msg = choice_data.get("message", {})
                choices.append(
                    ChatChoice(
                        index=choice_data.get("index", 0),
                        message=ChatMessage(
                            role=msg.get("role", "assistant"),
                            content=msg.get("content"),
                            tool_calls=msg.get("tool_calls"),
                        ),
                        finish_reason=choice_data.get("finish_reason"),
                    )
                )

            return ChatCompletionResponse(
                id=upstream_data.get("id", f"chatcmpl-{request_id[:12]}"),
                created=int(datetime.utcnow().timestamp()),
                model=model_id,
                choices=choices,
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                ),
            )
    except httpx.TimeoutException:
        elapsed = (time.monotonic() - start_time) * 1000
        await log_request(
            db, request_id, model_id,
            prompt_tokens=0, completion_tokens=0,
            latency_ms=elapsed, cost=0.0,
            is_error=True, error_message="Upstream timeout",
            model_used=model_id,
        )
        raise HTTPException(status_code=504, detail="Upstream request timed out")
    except httpx.RequestError as exc:
        elapsed = (time.monotonic() - start_time) * 1000
        await log_request(
            db, request_id, model_id,
            prompt_tokens=0, completion_tokens=0,
            latency_ms=elapsed, cost=0.0,
            is_error=True, error_message=f"Request error: {exc}",
            model_used=model_id,
        )
        raise HTTPException(status_code=502, detail=f"Upstream connection error: {exc}")


async def _handle_stream(
    request_id: str,
    model_id: str,
    model_obj: Model,
    base_url: str,
    upstream_url: str,
    payload: dict,
    headers: dict,
    db,
):
    async def event_generator():
        async def stream_body():
            async for event in _iter_stream_events(
                request_id, model_id, model_obj, upstream_url, payload, headers, db,
            ):
                yield event

        if settings.upstream_queue_enabled:
            async with upstream_queue_manager.acquire(base_url, request_id, model_id):
                async for event in stream_body():
                    yield event
        else:
            async for event in stream_body():
                yield event

    return EventSourceResponse(event_generator())


async def _iter_stream_events(
    request_id: str,
    model_id: str,
    model_obj: Model,
    upstream_url: str,
    payload: dict,
    headers: dict,
    db,
):
    start_time = time.monotonic()
    prompt_tokens = 0
    completion_tokens = 0
    first_chunk = True

    try:
        timeout = _get_timeout(model_obj)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", upstream_url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    error_detail = f"Upstream returned {resp.status_code}: {error_text[:500].decode()}"
                    logger.error("Stream request %s: %s", request_id, error_detail)
                    yield {"event": "error", "data": json.dumps({"error": error_detail})}
                    elapsed = (time.monotonic() - start_time) * 1000
                    await log_request(
                        db, request_id, model_id,
                        prompt_tokens=0, completion_tokens=0,
                        latency_ms=elapsed, cost=0.0,
                        is_error=True, error_message=error_detail,
                        model_used=model_id,
                    )
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

                    chunk_id = chunk.get("id", f"chatcmpl-{request_id[:12]}")
                    created = chunk.get("created", int(datetime.utcnow().timestamp()))
                    yield {
                        "event": "chunk",
                        "data": json.dumps({
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": chunk.get("choices", []),
                        }),
                    }
                    if first_chunk:
                        first_chunk = False

                yield {"event": "done", "data": "[DONE]"}

                elapsed = (time.monotonic() - start_time) * 1000
                total_cost = (
                    (prompt_tokens / 1000) * model_obj.cost_per_1k_input
                    + (completion_tokens / 1000) * model_obj.cost_per_1k_output
                )
                await log_request(
                    db, request_id, model_id,
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    latency_ms=elapsed, cost=total_cost,
                    model_used=model_id,
                )
    except Exception as exc:
        logger.exception("Stream error for request %s", request_id)
        yield {"event": "error", "data": json.dumps({"error": str(exc)})}
