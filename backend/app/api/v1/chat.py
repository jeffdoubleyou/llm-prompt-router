from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
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
from app.services.router_service import (
    classify_and_route,
    estimate_token_count,
    extract_features,
    get_model_by_id,
    log_request,
    parse_usage_from_response,
    store_prompt_debug,
)
from app.services.upstream_queue import upstream_queue_manager
from app.services.llamacpp_tools import prepare_llamacpp_upstream_payload
from app.services.upstream_errors import (
    openai_error_body,
    upstream_error_from_response,
    upstream_error_message,
)

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
    features = extract_features(messages_dicts, tools=chat_req.tools)

    model_id = await classify_and_route(chat_req, db)

    await store_prompt_debug(
        request_id,
        model_id,
        messages_dicts,
        features,
        tools=chat_req.tools,
        max_tokens=chat_req.max_tokens,
    )

    model_obj = await get_model_by_id(db, model_id)
    if not model_obj:
        model_obj = await get_model_by_id(db, settings.default_model)
    if not model_obj:
        return JSONResponse(
            status_code=503,
            content=openai_error_body(
                "No models configured",
                status_code=503,
                error_type="server_error",
                code="service_unavailable",
            ),
        )

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
    if chat_req.stream:
        payload["stream_options"] = {"include_usage": True}

    tool_limit = prepare_llamacpp_upstream_payload(
        payload, base_url, model_obj.provider,
    )
    if tool_limit.rejected and tool_limit.error_body:
        error_detail = upstream_error_message(tool_limit.error_body)
        logger.error("Request %s rejected before upstream: %s", request_id, error_detail)
        await log_request(
            db, request_id, model_id,
            prompt_tokens=0, completion_tokens=0,
            latency_ms=0.0, cost=0.0,
            is_error=True, error_message=error_detail,
            model_used=model_id,
        )
        return JSONResponse(status_code=400, content=tool_limit.error_body)
    payload = tool_limit.payload

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if chat_req.stream:
        return await _handle_stream(
            request_id, model_id, model_obj, base_url, upstream_url, payload, headers, db,
            estimated_prompt_tokens=features.token_count,
        )

    return await _handle_non_stream(
        request_id, model_id, model_obj, base_url, upstream_url, payload, headers, db,
        estimated_prompt_tokens=features.token_count,
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
    estimated_prompt_tokens: int = 0,
):
    async def _execute():
        return await _handle_non_stream_inner(
            request_id, model_id, model_obj, upstream_url, payload, headers, db,
            estimated_prompt_tokens=estimated_prompt_tokens,
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
    estimated_prompt_tokens: int = 0,
):
    start_time = time.monotonic()
    timeout = _get_timeout(model_obj)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(upstream_url, json=payload, headers=headers)
            elapsed = (time.monotonic() - start_time) * 1000

            if resp.status_code != 200:
                error_body = upstream_error_from_response(resp.status_code, resp.text)
                error_detail = upstream_error_message(error_body)
                logger.error("Request %s: %s", request_id, error_detail)
                await log_request(
                    db, request_id, model_id,
                    prompt_tokens=0, completion_tokens=0,
                    latency_ms=elapsed, cost=0.0,
                    is_error=True, error_message=error_detail,
                    model_used=model_id,
                )
                return JSONResponse(status_code=resp.status_code, content=error_body)

            upstream_data = resp.json()

            prompt_tokens, completion_tokens = parse_usage_from_response(upstream_data)
            if prompt_tokens == 0 and estimated_prompt_tokens > 0:
                prompt_tokens = estimated_prompt_tokens
            if completion_tokens == 0:
                completion_text_parts: list[str] = []
                for choice_data in upstream_data.get("choices", []):
                    msg = choice_data.get("message", {})
                    content = msg.get("content")
                    if isinstance(content, str):
                        completion_text_parts.append(content)
                    for tool_call in msg.get("tool_calls") or []:
                        fn = tool_call.get("function") or {}
                        if fn.get("name"):
                            completion_text_parts.append(str(fn["name"]))
                        if fn.get("arguments"):
                            completion_text_parts.append(str(fn["arguments"]))
                if completion_text_parts:
                    completion_tokens = estimate_token_count(" ".join(completion_text_parts))
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
        return JSONResponse(
            status_code=504,
            content=openai_error_body(
                "Upstream request timed out",
                status_code=504,
                error_type="server_error",
                code="upstream_timeout",
            ),
        )
    except httpx.RequestError as exc:
        elapsed = (time.monotonic() - start_time) * 1000
        await log_request(
            db, request_id, model_id,
            prompt_tokens=0, completion_tokens=0,
            latency_ms=elapsed, cost=0.0,
            is_error=True, error_message=f"Request error: {exc}",
            model_used=model_id,
        )
        return JSONResponse(
            status_code=502,
            content=openai_error_body(
                f"Upstream connection error: {exc}",
                status_code=502,
                error_type="server_error",
                code="upstream_connection_error",
            ),
        )


async def _handle_stream(
    request_id: str,
    model_id: str,
    model_obj: Model,
    base_url: str,
    upstream_url: str,
    payload: dict,
    headers: dict,
    db,
    estimated_prompt_tokens: int = 0,
):
    queue_slot = None
    if settings.upstream_queue_enabled:
        queue_slot = upstream_queue_manager.acquire(base_url, request_id, model_id)
        await queue_slot.__aenter__()

    async def event_generator():
        try:
            async for event in _iter_stream_events(
                request_id, model_id, model_obj, upstream_url, payload, headers, db,
                estimated_prompt_tokens=estimated_prompt_tokens,
            ):
                yield event
        finally:
            if queue_slot is not None:
                await queue_slot.__aexit__(None, None, None)

    return EventSourceResponse(event_generator())


async def _iter_stream_events(
    request_id: str,
    model_id: str,
    model_obj: Model,
    upstream_url: str,
    payload: dict,
    headers: dict,
    db,
    estimated_prompt_tokens: int = 0,
):
    start_time = time.monotonic()
    prompt_tokens = 0
    completion_tokens = 0
    completion_text_parts: list[str] = []
    first_chunk = True

    try:
        timeout = _get_timeout(model_obj)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", upstream_url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    error_body = upstream_error_from_response(
                        resp.status_code,
                        error_text.decode(errors="replace"),
                    )
                    error_detail = upstream_error_message(error_body)
                    logger.error("Stream request %s: %s", request_id, error_detail)
                    yield {"data": json.dumps(error_body)}
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
                        chunk_prompt, chunk_completion = parse_usage_from_response(chunk)
                        if chunk_prompt:
                            prompt_tokens = chunk_prompt
                        if chunk_completion:
                            completion_tokens = chunk_completion

                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content")
                        if isinstance(content, str):
                            completion_text_parts.append(content)
                        for tool_call in delta.get("tool_calls") or []:
                            fn = tool_call.get("function") or {}
                            if fn.get("name"):
                                completion_text_parts.append(str(fn["name"]))
                            if fn.get("arguments"):
                                completion_text_parts.append(str(fn["arguments"]))

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

                if prompt_tokens == 0 and estimated_prompt_tokens > 0:
                    prompt_tokens = estimated_prompt_tokens
                if completion_tokens == 0 and completion_text_parts:
                    completion_tokens = estimate_token_count("".join(completion_text_parts))

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
        yield {
            "data": json.dumps(openai_error_body(
                str(exc),
                status_code=500,
                error_type="server_error",
                code="stream_error",
            )),
        }
