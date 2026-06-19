"""OpenAI-compatible error bodies for upstream failures."""

from __future__ import annotations

import json
from typing import Any


def _error_type_for_status(status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "server_error"
    return "invalid_request_error"


def _error_code_for_status(status_code: int) -> str:
    if status_code == 429:
        return "rate_limit_exceeded"
    if status_code >= 500:
        return "upstream_error"
    return "invalid_request_error"


def openai_error_body(
    message: str,
    *,
    status_code: int = 400,
    error_type: str | None = None,
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    """Build ``{"error": {"message", "type", "param", "code"}}``."""
    return {
        "error": {
            "message": message,
            "type": error_type or _error_type_for_status(status_code),
            "param": param,
            "code": code or _error_code_for_status(status_code),
        }
    }


def parse_upstream_error_text(body_text: str) -> dict[str, Any] | None:
    """Return upstream JSON if it already looks like an OpenAI error response."""
    text = (body_text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        err = data["error"]
        if isinstance(err.get("message"), str):
            return data
    return None


def upstream_error_from_response(status_code: int, body_text: str) -> dict[str, Any]:
    """Normalize an upstream HTTP error into an OpenAI-style JSON body."""
    parsed = parse_upstream_error_text(body_text)
    if parsed is not None:
        return parsed

    message = (body_text or "").strip()
    if len(message) > 2000:
        message = message[:2000] + "…"
    if not message:
        message = f"Upstream request failed with status {status_code}"
    return openai_error_body(message, status_code=status_code)


def upstream_error_message(body: dict[str, Any]) -> str:
    """Extract a log-friendly message from an OpenAI-style error body."""
    err = body.get("error")
    if isinstance(err, dict) and err.get("message"):
        return str(err["message"])
    return json.dumps(body)[:500]
