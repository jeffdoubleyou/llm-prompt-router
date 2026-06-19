"""Tool-payload limits for llama.cpp upstream servers.

llama.cpp builds a GBNF grammar from the tools list; large tool schemas (e.g.
OpenCode + many MCP tools) can exceed parser limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from app.core.config import settings
from app.services.upstream_errors import openai_error_body

logger = logging.getLogger(__name__)


def _normalized_prefix(value: str) -> str:
    return value.rstrip("/").lower()


def tool_names_from_payload(tools: list[Any]) -> list[str]:
    """Extract function/tool names from an OpenAI-style tools array."""
    names: list[str] = []
    for entry in tools:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif entry.get("name"):
            names.append(str(entry["name"]))
        else:
            names.append("?")
    return names


def is_llamacpp_upstream(base_url: str, provider: str | None = None) -> bool:
    """True when the upstream should receive llama.cpp tool limits."""
    if not base_url:
        return False

    prefixes = settings.llamacpp_base_url_prefixes
    if prefixes:
        base_key = _normalized_prefix(base_url)
        return any(base_key.startswith(_normalized_prefix(p)) for p in prefixes)

    provider_key = (provider or "").strip().lower()
    allowed = {p.lower() for p in settings.llamacpp_providers}
    if provider_key in allowed:
        return True

    host = (urlparse(base_url).hostname or "").lower()
    if host in {"127.0.0.1", "localhost", "::1"}:
        return True
    if host.startswith("192.168.") or host.startswith("10."):
        return True

    return False


@dataclass
class LlamacppToolLimitResult:
    payload: dict[str, Any]
    rejected: bool = False
    error_body: dict[str, Any] | None = None
    tool_names: list[str] = field(default_factory=list)


def _log_requested_tools(base_url: str, tool_names: list[str]) -> None:
    logger.info(
        "llama.cpp upstream tools (%d) for %s: %s",
        len(tool_names),
        base_url,
        ", ".join(tool_names) if tool_names else "(none)",
    )


def _reject_body(tool_count: int, max_tools: int, tool_names: list[str]) -> dict[str, Any]:
    preview = ", ".join(tool_names[:25])
    if len(tool_names) > 25:
        preview += f" … +{len(tool_names) - 25} more"
    message = (
        f"Too many tools ({tool_count}) for llama.cpp backend (max {max_tools}). "
        "Disable MCP tools in the client, use a cloud model, raise LLAMACPP_MAX_TOOLS, "
        f"or set LLAMACPP_TOOL_LIMIT_MODE=truncate. Tools requested: {preview}"
    )
    return openai_error_body(
        message,
        status_code=400,
        error_type="invalid_request_error",
        code="too_many_tools",
    )


def prepare_llamacpp_upstream_payload(
    payload: dict[str, Any],
    base_url: str,
    provider: str | None = None,
) -> LlamacppToolLimitResult:
    """Log tools and apply reject/truncate policy for llama.cpp upstreams."""
    max_tools = settings.llamacpp_max_tools
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return LlamacppToolLimitResult(payload=payload)

    if max_tools <= 0 or not is_llamacpp_upstream(base_url, provider):
        return LlamacppToolLimitResult(payload=payload)

    tool_names = tool_names_from_payload(tools)
    _log_requested_tools(base_url, tool_names)

    if len(tools) <= max_tools:
        return LlamacppToolLimitResult(payload=payload, tool_names=tool_names)

    mode = settings.llamacpp_tool_limit_mode
    if mode == "reject":
        logger.warning(
            "llama.cpp tool limit: rejecting %d tools (max %d) for %s",
            len(tools),
            max_tools,
            base_url,
        )
        return LlamacppToolLimitResult(
            payload=payload,
            rejected=True,
            error_body=_reject_body(len(tools), max_tools, tool_names),
            tool_names=tool_names,
        )

    # truncate
    limited = dict(payload)
    limited["tools"] = tools[:max_tools]
    limited.pop("tool_choice", None)
    kept = tool_names[:max_tools]
    dropped = tool_names[max_tools:]
    logger.warning(
        "llama.cpp tool limit: truncating %d tools to %d for %s — "
        "kept: %s; dropped: %s",
        len(tools),
        max_tools,
        base_url,
        ", ".join(kept),
        ", ".join(dropped[:12]) + (f" … +{len(dropped) - 12} more" if len(dropped) > 12 else ""),
    )
    return LlamacppToolLimitResult(
        payload=limited,
        tool_names=kept,
    )


def apply_llamacpp_tool_limits(
    payload: dict[str, Any],
    base_url: str,
    provider: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper; raises if reject mode would apply."""
    result = prepare_llamacpp_upstream_payload(payload, base_url, provider)
    if result.rejected:
        return payload
    return result.payload
