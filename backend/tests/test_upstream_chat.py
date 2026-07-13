"""Tests for upstream error formatting and llama.cpp tool limits."""

from __future__ import annotations

import json
import sys
import os

import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.modules["asyncpg"] = MagicMock()

from app.core.config import settings
from app.services.upstream_errors import (
    openai_error_body,
    parse_upstream_error_text,
    upstream_error_from_response,
    upstream_error_message,
)
from app.core.models import ChatCompletionRequest, ChatMessage, Role
from app.models.db import Model
from app.api.v1.chat import (
    build_upstream_chat_payload,
    omit_null_fields,
    resolve_upstream_max_tokens,
)
from app.services.llamacpp_tools import (
    is_llamacpp_upstream,
    prepare_llamacpp_upstream_payload,
    tool_names_from_payload,
)


class TestChatCompletionRequestExtensions:
    def test_parses_llamacpp_fields(self):
        req = ChatCompletionRequest.model_validate({
            "messages": [{"role": "user", "content": "hi"}],
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking_budget_tokens": 1024,
        })
        assert req.cache_prompt is True
        assert req.chat_template_kwargs == {"enable_thinking": False}
        assert req.thinking_budget_tokens == 1024

    def test_llamacpp_fields_default_to_none(self):
        req = ChatCompletionRequest(
            messages=[ChatMessage(role=Role.user, content="hi")],
        )
        assert req.cache_prompt is None
        assert req.chat_template_kwargs is None
        assert req.thinking_budget_tokens is None


class TestUpstreamErrors:
    def test_openai_error_shape(self):
        body = openai_error_body("bad request", status_code=400)
        assert body["error"]["message"] == "bad request"
        assert body["error"]["type"] == "invalid_request_error"
        assert isinstance(body["error"]["code"], str)

    def test_passthrough_upstream_openai_error(self):
        raw = json.dumps({
            "error": {
                "message": "Failed to initialize samplers: failed to parse grammar",
                "type": "invalid_request_error",
                "code": 400,
            },
        })
        body = upstream_error_from_response(400, raw)
        assert body["error"]["message"].startswith("Failed to initialize")

    def test_wrap_plain_text_upstream_error(self):
        body = upstream_error_from_response(502, "connection reset")
        assert body["error"]["message"] == "connection reset"
        assert body["error"]["type"] == "server_error"

    def test_parse_upstream_error_text(self):
        assert parse_upstream_error_text('{"error":{"message":"x"}}') is not None
        assert parse_upstream_error_text("not json") is None

    def test_upstream_error_message(self):
        body = openai_error_body("hello", status_code=500)
        assert upstream_error_message(body) == "hello"


class TestLlamacppTools:
    def test_tool_names_from_payload(self):
        tools = [
            {"type": "function", "function": {"name": "bash"}},
            {"type": "function", "function": {"name": "read"}},
        ]
        assert tool_names_from_payload(tools) == ["bash", "read"]

    def test_detects_custom_provider(self):
        assert is_llamacpp_upstream("http://10.0.0.5:8080/v1", "custom") is True

    def test_detects_configured_prefix(self, monkeypatch):
        monkeypatch.setattr(
            settings,
            "llamacpp_base_url_prefixes",
            ["http://192.168.0.124:8000/v1"],
        )
        assert is_llamacpp_upstream("http://192.168.0.124:8000/v1", "openai") is True

    def test_skips_openai_cloud(self, monkeypatch):
        monkeypatch.setattr(settings, "llamacpp_base_url_prefixes", [])
        monkeypatch.setattr(settings, "llamacpp_providers", ["custom", "llama"])
        assert is_llamacpp_upstream("https://api.openai.com/v1", "openai") is False

    def test_rejects_when_over_limit(self, monkeypatch):
        monkeypatch.setattr(settings, "llamacpp_max_tools", 2)
        monkeypatch.setattr(settings, "llamacpp_tool_limit_mode", "reject")
        monkeypatch.setattr(settings, "llamacpp_base_url_prefixes", ["http://127.0.0.1"])
        payload = {
            "tools": [
                {"type": "function", "function": {"name": f"tool-{i}"}}
                for i in range(5)
            ],
            "tool_choice": "auto",
        }
        result = prepare_llamacpp_upstream_payload(
            payload, "http://127.0.0.1:8080/v1", "custom",
        )
        assert result.rejected is True
        assert result.error_body is not None
        assert result.error_body["error"]["code"] == "too_many_tools"
        assert "tool-0" in result.error_body["error"]["message"]
        assert "tool-4" in result.error_body["error"]["message"]

    def test_truncates_when_mode_truncate(self, monkeypatch):
        monkeypatch.setattr(settings, "llamacpp_max_tools", 2)
        monkeypatch.setattr(settings, "llamacpp_tool_limit_mode", "truncate")
        monkeypatch.setattr(settings, "llamacpp_base_url_prefixes", ["http://127.0.0.1"])
        payload = {
            "tools": [
                {"type": "function", "function": {"name": f"tool-{i}"}}
                for i in range(5)
            ],
            "tool_choice": "auto",
        }
        result = prepare_llamacpp_upstream_payload(
            payload, "http://127.0.0.1:8080/v1", "custom",
        )
        assert result.rejected is False
        assert len(result.payload["tools"]) == 2
        assert "tool_choice" not in result.payload
        assert result.payload["tools"][0]["function"]["name"] == "tool-0"

    def test_no_change_when_under_limit(self, monkeypatch):
        monkeypatch.setattr(settings, "llamacpp_max_tools", 10)
        monkeypatch.setattr(settings, "llamacpp_tool_limit_mode", "reject")
        payload = {
            "tools": [{"type": "function", "function": {"name": "bash"}}],
            "tool_choice": "auto",
        }
        result = prepare_llamacpp_upstream_payload(
            payload, "http://127.0.0.1:8080/v1", "custom",
        )
        assert result.rejected is False
        assert result.payload == payload

    def test_disabled_when_max_zero(self, monkeypatch):
        monkeypatch.setattr(settings, "llamacpp_max_tools", 0)
        payload = {
            "tools": [{"type": "function", "function": {"name": f"t{i}"}} for i in range(5)],
            "tool_choice": "auto",
        }
        result = prepare_llamacpp_upstream_payload(
            payload, "http://127.0.0.1:8080/v1", "custom",
        )
        assert result.rejected is False
        assert result.payload == payload

    def test_logs_tool_list(self, monkeypatch, caplog):
        import logging
        monkeypatch.setattr(settings, "llamacpp_max_tools", 10)
        monkeypatch.setattr(settings, "llamacpp_tool_limit_mode", "reject")
        caplog.set_level(logging.INFO)
        payload = {
            "tools": [
                {"type": "function", "function": {"name": "bash"}},
                {"type": "function", "function": {"name": "read"}},
            ],
        }
        prepare_llamacpp_upstream_payload(
            payload, "http://127.0.0.1:8080/v1", "custom",
        )
        assert "bash, read" in caplog.text


def _stub_model(**overrides) -> Model:
    values = {
        "id": "test-model",
        "display_name": "test",
        "provider": "custom",
        "capabilities": ["text"],
        "tags": [],
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "max_tokens": 4096,
        "context_window": 128000,
        "rpm_limit": 60,
        "tpm_limit": 100000,
        "is_active": True,
        "priority": 1,
    }
    values.update(overrides)
    return Model(**values)


class TestUpstreamPayloadSanitization:
    def test_omit_null_fields(self):
        assert omit_null_fields({"a": 1, "b": None, "c": False}) == {"a": 1, "c": False}

    def test_resolve_max_tokens_from_request(self):
        model = _stub_model(max_tokens=2048)
        assert resolve_upstream_max_tokens(512, model) == 512

    def test_resolve_max_tokens_falls_back_to_model(self):
        model = _stub_model(max_tokens=2048)
        assert resolve_upstream_max_tokens(None, model) == 2048

    def test_resolve_max_tokens_default_when_model_unset(self):
        model = _stub_model(max_tokens=0)
        assert resolve_upstream_max_tokens(None, model) == 4096

    def test_null_max_tokens_becomes_model_default(self):
        """Regression: llama.cpp rejects max_tokens: null with type_error.302."""
        model = _stub_model(max_tokens=4096)
        req = ChatCompletionRequest.model_validate({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": None,
            "stop": None,
            "user": None,
            "temperature": None,
        })
        payload = build_upstream_chat_payload(
            model_id=model.id,
            model_obj=model,
            chat_req=req,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert payload["max_tokens"] == 4096
        assert isinstance(payload["max_tokens"], int)
        assert "stop" not in payload
        assert "user" not in payload
        assert "temperature" not in payload

    def test_explicit_max_tokens_preserved(self):
        model = _stub_model(max_tokens=4096)
        req = ChatCompletionRequest(
            messages=[ChatMessage(role=Role.user, content="hi")],
            max_tokens=1024,
        )
        payload = build_upstream_chat_payload(
            model_id=model.id,
            model_obj=model,
            chat_req=req,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert payload["max_tokens"] == 1024
