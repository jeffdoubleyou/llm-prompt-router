import json

from app.services.router_service import (
    _filter_eligible_models,
    _prompt_exceeds_context_window,
    estimate_prompt_tokens,
    estimate_token_count,
    extract_features,
)
from tests.test_complexity_routing import _make_model


def test_estimate_prompt_tokens_includes_tool_definitions():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "Find docs"}]

    without_tools = estimate_prompt_tokens(messages)
    with_tools = estimate_prompt_tokens(messages, tools=tools)

    assert with_tools.tools_tokens > 0
    assert with_tools.prompt_tokens > without_tools.prompt_tokens


def test_estimate_prompt_tokens_includes_tool_call_arguments():
    messages = [
        {"role": "user", "content": "Run it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({"command": "echo hello world"}),
                    },
                }
            ],
        },
    ]

    estimate = estimate_prompt_tokens(messages)

    assert estimate.tool_call_tokens > 0
    assert estimate.prompt_tokens > estimate.message_tokens


def test_extract_features_counts_tools_and_tool_calls():
    tools = [{"type": "function", "function": {"name": "ping", "parameters": {}}}]
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "ping",
                        "arguments": '{"x": "' + ("y" * 200) + '"}',
                    }
                }
            ],
        },
    ]

    content_only = extract_features([{"role": "user", "content": "hello"}])
    full = extract_features(messages, tools=tools)

    assert full.token_count > content_only.token_count


def test_context_filter_reserves_output_tokens():
    model = _make_model(model_id="small", context_window=15000, max_tokens=4096)
    from app.core.models import PromptFeatures

    features = PromptFeatures(token_count=8000)

    assert not _prompt_exceeds_context_window(features.token_count, model)
    assert _prompt_exceeds_context_window(features.token_count, model, max_tokens=8000)

    eligible = _filter_eligible_models(features, [model], max_tokens=8000)
    assert eligible == []


def test_context_filter_uses_request_max_tokens():
    model = _make_model(model_id="medium", context_window=20000, max_tokens=4096)
    from app.core.models import PromptFeatures

    features = PromptFeatures(token_count=12000)

    assert _filter_eligible_models(features, [model], max_tokens=9000) == []
    assert _filter_eligible_models(features, [model], max_tokens=5000) == [model]


def test_template_overhead_scales_with_message_count():
    one = estimate_prompt_tokens([{"role": "user", "content": "hi"}])
    many = estimate_prompt_tokens(
        [{"role": "user", "content": "hi"} for _ in range(10)],
    )

    assert many.template_overhead > one.template_overhead
    assert many.prompt_tokens > one.prompt_tokens


def test_estimate_token_count_nonempty():
    assert estimate_token_count("hello world") > 0
