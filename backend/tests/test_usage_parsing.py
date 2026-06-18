from app.services.router_service import estimate_token_count, parse_usage_from_response


def test_parse_usage_openai_format():
    prompt, completion = parse_usage_from_response(
        {"usage": {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}}
    )
    assert prompt == 12
    assert completion == 34


def test_parse_usage_input_output_format():
    prompt, completion = parse_usage_from_response(
        {"usage": {"input_tokens": 8, "output_tokens": 16, "total_tokens": 24}}
    )
    assert prompt == 8
    assert completion == 16


def test_parse_usage_total_only():
    prompt, completion = parse_usage_from_response(
        {"usage": {"total_tokens": 100}}
    )
    assert prompt == 100
    assert completion == 0


def test_parse_usage_missing():
    prompt, completion = parse_usage_from_response({})
    assert prompt == 0
    assert completion == 0


def test_estimate_token_count_nonempty():
    assert estimate_token_count("hello world") > 0
