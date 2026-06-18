"""Tests for image detection in extract_features (JEF-157)."""

from __future__ import annotations

import pytest

from app.services.router_service import (
    extract_features,
    _detect_images,
    _has_urls_in_messages,
    _is_image_data_uri,
    analyze_images,
)


class TestAnalyzeImages:
    def test_no_images_returns_empty_detections(self):
        result = analyze_images([{"role": "user", "content": "hello"}])
        assert result["has_images"] is False
        assert result["detection_count"] == 0
        assert result["detections"] == []

    def test_openai_image_url_lists_detection(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ],
        }]
        result = analyze_images(messages)
        assert result["has_images"] is True
        assert result["detection_count"] == 1
        assert result["detections"][0]["match_type"] == "openai_image_url"
        assert result["detections"][0]["message_index"] == 0
        assert result["detections"][0]["part_index"] == 1
        assert "example.com" in result["detections"][0]["detail"]

    def test_multiple_detections(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/1.png"}},
                {"type": "image_url", "image_url": {"url": "https://example.com/2.png"}},
            ],
        }]
        result = analyze_images(messages)
        assert result["detection_count"] == 2

    def test_plain_url_not_listed(self):
        result = analyze_images([{
            "role": "user",
            "content": "https://example.com/photo.jpg",
        }])
        assert result["has_images"] is False


class TestDetectImages:
    """Tests for the _detect_images helper function."""

    def test_no_content(self):
        """Empty or missing content should return False."""
        assert _detect_images([]) is False
        assert _detect_images([{"role": "user", "content": None}]) is False
        assert _detect_images([{"role": "user"}]) is False

    def test_plain_text_no_images(self):
        """Plain text without any image references should return False."""
        messages = [{"role": "user", "content": "Hello, world!"}]
        assert _detect_images(messages) is False

    def test_mentioned_image_filename_not_detected(self):
        """Mentioning a filename should not count as an attached image."""
        messages = [{"role": "user", "content": "Save the output as chart.png"}]
        assert _detect_images(messages) is False

    def test_mentioned_image_url_not_detected(self):
        """A plain-text URL to an image is not multimodal image content."""
        messages = [{
            "role": "user",
            "content": "Check this image: https://example.com/photo.jpg",
        }]
        assert _detect_images(messages) is False

    def test_loose_base64_marker_not_detected(self):
        """Loose ;base64, mentions without a data URI should not trigger."""
        messages = [{"role": "user", "content": "iVBORw0KGgo;base64,someImageData"}]
        assert _detect_images(messages) is False

    # --- OpenAI array-of-parts: image_url ---

    def test_openai_image_url_format(self):
        """OpenAI array-of-parts with image_url type."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
            ],
        }]
        assert _detect_images(messages) is True

    def test_openai_image_url_only(self):
        """Single-part message with just image_url."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/photo.png"}},
            ],
        }]
        assert _detect_images(messages) is True

    # --- OpenAI inline image ---

    def test_openai_inline_image_type(self):
        """OpenAI inline image with type=image and image_url key."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image_url": {"url": "https://example.com/photo.jpg"}},
            ],
        }]
        assert _detect_images(messages) is True

    # --- Anthropic image format ---

    def test_anthropic_image_source(self):
        """Anthropic image format with type=image and source key."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image", "source": {"type": "base64", "data": "iVBORw0KGgo", "media_type": "image/png"}},
            ],
        }]
        assert _detect_images(messages) is True

    def test_anthropic_image_url_source(self):
        """Anthropic image with URL source."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/photo.jpg"}},
            ],
        }]
        assert _detect_images(messages) is True

    # --- Base64 data URIs in string content ---

    def test_base64_data_uri_in_string(self):
        """Base64 data URI in plain string content."""
        messages = [{
            "role": "user",
            "content": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        }]
        assert _detect_images(messages) is True

    def test_markdown_image_in_string(self):
        """Markdown image syntax in string content."""
        messages = [{
            "role": "user",
            "content": "Look at this: ![diagram](https://example.com/diagram.png)",
        }]
        assert _detect_images(messages) is True

    # --- Mixed messages ---

    def test_mixed_text_and_image_messages(self):
        """Conversation with one message containing an image."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": [
                {"type": "text", "text": "What is in this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ]},
        ]
        assert _detect_images(messages) is True

    def test_multiple_images(self):
        """Multiple images in a single message."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/1.png"}},
                {"type": "image_url", "image_url": {"url": "https://example.com/2.jpg"}},
                {"type": "text", "text": "Compare these"},
            ],
        }]
        assert _detect_images(messages) is True

    # --- Edge cases ---

    def test_non_dict_parts_ignored(self):
        """Non-dict parts in content list are safely ignored."""
        messages = [{
            "role": "user",
            "content": ["not a dict", None, {"type": "text", "text": "hello"}],
        }]
        assert _detect_images(messages) is False

    def test_image_url_nested_base64(self):
        """Base64 data URI nested inside image_url.url."""
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgo"
                    },
                },
            ],
        }]
        assert _detect_images(messages) is True


class TestIsImageDataUri:
    """Tests for the _is_image_data_uri helper function."""

    def test_data_uri_png(self):
        assert _is_image_data_uri("data:image/png;base64,ABC") is True

    def test_data_uri_jpeg(self):
        assert _is_image_data_uri("data:image/jpeg;base64,ABC") is True

    def test_loose_base64_marker(self):
        assert _is_image_data_uri("some;base64,data") is False

    def test_png_url(self):
        assert _is_image_data_uri("https://example.com/photo.png") is False

    def test_plain_text(self):
        assert _is_image_data_uri("hello world") is False


class TestHasUrls:
    def test_user_message_with_url(self):
        messages = [{"role": "user", "content": "See https://example.com/docs"}]
        assert _has_urls_in_messages(messages) is True

    def test_system_message_urls_ignored(self):
        messages = [
            {"role": "system", "content": "Follow rules at https://docs.example.com/rules"},
            {"role": "user", "content": "Hello"},
        ]
        assert _has_urls_in_messages(messages) is False

    def test_url_in_multimodal_text_part(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "Read https://example.com/page"}],
        }]
        assert _has_urls_in_messages(messages) is True


class TestExtractFeaturesImages:
    """Integration tests: extract_features should set has_images correctly."""

    def test_plain_text_no_images(self):
        features = extract_features([{"role": "user", "content": "Hello"}])
        assert features.has_images is False
        assert features.has_urls is False

    def test_system_urls_do_not_set_has_urls(self):
        features = extract_features([
            {"role": "system", "content": "Docs: https://example.com/guide"},
            {"role": "user", "content": "Summarize this function"},
        ])
        assert features.has_urls is False

    def test_openai_image_url_format(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
            ],
        }]
        features = extract_features(messages)
        assert features.has_images is True

    def test_base64_data_uri(self):
        messages = [{
            "role": "user",
            "content": "data:image/png;base64,ABC123",
        }]
        features = extract_features(messages)
        assert features.has_images is True

    def test_anthropic_image_format(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": "iVBORw0KGgo"}},
            ],
        }]
        features = extract_features(messages)
        assert features.has_images is True

    def test_image_url_in_text_not_flagged(self):
        messages = [{
            "role": "user",
            "content": "Check https://example.com/photo.jpg",
        }]
        features = extract_features(messages)
        assert features.has_images is False
        assert features.has_urls is True
