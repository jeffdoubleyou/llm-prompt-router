"""Tests for image detection in extract_features (JEF-157)."""

from __future__ import annotations

import pytest

from app.services.router_service import extract_features, _detect_images, _is_image_uri


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

    def test_base64_in_string_no_prefix(self):
        """Base64 data without data: prefix but with base64 marker."""
        messages = [{
            "role": "user",
            "content": "iVBORw0KGgo;base64,someImageData",
        }]
        assert _detect_images(messages) is True

    # --- Image URLs in string content ---

    def test_image_url_in_string(self):
        """Image URL embedded in string content."""
        messages = [{
            "role": "user",
            "content": "Check this image: https://example.com/photo.jpg",
        }]
        assert _detect_images(messages) is True

    def test_image_url_png(self):
        """PNG image URL in string content."""
        messages = [{
            "role": "user",
            "content": "See attached: https://cdn.example.com/image.png",
        }]
        assert _detect_images(messages) is True

    def test_image_url_webp(self):
        """WebP image URL in string content."""
        messages = [{
            "role": "user",
            "content": "View this: https://example.com/diagram.webp",
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


class TestIsImageUri:
    """Tests for the _is_image_uri helper function."""

    def test_data_uri_png(self):
        """data:image/png;base64,..."""
        assert _is_image_uri("data:image/png;base64,ABC") is True

    def test_data_uri_jpeg(self):
        """data:image/jpeg;base64,..."""
        assert _is_image_uri("data:image/jpeg;base64,ABC") is True

    def test_base64_marker(self):
        """String containing ;base64,"""
        assert _is_image_uri("some;base64,data") is True

    def test_png_url(self):
        """URL ending in .png."""
        assert _is_image_uri("https://example.com/photo.png") is True

    def test_jpeg_url(self):
        """URL ending in .jpg."""
        assert _is_image_uri("https://example.com/photo.jpg") is True

    def test_gif_url(self):
        """URL ending in .gif."""
        assert _is_image_uri("https://example.com/animation.gif") is True

    def test_webp_url(self):
        """URL ending in .webp."""
        assert _is_image_uri("https://example.com/image.webp") is True

    def test_svg_url(self):
        """URL ending in .svg."""
        assert _is_image_uri("https://example.com/logo.svg") is True

    def test_bmp_url(self):
        """URL ending in .bmp."""
        assert _is_image_uri("https://example.com/photo.bmp") is True

    def test_png_with_query_params(self):
        """PNG URL with query parameters."""
        assert _is_image_uri("https://example.com/photo.png?width=100") is True

    def test_plain_text_url(self):
        """Non-image URL should return False."""
        assert _is_image_uri("https://example.com/page.html") is False

    def test_plain_text(self):
        """Plain text should return False."""
        assert _is_image_uri("hello world") is False


class TestExtractFeaturesImages:
    """Integration tests: extract_features should set has_images correctly."""

    def test_plain_text_no_images(self):
        features = extract_features([{"role": "user", "content": "Hello"}])
        assert features.has_images is False

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

    def test_image_url_in_text(self):
        messages = [{
            "role": "user",
            "content": "Check https://example.com/photo.jpg",
        }]
        features = extract_features(messages)
        assert features.has_images is True
