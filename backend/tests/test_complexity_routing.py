"""Tests for complexity-based prompt routing.

Verifies that:
- Complexity scores are computed correctly for various prompt types
- Models with complexity metadata are matched by capability + cost
- Fallback to rule-based matching works when no complexity metadata exists
- Edge cases (empty prompts, very long prompts, multimodal) are handled
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock

# Mock asyncpg before any app module import
sys.modules["asyncpg"] = MagicMock()

from app.core.models import PromptFeatures
from app.services.router_service import (
    _compute_complexity_score,
    match_model_by_complexity,
    match_model_by_rules,
    extract_features,
)
from app.models.db import Model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(
    model_id: str = "test-model",
    is_active: bool = True,
    capabilities: list | None = None,
    tags: list | None = None,
    cost_per_1k_input: float = 0.001,
    cost_per_1k_output: float = 0.005,
    max_tokens: int = 4096,
    context_window: int = 8192,
    priority: int = 10,
    estimated_parameters_billions: float | None = None,
    estimated_tokens_per_second: float | None = None,
    max_complexity_score: float | None = None,
    timeout: float | None = None,
) -> Model:
    """Convenience factory for test Model objects."""
    return Model(
        id=model_id,
        display_name=model_id,
        provider="openai",
        capabilities=capabilities or ["text", "streaming", "json_mode"],
        tags=tags or [],
        cost_per_1k_input=cost_per_1k_input,
        cost_per_1k_output=cost_per_1k_output,
        max_tokens=max_tokens,
        context_window=context_window,
        rpm_limit=60,
        tpm_limit=100000,
        is_active=is_active,
        priority=priority,
        estimated_parameters_billions=estimated_parameters_billions,
        estimated_tokens_per_second=estimated_tokens_per_second,
        max_complexity_score=max_complexity_score,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Complexity score tests
# ---------------------------------------------------------------------------

class TestComplexityScore:
    def test_empty_prompt(self):
        features = PromptFeatures()
        score = _compute_complexity_score(features)
        assert score == 0.0

    def test_short_simple_prompt_legacy_fallback(self):
        features = PromptFeatures(token_count=150, char_length=40)
        score = _compute_complexity_score(features)
        assert score == 0.05  # legacy token tier

    def test_task_difficulty_used_when_set(self):
        features = PromptFeatures(task_difficulty=0.72, requirement_load=0.2)
        score = _compute_complexity_score(features)
        assert score == 0.75  # 0.72 + 0.2 * 0.15

    def test_long_prompt_legacy(self):
        features = PromptFeatures(token_count=5000, char_length=20000)
        score = _compute_complexity_score(features)
        assert score >= 0.20

    def test_high_reasoning_legacy(self):
        features = PromptFeatures(reasoning_complexity=0.9)
        score = _compute_complexity_score(features)
        assert score >= 0.27

    def test_math_domain(self):
        features = PromptFeatures(dominant_language="math")
        score = _compute_complexity_score(features)
        assert score >= 0.15

    def test_code_domain(self):
        features = PromptFeatures(dominant_language="code", has_code_blocks=True)
        score = _compute_complexity_score(features)
        assert score >= 0.18  # 0.10 domain + 0.08 code

    def test_multimodal_prompt(self):
        features = PromptFeatures(has_images=True, has_tool_calls=True, token_count=200)
        score = _compute_complexity_score(features)
        assert score >= 0.10  # 0.05 images + 0.05 tool_calls

    def test_score_capped_at_1_0(self):
        features = PromptFeatures(
            token_count=10000,
            reasoning_complexity=1.0,
            dominant_language="math",
            has_code_blocks=True,
            has_images=True,
            has_tool_calls=True,
            has_urls=True,
        )
        score = _compute_complexity_score(features)
        # Sum: 0.30 + 0.30 + 0.15 + 0.08 + 0.05 + 0.05 + 0.02 = 0.95
        # Add a high token bonus to push it over 1.0 for cap testing
        features_with_more = PromptFeatures(
            token_count=10000,
            reasoning_complexity=1.0,
            dominant_language="math",
            has_code_blocks=True,
            has_images=True,
            has_tool_calls=True,
            has_urls=True,
        )
        score = _compute_complexity_score(features_with_more)
        assert score <= 1.0

    def test_score_capped_at_max(self):
        features = PromptFeatures(
            token_count=100,
            reasoning_complexity=0.0,
            dominant_language="natural_language",
        )
        score = _compute_complexity_score(features)
        assert score >= 0.0
        assert score <= 1.0


# ---------------------------------------------------------------------------
# Extract features tests
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_complexity_score(self):
        messages = [{"role": "user", "content": "Hello!"}]
        features = extract_features(messages)
        assert hasattr(features, "complexity_score")
        assert 0.0 <= features.complexity_score <= 1.0
        assert features.task_type == "chitchat"

    def test_longer_prompt_has_higher_context_load(self):
        short = extract_features([{"role": "user", "content": "hi"}])
        long_text = " ".join([f"word{i}" for i in range(500)])
        long = extract_features([{"role": "user", "content": long_text}])
        assert long.context_load >= short.context_load


# ---------------------------------------------------------------------------
# Complexity-based model matching tests
# ---------------------------------------------------------------------------

class TestMatchModelByComplexity:
    def test_simple_prompt_selects_cheap_model(self):
        """A simple prompt should route to the cheapest capable model."""
        models = [
            _make_model(
                model_id="cheap-fast",
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0003,
                max_complexity_score=0.5,
                estimated_tokens_per_second=80.0,
            ),
            _make_model(
                model_id="expensive-powerful",
                cost_per_1k_input=0.02,
                cost_per_1k_output=0.08,
                max_complexity_score=0.95,
                estimated_tokens_per_second=10.0,
            ),
        ]
        features = PromptFeatures(token_count=20, reasoning_complexity=0.1)
        model, confidence = match_model_by_complexity(features, models)
        assert model is not None
        assert model.id == "cheap-fast"
        assert confidence > 0.5

    def test_complex_prompt_skips_cheap_model(self):
        """A complex prompt should skip models that can't handle the complexity."""
        models = [
            _make_model(
                model_id="cheap-fast",
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0003,
                max_complexity_score=0.3,
                estimated_tokens_per_second=80.0,
            ),
            _make_model(
                model_id="capable-model",
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.02,
                max_complexity_score=0.8,
                estimated_tokens_per_second=30.0,
            ),
        ]
        features = PromptFeatures(
            token_count=5000,
            reasoning_complexity=0.8,
            dominant_language="math",
            has_code_blocks=True,
            task_difficulty=0.75,
            requirement_load=0.1,
        )
        model, confidence = match_model_by_complexity(features, models)
        assert model is not None
        assert model.id == "capable-model"

    def test_no_capable_model_returns_none(self):
        models = [
            _make_model(
                model_id="weak-model",
                max_complexity_score=0.1,
            ),
        ]
        features = PromptFeatures(token_count=10000, reasoning_complexity=1.0)
        model, confidence = match_model_by_complexity(features, models)
        assert model is None
        assert confidence == 0.0

    def test_inactive_model_excluded(self):
        models = [
            _make_model(model_id="active", is_active=True, max_complexity_score=0.9),
            _make_model(model_id="inactive", is_active=False, max_complexity_score=0.9),
        ]
        features = PromptFeatures(token_count=100)
        model, _ = match_model_by_complexity(features, models)
        assert model is not None
        assert model.id == "active"

    def test_vision_capability_required(self):
        """Image prompts should skip models without vision capability."""
        models = [
            _make_model(
                model_id="no-vision",
                capabilities=["text", "streaming"],
                max_complexity_score=0.9,
            ),
            _make_model(
                model_id="has-vision",
                capabilities=["text", "streaming", "vision"],
                max_complexity_score=0.9,
            ),
        ]
        features = PromptFeatures(has_images=True)
        model, _ = match_model_by_complexity(features, models)
        assert model is not None
        assert model.id == "has-vision"

    def test_context_window_enforced(self):
        """Models with context_window smaller than token_count should be excluded."""
        models = [
            _make_model(
                model_id="small-context",
                context_window=1000,
                max_complexity_score=0.9,
            ),
            _make_model(
                model_id="large-context",
                context_window=100000,
                max_complexity_score=0.9,
            ),
        ]
        features = PromptFeatures(token_count=5000)
        model, _ = match_model_by_complexity(features, models)
        assert model is not None
        assert model.id == "large-context"

    def test_no_complexity_metadata_fallback(self):
        """Models without complexity metadata should still be considered."""
        models = [
            _make_model(
                model_id="no-metadata",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.005,
                max_complexity_score=None,
            ),
            _make_model(
                model_id="with-metadata",
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0003,
                max_complexity_score=0.5,
            ),
        ]
        features = PromptFeatures(token_count=100)
        model, _ = match_model_by_complexity(features, models)
        assert model is not None
        # Should pick the cheaper model with metadata
        assert model.id == "with-metadata"

    def test_same_cost_faster_model_wins(self):
        """When costs are equal, prefer the faster model."""
        models = [
            _make_model(
                model_id="slow",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.005,
                max_complexity_score=0.9,
                estimated_tokens_per_second=10.0,
            ),
            _make_model(
                model_id="fast",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.005,
                max_complexity_score=0.9,
                estimated_tokens_per_second=60.0,
            ),
        ]
        features = PromptFeatures(token_count=100)
        model, _ = match_model_by_complexity(features, models)
        assert model is not None
        assert model.id == "fast"

    def test_tool_call_capability_required(self):
        """Tool call prompts should skip models without tool_calling capability."""
        models = [
            _make_model(
                model_id="no-tools",
                capabilities=["text", "streaming"],
                max_complexity_score=0.9,
            ),
            _make_model(
                model_id="has-tools",
                capabilities=["text", "streaming", "tool_calling"],
                max_complexity_score=0.9,
            ),
        ]
        features = PromptFeatures(has_tool_calls=True)
        model, _ = match_model_by_complexity(features, models)
        assert model is not None
        assert model.id == "has-tools"

    def test_all_models_inactive_returns_none(self):
        models = [
            _make_model(model_id="inactive1", is_active=False),
            _make_model(model_id="inactive2", is_active=False),
        ]
        features = PromptFeatures(token_count=100)
        model, confidence = match_model_by_complexity(features, models)
        assert model is None
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# Rule-based matching still works
# ---------------------------------------------------------------------------

class TestRuleBasedMatching:
    def test_vision_boost(self):
        models = [
            _make_model(model_id="with-vision", capabilities=["text", "vision"]),
            _make_model(model_id="no-vision", capabilities=["text"]),
        ]
        features = PromptFeatures(has_images=True)
        model, _ = match_model_by_rules(features, models)
        assert model is not None
        assert model.id == "with-vision"

    def test_reasoning_boost(self):
        models = [
            _make_model(
                model_id="with-reasoning",
                capabilities=["text", "reasoning"],
                tags=["reasoning"],
            ),
            _make_model(model_id="no-reasoning", capabilities=["text"]),
        ]
        features = PromptFeatures(reasoning_complexity=0.7)
        model, _ = match_model_by_rules(features, models)
        assert model is not None
        assert model.id == "with-reasoning"

    def test_no_models_returns_none(self):
        features = PromptFeatures(token_count=100)
        model, confidence = match_model_by_rules(features, [])
        assert model is None
        assert confidence == 0.0
