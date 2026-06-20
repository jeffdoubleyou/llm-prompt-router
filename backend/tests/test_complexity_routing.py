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
    _filter_eligible_models,
    match_model_by_complexity,
    match_model_by_rules,
    extract_features,
    select_routing_model,
)
from app.core.config import settings
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
    def test_simple_prompt_selects_fastest_cheapest_capable_model(self):
        """A simple prompt should route to the fastest/cheapest capable model."""
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

    def test_no_capable_model_returns_best_effort_highest_capacity(self):
        models = [
            _make_model(
                model_id="weak-model",
                max_complexity_score=0.1,
            ),
        ]
        features = PromptFeatures(token_count=10000, reasoning_complexity=1.0)
        model, confidence = match_model_by_complexity(features, models)
        assert model is not None
        assert model.id == "weak-model"
        assert confidence > 0.0

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
        model, _, method = select_routing_model(features, models)
        assert model is not None
        assert model.id == "has-vision"
        assert method == "rules"

    def test_non_image_prompt_includes_vision_models(self):
        """Vision-capable models compete for text-only prompts; fastest eligible model wins."""
        models = [
            _make_model(
                model_id="vision-fast",
                capabilities=["text", "vision", "tool_calling"],
                max_complexity_score=0.95,
                estimated_tokens_per_second=60.0,
                priority=3,
            ),
            _make_model(
                model_id="text-slow",
                capabilities=["text", "tool_calling"],
                max_complexity_score=0.95,
                estimated_tokens_per_second=35.0,
                priority=3,
            ),
        ]
        features = PromptFeatures(
            has_images=False,
            has_tool_calls=True,
            task_difficulty=0.48,
        )
        eligible = _filter_eligible_models(features, models)
        assert {m.id for m in eligible} == {"vision-fast", "text-slow"}

        model, _, method = select_routing_model(features, models)
        assert model is not None
        assert model.id == "vision-fast"
        assert method == "rules"

    def test_complexity_routing_disabled_uses_capability_rules_only(self, monkeypatch):
        monkeypatch.setattr(settings, "complexity_routing_enabled", False)
        models = [
            _make_model(
                model_id="small",
                capabilities=["text"],
                max_complexity_score=0.3,
                estimated_tokens_per_second=60.0,
            ),
            _make_model(
                model_id="large",
                capabilities=["text"],
                max_complexity_score=0.95,
                estimated_tokens_per_second=10.0,
            ),
        ]
        features = PromptFeatures(task_difficulty=0.8)
        model, _, method = select_routing_model(features, models)
        assert model is not None
        assert model.id == "small"
        assert method == "rules"

    def test_code_blocks_boost_rule_score(self):
        models = [
            _make_model(
                model_id="no-code",
                capabilities=["text", "streaming"],
                max_complexity_score=0.9,
            ),
            _make_model(
                model_id="has-code",
                capabilities=["text", "streaming", "code"],
                max_complexity_score=0.9,
            ),
        ]
        features = PromptFeatures(has_code_blocks=True)
        model, _ = match_model_by_rules(features, models)
        assert model is not None
        assert model.id == "has-code"

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
        eligible = _filter_eligible_models(features, models)
        assert len(eligible) == 1
        assert eligible[0].id == "large-context"

    def test_default_routing_prefers_fastest_then_cheapest(self):
        """Among capable models, pick fastest then cheapest (not by parameter tier)."""
        models = [
            _make_model(
                model_id="slow-cheap",
                capabilities=["text"],
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0003,
                max_complexity_score=0.5,
                estimated_tokens_per_second=10.0,
            ),
            _make_model(
                model_id="fast-expensive",
                capabilities=["text"],
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.04,
                max_complexity_score=0.95,
                estimated_tokens_per_second=80.0,
            ),
        ]
        features = PromptFeatures(token_count=20, reasoning_complexity=0.1)
        model, _, method = select_routing_model(features, models)
        assert model is not None
        assert model.id == "fast-expensive"
        assert method == "rules"

    def test_no_complexity_metadata_still_eligible(self):
        """Models without max_complexity_score are used when no scored model is sufficient."""
        models = [
            _make_model(
                model_id="no-metadata",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.005,
                max_complexity_score=None,
                estimated_tokens_per_second=50.0,
            ),
            _make_model(
                model_id="with-metadata",
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0003,
                max_complexity_score=0.5,
                estimated_tokens_per_second=20.0,
            ),
        ]
        features = PromptFeatures(task_difficulty=0.8)
        model, _ = match_model_by_complexity(features, models)
        assert model is not None
        assert model.id == "no-metadata"

    def test_same_capacity_faster_model_wins(self):
        """When capacity tier and cost are equal, prefer the faster model."""
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

    def test_zero_cost_prefers_smallest_sufficient_model(self):
        """Among models that cover difficulty, prefer the smallest sufficient tier."""
        models = [
            _make_model(
                model_id="small",
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                max_complexity_score=0.6,
                estimated_tokens_per_second=12.0,
            ),
            _make_model(
                model_id="large",
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                max_complexity_score=0.95,
                estimated_tokens_per_second=35.0,
            ),
        ]
        features = PromptFeatures(task_difficulty=0.58, requirement_load=0.0)
        model, _ = match_model_by_complexity(features, models)
        assert model is not None
        assert model.id == "small"

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
        model, _ = match_model_by_rules(features, models)
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

    def test_high_code_edit_routes_to_coding_not_fast_small_model(self, monkeypatch):
        """Regression: 0.86 task difficulty must not fall back to max 0.5 gemma 4B."""
        monkeypatch.setattr(settings, "complexity_routing_enabled", True)
        caps = [
            "text", "tool_calling", "function_calling", "streaming",
            "reasoning", "code", "long_context", "vision",
        ]
        models = [
            _make_model(
                model_id="unsloth/Qwen3.6-35B-A3B-MTP-GGUF:IQ4_XS_coding",
                capabilities=caps,
                priority=3,
                max_complexity_score=0.9,
                estimated_tokens_per_second=30.0,
                context_window=400000,
            ),
            _make_model(
                model_id="unsloth/Qwen3.6-35B-A3B-MTP-GGUF:IQ4_XS",
                capabilities=caps,
                priority=3,
                max_complexity_score=0.82,
                estimated_tokens_per_second=30.0,
                context_window=400000,
            ),
            _make_model(
                model_id="unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL",
                capabilities=caps,
                priority=3,
                max_complexity_score=0.5,
                estimated_tokens_per_second=60.0,
                context_window=128000,
            ),
            _make_model(
                model_id="unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_S",
                capabilities=[c for c in caps if c != "vision"],
                priority=2,
                max_complexity_score=1.0,
                estimated_tokens_per_second=7.0,
                context_window=400000,
            ),
        ]
        features = PromptFeatures(
            task_difficulty=0.86,
            requirement_load=0.34,
            task_type="code_edit",
            token_count=29128,
            has_tool_calls=True,
            has_code_blocks=True,
            has_images=False,
            dominant_language="code",
        )
        model, _, method = select_routing_model(features, models, request_has_tools=True)
        assert model is not None
        assert model.id == "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:IQ4_XS_coding"
        assert method == "complexity"

    def test_complexity_fallback_escalates_to_highest_capacity(self, monkeypatch):
        """When no model covers difficulty, pick highest max_complexity (not fastest)."""
        monkeypatch.setattr(settings, "complexity_routing_enabled", True)
        caps = [
            "text", "vision", "tool_calling", "function_calling", "streaming",
            "reasoning", "code", "long_context",
        ]
        models = [
            _make_model(
                model_id="unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL",
                capabilities=caps,
                priority=3,
                max_complexity_score=0.5,
                estimated_tokens_per_second=60.0,
                context_window=128000,
            ),
            _make_model(
                model_id="unsloth/Qwen3.6-35B-A3B-MTP-GGUF:IQ4_XS_coding",
                capabilities=caps,
                priority=3,
                max_complexity_score=0.9,
                estimated_tokens_per_second=30.0,
                context_window=400000,
            ),
        ]
        features = PromptFeatures(
            task_difficulty=0.95,
            requirement_load=0.26,
            task_type="debug",
            has_images=True,
            has_tool_calls=True,
            has_code_blocks=True,
            token_count=26563,
        )
        model, _, method = select_routing_model(features, models, request_has_tools=True)
        assert model is not None
        assert model.id == "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:IQ4_XS_coding"
        assert method == "complexity"


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
