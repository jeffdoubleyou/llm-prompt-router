"""Tests for content-aware prompt complexity (Phase 1 & 2)."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock

sys.modules["asyncpg"] = MagicMock()

from app.services.prompt_complexity import (
    TASK_DEBUG,
    TASK_CHITCHAT,
    TASK_SUMMARIZATION,
    analyze_prompt_complexity,
    classify_task_type,
    compute_context_load,
    compute_reasoning_complexity,
    get_routing_difficulty,
)
from app.services.router_service import extract_features


class TestContextLoad:
    def test_empty_is_zero(self):
        assert compute_context_load(0) == 0.0

    def test_short_prompt_low(self):
        assert compute_context_load(3) < 0.15

    def test_long_prompt_higher_but_capped(self):
        short = compute_context_load(200)
        long = compute_context_load(8000)
        huge = compute_context_load(100_000)
        assert long > short
        assert huge <= 1.0


class TestTaskClassification:
    def test_chitchat(self):
        assert classify_task_type("Hello!", "", False) == TASK_CHITCHAT

    def test_debug(self):
        task = classify_task_type(
            "Find the race condition in this handler and explain root cause",
            "",
            True,
        )
        assert task == TASK_DEBUG

    def test_summarization(self):
        assert classify_task_type("Summarize this document in bullet points", "", False) == TASK_SUMMARIZATION

    def test_planning(self):
        task = classify_task_type(
            "Design a migration plan for our auth system with trade-offs",
            "",
            False,
        )
        assert task in ("planning", "multi_step_research")


class TestReasoningComplexity:
    def test_no_false_positive_anyway(self):
        """'why' in 'anyway' should not inflate score (word boundaries)."""
        low = compute_reasoning_complexity("Anyway this is fine")
        high = compute_reasoning_complexity("Explain why this algorithm fails step by step")
        assert high > low

    def test_capped(self):
        text = "explain why analyze compare prove derive " * 5
        assert compute_reasoning_complexity(text) <= 1.0


class TestTaskDifficultyVsSize:
    def test_short_debug_harder_than_long_summary(self):
        debug = extract_features([{
            "role": "user",
            "content": "Debug this race condition in the async handler",
        }])
        summary = extract_features([{
            "role": "user",
            "content": " ".join(["word"] * 2000) + " Summarize this policy document",
        }])
        assert debug.task_type == TASK_DEBUG
        assert summary.task_type == TASK_SUMMARIZATION
        assert debug.task_difficulty > summary.task_difficulty

    def test_long_text_raises_context_not_task_difficulty_as_much(self):
        short = extract_features([{"role": "user", "content": "Hi"}])
        long = extract_features([{"role": "user", "content": " ".join(["word"] * 3000)}])
        assert long.context_load > short.context_load
        # Task difficulty should not jump proportionally to length alone
        assert (long.task_difficulty - short.task_difficulty) < (long.context_load - short.context_load)


class TestRequirementLoad:
    def test_constraints_increase_requirement_load(self):
        plain = extract_features([{"role": "user", "content": "Say hello"}])
        strict = extract_features([{
            "role": "user",
            "content": (
                "Implement OAuth2 refresh rotation. Must preserve public API. "
                "Output valid JSON schema. Include unit tests."
            ),
        }])
        assert strict.requirement_load > plain.requirement_load
        assert strict.task_difficulty > plain.task_difficulty


class TestRoutingDifficulty:
    def test_uses_task_difficulty_when_set(self):
        d = get_routing_difficulty(0.7, 0.4, legacy_complexity_score=0.2)
        assert d > 0.7
        assert d == 0.76  # 0.7 + 0.4 * 0.15

    def test_falls_back_to_legacy(self):
        assert get_routing_difficulty(0.0, 0.0, legacy_complexity_score=0.55) == 0.55


class TestExtractFeaturesFields:
    def test_new_fields_present(self):
        features = extract_features([{"role": "user", "content": "Hello"}])
        assert hasattr(features, "context_load")
        assert hasattr(features, "task_difficulty")
        assert hasattr(features, "requirement_load")
        assert hasattr(features, "task_type")
        assert features.task_type == TASK_CHITCHAT

    def test_analyze_directly(self):
        analysis = analyze_prompt_complexity(
            [{"role": "user", "content": "Fix the bug in this function"}],
            token_count=50,
            dominant_language="code",
            has_code_blocks=True,
            has_images=False,
            has_tool_calls=False,
            full_text="def foo(): pass",
        )
        assert analysis.task_type == TASK_DEBUG
        assert 0.0 < analysis.task_difficulty <= 1.0
