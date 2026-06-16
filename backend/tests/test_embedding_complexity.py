"""Tests for Phase 3 embedding-based complexity routing."""

from __future__ import annotations

import json
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sys.modules["asyncpg"] = MagicMock()

from app.core.config import settings
from app.services.embedding_complexity import (
    EmbeddingComplexityScorer,
    blend_task_difficulty,
    embedding_difficulty_for_text,
    embedding_status,
)
from app.services.prompt_complexity import analyze_prompt_complexity


class TestBlendTaskDifficulty:
    def test_all_heuristic(self):
        assert blend_task_difficulty(0.4, 0.9, 0.0) == 0.4

    def test_all_embedding(self):
        assert blend_task_difficulty(0.4, 0.9, 1.0) == 0.9

    def test_weighted_blend(self):
        assert blend_task_difficulty(0.4, 0.8, 0.5) == 0.6


class TestEmbeddingDisabledByDefault:
    def test_disabled_returns_none(self):
        with patch.object(settings, "embedding_routing_enabled", False):
            assert embedding_difficulty_for_text("debug this bug") is None

    def test_status_when_disabled(self):
        with patch.object(settings, "embedding_routing_enabled", False):
            status = embedding_status()
            assert status["embedding_routing_enabled"] is False
            assert status["embedding_model_loaded"] is False


class TestEmbeddingScorerKNN:
    def test_knn_picks_harder_neighbor(self, tmp_path):
        exemplars = {
            "exemplars": [
                {"text": "hello", "difficulty": 0.1},
                {"text": "fix race condition in async code", "difficulty": 0.85},
            ]
        }
        path = tmp_path / "exemplars.json"
        path.write_text(json.dumps(exemplars))

        scorer = EmbeddingComplexityScorer()
        easy = np.array([[1.0, 0.0]])
        hard = np.array([[0.0, 1.0]])

        def fake_encode(texts, show_progress_bar=False):
            if texts[0] == "hello":
                return easy
            if texts[0] == "fix race condition in async code":
                return hard
            # query closer to hard
            return np.array([[0.1, 0.9]])

        scorer._model = MagicMock()
        scorer._model.encode = fake_encode
        scorer._exemplar_embeddings = np.vstack([easy, hard])
        scorer._exemplar_difficulties = np.array([0.1, 0.85])
        scorer._loaded = True

        with patch.object(settings, "embedding_exemplars_path", str(path)):
            score = scorer.score("find the bug in the async handler")
        assert score is not None
        assert score > 0.6


class TestAnalyzeWithEmbeddings:
    def test_embedding_blends_into_task_difficulty(self):
        with patch.object(settings, "embedding_routing_enabled", True), patch(
            "app.services.embedding_complexity.embedding_difficulty_for_text",
            return_value=0.85,
        ), patch.object(settings, "embedding_blend_weight", 0.5):
            analysis = analyze_prompt_complexity(
                [{"role": "user", "content": "Hello!"}],
                token_count=10,
                dominant_language="natural_language",
                has_code_blocks=False,
                has_images=False,
                has_tool_calls=False,
                full_text="Hello!",
            )
            assert analysis.embedding_routing_applied is True
            assert analysis.embedding_difficulty == 0.85
            assert analysis.heuristic_task_difficulty < analysis.task_difficulty
            assert analysis.task_difficulty == blend_task_difficulty(
                analysis.heuristic_task_difficulty, 0.85, 0.5
            )

    def test_embedding_failure_uses_heuristic_only(self):
        with patch.object(settings, "embedding_routing_enabled", True), patch(
            "app.services.embedding_complexity.embedding_difficulty_for_text",
            return_value=None,
        ):
            analysis = analyze_prompt_complexity(
                [{"role": "user", "content": "Hello!"}],
                token_count=10,
                dominant_language="natural_language",
                has_code_blocks=False,
                has_images=False,
                has_tool_calls=False,
                full_text="Hello!",
            )
            assert analysis.embedding_routing_applied is False
            assert analysis.embedding_difficulty is None
            assert analysis.task_difficulty == analysis.heuristic_task_difficulty
