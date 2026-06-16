"""Phase 3: embedding-based task difficulty via k-NN over labeled exemplars.

Gated by settings.embedding_routing_enabled (default False). When disabled, no
model is loaded and heuristics from prompt_complexity are used unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SCORER: EmbeddingComplexityScorer | None = None
_INIT_FAILED = False


def _default_exemplars_path() -> str:
    return str(
        Path(__file__).resolve().parent.parent.parent.parent / "ml" / "complexity_exemplars.json"
    )


def _text_cache_key(text: str) -> str:
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class _EmbeddingLRUCache:
    """Thread-safe in-memory cache for text -> difficulty scores."""

    def __init__(self, max_size: int) -> None:
        self._max_size = max(1, max_size)
        self._data: OrderedDict[str, float] = OrderedDict()

    def get(self, key: str) -> float | None:
        with _LOCK:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: float) -> None:
        with _LOCK:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)


class EmbeddingComplexityScorer:
    """k-NN difficulty scorer using sentence-transformer embeddings."""

    def __init__(self) -> None:
        self._model: Any = None
        self._exemplar_embeddings: np.ndarray | None = None
        self._exemplar_difficulties: np.ndarray | None = None
        self._cache = _EmbeddingLRUCache(settings.embedding_cache_size)
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def exemplar_count(self) -> int:
        if self._exemplar_difficulties is None:
            return 0
        return int(len(self._exemplar_difficulties))

    def initialize(self) -> bool:
        if self._loaded:
            return True
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error(
                "embedding_routing_enabled is True but sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
            return False

        exemplars = self._load_exemplars()
        if not exemplars:
            logger.error("No complexity exemplars found — embedding routing disabled")
            return False

        try:
            logger.info("Loading embedding model %s ...", settings.embedding_model_name)
            self._model = SentenceTransformer(settings.embedding_model_name)
            texts = [e["text"] for e in exemplars]
            difficulties = np.array([float(e["difficulty"]) for e in exemplars], dtype=np.float64)
            self._exemplar_embeddings = np.array(self._model.encode(texts, show_progress_bar=False))
            self._exemplar_difficulties = difficulties
            self._loaded = True
            logger.info(
                "Embedding complexity scorer ready (%d exemplars)",
                len(exemplars),
            )
            return True
        except Exception:
            logger.exception("Failed to initialize embedding complexity scorer")
            return False

    def _load_exemplars(self) -> list[dict[str, Any]]:
        path = Path(settings.embedding_exemplars_path or _default_exemplars_path())
        if not path.exists():
            logger.error("Exemplar file not found: %s", path)
            return []
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("exemplars", data if isinstance(data, list) else [])
        exemplars: list[dict[str, Any]] = []
        for item in raw:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            exemplars.append({
                "text": text,
                "difficulty": float(item.get("difficulty", 0.5)),
            })
        return exemplars

    def score(self, text: str) -> float | None:
        """Return embedding-based difficulty in [0, 1], or None if unavailable."""
        if not text.strip():
            return None
        if not self._loaded:
            if not self.initialize():
                return None

        cache_key = _text_cache_key(text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        assert self._model is not None
        assert self._exemplar_embeddings is not None
        assert self._exemplar_difficulties is not None

        try:
            query = np.array(self._model.encode([text], show_progress_bar=False))
            sims = cosine_similarity(query, self._exemplar_embeddings)[0]
            k = min(settings.embedding_k_neighbors, len(sims))
            top_idx = np.argsort(sims)[-k:][::-1]
            weights = np.maximum(sims[top_idx], 0.0)
            if weights.sum() <= 0:
                difficulty = float(np.mean(self._exemplar_difficulties[top_idx]))
            else:
                difficulty = float(np.average(self._exemplar_difficulties[top_idx], weights=weights))
            difficulty = min(max(round(difficulty, 3), 0.0), 1.0)
            self._cache.set(cache_key, difficulty)
            return difficulty
        except Exception:
            logger.exception("Embedding difficulty scoring failed")
            return None


def blend_task_difficulty(heuristic: float, embedding: float, blend_weight: float) -> float:
    """Blend heuristic and embedding difficulties. blend_weight is embedding share."""
    w = min(max(blend_weight, 0.0), 1.0)
    blended = (1.0 - w) * heuristic + w * embedding
    return min(round(blended, 3), 1.0)


def get_embedding_scorer() -> EmbeddingComplexityScorer | None:
    """Return a shared scorer instance when embedding routing is enabled."""
    global _SCORER, _INIT_FAILED
    if not settings.embedding_routing_enabled:
        return None
    if _INIT_FAILED:
        return None
    with _LOCK:
        if _SCORER is None:
            _SCORER = EmbeddingComplexityScorer()
        return _SCORER


def embedding_difficulty_for_text(text: str) -> float | None:
    """Score text when embedding routing is enabled; otherwise None."""
    if not settings.embedding_routing_enabled:
        return None
    scorer = get_embedding_scorer()
    if scorer is None:
        return None
    result = scorer.score(text)
    return result


def warm_up_embedding_scorer() -> bool:
    """Eager-load model and exemplars (call from app startup when enabled)."""
    if not settings.embedding_routing_enabled:
        return False
    scorer = get_embedding_scorer()
    if scorer is None:
        return False
    return scorer.initialize()


def embedding_status() -> dict[str, Any]:
    """Status snapshot for monitoring endpoints."""
    scorer = get_embedding_scorer() if settings.embedding_routing_enabled else None
    return {
        "embedding_routing_enabled": settings.embedding_routing_enabled,
        "embedding_model_name": settings.embedding_model_name,
        "embedding_blend_weight": settings.embedding_blend_weight,
        "embedding_model_loaded": bool(scorer and scorer.is_loaded),
        "embedding_exemplar_count": scorer.exemplar_count if scorer else 0,
    }
