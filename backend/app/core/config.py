from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "LLM Prompt Router"
    debug: bool = False
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://router:router@db:5432/router"
    database_url_sync: str = "postgresql://router:router@db:5432/router"
    redis_url: str = "redis://redis:6379/0"

    encryption_key: str = ""

    cors_origins: list[str] = ["*"]

    classifier_model_path: str = str(
        Path(__file__).resolve().parent.parent.parent.parent / "ml" / "model.joblib"
    )
    classifier_min_confidence: float = 0.6
    classifier_features_path: str = str(
        Path(__file__).resolve().parent.parent.parent.parent / "ml" / "features.joblib"
    )

    default_model: str = "gpt-4o-mini"

    queue_poll_interval: float = 1.0
    worker_concurrency: int = 4

    upstream_timeout: float = 120.0
    max_retries: int = 3

    stats_window_minutes: int = 60

    prompt_debug_max_stored: int = 100
    prompt_debug_ttl_seconds: int = 86400

    # Phase 3: embedding-based task difficulty (off by default)
    embedding_routing_enabled: bool = False
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_blend_weight: float = 0.55
    embedding_k_neighbors: int = 5
    embedding_exemplars_path: str = ""
    embedding_cache_size: int = 512

    # Serialize upstream requests per base URL (e.g. llama.cpp single-model servers)
    upstream_queue_enabled: bool = False

    # When False, rank eligible models by speed then cost only (no max_complexity_score filter)
    complexity_routing_enabled: bool = False


settings = Settings()
