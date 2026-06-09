from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PromptFeatures:
    token_count: int = 0
    char_length: int = 0
    has_code_blocks: bool = False
    has_urls: bool = False
    has_images: bool = False
    has_tool_calls: bool = False
    dominant_language: str = "unknown"
    reasoning_complexity: float = 0.0
    hour_of_day: int = 0


@dataclass
class PredictionResult:
    model_id: str
    confidence: float
    all_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingExample:
    features: PromptFeatures
    selected_model: str
    is_correct: bool | None = None


@dataclass
class TrainingMetrics:
    accuracy: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1_score: dict[str, float]
    training_samples: int
    feature_importance: dict[str, float]
    model_version: str
    trained_at: str


LANGUAGE_MAP = {
    "code": 0,
    "natural_language": 1,
    "math": 2,
    "translation": 3,
    "unknown": 4,
}
