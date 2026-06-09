from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from app.core.config import settings
from ml.feature_extraction import (
    extract_features_dict,
    feature_names,
    features_to_vector,
)
from ml.schema import PredictionResult, PromptFeatures

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = settings.classifier_min_confidence


class LLMClassifier:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or settings.classifier_model_path
        self.features_path = settings.classifier_features_path
        self._model: HistGradientBoostingClassifier | None = None
        self._feature_names: list[str] | None = None
        self._model_classes: list[str] | None = None
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                if isinstance(data, dict):
                    self._model = data.get("model")
                    self._model_classes = data.get("classes")
                    self._feature_names = data.get("feature_names")
                else:
                    self._model = data
                    self._model_classes = (
                        self._model.classes_.tolist() if hasattr(self._model, "classes_") else None
                    )
                    self._feature_names = feature_names()
                logger.info(
                    "Loaded classifier model from %s (classes: %s)",
                    self.model_path,
                    self._model_classes,
                )
            except Exception:
                logger.exception("Failed to load model from %s", self.model_path)
                self._model = None
        else:
            logger.warning("No model found at %s — will create on first train", self.model_path)

    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, features: dict[str, Any] | PromptFeatures) -> PredictionResult:
        if self._model is None:
            return PredictionResult(model_id="unknown", confidence=0.0)

        if isinstance(features, dict):
            pf = extract_features_dict(features)
        else:
            pf = features

        vector = features_to_vector(pf).reshape(1, -1)

        if self._feature_names is not None and len(self._feature_names) == vector.shape[1]:
            pass

        try:
            probas = self._model.predict_proba(vector)[0]
            pred_idx = int(np.argmax(probas))
            confidence = float(probas[pred_idx])

            if self._model_classes and pred_idx < len(self._model_classes):
                model_id = self._model_classes[pred_idx]
            else:
                model_id = str(pred_idx)

            all_scores = {}
            if self._model_classes:
                for i, cls in enumerate(self._model_classes):
                    if i < len(probas):
                        all_scores[cls] = float(probas[i])

            return PredictionResult(
                model_id=model_id,
                confidence=confidence,
                all_scores=all_scores,
            )
        except Exception:
            logger.exception("Prediction failed")
            return PredictionResult(model_id="unknown", confidence=0.0)

    def predict_with_threshold(
        self, features: dict[str, Any] | PromptFeatures
    ) -> PredictionResult:
        result = self.predict(features)
        if result.confidence >= MIN_CONFIDENCE:
            return result
        return PredictionResult(
            model_id=result.model_id,
            confidence=result.confidence,
            all_scores=result.all_scores,
        )

    @property
    def model(self) -> HistGradingBoostingClassifier | None:
        return self._model


def create_default_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_leaf=10,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
    )
