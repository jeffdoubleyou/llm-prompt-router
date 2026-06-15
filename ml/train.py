#!/usr/bin/env python3
"""
Training script for the ML classifier.
Loads training data from the database, extracts features, trains a
HistGradientBoostingClassifier, and saves the model with joblib.

Usage:
    python -m ml.train [--db-url <postgresql://...>]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("train")

try:
    from app.core.config import settings
    DB_URL = settings.database_url_sync
except ImportError:
    DB_URL = os.environ.get("DATABASE_URL", "postgresql://router:router@db:5432/router")

from ml.feature_extraction import (
    extract_features_dict,
    feature_names,
    features_to_vector,
)
from ml.schema import LANGUAGE_MAP, TrainingMetrics


def _normalize_sync_db_url(db_url: str) -> str:
    """Convert async or driverless PostgreSQL URLs for sync SQLAlchemy."""
    if "+asyncpg" in db_url:
        return db_url.replace("+asyncpg", "+psycopg2", 1)
    if db_url.startswith("postgresql://"):
        return db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    if db_url.startswith("postgres://"):
        return db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    return db_url


def load_training_data(db_url: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from sqlalchemy import create_engine, text

    engine = create_engine(_normalize_sync_db_url(db_url))
    query = text("""
        SELECT cs.prompt_text, cs.selected_model, cs.features
        FROM classifier_samples cs
        WHERE cs.selected_model IS NOT NULL
        ORDER BY cs.created_at DESC
    """)
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    if not rows:
        logger.warning("No training data found in database")
        return np.array([]), np.array([]), []

    X_list = []
    y_list = []
    model_labels: list[str] = []

    for row in rows:
        prompt_text = row[0]
        selected_model = row[1]
        features_json = row[2] or {}

        if isinstance(features_json, str):
            import json
            try:
                features_json = json.loads(features_json)
            except json.JSONDecodeError:
                continue

        if not features_json:
            features_json = {"prompt_text": prompt_text}

        pf = extract_features_dict(features_json)
        vec = features_to_vector(pf)
        X_list.append(vec)
        y_list.append(selected_model)
        if selected_model not in model_labels:
            model_labels.append(selected_model)

    if not X_list:
        logger.warning("No valid training examples after processing")
        return np.array([]), np.array([]), []

    X = np.array(X_list)
    y = np.array(y_list)
    logger.info("Loaded %d training examples with %d classes", len(X), len(model_labels))
    return X, y, model_labels


def train_classifier(X: np.ndarray, y: np.ndarray) -> tuple[HistGradientBoostingClassifier, dict]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    logger.info("Training split: %d train, %d test", len(X_train), len(X_test))

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=8,
        learning_rate=0.08,
        min_samples_leaf=5,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        categorical_features=None,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    classes = clf.classes_.tolist() if hasattr(clf, "classes_") else []

    importances = clf.feature_importances_.tolist() if hasattr(clf, "feature_importances_") else []
    feature_importance = dict(zip(feature_names(), importances))

    metrics = TrainingMetrics(
        accuracy=accuracy,
        precision={cls: float(p) for cls, p in zip(classes, precision)},
        recall={cls: float(r) for cls, r in zip(classes, recall)},
        f1_score={cls: float(f) for cls, f in zip(classes, f1)},
        training_samples=len(X),
        feature_importance=feature_importance,
        model_version="v1.0",
        trained_at=datetime.utcnow().isoformat(),
    )
    logger.info(
        "Training complete — accuracy: %.4f, samples: %d, classes: %d",
        accuracy,
        len(X),
        len(classes),
    )
    return clf, metrics


def save_model(
    clf: HistGradientBoostingClassifier,
    metrics: TrainingMetrics,
    model_path: str | None = None,
    features_path: str | None = None,
) -> None:
    model_path = model_path or settings.classifier_model_path
    features_path = features_path or settings.classifier_features_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model_data = {
        "model": clf,
        "classes": clf.classes_.tolist() if hasattr(clf, "classes_") else [],
        "feature_names": feature_names(),
        "metrics": {
            "accuracy": metrics.accuracy,
            "training_samples": metrics.training_samples,
            "trained_at": metrics.trained_at,
            "feature_importance": metrics.feature_importance,
        },
    }
    joblib.dump(model_data, model_path)
    logger.info("Model saved to %s", model_path)

    joblib.dump(feature_names(), features_path)
    logger.info("Feature names saved to %s", features_path)


def print_metrics(metrics: TrainingMetrics) -> None:
    print("\n" + "=" * 60)
    print("CLASSIFIER TRAINING REPORT")
    print("=" * 60)
    print(f"  Model version:     {metrics.model_version}")
    print(f"  Training samples:  {metrics.training_samples}")
    print(f"  Accuracy:          {metrics.accuracy:.4f}")
    print(f"  Trained at:        {metrics.trained_at}")
    print("-" * 60)
    print("  Per-class metrics:")
    for cls in sorted(set(list(metrics.precision.keys()) + list(metrics.recall.keys()))):
        p = metrics.precision.get(cls, 0)
        r = metrics.recall.get(cls, 0)
        f = metrics.f1_score.get(cls, 0)
        print(f"    {cls:30s}  p={p:.3f}  r={r:.3f}  f1={f:.3f}")
    print("-" * 60)
    print("  Top feature importances:")
    sorted_importances = sorted(
        metrics.feature_importance.items(), key=lambda x: x[1], reverse=True
    )
    for name, importance in sorted_importances[:5]:
        print(f"    {name:25s}  {importance:.4f}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ML classifier")
    parser.add_argument("--db-url", default=DB_URL, help="Database URL")
    parser.add_argument("--model-path", default=None, help="Output path for model.joblib")
    parser.add_argument("--features-path", default=None, help="Output path for features.joblib")
    args = parser.parse_args()

    logger.info("Loading training data from database...")
    X, y, _ = load_training_data(args.db_url)

    if len(X) == 0:
        logger.error("No training data available. Exiting.")
        sys.exit(1)

    logger.info("Training classifier...")
    clf, metrics = train_classifier(X, y)

    save_model(
        clf,
        metrics,
        model_path=args.model_path,
        features_path=args.features_path,
    )

    print_metrics(metrics)

    print(f"\nModel saved. To use in the router, restart the backend.")
    print(f"  model_path:    {args.model_path or settings.classifier_model_path}")
    print(f"  features_path: {args.features_path or settings.classifier_features_path}")


if __name__ == "__main__":
    main()
