ML Classifier Training Guide

This document explains how the ML classifier training pipeline works in the LLM Router, including data collection, training, model deployment, and feature extraction.

---

## Overview

The ML classifier is a machine learning model that helps route prompts to the best-fit LLM by learning from historical routing decisions. It sits alongside the rule-based router and is trained on labeled data collected from past requests where the rule-based confidence was low.

**Key files:**

- `ml/train.py` — Training script
- `ml/classifier.py` — Model loading and prediction
- `ml/feature_extraction.py` — Feature extraction functions
- `ml/schema.py` — Data classes for features, predictions, and metrics
- `backend/app/workers/classifier_worker.py` — Async worker that collects training data
- `backend/app/services/router_service.py` — Routing logic that feeds the classifier

---

## 1. How Training Data Is Collected

### When Data Is Collected

The router uses a two-stage classification process:

1. **Rule-based routing** (`router_service.py:match_model_by_rules`): Assigns a model based on hand-crafted rules (vision capability, tool calling, context length, etc.) and calculates a confidence score.
2. **Threshold check**: If confidence is below `classifier_min_confidence` (default `0.6`), the request is enqueued for ML training.

### The Collection Pipeline

```
Request arrives
    ↓
Rule-based match → confidence = 0.45
    ↓
Confidence < 0.6 → enqueue to Redis queue
    ↓
Classifier worker dequeues, runs ML classifier
    ↓
Stores result in classifier_samples table
```

**Key configuration** (from `backend/app/core/config.py`):

| Setting | Default | Description |
|---|---|---|
| `classifier_min_confidence` | `0.6` | Minimum rule-based confidence before skipping ML |
| `worker_concurrency` | `4` | Number of concurrent classifier worker tasks |

### The Classifier Worker

The classifier worker (`classifier_worker.py`) is a set of async tasks that:

1. Dequeue items from Redis (key: `classifier_queue`)
2. Run the ML classifier on the prompt features
3. Store the result in the `classifier_samples` PostgreSQL table with:
   - `prompt_text` — The prompt features as JSON
   - `selected_model` — The model the router chose
   - `features` — Raw feature dict
   - `confidence` — The classifier's confidence score
   - `is_correct` — `NULL` initially (can be set later if user feedback is available)

### The Training Data Table

The `classifier_samples` table schema:

| Column | Type | Description |
|---|---|---|
| `id` | String(36) | UUID |
| `prompt_text` | Text | Serialized prompt features |
| `selected_model` | String(255) | The model chosen by the router |
| `features` | JSON | Feature dictionary |
| `confidence` | Float | Classifier confidence score |
| `is_correct` | Boolean (nullable) | Whether the choice was correct (for supervised feedback) |
| `created_at` | DateTime | When the sample was recorded |

---

## 2. How to Trigger Training

### Manual Training

Training is a standalone CLI command — there is no REST API endpoint to trigger it remotely:

```bash
cd /path/to/llm-prompt-router
python -m ml.train
```

With custom database URL:

```bash
python -m ml.train --db-url postgresql://user:pass@host:5432/router
```

With custom output paths:

```bash
python -m ml.train --model-path /custom/path/model.joblib --features-path /custom/path/features.joblib
```

### What Training Does

1. **Load data**: Queries `classifier_samples` for all rows where `selected_model IS NOT NULL`, ordered by `created_at DESC`.
2. **Extract features**: For each row, extracts a 9-element feature vector using `ml/feature_extraction.py:extract_features_dict()`.
3. **Split data**: 80/20 train/test split with stratification (`random_state=42`).
4. **Train model**: Trains a `HistGradientBoostingClassifier` with these hyperparameters:

   | Parameter | Value |
   |---|---|
   | `max_iter` | 300 |
   | `max_depth` | 8 |
   | `learning_rate` | 0.08 |
   | `min_samples_leaf` | 5 |
   | `l2_regularization` | 1.0 |
   | `early_stopping` | `True` |
   | `validation_fraction` | 0.15 |
   | `n_iter_no_change` | 20 |
   | `random_state` | 42 |

5. **Evaluate**: Computes accuracy, precision, recall, F1 per class, and feature importance.
6. **Save model**: Serializes the model and metadata with `joblib`.

### Training Output

After training, the script prints a report:

```
============================================================
CLASSIFIER TRAINING REPORT
============================================================
  Model version:     v1.0
  Training samples:  1500
  Accuracy:          0.8750
  Trained at:        2026-06-12T16:00:00
------------------------------------------------------------
  Per-class metrics:
    claude-opus-4-20250514     p=0.850  r=0.820  f1=0.835
    gemini-2.5-pro             p=0.910  r=0.890  f1=0.900
    gpt-4o                     p=0.880  r=0.860  f1=0.870
    gpt-4o-mini                p=0.920  r=0.930  f1=0.925
------------------------------------------------------------
  Top feature importances:
    token_count               0.2840
    reasoning_complexity      0.2100
    dominant_language         0.1850
    has_tool_calls            0.1200
    char_length               0.0950
============================================================

Model saved. To use in the router, restart the backend.
```

---

## 3. Where the Model Is Saved

### Default Model Path

The model is saved to `ml/model.joblib` relative to the project root, determined by `settings.classifier_model_path`:

```python
# In backend/app/core/config.py
classifier_model_path = str(
    Path(__file__).resolve().parent.parent.parent.parent / "ml" / "model.joblib"
)
# Resolves to: <project_root>/ml/model.joblib
```

The features file is saved alongside it:

```python
classifier_features_path = str(
    Path(__file__).resolve().parent.parent.parent.parent / "ml" / "features.joblib"
)
# Resolves to: <project_root>/ml/features.joblib
```

### Model File Contents

The `model.joblib` file contains a dictionary with:

| Key | Type | Description |
|---|---|---|
| `model` | `HistGradientBoostingClassifier` | The trained scikit-learn model |
| `classes` | `list[str]` | Available model IDs (e.g., `["gpt-4o-mini", "claude-opus-4-..."]`) |
| `feature_names` | `list[str]` | The 9 feature names |
| `metrics` | `dict` | Training metrics including accuracy, training_samples, trained_at, feature_importance |

---

## 4. How to Make the Router Use the New Model

### Reload Requires Restart

The ML classifier loads the model at import time (`ml/classifier.py:__init__`). To use a newly trained model:

1. **Train the model** (as shown above)
2. **Restart the backend server** so the classifier reloads the model from disk

```bash
# If using Docker Compose
docker compose restart backend

# If running manually
cp ml/model.joblib backend/app/ml/model.joblib  # Adjust path as needed
# Then restart the uvicorn/gunicorn process
```

### How the Classifier Loads the Model

The `LLMClassifier` class in `ml/classifier.py`:

1. Checks if `model.joblib` exists at the configured path
2. Loads it with `joblib.load()`
3. Extracts the model, class labels, and feature names
4. On prediction, converts input features to a vector, runs `model.predict_proba()`, and returns the highest-confidence model

If no model file exists, the classifier falls back with `confidence=0.0` and `model_id="unknown"`.

---

## 5. Feature Extraction Details

The ML classifier uses **9 features** extracted from each prompt. These features are defined in `ml/schema.py:PromptFeatures` and computed in `ml/feature_extraction.py`.

### Feature List

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `token_count` | `int` | Number of tokens in the prompt, computed with tiktoken `cl100k_base` encoding |
| 2 | `char_length` | `int` | Total character count of the prompt text |
| 3 | `has_code_blocks` | `bool` | Whether the prompt contains code blocks (`` ``` `` markers or code keywords) |
| 4 | `has_urls` | `bool` | Whether the prompt contains URLs (`http://` or `https://`) |
| 5 | `has_images` | `bool` | Whether the prompt contains image references (`image_url` parts or markdown/img tags) |
| 6 | `has_tool_calls` | `bool` | Whether the prompt contains tool/function call patterns |
| 7 | `dominant_language` | `str` | One of: `"code"`, `"natural_language"`, `"math"`, `"translation"`, `"unknown"` |
| 8 | `reasoning_complexity` | `float` | Score from 0.0 to 1.0 based on reasoning trigger words, code density, question marks, and complexity keywords |
| 9 | `hour_of_day` | `int` | Hour of day (0-23) when the prompt was received |

### Feature Vector Encoding

Features are converted to a 9-element float64 numpy array:

```python
# ml/feature_extraction.py:features_to_vector()
vector = [
    token_count,           # float
    char_length,           # float
    has_code_blocks,       # 0.0 or 1.0
    has_urls,              # 0.0 or 1.0
    has_images,            # 0.0 or 1.0
    has_tool_calls,        # 0.0 or 1.0
    language_encoded,      # 0=code, 1=natural_language, 2=math, 3=translation, 4=unknown
    reasoning_complexity,  # 0.0 to 1.0
    hour_of_day,           # 0 to 23
]
```

### Language Detection

The `detect_dominant_language()` function classifies prompts into 5 categories:

- **`code`**: ≥4 code-related keywords (def, class, import, fn, func, const, let, etc.)
- **`math`**: ≥2 math-related keywords (solve, equation, derivative, integral, matrix, etc.)
- **`translation`**: Contains translation indicators ("translate", "in french", "en español", etc.)
- **`natural_language`**: Everything else
- **`unknown`**: Empty or None text

### Reasoning Complexity Scoring

The `compute_reasoning_complexity()` function assigns a score (0.0 to 1.0) based on:

- **Reasoning trigger words** (+0.15 per match): "explain", "reason", "think step by step", "analyze", "compare", "why", "how does", "what if", "derive", "prove", "evaluate", "synthesize", etc.
- **Code density** (+0.2 if >10% of text is newlines)
- **Question marks** (+0.05 per question mark, capped at 0.3)
- **Complexity keywords** (+0.1 per match): "comprehensive", "detailed", "in-depth", "complex", "sophisticated", "multi-step", "hierarchical", etc.

---

## 6. Optional Enhancements

The issue description mentions these as potential future work:

### API Endpoint for Training

Add `POST /api/v1/classifier/train` to trigger training remotely without CLI access.

### UI Training Button

Add a "Train Classifier" button on the Classifier UI page (`ui/`) that calls the API endpoint.

### Scheduled Retraining

Set up a cron job or background task to retrain the model periodically (e.g., daily or weekly) using accumulated data.

### Active Learning

Implement the `is_correct` field in `classifier_samples` to capture user feedback, then use only correct samples for training to improve model accuracy over time.

---

## Quick Reference

| Task | Command |
|---|---|
| Train the classifier | `python -m ml.train` |
| Train with custom DB | `python -m ml.train --db-url postgresql://user:pass@host:5432/router` |
| Restart backend to load new model | `docker compose restart backend` |
| View training data count | `SELECT COUNT(*) FROM classifier_samples WHERE selected_model IS NOT NULL;` |
| View model metadata | `python -c "import joblib; print(joblib.load('ml/model.joblib')['metrics'])"` |

---

## Architecture Diagram

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Request   │────▶│ Rule-based       │────▶│ Confidence < 0.6?│
│   arrives   │     │ Router           │     └────────┬────────┘
└─────────────┘     └──────────────────┘              │
                                                      │
                              ┌───────────────────────┘
                              ▼
                     ┌───────────────┐
                     │ Redis Queue   │
                     │ (classifier)  │
                     └───────┬───────┘
                             ▼
                     ┌───────────────┐
                     │ Classifier    │────▶ ml/classifier.py
                     │ Worker        │         (predict & store)
                     └───────┬───────┘
                             ▼
                     ┌───────────────┐
                     │ PostgreSQL    │
                     │ classifier_   │
                     │ samples       │
                     └───────────────┘
                             │
                     (manual trigger)
                             ▼
                     ┌───────────────┐
                     │ Training      │────▶ ml/train.py
                     │ Script        │
                     └───────┬───────┘
                             ▼
                     ┌───────────────┐
                     │ model.joblib  │────▶ ml/classifier.py
                     │ features.     │         (_load)
                     │ joblib        │
                     └───────────────┘
                             │
                     (restart server)
                             │
                     ┌───────┴───────┐
                     │ Classifier    │────▶ predict()
                     │ loads model   │
                     └───────────────┘
```
