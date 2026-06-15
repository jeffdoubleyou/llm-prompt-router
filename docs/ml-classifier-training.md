# ML Classifier: How It Works

This guide explains how request routing, training-data collection, and classifier training fit together in the LLM Prompt Router.

If you only read one section, read **[The mental model](#the-mental-model)** and **[What actually routes a request today](#what-actually-routes-a-request-today)**.

---

## The mental model

Think of three separate layers:

| Layer | What it does | When it runs |
|-------|----------------|--------------|
| **1. Rule / complexity router** | Picks a model from your registry using hand-crafted scoring | **Synchronously on every request** — this is what users hit |
| **2. Sample collector** | Saves prompt features to Postgres for later training | **Asynchronously** when rule confidence is low |
| **3. Training script** | Reads saved samples, trains `model.joblib`, writes it to disk | **Manually** — you run `python -m ml.train` |

The trained ML model is loaded by background workers and used when **recording** samples. It does **not** currently change which model serves a live request.

```
  Live request                Background                 Manual (you)
  ───────────                 ──────────                 ────────────

  POST /v1/chat/completions
         │
         ▼
  extract_features()
         │
         ▼
  complexity / rule router ──────────────▶ upstream LLM  (user gets response)
         │
         │  (if confidence < 0.6)
         ▼
  Redis queue ──────────────────▶ classifier worker ──▶ classifier_samples table
                                           │
                                           │ uses model.joblib
                                           │ (if it exists)
                                           ▼
                                    stores features + prediction

                                                              python -m ml.train
                                                                     │
                                                                     ▼
                                                              ml/model.joblib
                                                              (restart app to reload)
```

---

## End-to-end request lifecycle

Here is what happens when a client calls `POST /v1/chat/completions`:

### Step 1 — Feature extraction

`backend/app/api/v1/chat.py` receives the request and calls `extract_features()` in `router_service.py`.

From the message list it computes a `PromptFeatures` object:

- `token_count`, `char_length`
- `has_code_blocks`, `has_urls`, `has_images`, `has_tool_calls`
- `dominant_language`, `reasoning_complexity`, `hour_of_day`
- `complexity_score` (used by routing only; not part of the ML feature vector)

The same features are also stored in Redis (see **Prompt Debug** on the `/prompts` UI page) for inspection.

### Step 2 — Model selection (this decides routing)

`classify_and_route()` in `router_service.py` picks a model:

1. **Complexity-based routing** (if any active model has `max_complexity_score` set)  
   Filters to capable models, then prefers the cheapest/fastest that can handle the prompt's complexity.

2. **Rule-based routing** (fallback)  
   Scores each active model on capabilities (vision +3, tools +2, long context +2, code +1.5, reasoning +2, etc.) and picks the highest score.

Both paths return a `(model, confidence)` pair where confidence is a normalized score in `[0, 1]`.

### Step 3 — Confidence gate

| Condition | What happens |
|-----------|----------------|
| `confidence >= classifier_min_confidence` (default **0.6**) | Return the matched model immediately. Request is **not** enqueued. |
| `confidence < 0.6` | Enqueue prompt features to Redis **and still return the same rule-matched model**. |

**Important:** even when a request is enqueued, the user is routed by rules/complexity — not by the ML classifier.

### Step 4 — Proxy to upstream

`chat.py` forwards the request to the chosen model's `base_url` and logs tokens/cost/latency to `request_logs`.

### Step 5 — Background sample collection (low-confidence only)

If the request was enqueued, a classifier worker (`backend/app/workers/classifier_worker.py`) eventually:

1. Dequeues from Redis key `router:unclassified_queue`
2. Runs `LLMClassifier.predict()` on the stored feature dict (if `ml/model.joblib` exists)
3. Inserts a row into `classifier_samples`

---

## What actually routes a request today

| Component | Affects live routing? | Notes |
|-----------|----------------------|-------|
| Complexity router | **Yes** | When models have `max_complexity_score` |
| Rule-based router | **Yes** | Always available as fallback |
| ML classifier (`model.joblib`) | **No** | Only used when saving training samples |
| Classifier workers | **No** | Async; never on the hot path |

The README's "ML classifier fallback" description reflects the **intended** design, but the current implementation stops at data collection. Wiring `classify_and_route()` to call `LLMClassifier.predict()` when rule confidence is low would be the next step to make training affect routing.

---

## Training data: what gets stored

### When is a sample created?

Only when rule/complexity confidence is **below** `CLASSIFIER_MIN_CONFIDENCE` (default `0.6`).

High-confidence requests are routed and logged to `request_logs`, but **never** become classifier training samples.

### What is stored?

Table: `classifier_samples`

| Column | Contents |
|--------|----------|
| `features` | The `PromptFeatures` dict extracted at request time |
| `prompt_text` | JSON serialization of the same features (legacy/debug) |
| `selected_model` | The **ML classifier's prediction** if a model is loaded; otherwise the rule-based fallback model from the queue item |
| `confidence` | The classifier's prediction confidence |
| `is_correct` | `NULL` by default; can be set via the Classifier UI for human feedback |

### What label does training use?

`ml/train.py` trains on `selected_model` from each row:

```sql
SELECT cs.prompt_text, cs.selected_model, cs.features
FROM classifier_samples cs
WHERE cs.selected_model IS NOT NULL
```

So today the model learns to reproduce **past classifier predictions** (or rule-based fallbacks when no model file exists), not necessarily the model that gave the best response.

The `is_correct` column exists for supervised feedback (mark samples ✓/✗ in the **Classifier → Training Data** UI), but the training script does **not** yet filter to `is_correct = true` only. That is listed as a future improvement below.

---

## How to train the classifier

Training is a **manual CLI step** inside the app container (or locally with DB access).

### Prerequisites

1. **Traffic** — Send requests through the router so low-confidence samples accumulate.
2. **Check sample count** — Classifier UI → Training Data tab, or:
   ```sql
   SELECT COUNT(*) FROM classifier_samples WHERE selected_model IS NOT NULL;
   ```
   You need enough rows for meaningful classes (rough guide: tens per model minimum).
3. **Model registry** — Active models in `models` table should match the `selected_model` values you want the classifier to predict.

### Run training (Docker)

```bash
# Rebuild if you recently changed requirements (needs psycopg2-binary)
docker compose build app
docker compose up -d app

# Train (reads from Postgres, writes ml/model.joblib)
docker compose exec app python -m ml.train
```

### Run training (local)

```bash
cd /path/to/llm-prompt-router
pip install -r backend/requirements.txt
python -m ml.train --db-url postgresql://router:router@localhost:5432/router
```

### What the script does

1. Loads all `classifier_samples` rows with a non-null `selected_model`
2. Converts each `features` JSON blob to a 9-element numeric vector (`ml/feature_extraction.py`)
3. 80/20 stratified train/test split
4. Fits `HistGradientBoostingClassifier`
5. Prints accuracy / per-class metrics / feature importances
6. Saves `ml/model.joblib` and `ml/features.joblib`

### After training

```bash
# Reload workers so they pick up the new model file
docker compose restart app
```

The classifier is loaded **once at worker startup** (`ml/classifier.py`). A restart is required after every train run.

**What changes after restart:**

- Background workers will store **new** ML predictions in `classifier_samples`
- **Live routing behavior is unchanged** until code is added to consult the classifier on the request path

---

## The 9 ML features

These are defined in `ml/schema.py` and extracted at request time by `router_service.extract_features()`. The training pipeline reads them from the stored `features` column.

| # | Feature | Description |
|---|---------|-------------|
| 1 | `token_count` | tiktoken `cl100k_base` count (or char/4 fallback) |
| 2 | `char_length` | Total characters |
| 3 | `has_code_blocks` | Code fences or code keywords |
| 4 | `has_urls` | URLs in user/assistant/tool messages (system prompts excluded) |
| 5 | `has_images` | Actual multimodal image parts or embedded image data |
| 6 | `has_tool_calls` | Tool call fields present on messages |
| 7 | `dominant_language` | Encoded: code, natural_language, math, translation, unknown |
| 8 | `reasoning_complexity` | 0.0–1.0 heuristic score |
| 9 | `hour_of_day` | UTC hour when the request was processed |

`complexity_score` is computed for routing but is **not** included in the ML vector.

---

## Configuration reference

| Setting | Env var | Default | Effect |
|---------|---------|---------|--------|
| Confidence threshold | `CLASSIFIER_MIN_CONFIDENCE` | `0.6` | Below this → enqueue sample; at/above → skip enqueue |
| Worker count | `WORKER_CONCURRENCY` | `4` | Parallel classifier workers |
| Model file path | (in config) | `ml/model.joblib` | Where train writes / workers read |
| Default fallback model | `DEFAULT_MODEL` | `gpt-4o-mini` | Used when no models match |
| Prompt debug retention | `PROMPT_DEBUG_MAX_STORED` | `100` | Recent prompts in Redis for debugging |

---

## Inspecting the pipeline

| What you want | Where to look |
|---------------|---------------|
| Which model served a request | **Logs** UI → `request_logs` |
| Extracted features for recent requests | **Prompts** UI (Redis) |
| Training samples | **Classifier → Training Data** UI |
| Queue backlog | **Queue** UI or `GET /api/v1/queue` |
| Classifier status / sample count | **Classifier → Status** or `GET /api/v1/classifier` |
| Model file metadata | `python -c "import joblib; print(joblib.load('ml/model.joblib')['metrics'])"` |

---

## Improving training quality

### 1. Mark correct / incorrect samples

In the Classifier UI, expand a training sample and click ✓ or ✗. This sets `is_correct`. Intended for future training filters; not applied by `ml/train.py` yet.

### 2. Ensure diverse traffic

The classifier only sees **ambiguous** requests (low rule confidence). If everything scores above 0.6, `classifier_samples` stays empty.

To collect more samples temporarily, raise `CLASSIFIER_MIN_CONFIDENCE` (e.g. to `0.9`) — but remember to lower it again for production.

### 3. Align labels with reality

Today `selected_model` reflects the classifier's own guess (or rules), not ground truth. For high-quality training you eventually want labels from:

- Human review (`is_correct`)
- Explicit model override from the client
- Offline evaluation (quality scores per model)

---

## Architecture diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         LIVE REQUEST PATH (sync)                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Client ──▶ POST /v1/chat/completions                                    │
│                  │                                                       │
│                  ▼                                                       │
│            extract_features(messages)                                    │
│                  │                                                       │
│                  ▼                                                       │
│     ┌────────────────────────────┐                                       │
│     │ Complexity router          │──┐                                    │
│     │ (if max_complexity_score   │  │                                    │
│     │  set on any model)         │  │                                    │
│     └────────────────────────────┘  │                                    │
│                  │ no match         │                                    │
│                  ▼                  │                                    │
│     ┌────────────────────────────┐  │                                    │
│     │ Rule-based router          │  │                                    │
│     │ (capability scoring)       │  │                                    │
│     └────────────────────────────┘  │                                    │
│                  │                  │                                    │
│                  ▼                  ▼                                    │
│         confidence >= 0.6?     matched model ID                          │
│            │        │                  │                                 │
│           yes       no                 │                                 │
│            │        │                  │                                 │
│            │        ▼                  │                                 │
│            │   Redis LPUSH            │                                 │
│            │   router:unclassified_queue                                │
│            │        │                  │                                 │
│            └────────┴──────────────────┼──▶ proxy to upstream LLM        │
│                                      │         │                         │
│                                      │         ▼                         │
│                                      │    request_logs (tokens, cost)    │
└──────────────────────────────────────┼─────────────────────────────────┘
                                       │
┌──────────────────────────────────────┼─────────────────────────────────┐
│                    BACKGROUND PATH (async)                            │
├──────────────────────────────────────┼─────────────────────────────────┤
│                                      │                                  │
│                    classifier_worker (×N)                               │
│                              │                                          │
│                              ▼                                          │
│                    LLMClassifier.predict(features)                      │
│                    (reads ml/model.joblib)                              │
│                              │                                          │
│                              ▼                                          │
│                    INSERT classifier_samples                            │
│                    (features, selected_model, confidence)               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINING PATH (manual)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  python -m ml.train                                                      │
│       │                                                                  │
│       ▼                                                                  │
│  SELECT * FROM classifier_samples                                        │
│       │                                                                  │
│       ▼                                                                  │
│  HistGradientBoostingClassifier.fit()                                    │
│       │                                                                  │
│       ▼                                                                  │
│  ml/model.joblib  ──▶  docker compose restart app  ──▶  workers reload │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Future work (not implemented yet)

| Enhancement | Benefit |
|-------------|---------|
| Use ML prediction in `classify_and_route()` when rule confidence is low | Training actually affects routing |
| Train only on `is_correct = true` samples | Human feedback improves model |
| `POST /api/v1/classifier/train` | Trigger training without shell access |
| Scheduled retraining | Keep model fresh automatically |
| Store `routed_model` vs `predicted_model` separately | Clearer training labels |

---

## Quick reference

| Task | Command |
|------|---------|
| Train | `docker compose exec app python -m ml.train` |
| Reload model in workers | `docker compose restart app` |
| Count training samples | `SELECT COUNT(*) FROM classifier_samples;` |
| View model metrics | `docker compose exec app python -c "import joblib; print(joblib.load('ml/model.joblib')['metrics'])"` |
| Temporarily collect more samples | Set `CLASSIFIER_MIN_CONFIDENCE=0.9` in compose env |

---

## Key source files

| File | Role |
|------|------|
| `backend/app/api/v1/chat.py` | Entry point; calls routing then proxies |
| `backend/app/services/router_service.py` | Feature extraction, routing, enqueue logic |
| `backend/app/workers/classifier_worker.py` | Dequeues, predicts, saves samples |
| `backend/app/services/redis_queue.py` | Redis queue (`router:unclassified_queue`) |
| `ml/classifier.py` | Load model, predict |
| `ml/train.py` | Training CLI |
| `ml/feature_extraction.py` | Feature vector for sklearn |
| `backend/app/models/db.py` | `ClassifierSample`, `RequestLog` tables |
