# LLM Prompt Router

🚧 Work In Progress

An OpenAI-compatible proxy with intelligent prompt routing, real-time metrics, and ML-based classifier fallback.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client    │────▶│  FastAPI App │────▶│   Upstream   │
│  (OpenAI    │     │  (Backend)   │     │   LLM APIs   │
│   SDK)      │     │              │     │              │
└─────────────┘     └──────┬───────┘     └──────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │PostgreSQL│ │  Redis   │ │   ML     │
        │  (logs,  │ │ (queue,  │ │Classifier│
        │ models,  │ │ metrics) │ │(fallback)│
        │ samples) │ │          │ │          │
        └──────────┘ └──────────┘ └──────────┘

┌────────────────────────────────────────────┐
│            React SPA Dashboard             │
│  Metrics │ Models │ Logs │ Classifier │ ...│
└────────────────────────────────────────────┘
```

## Project Structure

```
llm-router/
├── backend/               # FastAPI backend
│   ├── app/
│   │   ├── main.py        # App entry point
│   │   ├── api/v1/        # REST endpoints
│   │   │   ├── chat.py    # OpenAI-compatible proxy
│   │   │   └── router.py  # CRUD + metrics + admin
│   │   ├── core/          # Config, database, security, models
│   │   ├── models/        # SQLAlchemy ORM
│   │   ├── services/      # Routing engine, Redis queue
│   │   └── workers/       # Classifier background worker
│   └── requirements.txt
├── ml/                    # ML Classifier
│   ├── feature_extraction.py
│   ├── classifier.py
│   ├── train.py
│   └── schema.py
├── ui/                    # React SPA
│   ├── src/
│   │   ├── pages/         # Dashboard, Models, Logs, Metrics, ...
│   │   ├── hooks/         # SSE live metrics hook
│   │   └── lib/           # API client
│   └── ...
├── docker-compose.yml
├── Dockerfile             # Backend
├── Dockerfile.ui          # UI (nginx)
└── README.md
```

## Quick Start

### Prerequisites

- Docker and Docker Compose (recommended)
- Python 3.12+ (for local dev)
- Node.js 20+ (for local UI dev)

### Run with Docker Compose

```bash
cd llm-router
docker compose up -d --build
```

All services communicate over the `llm-router` network. Only the backend API and UI are exposed to the host (on ports `APP_PORT` and `UI_PORT` respectively). PostgreSQL and Redis are not exposed to the host — they communicate only over the internal `llm-router` network, reducing the attack surface.

- Backend API: `http://localhost:${APP_PORT:-8080}`
- UI Dashboard: `http://localhost:${UI_PORT:-3000}`
- Swagger docs: `http://localhost:${APP_PORT:-8080}/docs`

### Deploy and Upgrade

PostgreSQL data is stored in a **bind mount** at `./data/postgres` so it survives redeployments.

**Safe upgrade command** (preserves all data):

```bash
docker compose down && docker compose up -d --build
```

**Do NOT use** `docker compose down -v` — the `-v` flag removes named volumes and will **permanently delete** all database contents including models, request logs, and training data.

**First-time setup** (fresh clone):

```bash
docker compose up -d --build
python scripts/seed_models.py   # populate with default models if DB is empty
```

**Data persistence**

- Model configs, request logs, and classifier training data all live in `./data/postgres/` on the host.
- This directory is gitignored — do not commit it.
- To back up your data, copy the `./data/postgres/` directory.

### Local Development

**Backend:**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start services:
docker compose up -d db redis

uvicorn app.main:app --reload --host 0.0.0.0 --port ${APP_PORT:-8080}
```

**UI:**

```bash
cd ui
npm install
npm run dev
```

**ML Classifier Training:**

See **[docs/ml-classifier-training.md](docs/ml-classifier-training.md)** for how routing, sample collection, and training connect.

```bash
# Inside the app container
docker compose exec app python -m ml.train
docker compose restart app   # reload model.joblib in workers
```

## API Endpoints

### Chat (OpenAI-compatible proxy)


| Method | Path                   | Description                                 |
| ------ | ---------------------- | ------------------------------------------- |
| POST   | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |


### Model Registry


| Method | Path                  | Description      |
| ------ | --------------------- | ---------------- |
| GET    | `/api/v1/models`      | List all models  |
| POST   | `/api/v1/models`      | Register a model |
| PUT    | `/api/v1/models/{id}` | Update model     |
| DELETE | `/api/v1/models/{id}` | Delete model     |


### Monitoring


| Method | Path                          | Description                   |
| ------ | ----------------------------- | ----------------------------- |
| GET    | `/api/v1/logs`                | Paginated request logs        |
| GET    | `/api/v1/metrics/summary`     | Aggregated metrics per model  |
| GET    | `/api/v1/metrics/time-series` | Time-series metrics data      |
| GET    | `/api/v1/metrics/live`        | SSE real-time metrics stream  |
| GET    | `/api/v1/metrics/dashboard`   | Dashboard aggregate data      |
| GET    | `/api/v1/classifier`          | Classifier status             |
| GET    | `/api/v1/queue`               | Queue depth and worker status |
| GET    | `/health`                     | Health check                  |


## Routing Logic

Every `POST /v1/chat/completions` request goes through feature extraction and rule-based routing. See **[docs/ml-classifier-training.md](docs/ml-classifier-training.md)** for the full picture including how ML training fits in.

### What routes requests (live path)

1. **Feature extraction** — Token count, code/URL/image/tool signals, language, task type, etc. (`router_service.extract_features`). Image rules: **[docs/image-detection.md](docs/image-detection.md)**; per-request debug on the Prompt Debug UI page.
2. **Capability filter** — Hard requirements first: vision required when images are present (non-vision models excluded), tool calling for tools, context window limits.
3. **Rule-based scoring** — Score remaining models by capability match (vision +3, tools +2, long context +2, code +1.5, reasoning +2, priority bias). Keep the highest-scoring tier.
4. **Speed/cost ranking** — Among that tier, pick the **fastest** model (`estimated_tokens_per_second`), then the **cheapest** (input + output cost per 1k tokens).
5. **Complexity routing (optional)** — When `COMPLEXITY_ROUTING_ENABLED=true`, also require `max_complexity_score` to cover the prompt's routing difficulty before speed/cost ranking. **Default is off** — no parameter/complexity-tier selection unless you enable it.
6. **Route** — The selected model is used immediately. If confidence ≥ `CLASSIFIER_MIN_CONFIDENCE` (default 0.6), done.

### What the ML classifier does today

When confidence is **below** the threshold, the request is also enqueued to Redis. Background workers run the trained classifier and save a row to `classifier_samples` for later training. **The ML model does not change which model serves the request** — that is still the rule/complexity pick.

To train: `python -m ml.train` (see [ML classifier guide](docs/ml-classifier-training.md)). Restart the app after training so workers reload `ml/model.joblib`.

## Configuration


| Environment Variable        | Default                                             | Description                                       |
| --------------------------- | --------------------------------------------------- | ------------------------------------------------- |
| `APP_PORT`                  | `8080`                                              | Port for the backend FastAPI server               |
| `UI_PORT`                   | `3000`                                              | Port for the frontend Nginx/UI server             |
| `DATABASE_URL`              | `postgresql+asyncpg://router:router@db:5432/router` | Async DB URL                                      |
| `REDIS_URL`                 | `redis://redis:6379/0`                              | Redis URL                                         |
| `ENCRYPTION_KEY`            | (required)                                          | Fernet key for API key encryption                 |
| `LOG_LEVEL`                 | `INFO`                                              | Logging level                                     |
| `CLASSIFIER_MIN_CONFIDENCE` | `0.6`                                               | Minimum confidence for routing                    |
| `EMBEDDING_ROUTING_ENABLED` | `false`                                             | Blend embedding k-NN difficulty into task routing |
| `EMBEDDING_BLEND_WEIGHT`    | `0.55`                                              | Weight on embedding vs heuristic difficulty (0–1) |
| `EMBEDDING_MODEL_NAME`      | `sentence-transformers/all-MiniLM-L6-v2`            | Local embedding model                             |
| `UPSTREAM_QUEUE_ENABLED`    | `false`                                             | FIFO queue per upstream base URL (llama.cpp)      |
| `COMPLEXITY_ROUTING_ENABLED`| `false`                                             | Filter by max_complexity_score before speed/cost  |
| `LLAMACPP_MAX_TOOLS`        | `20`                                                | Max tools for llama.cpp (`0` = no limit)          |
| `LLAMACPP_TOOL_LIMIT_MODE`  | `reject`                                            | `reject` (400 to client) or `truncate`            |
| `LLAMACPP_BASE_URL_PREFIXES`| (empty)                                             | Comma-separated URL prefixes for tool limiting    |
| `LLAMACPP_PROVIDERS`        | `custom,llama,ollama,llamacpp`                      | Providers treated as llama.cpp when prefixes empty |
| `WORKER_CONCURRENCY`        | `4`                                                 | ML worker count                                   |
| `UPSTREAM_TIMEOUT`          | `120.0`                                             | Upstream API timeout (seconds)                    |
| `DEFAULT_MODEL`             | `gpt-4o-mini`                                       | Fallback model                                    |

## Screenshots

### Dashboard
<img width="953" height="997" alt="image" src="https://github.com/user-attachments/assets/3ab81324-e9bd-4724-b7be-09781f249b6a" />

### Models
<img width="962" height="773" alt="image" src="https://github.com/user-attachments/assets/451775c6-792c-4fbb-9191-2390b28583e9" />

### Logs
<img width="962" height="773" alt="image" src="https://github.com/user-attachments/assets/7f50637d-6ce7-42ec-8fed-0066a71bb84e" />

### Prompt Debug
<img width="962" height="773" alt="image" src="https://github.com/user-attachments/assets/668557ce-e709-4911-a09e-c1d58b5aba82" />

### Prompt Debug -- Detail / Parsed
<img width="947" height="994" alt="image" src="https://github.com/user-attachments/assets/d32f8c34-eb64-4c34-a864-c0e09e38de2f" />

### Complexity Debug
<img width="953" height="1002" alt="image" src="https://github.com/user-attachments/assets/80103ef6-4560-4ea2-a98b-8ce0153073a1" />

### Metrics
<img width="962" height="837" alt="image" src="https://github.com/user-attachments/assets/68cbda4e-333d-4ce4-9b2e-c9ced653c8e2" />

### Request Queue
<img width="963" height="717" alt="image" src="https://github.com/user-attachments/assets/02cfb7b4-9073-4011-835d-8df8b231fbae" />

