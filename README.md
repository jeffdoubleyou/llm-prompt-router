# LLM Prompt Router

A production-ready OpenAI-compatible proxy with intelligent prompt routing, real-time metrics, and ML-based classifier fallback.

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
│            React SPA Dashboard              │
│  Metrics │ Models │ Logs │ Classifier │ ... │
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

```bash
python -m ml.train
```

## API Endpoints

### Chat (OpenAI-compatible proxy)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |

### Model Registry

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/models` | List all models |
| POST | `/api/v1/models` | Register a model |
| PUT | `/api/v1/models/{id}` | Update model |
| DELETE | `/api/v1/models/{id}` | Delete model |

### Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/logs` | Paginated request logs |
| GET | `/api/v1/metrics/summary` | Aggregated metrics per model |
| GET | `/api/v1/metrics/time-series` | Time-series metrics data |
| GET | `/api/v1/metrics/live` | SSE real-time metrics stream |
| GET | `/api/v1/metrics/dashboard` | Dashboard aggregate data |
| GET | `/api/v1/classifier` | Classifier status |
| GET | `/api/v1/queue` | Queue depth and worker status |
| GET | `/health` | Health check |

## Routing Logic

1. **Feature extraction** — Every incoming prompt is analyzed for:
   - Token count, character length
   - Code blocks, URLs, images
   - Tool/function calls
   - Dominant language (code, math, translation, natural language)
   - Reasoning complexity score
   - Hour of day

2. **Rule-based matching** — Active models are scored against prompt features:
   - Vision models get +3 for image prompts
   - Tool-calling models get +2 for tool calls
   - Long-context models get +2 for large prompts
   - Code-optimized models get +1.5 for code
   - Reasoning models get +2 for complex prompts
   - Priority field adds a bias

3. **Confidence check** — If rule confidence ≥ threshold (default 0.60), route directly.

4. **ML classifier fallback** — If confidence < threshold, enqueue to Redis. Background workers predict the best model using a trained HistGradientBoostingClassifier.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `APP_PORT` | `8080` | Port for the backend FastAPI server |
| `UI_PORT` | `3000` | Port for the frontend Nginx/UI server |
| `DATABASE_URL` | `postgresql+asyncpg://router:router@db:5432/router` | Async DB URL |
| `REDIS_URL` | `redis://redis:6379/0` | Redis URL |
| `ENCRYPTION_KEY` | (required) | Fernet key for API key encryption |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CLASSIFIER_MIN_CONFIDENCE` | `0.6` | Minimum confidence for routing |
| `WORKER_CONCURRENCY` | `4` | ML worker count |
| `UPSTREAM_TIMEOUT` | `120.0` | Upstream API timeout (seconds) |
| `DEFAULT_MODEL` | `gpt-4o-mini` | Fallback model |
