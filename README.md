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

- Backend API: http://localhost:8000
- UI Dashboard: http://localhost:80
- Swagger docs: http://localhost:8000/docs

### Local Development

**Backend:**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start services:
docker compose up -d db redis

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
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
| `DATABASE_URL` | `postgresql+asyncpg://router:router@db:5432/router` | Async DB URL |
| `REDIS_URL` | `redis://redis:6379/0` | Redis URL |
| `ENCRYPTION_KEY` | (required) | Fernet key for API key encryption |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CLASSIFIER_MIN_CONFIDENCE` | `0.6` | Minimum confidence for routing |
| `WORKER_CONCURRENCY` | `4` | ML worker count |
| `UPSTREAM_TIMEOUT` | `120.0` | Upstream API timeout (seconds) |
| `DEFAULT_MODEL` | `gpt-4o-mini` | Fallback model |
