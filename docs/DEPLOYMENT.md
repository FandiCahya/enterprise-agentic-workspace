# Deployment Guide (Neon + Render + Vercel)

This guide provides a practical deployment setup for:
- **Database:** Neon PostgreSQL (with `pgvector`)
- **Backend API:** Render Web Service
- **Frontend:** Vercel (React/Vite app in `frontend/`)

> The backend and frontend are deployed separately. The frontend calls the backend via `VITE_API_URL`.

## 1) Neon (PostgreSQL + pgvector)

1. Create a Neon project and database.
2. Enable `pgvector` (Neon supports pgvector on compatible plans/configurations). In the SQL editor, run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

3. Copy the connection string from Neon (usually starts with `postgresql://...`).
4. Use that value as `DATABASE_URL` in the backend environment.

**Notes**
- The backend normalizes `postgresql://` / `postgres://` to `postgresql+psycopg://` automatically.
- Use the same `COLLECTION_NAME` across ingestion and runtime.

## 2) Render (Backend API)

Deploy the repository root as a Python web service.

### Recommended Render settings
- **Build command:** `pip install --upgrade pip && pip install -r requirements.txt`
- **Start command:** `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
- **Health check path:** `/health`
- **Python version:** 3.12.x (set `PYTHON_VERSION` or Render runtime config)

You can also use `render.yaml` in this repo as a starting point.

### Backend environment variables (example)

```env
# Required (LLM/chat + eval)
GOOGLE_API_KEY=<your-google-gemini-api-key>

# Database (recommended: Neon connection string)
DATABASE_URL=postgresql://<user>:<password>@<host>/<db>?sslmode=require

# RAG / models
LLM_MODEL=gemini-2.0-flash
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL=models/text-embedding-004
COLLECTION_NAME=enterprise_knowledge_base
RERANKER_MODEL=BAAI/bge-reranker-base

# API security / browser access
ALLOWED_ORIGINS=https://<your-frontend>.vercel.app
API_KEY=<generate-a-strong-random-secret>
MAX_UPLOAD_MB=20

# Observability (disable unless you have a reachable Phoenix/OTLP endpoint)
ENABLE_TRACING=false
PHOENIX_ENDPOINT=http://localhost:6006/v1/traces
```

### Important backend notes
- **Cold start / first request latency:** the retrieval engine and models may initialize lazily on first use (embedding/reranker downloads), which can increase first-request time.
- **Document ingestion:** deploy does not auto-ingest your local `data/` files. You must ingest into the deployed database (CLI/job/script against the same `DATABASE_URL` and `COLLECTION_NAME`).
- **API key:** if `API_KEY` is set, clients must send header `x-api-key: <API_KEY>` for protected endpoints (`/api/v1/documents/*`, `/api/v1/agent/chat`). `/health` remains public.

## 3) Vercel (Frontend)

Deploy the **frontend app** (`frontend/`) as a Vite project.

### Option A: Use root `vercel.json` (repo-level build)
This repository includes a root `vercel.json` configured to:
- run build in `frontend/`
- publish `frontend/dist`
- rewrite SPA routes to `index.html`

In Vercel project settings, ensure the configuration matches your preferred root/build behavior.

### Option B: Deploy `frontend/` as the project root
If you set the Vercel project root to `frontend/`, use `frontend/vercel.json` (SPA rewrite) and standard Vite build settings.

### Frontend environment variables (example)

```env
# Backend base URL (Render)
VITE_API_URL=https://<your-backend>.onrender.com

# Optional but recommended when backend API_KEY is enabled
VITE_API_KEY=<same-strong-random-secret-as-backend-API_KEY>
```

**Notes**
- The frontend now automatically attaches `x-api-key` (when `VITE_API_KEY` is set) to protected API requests (upload/ingest/search/chat).
- If you enable `API_KEY` on the backend, configure `VITE_API_KEY` in Vercel or requests will return `401`.

## 4) Post-deploy checklist

1. **Health check:** `GET https://<backend>/health` returns `200` and database connected.
2. **CORS:** browser requests from your Vercel domain succeed (no CORS errors).
3. **Auth:** protected endpoints accept requests with valid `x-api-key` and reject missing/invalid keys.
4. **Ingestion:** upload or ingest at least one document into the deployed DB/collection.
5. **Chat:** test `/api/v1/agent/chat` (streaming and non-streaming) from the frontend.

## 5) Troubleshooting

- **`503` on `/health`**: check `DATABASE_URL`, Neon SSL settings, and `vector` extension availability.
- **`401 Invalid or missing API key`**: ensure frontend `VITE_API_KEY` matches backend `API_KEY` exactly.
- **CORS error in browser**: set backend `ALLOWED_ORIGINS` to the exact frontend origin (e.g. `https://app.vercel.app`).
- **Slow first response**: expected during cold start/model initialization; consider warm-up strategy or always-on instance for demos.
- **Poor/empty answers**: confirm documents were ingested into the same DB and `COLLECTION_NAME`, and embedding config is consistent.
