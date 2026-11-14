# Enterprise Agentic RAG Workspace

Production-oriented Document Intelligence & Agentic RAG pipeline built with **FastAPI**, **LangGraph**, **PostgreSQL + PGVector**, and **Google Gemini**.

This repository includes:
- **Backend API** for document ingestion, hybrid search, and agentic chat (streaming + non-streaming)
- **RAG pipeline** with BM25 + dense vector retrieval, Reciprocal Rank Fusion (RRF), optional cross-encoder reranking, relevance evaluation, and query rewriting
- **Lite deployment mode** for low-resource/free-tier hosts (Google embeddings + reranker disabled, no local `torch`/`sentence-transformers` required)
- **Observability hooks** for Arize Phoenix / OpenTelemetry
- **RAGAS evaluation script** for automated quality checks
- **React + Vite frontend** and a static **portfolio landing page**

## Architecture

### Core components
- `src/main.py` — FastAPI app, CORS/auth, upload/search/chat endpoints, SSE streaming, DB health checks
- `src/core/config.py` — centralized environment/config resolution (`DATABASE_URL`, embeddings, LLM, CORS, API key)
- `src/core/ingestion.py` — document loading, chunking, and batch ingestion into PGVector
- `src/core/retrieval.py` — hybrid retrieval engine (BM25 + PGVector) with RRF and cross-encoder reranking
- `src/core/graph.py` — LangGraph workflow: retrieve → evaluate → rewrite (max 1 retry) → generate / fast-fail
- `src/eval/evaluate_ragas.py` — runs evaluation dataset through the pipeline and computes RAGAS metrics
- `frontend/` — chat UI, document upload, health/status panels
- `portfolio/` — static showcase page

### Request flow (Agentic Chat)
1. Retrieve candidate chunks via hybrid search (BM25 + PGVector, RRF)
2. (Optional) Re-rank candidates with a cross-encoder (`BAAI/bge-reranker-base` by default, can be disabled with `ENABLE_RERANKER=false`)
3. Evaluate context relevance with an LLM
4. If not relevant, rewrite the query once and retry retrieval/evaluation
5. Generate a grounded answer from validated context (or return a fast-fail response)

## Tech Stack
- **Backend:** FastAPI, Uvicorn, Pydantic
- **RAG / Agents:** LangChain, LangGraph, LangChain Google GenAI
- **Vector DB:** PostgreSQL + PGVector (`langchain-postgres`, `psycopg`)
- **Retrieval/ML:** BM25 (`rank-bm25`), Sentence Transformers, HuggingFace Cross-Encoder, PyTorch
- **Document loaders:** PyPDF, docx2txt
- **Observability/Eval:** Arize Phoenix, OpenTelemetry, RAGAS
- **Frontend:** React, Vite, Tailwind CSS
- **Infra:** Docker Compose, GitHub Actions, Render, Vercel

## Project Structure

```text
.
├─ src/
│  ├─ main.py
│  ├─ core/
│  │  ├─ config.py
│  │  ├─ graph.py
│  │  ├─ ingestion.py
│  │  └─ retrieval.py
│  └─ eval/
│     └─ evaluate_ragas.py
├─ frontend/
├─ portfolio/
├─ tests/
├─ docker-compose.yml
├─ render.yaml
├─ vercel.json
├─ requirements.txt
└─ .env.example
```

## Prerequisites
- **Python** 3.11+ (3.12 recommended)
- **Node.js** 20+ (for frontend build/dev)
- **Docker** (recommended for local PostgreSQL + Phoenix)
- **Google API key** (for Gemini LLM; embeddings can run locally by default)

> Notes:
> - **Full/local mode (default in `.env.example`)**: `USE_LOCAL_EMBEDDINGS=true`, `ENABLE_RERANKER=true` (higher quality, heavier dependencies/resources)
> - **Lite mode (recommended for free-tier deploy)**: `USE_LOCAL_EMBEDDINGS=false`, `ENABLE_RERANKER=false` with `requirements-lite.txt` (avoids `torch`/`sentence-transformers`)
> - If you switch embedding models/providers for an existing collection, re-ingest documents to avoid vector-space mismatch.

## Environment Variables

Copy `.env.example` to `.env` and configure as needed.

Key variables:
- `GOOGLE_API_KEY` — Gemini API key (required for chat/eval and for Google embeddings in lite mode)
- `LLM_MODEL` — chat/eval model (default: `gemini-2.0-flash`)
- `USE_LOCAL_EMBEDDINGS` — `true`/`false` (default: `true` in code; set `false` for lite deploy)
- `ENABLE_RERANKER` — `true`/`false` (default: `true`; set `false` for lite deploy)
- `EMBEDDING_MODEL` — Google embedding model when local embeddings disabled
- `LOCAL_EMBEDDING_MODEL` — local sentence-transformers model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `ALLOW_LOCAL_EMBEDDING_FALLBACK` — allow fallback to local embeddings if Google embedding init fails (default: `true`)
- `DATABASE_URL` — preferred in hosted environments (Neon/Render); auto-normalized to `postgresql+psycopg://`
- `POSTGRES_*` — used when `DATABASE_URL` is not provided
- `COLLECTION_NAME` — PGVector collection name (default: `enterprise_knowledge_base`)
- `ALLOWED_ORIGINS` — comma-separated CORS origins (use explicit origins in production)
- `API_KEY` — optional API key; when set, protected endpoints require `x-api-key`
- `MAX_UPLOAD_MB` — upload size limit (default: `20`)
- `ENABLE_TRACING` / `PHOENIX_ENDPOINT` — observability settings

## Local Development

### 1. Start infrastructure (PostgreSQL + Phoenix)
```bash
docker compose up -d postgres-pgvector phoenix-observability
```

### 2. Backend setup
```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
# Full mode (local embeddings + reranker, heavier)
pip install -r requirements.txt
# OR Lite mode (recommended for low-resource hosts; use Google embeddings)
# pip install -r requirements-lite.txt

cp .env.example .env  # then edit values
```

### 3. Ingest documents
Place files (`.pdf`, `.docx`, `.txt`, `.md`) in `data/`, then run:
```bash
python -m src.core.ingestion --data-dir data
```

### 4. Run backend API
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Phoenix UI (if enabled): `http://localhost:6006`

### 5. Run frontend (dev)
```bash
cd frontend
npm ci
npm run dev
```
Vite dev server proxies `/api` and `/health` to `http://localhost:8000` by default.

## API Overview

### `GET /health`
Checks database connectivity and returns service status.

### `POST /api/v1/documents/ingest`
Ingest a raw text chunk into PGVector.
- Protected by `x-api-key` if `API_KEY` is configured.

### `POST /api/v1/documents/upload`
Upload and ingest a document file (`pdf`, `docx`, `txt`, `md`).
- Protected by `x-api-key` if `API_KEY` is configured.
- Enforces `MAX_UPLOAD_MB`.

### `POST /api/v1/documents/search`
Vector similarity search over stored chunks.
- Protected by `x-api-key` if `API_KEY` is configured.

### `POST /api/v1/agent/chat`
Agentic RAG chat endpoint.
- `stream=true` (default): SSE stream with status/content/done events
- `stream=false`: JSON response `{ answer, retry_count, pipeline }`
- Protected by `x-api-key` if `API_KEY` is configured.

## Evaluation (RAGAS)

Run the evaluation script after the database is populated and model access is configured:
```bash
python -m src.eval.evaluate_ragas
```

The script computes:
- **Faithfulness**
- **Answer Relevancy**
- **Context Precision**

Results are printed to stdout and saved to `reports/ragas_evaluation_report.csv`.

> Note: metric names/targets reported in older README or portfolio content may differ from the current evaluation script. Treat the generated report as the source of truth for your environment.

## Testing & CI

### Local tests
```bash
pip install pytest httpx
pytest -q
```
Current tests focus on configuration behavior and API validation/auth basics (with external dependencies mocked/guarded).

### CI
GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
- Backend syntax checks and unit tests
- Frontend install + production build

## Deployment Guide

For a step-by-step setup (including environment variable examples for **Neon + Render + Vercel**), see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

### Quick summary
- **Backend (Render, lite/free-tier recommended) + DB (Neon/PostgreSQL + pgvector)**
  - Build: `pip install --upgrade pip && pip install -r requirements-lite.txt`
  - Start: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
  - Health check: `/health`
  - Env: `DATABASE_URL`, `GOOGLE_API_KEY`, `USE_LOCAL_EMBEDDINGS=false`, `ENABLE_RERANKER=false`, `ALLOWED_ORIGINS`, `API_KEY`
- **Frontend (Vercel, `frontend/`)**
  - Set `VITE_API_URL` to backend URL
  - Set `VITE_API_KEY` if backend `API_KEY` is enabled (frontend now attaches `x-api-key` automatically)
- **Portfolio**
  - `portfolio/` can be deployed separately as a static site

> After deployment, ingest documents into the deployed database/collection before testing chat/search.
> If you switch embedding mode/provider, re-ingest into the same or a new `COLLECTION_NAME`.

## Operational Notes & Limitations
- **Lite vs full mode:** lite mode (`USE_LOCAL_EMBEDDINGS=false`, `ENABLE_RERANKER=false`, `requirements-lite.txt`) is easier to deploy on free/low-resource tiers but may reduce retrieval quality compared to full local embeddings + cross-encoder reranking.
- **Embedding compatibility:** changing embedding models/providers requires re-indexing/re-ingestion for the target collection (or use a different `COLLECTION_NAME`).
- **Hybrid BM25 index:** BM25 is initialized from a sampled subset of stored documents on engine initialization; very large corpora may benefit from a persisted/full-corpus lexical index strategy.
- **Stateful in-memory components:** retrieval engine and vector store are lazily cached in-process; multi-instance deployments should consider shared indexing/refresh strategies.
- **LLM dependency:** chat and evaluation require access to a configured LLM provider/model; deployment environments must provide valid credentials and outbound access.
- **Security:** use explicit `ALLOWED_ORIGINS`, set `API_KEY`, and place the service behind HTTPS/reverse-proxy protections for production use.

## Troubleshooting
- **`/health` returns 503:** verify `DATABASE_URL` / Postgres credentials and pgvector extension availability.
- **Empty or poor answers:** ensure documents were ingested into the same collection and embedding space used at query time.
- **Upload fails (413):** increase `MAX_UPLOAD_MB` (and proxy limits, if applicable).
- **CORS errors in browser:** set `ALLOWED_ORIGINS` to the exact frontend origin(s).
- **Tracing warnings:** set `ENABLE_TRACING=false` when Phoenix is unavailable.