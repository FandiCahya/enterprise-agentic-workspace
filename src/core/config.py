"""Shared configuration and factory helpers for the Enterprise Agentic RAG workspace.

Centralizes environment handling so API, ingestion, retrieval, graph, and eval
use consistent database URLs, embedding models, and LLM configuration.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root when available (safe no-op if missing)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_knowledge_base")
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_database_url() -> str:
    """Resolve a SQLAlchemy/psycopg database URL.

    Priority:
    1. DATABASE_URL (Neon/Render/etc.), normalized to postgresql+psycopg://
    2. Individual POSTGRES_* components
    """
    raw = (os.getenv("DATABASE_URL") or "").strip()
    if raw:
        if raw.startswith("postgres://"):
            raw = "postgresql://" + raw[len("postgres://"):]
        if raw.startswith("postgresql://"):
            raw = "postgresql+psycopg://" + raw[len("postgresql://"):]
        elif raw.startswith("postgresql+psycopg2://"):
            raw = "postgresql+psycopg://" + raw[len("postgresql+psycopg2://"):]
        return raw

    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "adminpassword")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "enterprise_rag")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


def resolve_llm_model() -> str:
    preferred = (os.getenv("LLM_MODEL") or "").strip()
    if preferred:
        return preferred
    # Reasonable default for Gemini API deployments
    return "gemini-2.0-flash"


def resolve_google_embedding_model() -> str:
    preferred = (os.getenv("EMBEDDING_MODEL") or "").strip()
    if preferred:
        return preferred
    return "models/text-embedding-004"


@lru_cache(maxsize=2)
def build_embeddings(use_local: Optional[bool] = None):
    """Build embeddings with a safe local-first default.

    Set USE_LOCAL_EMBEDDINGS=false and provide GOOGLE_API_KEY to use Google.
    Falls back to local HuggingFace embeddings if Google init fails.
    """
    if use_local is None:
        use_local = env_bool("USE_LOCAL_EMBEDDINGS", default=True)

    if use_local:
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
        return HuggingFaceEmbeddings(model_name=model_name)

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=resolve_google_embedding_model())
    except Exception as exc:
        print(f"[WARNING] Google embeddings unavailable, falling back to local embeddings: {exc}")
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
        return HuggingFaceEmbeddings(model_name=model_name)


def build_llm(*, streaming: bool = False, temperature: float = 0):
    """Build the chat LLM used by the app/graph/eval."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = resolve_llm_model()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    kwargs = {
        "model": model,
        "temperature": temperature,
        "streaming": streaming,
    }
    if google_api_key:
        kwargs["google_api_key"] = google_api_key
    return ChatGoogleGenerativeAI(**kwargs)


def get_cors_origins() -> list[str]:
    raw = (os.getenv("ALLOWED_ORIGINS") or "").strip()
    if not raw:
        return ["*"]
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]


def get_api_key() -> Optional[str]:
    key = (os.getenv("API_KEY") or "").strip()
    return key or None
