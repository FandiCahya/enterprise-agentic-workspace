import asyncio
import json
import os
import shutil
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

load_dotenv()

# ------------------------------------------------------------------
# Observability (best-effort)
# ------------------------------------------------------------------
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
try:
    if os.getenv("ENABLE_TRACING", "true").lower() in {"1", "true", "yes", "y"}:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        tracer_provider = register(
            project_name=os.getenv("PHOENIX_PROJECT_NAME", "enterprise-rag-workspace"),
            endpoint=PHOENIX_ENDPOINT,
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print(f"[OK] Tracing Arize Phoenix aktif: {PHOENIX_ENDPOINT}")
except Exception as e:
    print(f"[WARNING] Tracing Phoenix tidak aktif: {e}")

from langchain_core.documents import Document
from langchain_postgres import PGVector

from src.core.config import (
    COLLECTION_NAME,
    build_embeddings,
    get_api_key,
    get_cors_origins,
    get_database_url,
)

# ------------------------------------------------------------------
# App & shared state
# ------------------------------------------------------------------
app = FastAPI(
    title="Enterprise Document Intelligence & Agentic Workspace API",
    description="Backend API untuk ingestion, hybrid retrieval, dan agentic RAG",
    version="1.1.0",
)

allowed_origins = get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_embeddings = None
_vector_store = None
_graph_executor = ThreadPoolExecutor(max_workers=4)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = build_embeddings()
    return _embeddings


def get_vector_store() -> PGVector:
    global _vector_store
    if _vector_store is None:
        _vector_store = PGVector(
            embeddings=get_embeddings(),
            collection_name=COLLECTION_NAME,
            connection=get_database_url(),
            use_jsonb=True,
        )
    return _vector_store


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = get_api_key()
    if not expected:
        return True
    if x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return True


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------
class DocumentIngestRequest(BaseModel):
    content: str = Field(..., min_length=1)
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)


class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    stream: bool = Field(default=True)


class AgentChatResponse(BaseModel):
    answer: str
    retry_count: int = 0
    pipeline: str = "LangGraph Agentic RAG (Hybrid Search + Cross-Encoder + Self-Correction)"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _sse(data: dict | str) -> str:
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _check_db() -> None:
    engine = create_engine(get_database_url())
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    finally:
        engine.dispose()


async def _run_graph(query: str) -> dict:
    from src.core.graph import agentic_app

    loop = asyncio.get_event_loop()
    initial_state = {
        "original_query": query,
        "current_query": query,
        "retry_count": 0,
    }
    return await loop.run_in_executor(_graph_executor, agentic_app.invoke, initial_state)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    try:
        _check_db()
        return {
            "status": "online",
            "database": "connected",
            "collection": COLLECTION_NAME,
            "tracing_endpoint": PHOENIX_ENDPOINT,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection error: {str(e)}")


@app.post("/api/v1/documents/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_document(payload: DocumentIngestRequest, _: bool = Depends(require_api_key)):
    try:
        vector_store = get_vector_store()
        doc = Document(page_content=payload.content, metadata=payload.metadata or {})
        vector_store.add_documents([doc])
        return {"message": "Dokumen berhasil disimpan ke PGVector", "content": payload.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan dokumen: {str(e)}")


@app.post("/api/v1/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...), _: bool = Depends(require_api_key)):
    supported = {".pdf", ".docx", ".txt", ".md"}
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: '{ext}'. Gunakan: {', '.join(sorted(supported))}",
        )

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        size = os.path.getsize(tmp_path)
        if size > max_bytes:
            raise HTTPException(status_code=413, detail=f"Ukuran file melebihi batas {MAX_UPLOAD_MB} MB")

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_path)
        elif ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(tmp_path)
        else:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()
        for doc in docs:
            doc.metadata["source_filename"] = filename
            doc.metadata["file_type"] = ext.lstrip(".")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        if not chunks:
            raise HTTPException(status_code=422, detail="Tidak ada teks yang bisa diekstrak dari file.")

        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        return {
            "message": f"File '{filename}' berhasil diproses",
            "filename": filename,
            "pages": len(docs),
            "chunks_stored": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses file: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/v1/documents/search")
async def search_documents(payload: SearchQueryRequest, _: bool = Depends(require_api_key)):
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search(payload.query, k=payload.top_k)
        output = [{"content": d.page_content, "metadata": d.metadata} for d in results]
        return {"query": payload.query, "results_count": len(output), "data": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan pencarian: {str(e)}")


async def stream_agentic_response(query: str):
    """Stream status updates and final generation tokens."""
    try:
        from src.core.graph import (
            GENERATE_PROMPT,
            EVAL_PROMPT,
            REWRITE_PROMPT,
            extract_text,
            format_context,
            get_llm,
            get_retrieval_engine,
        )

        yield _sse({"status": "thinking", "message": "Memulai Agentic RAG Pipeline..."})

        engine = get_retrieval_engine()
        original_query = query
        current_query = query
        retry_count = 0
        docs = []
        is_relevant = False

        while True:
            yield _sse({"status": "thinking", "message": f"Mencari dokumen untuk: {current_query}"})
            docs = engine.search(current_query)

            if not docs:
                is_relevant = False
            else:
                context_eval = "\n\n".join([f"- {d.page_content}" for d in docs])
                eval_llm = get_llm(streaming=False)
                eval_resp = extract_text(
                    (EVAL_PROMPT | eval_llm).invoke({"query": current_query, "context": context_eval}).content
                ).strip().upper()
                is_relevant = "YA" in eval_resp

            if is_relevant or retry_count >= 1:
                break

            retry_count += 1
            yield _sse({"status": "thinking", "message": "Dokumen kurang relevan, menulis ulang query..."})
            rewrite_llm = get_llm(streaming=False)
            rewritten = extract_text((REWRITE_PROMPT | rewrite_llm).invoke({"query": current_query}).content).strip()
            if rewritten:
                current_query = rewritten

        if not is_relevant:
            answer = "Maaf, informasi tersebut tidak ditemukan di dalam dokumen yang tersedia."
            yield _sse({"content": answer})
            yield _sse({"status": "done", "retry_count": retry_count, "pipeline": "LangGraph Agentic RAG"})
            yield "data: [DONE]\n\n"
            return

        yield _sse({"status": "thinking", "message": "Menyusun jawaban berbasis konteks..."})
        context = format_context(docs)
        stream_llm = get_llm(streaming=True)
        chain = GENERATE_PROMPT | stream_llm
        async for chunk in chain.astream({"query": original_query, "context": context}):
            token = extract_text(getattr(chunk, "content", chunk))
            if token:
                yield _sse({"content": token})

        yield _sse({"status": "done", "retry_count": retry_count, "pipeline": "LangGraph Agentic RAG"})
        yield "data: [DONE]\n\n"

    except Exception as e:
        traceback.print_exc()
        yield _sse({"error": str(e)})


@app.post("/api/v1/agent/chat", response_model=None)
async def agent_chat(payload: ChatRequest, _: bool = Depends(require_api_key)):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")

    try:
        if payload.stream:
            return StreamingResponse(
                stream_agentic_response(query),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        result = await _run_graph(query)
        return AgentChatResponse(
            answer=result.get("generation", ""),
            retry_count=int(result.get("retry_count", 0) or 0),
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail=f"Agentic pipeline gagal. Periksa konfigurasi LLM/DB. Detail: {exc}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
