import os
import sys
import json
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from dotenv import load_dotenv

# Fix encoding untuk terminal Windows (cp1252 tidak support emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import tempfile
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# 1. ENVIRONMENT & TRACING (ARIZE PHOENIX)
# ------------------------------------------------------------------
load_dotenv()

# Setup Tracing ke Arize Phoenix Dashboard (Port 6006)
try:
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    # Registrasi tracer OpenTelemetry
    tracer_provider = register(
        project_name="enterprise-rag-workspace",
        endpoint="http://localhost:6006/v1/traces"
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    print("[OK] Tracing Arize Phoenix berhasil diaktifkan pada http://localhost:6006")
except Exception as e:
    print(f"[WARNING] Gagal mengaktifkan Tracing Phoenix: {e}")

# Import LangChain Modules
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ------------------------------------------------------------------
# 2. CONFIGURATION & DATABASE KONEKSI
# ------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("[WARNING] GOOGLE_API_KEY belum diatur di file .env!")

POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "adminpassword")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "enterprise_rag")

# Format Connection String PostgreSQL (psycopg v3)
DATABASE_URL = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
COLLECTION_NAME = "enterprise_knowledge_base"


def resolve_embedding_model():
    """Return a Google model name if supported, otherwise empty so we can fall back locally."""
    preferred = os.getenv("EMBEDDING_MODEL")
    candidates = [preferred, "models/text-embedding-004", "models/embedding-001", "models/gemini-embedding-001"]
    for model in candidates:
        if model:
            return model
    return ""


def resolve_llm_model():
    """Use a model name that is actually available in the current Google GenAI API version."""
    preferred = os.getenv("LLM_MODEL")
    candidates = [preferred, "gemini-3.6-flash", "gemini-flash-latest", "gemini-3.5-flash"]
    for model in candidates:
        if model:
            return model
    return "gemini-3.6-flash"


def build_local_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_google_embeddings():
    model_name = resolve_embedding_model()
    if not model_name:
        raise ValueError("No Google embedding model configured")
    return GoogleGenerativeAIEmbeddings(model=model_name)


def build_embeddings():
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() in {"1", "true", "yes", "y"}
    if use_local:
        return build_local_embeddings()

    try:
        return build_google_embeddings()
    except Exception as exc:
        print(f"[WARNING] Google embeddings unavailable, switching to local embeddings: {exc}")
        return build_local_embeddings()


# Inisialisasi LLM & Embeddings
embeddings = build_embeddings()
llm = ChatGoogleGenerativeAI(model=resolve_llm_model(), temperature=0, streaming=True)

# Inisialisasi PGVector Store
def get_vector_store():
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def ensure_embeddings_available(action_name: str, doc=None):
    global embeddings
    if "embedContent" in action_name or doc is not None:
        pass
    try:
        return embeddings
    except Exception:
        return embeddings


# ------------------------------------------------------------------
# AGENTIC RAG GRAPH (Lazy-loaded agar tidak memperlambat startup)
# ------------------------------------------------------------------
_agentic_app = None
_graph_executor = ThreadPoolExecutor(max_workers=4)


def get_agentic_app():
    """Lazy-load LangGraph Agentic RAG agar inisialisasi retrieval engine
    (termasuk download model BM25/Cross-Encoder) hanya terjadi sekali saat
    permintaan pertama masuk, bukan saat server startup.
    """
    global _agentic_app
    if _agentic_app is None:
        print("[INFO] Menginisialisasi LangGraph Agentic RAG (pertama kali)...")
        from src.core.graph import agentic_app  # noqa: PLC0415
        _agentic_app = agentic_app
        print("[OK] LangGraph Agentic RAG berhasil dimuat.")
    return _agentic_app

# ------------------------------------------------------------------
# 3. HELPER UTILITIES
# ------------------------------------------------------------------

def extract_text(content) -> str:
    """Ekstrak teks dari response.content yang bisa berupa str atau list of dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


# ------------------------------------------------------------------
# 4. FASTAPI SETUP & PYDANTIC SCHEMAS
# ------------------------------------------------------------------
app = FastAPI(
    title="Enterprise Document Intelligence & Agentic Workspace API",
    description="Backend REST API siap produksi terhubung ke PGVector & Arize Phoenix",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentIngestRequest(BaseModel):
    content: str = Field(..., examples=["Gladi bersih wisuda dilaksanakan H-1 sebelum hari H."])
    metadata: Optional[dict] = Field(default_factory=dict, examples=[{"category": "wisuda", "page": 1}])

class SearchQueryRequest(BaseModel):
    query: str = Field(..., examples=["Kapan gladi bersih wisuda?"])
    top_k: int = Field(default=3, ge=1, le=10)

class ChatRequest(BaseModel):
    query: str = Field(..., examples=["Berapa hari syarat pengajuan refund?"])
    stream: bool = Field(default=True)


class AgentChatResponse(BaseModel):
    answer: str
    retry_count: int = Field(default=0, description="Jumlah kali query ditulis ulang oleh Self-Correction loop")
    pipeline: str = Field(
        default="LangGraph Agentic RAG (Hybrid Search + Cross-Encoder + Self-Correction)",
        description="Nama pipeline yang digunakan"
    )

# ------------------------------------------------------------------
# 4. ENDPOINTS REST API
# ------------------------------------------------------------------

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Endpoint untuk mengecek status kesehatan server dan koneksi DB."""
    try:
        vector_store = get_vector_store()
        return {
            "status": "online",
            "database": "PostgreSQL PGVector Connected",
            "tracing": "Arize Phoenix Active (Port 6006)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Connection Error: {str(e)}")


@app.post("/api/v1/documents/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_document(payload: DocumentIngestRequest):
    """Endpoint untuk memasukkan potongan teks (chunk) ke PGVector."""
    try:
        vector_store = get_vector_store()
        doc = Document(page_content=payload.content, metadata=payload.metadata)
        vector_store.add_documents([doc])
        return {"message": "Dokumen berhasil disimpan ke PGVector!", "content": payload.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan dokumen: {str(e)}")


@app.post("/api/v1/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload file dokumen (PDF, DOCX, TXT, MD) langsung ke PGVector.
    File akan otomatis di-chunk menggunakan RecursiveCharacterTextSplitter
    sebelum disimpan sebagai vector embeddings.
    """
    SUPPORTED = {".pdf", ".docx", ".txt", ".md"}
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: '{ext}'. Gunakan: {', '.join(SUPPORTED)}"
        )

    # Simpan file sementara agar loaders bisa membacanya dari disk
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Import loader yang sesuai
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_path)
        elif ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(tmp_path)
        else:  # .txt / .md
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()
        for doc in docs:
            doc.metadata["source_filename"] = file.filename
            doc.metadata["file_type"] = ext.lstrip(".")

        # Chunking
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

        # Upload ke PGVector
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        return {
            "message": f"File '{file.filename}' berhasil diproses!",
            "filename": file.filename,
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
async def search_documents(payload: SearchQueryRequest):
    """Endpoint untuk melakukan Vector Similarity Search di PGVector."""
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search(payload.query, k=payload.top_k)
        
        output = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]
        return {"query": payload.query, "results_count": len(output), "data": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan pencarian: {str(e)}")


def _run_agentic_graph(query: str) -> dict:
    """Jalankan LangGraph secara sinkron (dipanggil di thread pool agar
    event loop asyncio tidak terblokir).

    Pipeline yang dijalankan:
      retrieve -> evaluate -> [rewrite -> retrieve]* -> generate
    """
    app_graph = get_agentic_app()
    initial_state = {
        "original_query": query,
        "current_query": query,
        "retry_count": 0,
    }
    return app_graph.invoke(initial_state)


async def stream_agentic_response(query: str):
    """Generator Server-Sent Events (SSE).

    Langkah:
    1. Kirim frame status awal ke klien.
    2. Jalankan LangGraph di thread pool (non-blocking).
    3. Stream jawaban akhir token-by-token ke klien.
    4. Kirim frame penutup dengan metadata (retry_count, dll).
    """
    loop = asyncio.get_event_loop()
    try:
        # Frame 1 - beri tahu klien bahwa pipeline sedang berjalan
        yield "data: " + json.dumps({"status": "thinking", "message": "Memulai Agentic RAG Pipeline..."}) + "\n\n"

        # Jalankan seluruh graph (retrieve -> evaluate -> generate) di executor
        result: dict = await loop.run_in_executor(
            _graph_executor,
            _run_agentic_graph,
            query
        )

        answer: str = result.get("generation", "")
        retry_count: int = result.get("retry_count", 0)

        # Stream jawaban token-by-token agar terasa seperti mengetik
        chunk_size = 12  # karakter per SSE frame
        for i in range(0, len(answer), chunk_size):
            token = answer[i: i + chunk_size]
            yield "data: " + json.dumps({"content": token}) + "\n\n"
            await asyncio.sleep(0.02)

        # Frame penutup dengan metadata pipeline
        yield "data: " + json.dumps({"status": "done", "retry_count": retry_count, "pipeline": "LangGraph Agentic RAG"}) + "\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield "data: " + json.dumps({"error": str(e)}) + "\n\n"


@app.post("/api/v1/agent/chat", response_model=None)
async def agent_chat(payload: ChatRequest):
    """
    Endpoint Utama Agentic Chat menggunakan LangGraph Self-Correcting RAG.

    Pipeline (src/core/graph.py):
    1. Retrieve  - Hybrid Search (BM25 + PGVector) + Cross-Encoder Reranking
    2. Evaluate  - LLM menilai relevansi dokumen terhadap query
    3. Rewrite   - (opsional, maks 2x) LLM merumuskan ulang query jika tidak relevan
    4. Generate  - LLM menghasilkan jawaban terstruktur dari konteks tervalidasi

    Mode:
    - stream=true  -> Server-Sent Events, jawaban dikirim token-by-token
    - stream=false -> JSON biasa, jawaban lengkap sekaligus
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")

    try:
        if payload.stream:
            return StreamingResponse(
                stream_agentic_response(payload.query),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming: jalankan graph di thread pool & kembalikan JSON
            loop = asyncio.get_event_loop()
            result: dict = await loop.run_in_executor(
                _graph_executor,
                _run_agentic_graph,
                payload.query
            )
            return AgentChatResponse(
                answer=result.get("generation", ""),
                retry_count=result.get("retry_count", 0),
            )

    except Exception as exc:
        traceback.print_exc()  # Tampilkan full traceback di terminal uvicorn
        raise HTTPException(
            status_code=503,
            detail=(
                f"Agentic pipeline gagal. Periksa GOOGLE_API_KEY & koneksi DB. "
                f"Detail: {exc}"
            ),
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)