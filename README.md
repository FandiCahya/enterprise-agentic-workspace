# Enterprise Agentic RAG Workspace

Production-grade modular Document Intelligence and Agentic RAG pipeline built with FastAPI, LangGraph, and PostgreSQL (PGVector).

## 🚀 Key Features
- **Document Intelligence Ingestion:** Auto-chunking for PDF, DOCX, and TXT with structured metadata.
- **Advanced Hybrid Search:** Combines dense semantic vectors (PGVector) and sparse lexical search (BM25) via Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Re-ranking:** Re-scores context chunks using `BAAI/bge-reranker-base`.
- **Self-Correcting Agentic Workflow:** Built on LangGraph with automatic document evaluation and dynamic query rewriting loops.
- **Observability & Tracing:** Integrated OpenTelemetry traces sent directly to Arize Phoenix.
- **Quantitative Evaluation:** Automated scoring for Faithfulness, Answer Relevancy, and Context Precision using RAGAS.

## 🛠️ Tech Stack
- **Framework:** FastAPI, LangChain, LangGraph
- **LLM & Embeddings:** Google Gemini 3.6-Flash, `text-embedding-004`
- **Vector Database:** PostgreSQL with PGVector
- **Monitoring & Eval:** Arize Phoenix, RAGAS, OpenTelemetry
- **Infrastructure:** Docker & Docker Compose

## ⚡ Quick Start

1. **Clone & Setup Environment**
   ```bash
   git clone <repo-url>
   cd enterprise-agentic-workspace
   python -m venv venv
   source venv/Scripts/activate  # atau .\venv\Scripts\Activate.ps1 di PowerShell
   pip install -r requirements.txt

2. Configure Environment Variables
    Salin .env.example ke .env dan isi kredensial

3. Start Infrastructure
   docker compose up -d

4. Ingest Documents
    Simpan berkas dokumen ke folder data/, lalu jalankan:
    python src/core/ingestion.py

5. Run FastAPI Server
    uvicorn src.main:app --reload --port 8000
    a. Swagger UI: http://localhost:8000/docs
    b. Arize Phoenix Dashboard: http://localhost:6006


## 📊 Pipeline Evaluation (RAGAS Benchmark)

Evaluasi kuantitatif dilakukan secara terotomatisasi menggunakan framework RAGAS dengan model evaluator Gemini 3.6-Flash:

| Metric | Score | Target | Status | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Faithfulness** | **1.0000** | > 0.85 | ✅ Passed | 0% Halusinasi, jawaban 100% grounded pada dokumen |
| **Answer Relevancy** | **0.8313** | > 0.80 | ✅ Passed | Jawaban sangat relevan dengan intent pengguna |
| **Context Recall** | **0.8500** | > 0.75 | ✅ Passed | Konteks faktual berhasil diekstrak oleh hybrid retriever |