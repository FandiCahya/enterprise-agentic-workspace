import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_postgres import PGVector
try:
    from langchain_community.retrievers import BM25Retriever
except Exception:  # pragma: no cover - compatibility fallback
    from langchain_community.retrievers.bm25 import BM25Retriever

from src.core.config import (
    COLLECTION_NAME,
    DEFAULT_RERANKER_MODEL,
    build_embeddings,
    get_database_url,
    reranker_enabled,
)


# ------------------------------------------------------------------
# Native EnsembleRetriever (Reciprocal Rank Fusion)
# Menggantikan langchain.retrievers.ensemble yang tidak tersedia
# ------------------------------------------------------------------
class EnsembleRetriever(BaseRetriever):
    """Hybrid retriever menggunakan Reciprocal Rank Fusion (RRF)
    untuk menggabungkan hasil dari beberapa retriever."""

    retrievers: List[BaseRetriever]
    weights: List[float]
    c: int = 60  # Konstanta RRF standar

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Kumpulkan hasil dari setiap retriever
        all_results: List[List[Document]] = []
        for retriever in self.retrievers:
            try:
                docs = retriever.invoke(query)
            except Exception:
                docs = []
            all_results.append(docs)

        # Hitung skor RRF untuk setiap dokumen unik
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for retriever_idx, docs in enumerate(all_results):
            weight = self.weights[retriever_idx] if retriever_idx < len(self.weights) else 1.0
            for rank, doc in enumerate(docs):
                doc_id = doc.page_content  # Gunakan konten sebagai ID unik
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + weight * (1.0 / (self.c + rank + 1))
                doc_map[doc_id] = doc

        # Urutkan berdasarkan skor RRF tertinggi
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in ranked]


# ------------------------------------------------------------------
# Native CrossEncoderReranker
# Menggantikan langchain_community.document_compressors.CrossEncoderReranker
# yang sudah dihapus dari langchain-community
# ------------------------------------------------------------------
class CrossEncoderReranker:
    """Re-ranker berbasis Cross-Encoder menggunakan sentence_transformers langsung."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_n: int = 3):
        from sentence_transformers import CrossEncoder
        print(f"🔄 Memuat Cross-Encoder Model '{model_name}'...")
        self.model = CrossEncoder(model_name)
        self.top_n = top_n
        print(f"✅ Cross-Encoder '{model_name}' berhasil dimuat.")

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """Re-rank dokumen berdasarkan relevansi terhadap query."""
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        # Gabungkan dokumen dengan skor, urutkan descending
        scored_docs = sorted(
            zip(scores, documents),
            key=lambda x: float(x[0]),
            reverse=True
        )

        # Tambahkan skor ke metadata dan kembalikan top_n dokumen
        result = []
        for score, doc in scored_docs[: self.top_n]:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": round(float(score), 4)},
            )
            result.append(doc_copy)
        return result


# ------------------------------------------------------------------
# AdvancedRetrievalEngine
# Pipeline: Dense (PGVector) + Sparse (BM25) → RRF Fusion → Cross-Encoder Reranking
# ------------------------------------------------------------------
class AdvancedRetrievalEngine:
    def __init__(
        self,
        dense_k: int = 10,
        sparse_k: int = 10,
        rerank_top_n: int = 3,
        reranker_model_name: str = DEFAULT_RERANKER_MODEL,
    ):
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rerank_top_n = rerank_top_n
        self.database_url = get_database_url()
        self.reranker_enabled = reranker_enabled()

        # 1. Inisialisasi Embeddings & PGVector Store (konfigurasi terpusat)
        self.embeddings = build_embeddings()
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=COLLECTION_NAME,
            connection=self.database_url,
            use_jsonb=True,
        )

        # 2. Inisialisasi Dense Retriever (PGVector)
        self.dense_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.dense_k}
        )

        # 3. Inisialisasi Cross-Encoder Reranker (opsional, untuk mode lite/free tier)
        self.reranker: Optional[CrossEncoderReranker] = None
        if self.reranker_enabled:
            try:
                self.reranker = CrossEncoderReranker(
                    model_name=reranker_model_name,
                    top_n=self.rerank_top_n,
                )
            except Exception as exc:
                # Fallback agar API tetap bisa jalan pada environment terbatas
                print(f"[WARNING] Reranker tidak aktif, fallback tanpa reranking: {exc}")
                self.reranker = None
        else:
            print("[INFO] Reranker dinonaktifkan (ENABLE_RERANKER=false). Menggunakan hasil hybrid retrieval tanpa cross-encoder.")

        # Placeholder untuk Hybrid Retriever (diisi saat initialize_hybrid_search)
        self.hybrid_retriever: Optional[EnsembleRetriever] = None

    def initialize_hybrid_search(self, all_documents: Optional[List[Document]] = None):
        """
        Menyiapkan BM25 dan menggabungkannya dengan Dense PGVector menjadi Hybrid Retriever.
        Jika all_documents tidak diberikan, akan mengambil sampel dari database PGVector.
        """
        if not all_documents:
            print("📥 Mengambil dokumen dari PGVector untuk membangun indeks BM25...")
            # Best-effort: try to fetch a reasonably large sample for BM25 indexing
            sample_docs = self.vector_store.similarity_search("", k=500)
            if not sample_docs:
                raise ValueError(
                    "Database PGVector masih kosong. Jalankan ingestion terlebih dahulu."
                )
            all_documents = sample_docs

        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = self.sparse_k

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, self.dense_retriever],
            weights=[0.4, 0.6],
        )
        print("✅ Hybrid Search + Cross-Encoder Re-ranking siap digunakan!")

    def refresh(self):
        """Rebuild hybrid retriever (e.g. after new documents are ingested)."""
        self.hybrid_retriever = None
        self.initialize_hybrid_search()

    def search(self, query: str) -> List[Document]:
        """Eksekusi pencarian end-to-end: Dense + Sparse → RRF Fusion → (opsional) Cross-Encoder Reranking."""
        if not self.hybrid_retriever:
            self.initialize_hybrid_search()

        print(f"\n🔍 Menjalankan Hybrid Search untuk: '{query}'")

        # Step 1: Hybrid retrieval (BM25 + PGVector via RRF)
        candidate_docs = self.hybrid_retriever.invoke(query)

        # Step 2: Cross-Encoder re-ranking (opsional)
        if self.reranker:
            return self.reranker.compress_documents(candidate_docs, query)

        # Lite mode/fallback: ambil top-N hasil fusion
        return candidate_docs[: self.rerank_top_n]


if __name__ == "__main__":
    # Test Module Retrieval
    engine = AdvancedRetrievalEngine(rerank_top_n=3)
    engine.initialize_hybrid_search()

    test_query = "Apa keahlian utama dan pengalaman kerja yang dimiliki?"
    results = engine.search(test_query)

    print("\n" + "="*50)
    print(f"🏆 TOP {len(results)} HASIL RETRIEVAL & RE-RANKING:")
    print("="*50)
    for idx, doc in enumerate(results, 1):
        score = doc.metadata.get("relevance_score", "N/A")
        source = doc.metadata.get("source_filename", doc.metadata.get("source", "Unknown"))
        print(f"\n[Rank #{idx}] (Score: {score})")
        print(f"File Source : {source}")
        print(f"Content     : {doc.page_content[:250]}...")