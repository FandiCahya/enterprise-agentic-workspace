from __future__ import annotations

from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

from src.core.config import build_llm
from src.core.retrieval import AdvancedRetrievalEngine


def extract_text(content) -> str:
    """Extract text from provider responses that may be str or structured parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


# Lazy singletons
_llm = None
_retrieval_engine: AdvancedRetrievalEngine | None = None


def get_llm(streaming: bool = False):
    global _llm
    if streaming:
        # streaming LLM instances are cheap to create; keep non-streaming cached
        return build_llm(streaming=True)
    if _llm is None:
        _llm = build_llm(streaming=False)
    return _llm


def get_retrieval_engine() -> AdvancedRetrievalEngine:
    """Lazy-load AdvancedRetrievalEngine so heavy init happens on first use."""
    global _retrieval_engine
    if _retrieval_engine is None:
        print("[INFO] Menginisialisasi AdvancedRetrievalEngine (pertama kali)...")
        engine = AdvancedRetrievalEngine(rerank_top_n=3)
        engine.initialize_hybrid_search()
        _retrieval_engine = engine
        print("[OK] AdvancedRetrievalEngine siap digunakan.")
    return _retrieval_engine


EVAL_PROMPT = PromptTemplate.from_template(
    "Kamu adalah evaluator relevansi dokumen.\n"
    "Pertanyaan: {query}\n"
    "Dokumen Pendukung:\n{context}\n\n"
    "Apakah dokumen pendukung di atas memuat informasi yang cukup untuk menjawab pertanyaan? "
    "Jawab hanya dengan satu kata: 'YA' atau 'TIDAK'."
)

REWRITE_PROMPT = PromptTemplate.from_template(
    "Optimalkan pertanyaan berikut agar menjadi kata kunci pencarian dokumen teknis yang lebih efektif.\n"
    "Pertanyaan Asli: {query}\n"
    "Pertanyaan yang Dioptimalkan (hanya berikan kalimat barunya saja):"
)

GENERATE_PROMPT = PromptTemplate.from_template(
    "Kamu adalah asisten enterprise document intelligence yang akurat dan profesional.\n"
    "Jawab pertanyaan berikut secara terstruktur berdasarkan konteks yang diberikan.\n"
    "Jika informasi tidak terdapat pada konteks, katakan bahwa informasi tidak tersedia di dokumen.\n\n"
    "Konteks Dokumen:\n{context}\n\n"
    "Pertanyaan: {query}\n\n"
    "Jawaban:"
)


# ------------------------------------------------------------------
# 1. DEFINISI STATE GRAPH
# ------------------------------------------------------------------
class AgenticRAGState(TypedDict):
    original_query: str
    current_query: str
    documents: List[Document]
    generation: str
    retry_count: int
    is_relevant: bool


# ------------------------------------------------------------------
# 2. DEFINISI NODES (ACTION STEPS)
# ------------------------------------------------------------------
def format_context(docs: List[Document]) -> str:
    return "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])


def retrieve_node(state: AgenticRAGState):
    """Retrieve relevant docs via Hybrid Search + Cross-Encoder Reranker."""
    query = state["current_query"]
    print(f"\n🔍 [Graph Node] Retrieving docs for: '{query}'")
    docs = get_retrieval_engine().search(query)
    return {
        "documents": docs,
        "retry_count": state.get("retry_count", 0),
        "original_query": state.get("original_query", query),
        "current_query": query,
    }


def evaluate_documents_node(state: AgenticRAGState):
    """Evaluate whether retrieved chunks are sufficient to answer the query."""
    query = state["current_query"]
    docs = state.get("documents", [])

    print("⚖️ [Graph Node] Evaluating document relevance...")
    if not docs:
        return {"is_relevant": False}

    context = "\n\n".join([f"- {d.page_content}" for d in docs])
    llm = get_llm(streaming=False)
    chain = EVAL_PROMPT | llm
    response = extract_text(chain.invoke({"query": query, "context": context}).content).strip().upper()
    is_relevant = "YA" in response
    print(f"  └─ Status Relevansi: {'✅ RELEVAN' if is_relevant else '❌ TIDAK RELEVAN'}")
    return {"is_relevant": is_relevant}


def rewrite_query_node(state: AgenticRAGState):
    """Rewrite search query when initial documents are not relevant."""
    query = state["current_query"]
    retry = state.get("retry_count", 0) + 1
    print(f"✏️ [Graph Node] Rewriting query (Percobaan ke-{retry})...")
    llm = get_llm(streaming=False)
    chain = REWRITE_PROMPT | llm
    response = extract_text(chain.invoke({"query": query}).content).strip()
    print(f"  └─ Query Baru: '{response}'")
    return {"current_query": response or query, "retry_count": retry}


def generate_answer_node(state: AgenticRAGState):
    """Generate final answer using validated document context."""
    query = state.get("original_query") or state.get("current_query") or ""
    docs = state.get("documents", [])
    print("🤖 [Graph Node] Generating final verified answer...")
    context = format_context(docs)
    llm = get_llm(streaming=False)
    chain = GENERATE_PROMPT | llm
    response = chain.invoke({"query": query, "context": context})
    return {"generation": extract_text(response.content)}


def fast_fail_node(state: AgenticRAGState):
    """Menghasilkan pesan default jika dokumen tidak relevan setelah batas retry."""
    print("🚫 [Graph Node] Fast-fail: Dokumen tetap tidak relevan. Membatalkan generasi jawaban LLM.")
    return {"generation": "Maaf, informasi tersebut tidak ditemukan di dalam dokumen yang tersedia."}


# ------------------------------------------------------------------
# 3. CONDITIONAL ROUTING
# ------------------------------------------------------------------
def route_after_evaluation(state: AgenticRAGState):
    """Menentukan apakah lanjut ke penulisan jawaban atau pencarian ulang."""
    if state["is_relevant"]:
        return "generate"
    
    # Maksimal retry 1 kali agar tidak lambat
    if state.get("retry_count", 0) >= 1:
        print("  └─ Batas retry tercapai. Menghentikan pencarian dan mengembalikan respons default.")
        return "fast_fail"
    
    return "rewrite"


# ------------------------------------------------------------------
# 4. MEMBANGUN WORKFLOW GRAPH
# ------------------------------------------------------------------
def create_agentic_rag_graph():
    workflow = StateGraph(AgenticRAGState)
    
    # Daftarkan Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate", evaluate_documents_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("fast_fail", fast_fail_node)
    
    # Hubungkan Alur (Edges)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "evaluate")
    
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fast_fail": "fast_fail"
        }
    )
    
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("fast_fail", END)
    
    return workflow.compile()


# Instance Graph yang siap diimpor ke FastAPI
# Graph hanya dikompilasi sekali; retrieval_engine di-lazy-load saat request pertama.
agentic_app = create_agentic_rag_graph()


if __name__ == "__main__":
    # Test Workflow
    test_input = {
        "original_query": "Apa saja proyek atau pengalaman kerja yang pernah diselesaikan?",
        "current_query": "Apa saja proyek atau pengalaman kerja yang pernah diselesaikan?",
        "retry_count": 0
    }
    
    result = agentic_app.invoke(test_input)
    
    print("\n" + "="*50)
    print("💡 HASIL GENERASI LANGGRAPH AGENTIC RAG:")
    print("="*50)
    print(result["generation"])