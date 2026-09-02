import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from src.core.retrieval import AdvancedRetrievalEngine

load_dotenv()


def extract_text(content) -> str:
    """Ekstrak teks dari response.content yang bisa berupa str atau list of dicts
    (format baru google-genai SDK)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in content]
        return "".join(parts)
    return str(content)

# Inisialisasi LLM (lazy retrieval_engine — diinisialisasi saat request pertama)
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
_retrieval_engine: AdvancedRetrievalEngine | None = None


def get_retrieval_engine() -> AdvancedRetrievalEngine:
    """Lazy-load AdvancedRetrievalEngine agar initialize_hybrid_search()
    (download model + query DB) tidak berjalan saat import modul."""
    global _retrieval_engine
    if _retrieval_engine is None:
        print("[INFO] Menginisialisasi AdvancedRetrievalEngine (pertama kali)...")
        engine = AdvancedRetrievalEngine(rerank_top_n=3)
        engine.initialize_hybrid_search()
        _retrieval_engine = engine
        print("[OK] AdvancedRetrievalEngine siap digunakan.")
    return _retrieval_engine


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
def retrieve_node(state: AgenticRAGState):
    """Mengambil dokumen relevan via Hybrid Search + Cross-Encoder Reranker."""
    query = state["current_query"]
    print(f"\n🔍 [Graph Node] Retrieving docs for: '{query}'")
    docs = get_retrieval_engine().search(query)
    return {
        "documents": docs,
        "retry_count": state.get("retry_count", 0),
        "original_query": state.get("original_query", query),
        "current_query": query
    }


def evaluate_documents_node(state: AgenticRAGState):
    """Menilai apakah potongan dokumen yang diambil cukup menjawab query."""
    query = state["current_query"]
    docs = state["documents"]
    
    print("⚖️ [Graph Node] Evaluating document relevance...")
    
    if not docs:
        return {"is_relevant": False}

    context = "\n\n".join([f"- {d.page_content}" for d in docs])
    
    prompt = PromptTemplate.from_template(
        "Kamu adalah evaluator relevansi dokumen.\n"
        "Pertanyaan: {query}\n"
        "Dokumen Pendukung:\n{context}\n\n"
        "Apakah dokumen pendukung di atas memuat informasi yang cukup untuk menjawab pertanyaan? "
        "Jawab hanya dengan satu kata: 'YA' atau 'TIDAK'."
    )
    
    chain = prompt | llm
    response = extract_text(chain.invoke({"query": query, "context": context}).content).strip().upper()
    is_relevant = "YA" in response
    
    print(f"  └─ Status Relevansi: {'✅ RELEVAN' if is_relevant else '❌ TIDAK RELEVAN'}")
    return {"is_relevant": is_relevant}


def rewrite_query_node(state: AgenticRAGState):
    """Merumuskan ulang query pencarian jika dokumen awal tidak relevan."""
    query = state["current_query"]
    retry = state.get("retry_count", 0) + 1
    
    print(f"✏️ [Graph Node] Rewriting query (Percobaan ke-{retry})...")
    
    prompt = PromptTemplate.from_template(
        "Optimalkan pertanyaan berikut agar menjadi kata kunci pencarian dokumen teknis yang lebih efektif.\n"
        "Pertanyaan Asli: {query}\n"
        "Pertanyaan yang Dioptimalkan (hanya berikan kalimat barunya saja):"
    )
    
    chain = prompt | llm
    response = extract_text(chain.invoke({"query": query}).content).strip()
    
    print(f"  └─ Query Baru: '{response}'")
    return {"current_query": response, "retry_count": retry}


def generate_answer_node(state: AgenticRAGState):
    """Menghasilkan jawaban akhir menggunakan konteks dokumen yang tervalidasi."""
    query = state["original_query"]
    docs = state["documents"]
    
    print("🤖 [Graph Node] Generating final verified answer...")
    
    context = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
    
    prompt = PromptTemplate.from_template(
        "Kamu adalah asisten enterprise document intelligence yang akurat dan profesional.\n"
        "Jawab pertanyaan berikut secara terstruktur berdasarkan konteks yang diberikan.\n"
        "Jika informasi tidak terdapat pada konteks, katakan bahwa informasi tidak tersedia di dokumen.\n\n"
        "Konteks Dokumen:\n{context}\n\n"
        "Pertanyaan: {query}\n\n"
        "Jawaban:"
    )
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context})
    return {"generation": extract_text(response.content)}


# ------------------------------------------------------------------
# 3. CONDITIONAL ROUTING
# ------------------------------------------------------------------
def route_after_evaluation(state: AgenticRAGState):
    """Menentukan apakah lanjut ke penulisan jawaban atau pencarian ulang."""
    if state["is_relevant"]:
        return "generate"
    
    # Maksimal retry 2 kali agar tidak terjadi looping tanpa batas
    if state.get("retry_count", 0) >= 2:
        print("  └─ Batas retry tercapai. Melanjutkan ke generasi jawaban dengan dokumen yang ada.")
        return "generate"
    
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
    
    # Hubungkan Alur (Edges)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "evaluate")
    
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    
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