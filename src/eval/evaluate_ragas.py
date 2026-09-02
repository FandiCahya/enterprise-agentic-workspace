import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.core.graph import agentic_app
from src.core.retrieval import AdvancedRetrievalEngine

load_dotenv()

# Setup Evaluator Model & Embeddings
evaluator_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
evaluator_embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

# Inisialisasi retrieval engine untuk mengambil context mentah
retrieval_engine = AdvancedRetrievalEngine(rerank_top_n=3)
retrieval_engine.initialize_hybrid_search()


# ------------------------------------------------------------------
# 1. MENYIAPKAN DATASET PENGUJIAN (GOLDEN TESTSET)
# ------------------------------------------------------------------
# Daftar pertanyaan uji, ground truth (kunci jawaban ideal), dan ekspektasi
eval_data_samples = [
    {
        "question": "Apa keahlian teknis dan teknologi utama yang dikuasai?",
        "ground_truth": "Keahlian utama mencakup Python, Django, Laravel, PostgreSQL, FastAPI, dan AI Engineering (RAG, Vector DB)."
    },
    {
        "question": "Jelaskan proyek sistem informasi atau aplikasi yang pernah dibangun.",
        "ground_truth": "Membangun sistem SI-RESPAN menggunakan Django untuk survei harga pangan dan SOPilot berbasis Laravel untuk alur submission testing software."
    },
    {
        "question": "Bagaimana prosedur verifikasi bebas tanggungan untuk wisuda?",
        "ground_truth": "Mahasiswa wajib mengunggah berkas bebas tanggungan perpustakaan dan administrasi ke portal sebelum pelaksanaan wisuda."
    }
]


def run_pipeline_for_evaluation(testset):
    """Menjalankan Agentic RAG untuk menghasilkan respons dan mengumpulkan konteks."""
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print(f"🧪 Menjalankan evaluasi pada {len(testset)} sampel pertanyaan...\n")

    for i, item in enumerate(testset, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"[{i}/{len(testset)}] Memproses query: '{q}'")

        # 1. Eksekusi Workflow LangGraph
        result = agentic_app.invoke({
            "original_query": q,
            "current_query": q,
            "retry_count": 0
        })

        generated_answer = result.get("generation", "")
        retrieved_docs = result.get("documents", [])
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        questions.append(q)
        answers.append(generated_answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(gt)

    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    return Dataset.from_dict(dataset_dict)


# ------------------------------------------------------------------
# 2. EKSEKUSI EVALUASI RAGAS
# ------------------------------------------------------------------
def execute_ragas_evaluation():
    hf_dataset = run_pipeline_for_evaluation(eval_data_samples)

    print("\n📊 Menghitung skor metrik RAGAS (Faithfulness, Relevance, Precision)...")
    
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]

    results = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # Konversi hasil evaluasi ke DataFrame Pandas
    df_results = results.to_pandas()
    
    # Simpan laporan evaluasi ke file CSV
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/ragas_evaluation_report.csv"
    df_results.to_csv(report_path, index=False)
    
    print("\n" + "="*60)
    print("🏆 HASIL SKOR AKHIR EVALUASI RAGAS:")
    print("="*60)
    print(results)
    print(f"\n📁 Laporan lengkap tersimpan di: {report_path}")
    
    return results


if __name__ == "__main__":
    execute_ragas_evaluation()