import os
import sys
import types
from pathlib import Path

# Fix encoding untuk terminal Windows (cp1252 tidak support emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Pastikan direktori root project masuk ke sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# Compatibility shim: ragas tries to import ChatVertexAI from langchain_community.chat_models.vertexai
# which was relocated/deprecated in newer LangChain versions.
try:
    from langchain_google_vertexai import ChatVertexAI
except Exception:
    class ChatVertexAI:
        pass

if "langchain_community.chat_models.vertexai" not in sys.modules:
    vertexai_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    vertexai_mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_mod

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.graph import agentic_app
from src.core.retrieval import build_embeddings

load_dotenv()

# Setup Evaluator Model & Embeddings menggunakan model yang didukung
evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
)
evaluator_embeddings = LangchainEmbeddingsWrapper(build_embeddings())


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

    print(f"[TEST] Menjalankan evaluasi pada {len(testset)} sampel pertanyaan...\n")

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
        # Kolom standar Ragas v0.2+
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
        # Kompatibilitas versi terdahulu
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    return Dataset.from_dict(dataset_dict)


# ------------------------------------------------------------------
# 2. EKSEKUSI EVALUASI RAGAS
# ------------------------------------------------------------------
def execute_ragas_evaluation():
    hf_dataset = run_pipeline_for_evaluation(eval_data_samples)

    print("\n[EVAL] Menghitung skor metrik RAGAS (Faithfulness, Relevance, Precision)...")
    
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
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
    print("[HASIL] SKOR AKHIR EVALUASI RAGAS:")
    print("="*60)
    print(results)
    print(f"\n[INFO] Laporan lengkap tersimpan di: {report_path}")
    
    return results


if __name__ == "__main__":
    execute_ragas_evaluation()