"""
Document Ingestion Pipeline
============================
Pipeline otomatis untuk membaca dokumen (PDF, DOCX, TXT, MD) dari folder data/,
memotong teks menggunakan RecursiveCharacterTextSplitter, lalu menyimpan
potongan tersebut secara batch ke PGVector.

Jalankan:
    venv\\Scripts\\python.exe src/core/ingestion.py
    venv\\Scripts\\python.exe src/core/ingestion.py --data-dir data --batch-size 50
"""
import warnings
# Suppress langchain-community deprecation warning — loaders masih aktif digunakan
# sampai standalone replacement tersedia untuk semua format (PDF, DOCX, TXT)
warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*",
    category=DeprecationWarning,
)

import os
import sys
import glob
import argparse
import time
from typing import List, Tuple

# Fix encoding di terminal Windows (cp1252 tidak support emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

# ------------------------------------------------------------------
# Konfigurasi
# ------------------------------------------------------------------
POSTGRES_USER     = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "adminpassword")
POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB       = os.getenv("POSTGRES_DB", "enterprise_rag")

DATABASE_URL    = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
COLLECTION_NAME = "enterprise_knowledge_base"

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]


# ------------------------------------------------------------------
# Helpers: Embedding & Loader
# ------------------------------------------------------------------

def build_embeddings():
    """Inisialisasi model embedding — ikuti setting USE_LOCAL_EMBEDDINGS di .env."""
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() in {"1", "true", "yes", "y"}

    if use_local:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback
        print("[INFO] Menggunakan embedding lokal: sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
        print(f"[INFO] Menggunakan Google Embeddings: {model}")
        return GoogleGenerativeAIEmbeddings(model=model)
    except Exception as exc:
        print(f"[WARNING] Google embeddings gagal, fallback ke lokal: {exc}")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_file(file_path: str) -> List[Document]:
    """Deteksi ekstensi dan muat dokumen dengan metadata halaman."""
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
        elif ext in (".txt", ".md"):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            print(f"  [SKIP] Ekstensi tidak didukung: {file_path}")
            return []

        docs = loader.load()
        # Tambahkan nama file ke metadata setiap halaman/dokumen
        for doc in docs:
            doc.metadata["source_filename"] = os.path.basename(file_path)
            doc.metadata["file_type"] = ext.lstrip(".")
        return docs

    except Exception as exc:
        print(f"  [ERROR] Gagal memuat '{os.path.basename(file_path)}': {exc}")
        return []


# ------------------------------------------------------------------
# Kelas Utama
# ------------------------------------------------------------------

class DocumentIngestionEngine:
    """
    Pipeline ingestion dokumen lengkap:
    1. Baca file dari folder data/
    2. Chunking dengan RecursiveCharacterTextSplitter
    3. Batch upload ke PGVector
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 50,
    ):
        self.chunk_size   = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size   = batch_size

        # Text splitter — pemisah berbasis struktur bab & paragraf
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n",   # Antar bab / section besar
                "\n\n",     # Antar paragraf
                "\n",       # Antar baris
                ". ",       # Antar kalimat
                " ",        # Antar kata (fallback)
                "",         # Karakter (last resort)
            ],
            is_separator_regex=False,
        )

        print("[INFO] Inisialisasi model embedding...")
        self.embeddings = build_embeddings()

        print("[INFO] Menghubungkan ke PGVector...")
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
            use_jsonb=True,
        )
        print("[OK] Siap melakukan ingestion.\n")

    # ----------------------------------------------------------
    # Chunking
    # ----------------------------------------------------------

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Pecah dokumen menjadi chunks & lengkapi metadata."""
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"]   = i
            chunk.metadata["chunk_size"]    = len(chunk.page_content)
            # Pastikan source_filename ada (fallback)
            if "source_filename" not in chunk.metadata:
                src = chunk.metadata.get("source", "unknown")
                chunk.metadata["source_filename"] = os.path.basename(src)
        return chunks

    # ----------------------------------------------------------
    # Batch Upload
    # ----------------------------------------------------------

    def _upload_batch(self, batch: List[Document], batch_num: int, total_batches: int):
        """Upload satu batch chunk ke PGVector dengan retry sederhana."""
        for attempt in range(1, 4):
            try:
                self.vector_store.add_documents(batch)
                print(
                    f"  [OK] Batch {batch_num}/{total_batches} "
                    f"({len(batch)} chunks) tersimpan."
                )
                return
            except Exception as exc:
                print(f"  [WARNING] Batch {batch_num} attempt {attempt} gagal: {exc}")
                if attempt < 3:
                    time.sleep(2)
        print(f"  [ERROR] Batch {batch_num} gagal setelah 3 percobaan, dilewati.")

    def upload_chunks(self, chunks: List[Document]):
        """Upload semua chunks ke PGVector secara batch."""
        if not chunks:
            print("[INFO] Tidak ada chunk untuk diupload.")
            return

        total    = len(chunks)
        batches  = [chunks[i : i + self.batch_size] for i in range(0, total, self.batch_size)]
        n_batch  = len(batches)

        print(f"[INFO] Mengupload {total} chunks dalam {n_batch} batch (ukuran batch: {self.batch_size})...")
        for idx, batch in enumerate(batches, start=1):
            self._upload_batch(batch, idx, n_batch)

    # ----------------------------------------------------------
    # Entry Point
    # ----------------------------------------------------------

    def ingest_directory(self, data_dir: str = "data") -> Tuple[int, int]:
        """
        Proses seluruh file di dalam direktori data/.

        Returns:
            (jumlah file diproses, jumlah chunk tersimpan)
        """
        # Buat folder jika belum ada
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"[INFO] Folder '{data_dir}' dibuat.")
            print(f"       Masukkan file PDF/DOCX/TXT/MD ke dalamnya, lalu jalankan ulang.")
            return 0, 0

        # Kumpulkan semua file yang didukung
        all_files: List[str] = []
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
            all_files.extend(glob.glob(os.path.join(data_dir, "**", f"*{ext}"), recursive=True))

        # Hapus duplikat & urutkan
        all_files = sorted(set(all_files))

        if not all_files:
            print(f"[INFO] Tidak ada dokumen di folder '{data_dir}'.")
            print(f"       Format yang didukung: {', '.join(SUPPORTED_EXTENSIONS)}")
            return 0, 0

        print(f"[INFO] Ditemukan {len(all_files)} file di '{data_dir}'.")
        print("-" * 60)

        # Proses setiap file
        all_chunks: List[Document] = []
        files_ok = 0
        for file_path in all_files:
            fname = os.path.basename(file_path)
            docs  = load_file(file_path)
            if not docs:
                continue

            chunks = self.split_documents(docs)
            all_chunks.extend(chunks)
            files_ok += 1
            pages = len(docs)
            print(f"  [+] {fname}: {pages} halaman → {len(chunks)} chunks")

        print("-" * 60)
        print(f"[INFO] Total: {files_ok} file, {len(all_chunks)} chunks dihasilkan.\n")

        # Upload ke PGVector
        self.upload_chunks(all_chunks)

        print("\n" + "=" * 60)
        print(f"[DONE] Ingestion selesai!")
        print(f"       - File diproses : {files_ok}")
        print(f"       - Chunk tersimpan: {len(all_chunks)}")
        print("=" * 60)
        return files_ok, len(all_chunks)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Document Ingestion Pipeline — simpan dokumen ke PGVector"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Folder sumber dokumen (default: data/)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Ukuran maksimal setiap chunk dalam karakter (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap antar chunk dalam karakter (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Jumlah chunk per batch saat upload ke PGVector (default: 50)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  Document Ingestion Pipeline")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Chunk size : {args.chunk_size} chars  |  Overlap: {args.chunk_overlap} chars")
    print(f"  Batch size : {args.batch_size} chunks")
    print("=" * 60 + "\n")

    engine = DocumentIngestionEngine(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    engine.ingest_directory(data_dir=args.data_dir)