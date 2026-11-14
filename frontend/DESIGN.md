Berikut adalah draf spesifikasi dokumen **`DESIGN.md`** yang bisa kamu simpan di root proyek atau di dalam folder `frontend/` sebagai panduan arsitektur antarmuka pengguna:

---

# Design Specification: Enterprise Agentic RAG Workspace Frontend

Dokumen ini menjelaskan arsitektur antarmuka pengguna (UI/UX), tata letak komponen, hierarki state, dan kontrak integrasi API untuk frontend Enterprise Agentic Workspace.

---

## 1. Desain Filosofi & Prinsip UI/UX

* **Enterprise Minimalist:** Tema gelap (*slate-950*) dengan kontras tinggi untuk keterbacaan dokumen teknis yang panjang.
* **Transparansi Agen (Explainable AI):** Setiap jawaban wajib memperlihatkan metadata sitasi (*source document* dan *chunk index*), status evaluasi dokumen, dan penanda latensi.
* **Separation of Concerns:** Pemisahan tegas antara area interaksi percakapan (*Workspace Main Chat*) dan area ingestion/pemantauan infrastruktur (*Knowledge Ingestion Sidebar*).

---

## 2. Tata Letak Antarmuka (Layout Architecture)

Aplikasi mengadopsi struktur dua panel horizontal (*split-screen layout*):

```
+------------------------+-------------------------------------------------------------+
| SIDEBAR (w-80)         | MAIN WORKSPACE (flex-1)                                     |
|                        |                                                             |
| [Logo & Brand]         | [Header: Document Intelligence Status & Model Indicators]   |
|                        +-------------------------------------------------------------+
| [Quick Ingest Module]  |                                                             |
| - Textarea Input       | [Scrollable Chat History Area]                              |
| - Category Selector    | - User Message Bubble                                       |
| - Ingest Action Button | - Assistant Message Bubble (Markdown formatted)             |
|                        |   - Citations & Source Badges                               |
| [System Health Deck]   |   - Agentic Self-Correction indicator (Retry loop state)    |
| - FastAPI Status       |                                                             |
| - PGVector Metric      +-------------------------------------------------------------+
| - Phoenix Observability| [Prompt Input & Action Bar]                                 |
+------------------------+-------------------------------------------------------------+

```

---

## 3. Rincian Komponen & Hierarki

| Komponen | Path Berkas | Peran & Tanggung Jawab |
| --- | --- | --- |
| **`App.jsx`** | `src/App.jsx` | Layout root container, manajemen state global, dan orkestrasi event fetching. |
| **`Sidebar.jsx`** | `src/components/Sidebar.jsx` | Kontrol knowledge ingestion manual, tombol upload file, dan indikator service health. |
| **`ChatContainer.jsx`** | `src/components/ChatContainer.jsx` | Container scroll otomatis untuk pesan masuk dan indikator proses generasi. |
| **`MessageBubble.jsx`** | `src/components/MessageBubble.jsx` | Render tipografi teks chat, bubble styling per *role*, serta rendering sitasi dokumen. |
| **`SourceBadge.jsx`** | `src/components/SourceBadge.jsx` | Chip komponen kecil yang menampilkan nama file sumber dan index potongan teks (*chunk index*). |
| **`ChatInput.jsx`** | `src/components/ChatInput.jsx` | Input teks responsif dengan listener tombol `Enter` dan tombol kirim interaktif. |

---

## 4. Desain State & Data Flow

### A. State Diagram

```
[User Types Query] 
        │
        ▼
[Set isLoading = True] ──► [Show Loader with Agent Status: "Retrieving & Evaluating..."]
        │
        ▼
[POST /api/v1/agent/chat]
        │
 ┌──────┴────────────────────────┐
 │ Sukses                        │ Gagal
 ▼                               ▼
[Append Message with Citations]  [Show Error Toast / Fallback Bubble]
 │                               │
 └──────┬────────────────────────┘
        ▼
[Set isLoading = False] ──► [Auto Scroll to Bottom]

```

### B. Struktur Data Pesan (`messages`)

```typescript
interface SourceCitation {
  source: string;        // Nama dokumen (e.g. CV_Muhammad_Fandi.pdf)
  chunk_index: number;   // Indeks potongan dokumen
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: SourceCitation[];
  timestamp?: string;
}

```

---

## 5. Kontrak Integrasi API Backend

### A. Chat Agentic Endpoint

* **Endpoint:** `POST /api/v1/agent/chat`
* **Request Payload:**
```json
{
  "query": "Apa saja proyek atau pengalaman kerja yang pernah diselesaikan?",
  "stream": false
}

```


* **Response Payload (200 OK):**
```json
{
  "query": "Apa saja proyek...",
  "answer": "Berdasarkan dokumen, proyek yang pernah dikembangkan meliputi...",
  "sources": [
    {
      "source": "CV_Muhammad_Fandi_Nurcahya.pdf",
      "chunk_index": 2
    }
  ]
}

```



### B. Manual Knowledge Ingestion Endpoint

* **Endpoint:** `POST /api/v1/documents/ingest`
* **Request Payload:**
```json
{
  "content": "Isi dokumen teks baru untuk dimasukkan ke dalam PGVector...",
  "metadata": {
    "category": "manual_input",
    "source_filename": "dashboard_upload"
  }
}

```


* **Response Payload (200 OK):**
```json
{
  "status": "success",
  "chunks_added": 1,
  "message": "Dokumen berhasil di-vektorisasi dan disimpan."
}

```



---

## 6. Desain Token Warna & Tipografi (Tailwind CSS)

* **Background Utama:** `bg-slate-950` (Hitam pekat kebiruan)
* **Panel & Card:** `bg-slate-900` dengan border `border-slate-800`
* **Aksen Interaktif:** `bg-indigo-600` (hover: `bg-indigo-500`)
* **Status Berhasil / Sehat:** `text-emerald-400` / `bg-emerald-500/10`
* **Tipografi:** Sans-serif bawaan (Inter / Roboto), ukuran `text-sm` untuk chat umum dan `text-xs` untuk metadata dokumen.