import { useState, useCallback, useRef } from 'react';
import {
  Upload, Search, FileText, AlertCircle, CheckCircle,
  FileUp, X, FileBadge, Layers
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

const FILE_ICONS = {
  pdf:  '📄',
  docx: '📝',
  txt:  '📃',
  md:   '📋',
};

export function DocumentPanel() {
  const [tab, setTab] = useState('upload'); // 'upload' | 'ingest' | 'search'

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">📂 Document Manager</div>
        <div className="tabs">
          <button
            id="tab-upload"
            className={`tab-btn ${tab === 'upload' ? 'active' : ''}`}
            onClick={() => setTab('upload')}
          >
            <FileUp size={10} style={{ display: 'inline', marginRight: 4 }} />
            Upload
          </button>
          <button
            id="tab-ingest"
            className={`tab-btn ${tab === 'ingest' ? 'active' : ''}`}
            onClick={() => setTab('ingest')}
          >
            <Layers size={10} style={{ display: 'inline', marginRight: 4 }} />
            Text
          </button>
          <button
            id="tab-search"
            className={`tab-btn ${tab === 'search' ? 'active' : ''}`}
            onClick={() => setTab('search')}
          >
            <Search size={10} style={{ display: 'inline', marginRight: 4 }} />
            Search
          </button>
        </div>
      </div>

      <div className="panel-body">
        {tab === 'upload' && <UploadTab />}
        {tab === 'ingest' && <IngestTab />}
        {tab === 'search' && <SearchTab />}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* UPLOAD TAB                                                           */
/* ------------------------------------------------------------------ */
function UploadTab() {
  const [dragOver, setDragOver] = useState(false);
  const [files, setFiles] = useState([]); // { file, status, result, error }
  const inputRef = useRef(null);

  const addFiles = useCallback((newFiles) => {
    const valid = Array.from(newFiles).filter(f =>
      /\.(pdf|docx|txt|md)$/i.test(f.name)
    );
    setFiles(prev => [
      ...prev,
      ...valid.map(f => ({ file: f, status: 'pending', result: null, error: null })),
    ]);
  }, []);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  };

  const handleFileInput = (e) => addFiles(e.target.files);

  const removeFile = (idx) => setFiles(prev => prev.filter((_, i) => i !== idx));

  const uploadFile = useCallback(async (idx) => {
    setFiles(prev => prev.map((f, i) =>
      i === idx ? { ...f, status: 'uploading' } : f
    ));

    const { file } = files[idx];
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/api/v1/documents/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? 'Upload gagal');
      setFiles(prev => prev.map((f, i) =>
        i === idx ? { ...f, status: 'done', result: data } : f
      ));
    } catch (e) {
      setFiles(prev => prev.map((f, i) =>
        i === idx ? { ...f, status: 'error', error: e.message } : f
      ));
    }
  }, [files]);

  const uploadAll = useCallback(async () => {
    const pending = files.map((f, i) => ({ ...f, idx: i })).filter(f => f.status === 'pending');
    for (const { idx } of pending) {
      await uploadFile(idx);
    }
  }, [files, uploadFile]);

  const hasPending = files.some(f => f.status === 'pending');

  return (
    <>
      {/* Drop zone */}
      <div
        className={`dropzone ${dragOver ? 'dragover' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          id="file-upload-input"
          type="file"
          accept=".pdf,.docx,.txt,.md"
          multiple
          style={{ display: 'none' }}
          onChange={handleFileInput}
        />
        <div className="dropzone-icon">
          <FileUp size={24} color="#818cf8" />
        </div>
        <div className="dropzone-text">
          <strong>Klik atau drag & drop</strong>
        </div>
        <div className="dropzone-hint">
          PDF · DOCX · TXT · MD — Auto-chunking & embedding
        </div>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <>
          <div className="section-divider">{files.length} file dipilih</div>

          {files.map((item, idx) => {
            const ext = item.file.name.split('.').pop().toLowerCase();
            return (
              <div key={idx} className={`upload-file-card ${item.status}`}>
                <div className="upload-file-icon">{FILE_ICONS[ext] ?? '📄'}</div>
                <div className="upload-file-info">
                  <div className="upload-file-name">{item.file.name}</div>
                  <div className="upload-file-size">
                    {(item.file.size / 1024).toFixed(1)} KB
                    {item.status === 'done' && item.result && (
                      <span className="upload-file-meta">
                        · {item.result.pages} hal · {item.result.chunks_stored} chunks
                      </span>
                    )}
                    {item.status === 'error' && (
                      <span style={{ color: 'var(--danger)', marginLeft: 4 }}>
                        {item.error}
                      </span>
                    )}
                  </div>
                </div>
                <div className="upload-file-status">
                  {item.status === 'pending' && (
                    <button
                      className="upload-btn-single"
                      onClick={(e) => { e.stopPropagation(); uploadFile(idx); }}
                      title="Upload file ini"
                    >
                      <Upload size={12} />
                    </button>
                  )}
                  {item.status === 'uploading' && <span className="spinner" style={{ borderTopColor: 'var(--accent)' }} />}
                  {item.status === 'done' && <CheckCircle size={16} color="var(--success)" />}
                  {item.status === 'error' && <AlertCircle size={16} color="var(--danger)" />}
                  {item.status !== 'uploading' && (
                    <button
                      className="upload-remove-btn"
                      onClick={(e) => { e.stopPropagation(); removeFile(idx); }}
                      title="Hapus"
                    >
                      <X size={12} />
                    </button>
                  )}
                </div>
              </div>
            );
          })}

          {hasPending && (
            <button
              id="btn-upload-all"
              className="btn btn-primary btn-full"
              onClick={uploadAll}
            >
              <Upload size={13} /> Upload Semua ({files.filter(f => f.status === 'pending').length} file)
            </button>
          )}
        </>
      )}

      {/* Info card */}
      <div className="info-card">
        <FileBadge size={12} style={{ flexShrink: 0 }} />
        <div>
          File akan di-<strong>chunk</strong> otomatis (1000 char / 200 overlap), 
          lalu di-embed dan disimpan ke <strong>PGVector</strong>.
        </div>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* TEXT INGEST TAB                                                      */
/* ------------------------------------------------------------------ */
function IngestTab() {
  const [content, setContent] = useState('');
  const [metaStr, setMetaStr] = useState('{"category": "general"}');
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);

  const showToast = (type, message) => {
    setToast({ type, message });
    setTimeout(() => setToast(null), 3000);
  };

  const handleIngest = useCallback(async () => {
    if (!content.trim()) return;
    setLoading(true);
    try {
      let metadata = {};
      try { metadata = JSON.parse(metaStr); } catch { /* invalid JSON */ }

      const res = await fetch(`${API_BASE}/api/v1/documents/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: content.trim(), metadata }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? 'Gagal menyimpan');
      }

      showToast('success', 'Teks berhasil disimpan ke PGVector!');
      setContent('');
    } catch (e) {
      showToast('error', e.message);
    } finally {
      setLoading(false);
    }
  }, [content, metaStr]);

  return (
    <>
      {toast && (
        <div className={`toast ${toast.type}`}>
          {toast.type === 'success' ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
          {toast.message}
        </div>
      )}

      <div className="form-group">
        <label className="form-label">
          <FileText size={10} style={{ display: 'inline', marginRight: 4 }} />
          Konten Teks (chunk)
        </label>
        <textarea
          id="ingest-content"
          className="form-textarea"
          placeholder="Paste potongan teks ke sini..."
          value={content}
          onChange={(e) => setContent(e.target.value)}
          rows={5}
          style={{ minHeight: 100 }}
        />
      </div>

      <div className="form-group">
        <label className="form-label">Metadata (JSON)</label>
        <textarea
          id="ingest-metadata"
          className="form-textarea"
          value={metaStr}
          onChange={(e) => setMetaStr(e.target.value)}
          rows={2}
          style={{ minHeight: 56, fontFamily: 'monospace', fontSize: 12 }}
        />
      </div>

      <button
        id="btn-ingest-submit"
        className="btn btn-primary btn-full"
        onClick={handleIngest}
        disabled={loading || !content.trim()}
      >
        {loading ? <><span className="spinner" /> Menyimpan...</> : <><Upload size={13} /> Simpan ke Knowledge Base</>}
      </button>
    </>
  );
}

/* ------------------------------------------------------------------ */
/* SEARCH TAB                                                           */
/* ------------------------------------------------------------------ */
function SearchTab() {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(3);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/documents/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), top_k: topK }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? 'Pencarian gagal');
      }
      const data = await res.json();
      setResults(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [query, topK]);

  return (
    <>
      <div className="form-group">
        <label className="form-label">Query Pencarian</label>
        <input
          id="search-query"
          type="text"
          className="form-input"
          placeholder="Cari di knowledge base..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        />
      </div>

      <div className="form-group">
        <label className="form-label">Top-K &nbsp;<span style={{ color: 'var(--accent-hover)' }}>{topK}</span></label>
        <div className="range-wrap">
          <input id="search-top-k" type="range" className="form-range" min={1} max={10} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
          <span className="range-value">{topK}</span>
        </div>
      </div>

      <button
        id="btn-search-submit"
        className="btn btn-primary btn-full"
        onClick={handleSearch}
        disabled={loading || !query.trim()}
      >
        {loading ? <><span className="spinner" /> Mencari...</> : <><Search size={13} /> Cari Dokumen</>}
      </button>

      {error && (
        <div style={{ padding: '10px 12px', background: 'var(--danger-bg)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 'var(--radius-md)', fontSize: 12, color: 'var(--danger)' }}>
          <AlertCircle size={12} style={{ display: 'inline', marginRight: 6 }} />{error}
        </div>
      )}

      {results && (
        <>
          <div className="section-divider">{results.results_count} Hasil</div>
          {results.data.length === 0 ? (
            <p style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center' }}>Tidak ada dokumen relevan.</p>
          ) : (
            results.data.map((item, idx) => (
              <div className="search-result-card" key={idx}>
                <div className="search-result-idx">Hasil #{idx + 1}</div>
                <div className="search-result-content">
                  {item.content.length > 200 ? item.content.slice(0, 200) + '…' : item.content}
                </div>
                {item.metadata && Object.keys(item.metadata).length > 0 && (
                  <div className="search-result-meta">{JSON.stringify(item.metadata)}</div>
                )}
              </div>
            ))
          )}
        </>
      )}
    </>
  );
}
