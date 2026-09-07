import { useState } from 'react';
import { BarChart3, GitBranch, Trash2, TrendingUp, CheckCircle2 } from 'lucide-react';

const EVAL_METRICS = [
  {
    label: 'Faithfulness',
    value: 1.000,
    target: 0.85,
    desc: '0% Halusinasi · Grounded 100%',
    color: '#10b981',
    glow: 'rgba(16,185,129,0.2)',
  },
  {
    label: 'Answer Relevancy',
    value: 0.8313,
    target: 0.80,
    desc: 'Relevan dengan intent user',
    color: '#6366f1',
    glow: 'rgba(99,102,241,0.2)',
  },
  {
    label: 'Context Recall',
    value: 0.850,
    target: 0.75,
    desc: 'Konteks faktual terekstrak',
    color: '#a78bfa',
    glow: 'rgba(167,139,250,0.2)',
  },
];


const PIPELINE_STEPS = [
  { icon: '🔍', title: 'Hybrid Retrieval',      desc: 'BM25 + PGVector via Reciprocal Rank Fusion' },
  { icon: '⚖️', title: 'Cross-Encoder Re-rank', desc: 'BAAI/bge-reranker-base rescores chunks' },
  { icon: '🧠', title: 'LLM Evaluation',        desc: 'Gemini menilai relevansi konteks' },
  { icon: '🔄', title: 'Self-Correction',        desc: 'Auto query rewrite loop (maks 2×)' },
  { icon: '✨', title: 'Generation',             desc: 'Grounded answer dari konteks tervalidasi' },
];

/* ------------------------------------------------------------------ */
/* Compact metric card — clean & readable                              */
/* ------------------------------------------------------------------ */
function MetricCard({ metric }) {
  const pct = metric.value * 100;
  const targetPct = metric.target * 100;

  return (
    <div className="mc-card">
      {/* Top row: label + score */}
      <div className="mc-top">
        <span className="mc-label">{metric.label}</span>
        <span className="mc-score" style={{ color: metric.color }}>
          {pct.toFixed(1)}%
        </span>
      </div>

      {/* Progress track */}
      <div className="mc-track">
        {/* Target marker */}
        <div
          className="mc-target-mark"
          style={{ left: `${targetPct}%` }}
          title={`Target: ${targetPct.toFixed(0)}%`}
        />
        {/* Fill bar */}
        <div
          className="mc-fill"
          style={{
            width: `${Math.min(pct, 100)}%`,
            background: `linear-gradient(90deg, ${metric.color}99, ${metric.color})`,
            boxShadow: `0 0 8px ${metric.glow}`,
          }}
        />
      </div>

      {/* Bottom row: description + target badge */}
      <div className="mc-bottom">
        <span className="mc-desc">{metric.desc}</span>
        <span className="mc-target-badge">
          <CheckCircle2 size={9} />
          &gt;{targetPct.toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Panel                                                                */
/* ------------------------------------------------------------------ */
export function PipelinePanel({ lastRetryCount, messageCount, onClear }) {
  const [tab, setTab] = useState('scores');

  return (
    <div className="right-panel">
      <div className="panel">
        {/* Header */}
        <div className="panel-header">
          <div className="panel-title">⚡ Pipeline Info</div>
          <div className="tabs">
            <button
              className={`tab-btn ${tab === 'scores' ? 'active' : ''}`}
              onClick={() => setTab('scores')}
            >
              <TrendingUp size={10} style={{ display: 'inline', marginRight: 4 }} />
              Scores
            </button>
            <button
              className={`tab-btn ${tab === 'steps' ? 'active' : ''}`}
              onClick={() => setTab('steps')}
            >
              <GitBranch size={10} style={{ display: 'inline', marginRight: 4 }} />
              Steps
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="panel-body">

          {/* Session stats */}
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{messageCount}</div>
              <div className="stat-label">Messages</div>
            </div>
            <div className="stat-card">
              <div
                className="stat-value"
                style={lastRetryCount > 0
                  ? {
                      background: 'linear-gradient(135deg,#f59e0b,#ef4444)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                    }
                  : undefined}
              >
                {lastRetryCount}
              </div>
              <div className="stat-label">Retries</div>
            </div>
          </div>

          {/* ── SCORES TAB ── */}
          {tab === 'scores' && (
            <>
              <div className="section-divider">
                <BarChart3 size={10} style={{ display: 'inline', marginRight: 4 }} />
                RAGAS Benchmark · Gemini Flash
              </div>

              {EVAL_METRICS.map((m) => (
                <MetricCard key={m.label} metric={m} />
              ))}

              {/* Summary row */}
              <div className="mc-summary">
                <span className="mc-summary-dot" />
                <span>Semua metrik <strong>lulus</strong> · 0% halusinasi</span>
              </div>
            </>
          )}

          {/* ── STEPS TAB ── */}
          {tab === 'steps' && (
            <>
              <div className="section-divider">Pipeline Steps</div>
              {PIPELINE_STEPS.map((step) => (
                <div className="pipeline-step" key={step.title}>
                  <div className="pipeline-step-icon">{step.icon}</div>
                  <div className="pipeline-step-info">
                    <h4>{step.title}</h4>
                    <p>{step.desc}</p>
                  </div>
                </div>
              ))}

              <div className="section-divider">Tech Stack</div>
              {[
                ['🗄️', 'PostgreSQL + PGVector', 'Vector Database'],
                ['🔗', 'LangGraph',              'Agentic Workflow'],
                ['🤗', 'HuggingFace',            'Reranker · Embeddings'],
                ['📡', 'Arize Phoenix',           'Observability'],
              ].map(([icon, name, role]) => (
                <div key={name} className="tech-row">
                  <span className="tech-icon">{icon}</span>
                  <div>
                    <div className="tech-name">{name}</div>
                    <div className="tech-role">{role}</div>
                  </div>
                </div>
              ))}
            </>
          )}

          {/* Clear chat */}
          {messageCount > 0 && (
            <button
              id="btn-clear-chat"
              className="btn btn-secondary btn-full"
              onClick={onClear}
              style={{ marginTop: 4 }}
            >
              <Trash2 size={13} /> Hapus Riwayat Chat
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
