import { useEffect, useRef } from 'react';
import { MessageSquare, Sparkles } from 'lucide-react';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';

const EXAMPLE_PROMPTS = [
  'Kapan gladi bersih wisuda dilaksanakan?',
  'Apa saja persyaratan pengajuan refund?',
  'Jelaskan prosedur pendaftaran ulang.',
  'Berapa hari batas waktu pengumpulan tugas?',
];

export function ChatArea({ messages, status, onSend, onCancel }) {
  const bottomRef = useRef(null);

  // Auto-scroll on new message/token
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-wrapper">
      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 ? (
          <EmptyState onSelect={onSend} />
        ) : (
          messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <ChatInput onSend={onSend} onCancel={onCancel} status={status} />
    </div>
  );
}

function EmptyState({ onSelect }) {
  return (
    <div className="chat-empty">
      <div className="chat-empty-icon">
        <Sparkles size={28} color="#818cf8" />
      </div>
      <div className="chat-empty-title">Tanya dokumen Anda</div>
      <p style={{ fontSize: 13, color: 'var(--text-muted)', maxWidth: 360, textAlign: 'center' }}>
        Pipeline RAG akan mengambil konteks relevan dari knowledge base,
        mengevaluasi, dan menghasilkan jawaban yang grounded.
      </p>
      <div className="chat-empty-chips">
        {EXAMPLE_PROMPTS.map((p) => (
          <button
            key={p}
            className="chat-empty-chip"
            onClick={() => onSelect(p)}
          >
            <MessageSquare size={10} style={{ display: 'inline', marginRight: 5 }} />
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}
