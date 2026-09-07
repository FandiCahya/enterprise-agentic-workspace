import { useRef, useState, useCallback } from 'react';
import { Send, Square } from 'lucide-react';

export function ChatInput({ onSend, onCancel, status }) {
  const [value, setValue] = useState('');
  const textareaRef = useRef(null);

  const isStreaming = status === 'thinking' || status === 'streaming';

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed);
    setValue('');
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [value, isStreaming, onSend]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleInput = (e) => {
    setValue(e.target.value);
    // Auto resize
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 160) + 'px';
    }
  };

  return (
    <div className="chat-input-area">
      <div className="chat-input-container">
        <textarea
          ref={textareaRef}
          className="chat-textarea"
          rows={1}
          placeholder={isStreaming ? 'Menunggu respons...' : 'Tanyakan sesuatu tentang dokumen...'}
          value={value}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
        />

        {isStreaming ? (
          <button
            id="btn-cancel-stream"
            className="chat-send-btn"
            onClick={onCancel}
            style={{ background: 'linear-gradient(135deg,#ef4444,#b91c1c)' }}
            title="Batalkan"
          >
            <Square size={15} />
          </button>
        ) : (
          <button
            id="btn-send-message"
            className="chat-send-btn"
            onClick={handleSubmit}
            disabled={!value.trim()}
            title="Kirim (Enter)"
          >
            <Send size={15} />
          </button>
        )}
      </div>
      <p className="chat-input-hint">
        {isStreaming
          ? '⏳ Agentic RAG Pipeline sedang berjalan...'
          : 'Enter untuk kirim · Shift+Enter untuk baris baru'}
      </p>
    </div>
  );
}
