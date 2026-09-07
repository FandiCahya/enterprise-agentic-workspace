import { useState, useRef, useCallback } from 'react';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

/**
 * useAgentChat
 * Manages SSE streaming chat with the LangGraph Agentic RAG backend.
 *
 * Messages shape: { id, role: 'user'|'assistant', content, meta?: { retry_count, pipeline } }
 */
export function useAgentChat() {
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState('idle'); // 'idle' | 'thinking' | 'streaming' | 'error'
  const abortRef = useRef(null);

  const appendMessage = useCallback((msg) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const updateLastAssistant = useCallback((updater) => {
    setMessages((prev) => {
      const next = [...prev];
      const idx = next.findLastIndex((m) => m.role === 'assistant');
      if (idx !== -1) next[idx] = updater(next[idx]);
      return next;
    });
  }, []);

  const sendMessage = useCallback(async (query) => {
    if (!query.trim() || status !== 'idle') return;

    // Cancel any in-flight stream
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    // Add user message
    const userMsg = { id: Date.now(), role: 'user', content: query };
    appendMessage(userMsg);

    // Placeholder for assistant
    const assistantId = Date.now() + 1;
    appendMessage({ id: assistantId, role: 'assistant', content: '', streaming: true });

    setStatus('thinking');

    try {
      const res = await fetch(`${API_BASE}/api/v1/agent/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, stream: true }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') break;

          try {
            const frame = JSON.parse(raw);

            if (frame.status === 'thinking') {
              setStatus('thinking');
            } else if (frame.content) {
              setStatus('streaming');
              updateLastAssistant((m) => ({
                ...m,
                content: m.content + frame.content,
              }));
            } else if (frame.status === 'done') {
              updateLastAssistant((m) => ({
                ...m,
                streaming: false,
                meta: {
                  retry_count: frame.retry_count ?? 0,
                  pipeline: frame.pipeline ?? 'LangGraph Agentic RAG',
                },
              }));
              setStatus('idle');
            } else if (frame.error) {
              throw new Error(frame.error);
            }
          } catch {
            // Non-JSON line, skip
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      updateLastAssistant((m) => ({
        ...m,
        content: m.content || `❌ Terjadi kesalahan: ${err.message}`,
        streaming: false,
        error: true,
      }));
      setStatus('idle');
    } finally {
      // Ensure streaming flag is cleared
      updateLastAssistant((m) => ({ ...m, streaming: false }));
      setStatus('idle');
    }
  }, [status, appendMessage, updateLastAssistant]);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    updateLastAssistant((m) => ({ ...m, streaming: false }));
    setStatus('idle');
  }, [updateLastAssistant]);

  const clearMessages = useCallback(() => setMessages([]), []);

  return { messages, status, sendMessage, cancel, clearMessages };
}
