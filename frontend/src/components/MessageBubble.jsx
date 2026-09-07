import { Bot, User, RotateCcw, GitBranch } from 'lucide-react';

export function MessageBubble({ message }) {
  const { role, content, streaming, meta, error } = message;
  const isUser = role === 'user';

  return (
    <div className={`message-row ${role}`}>
      {/* Avatar */}
      <div className="message-avatar">
        {isUser ? <User size={15} color="white" /> : <Bot size={15} color="#818cf8" />}
      </div>

      {/* Content */}
      <div className="message-content-wrap">
        {/* Thinking dots */}
        {!isUser && streaming && !content && (
          <div className="thinking-dots">
            <span /><span /><span />
          </div>
        )}

        {/* Message bubble */}
        {(content || isUser) && (
          <div className={`message-bubble ${error ? 'error' : ''}`}>
            {content}
            {streaming && content && <span className="cursor" />}
          </div>
        )}

        {/* Meta info */}
        {!isUser && meta && (
          <div className="message-meta">
            <span className="pipeline-badge">
              <GitBranch size={9} />
              {meta.pipeline ?? 'Agentic RAG'}
            </span>
            {meta.retry_count > 0 && (
              <span className="retry-badge">
                <RotateCcw size={9} />
                {meta.retry_count}x rewrite
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
