import './App.css';
import { useHealth } from './hooks/useHealth';
import { useAgentChat } from './hooks/useAgentChat';
import { Header } from './components/Header';
import { ChatArea } from './components/ChatArea';
import { DocumentPanel } from './components/DocumentPanel';
import { PipelinePanel } from './components/PipelinePanel';

function App() {
  const { status: healthStatus } = useHealth();
  const { messages, status, sendMessage, cancel, clearMessages } = useAgentChat();

  // Find last retry count from most recent assistant message
  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant' && m.meta);
  const lastRetryCount = lastAssistant?.meta?.retry_count ?? 0;
  const messageCount = messages.filter((m) => m.role === 'user').length;

  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <Header healthStatus={healthStatus} />

      {/* ── Content ── */}
      <div className="content-area">
        {/* Left sidebar — Document Manager */}
        <aside className="sidebar">
          <DocumentPanel />
        </aside>

        {/* Center — Chat */}
        <main className="chat-wrapper">
          <ChatArea
            messages={messages}
            status={status}
            onSend={sendMessage}
            onCancel={cancel}
          />
        </main>

        {/* Right panel — Pipeline Info */}
        <PipelinePanel
          lastRetryCount={lastRetryCount}
          messageCount={messageCount}
          onClear={clearMessages}
        />
      </div>
    </div>
  );
}

export default App;
