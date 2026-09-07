import { Cpu, Zap } from 'lucide-react';
import { StatusBadge } from './StatusBadge';

export function Header({ healthStatus }) {
  return (
    <header className="header">
      {/* Logo */}
      <div className="header-logo">
        <div className="header-logo-icon">
          <Cpu size={16} color="white" />
        </div>
        <div>
          <div className="header-logo-title">
            <span className="gradient-text">Enterprise</span>
            {' '}Agentic Workspace
          </div>
          <div className="header-logo-sub">
            LangGraph · Hybrid RAG · Self-Correction
          </div>
        </div>
      </div>

      {/* Right side */}
      <div className="header-right">
        {/* Model badge */}
        <span className="status-badge checking" style={{ animationName: 'none' }}>
          <Zap size={10} />
          Gemini Flash · text-embedding-004
        </span>

        {/* Health */}
        <StatusBadge status={healthStatus} />
      </div>
    </header>
  );
}
