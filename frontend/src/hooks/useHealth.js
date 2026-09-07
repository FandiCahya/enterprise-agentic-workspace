import { useState, useEffect, useCallback } from 'react';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

export function useHealth() {
  const [status, setStatus] = useState('checking'); // 'checking' | 'online' | 'offline'

  const check = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
      setStatus(res.ok ? 'online' : 'offline');
    } catch {
      setStatus('offline');
    }
  }, []);

  useEffect(() => {
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, [check]);

  return { status, refetch: check };
}
