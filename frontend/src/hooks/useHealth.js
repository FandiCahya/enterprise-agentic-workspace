import { useState, useEffect, useCallback } from 'react';
import { apiUrl } from '../lib/api';

export function useHealth() {
  const [status, setStatus] = useState('checking'); // 'checking' | 'online' | 'offline'

  const check = useCallback(async () => {
    try {
      const res = await fetch(apiUrl('/health'), { signal: AbortSignal.timeout(4000) });
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

