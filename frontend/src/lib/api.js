const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '');
const API_KEY = import.meta.env.VITE_API_KEY;

export function apiUrl(path) {
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE}${p}`;
}

export function apiHeaders(extra = {}) {
  const headers = { ...extra };
  if (API_KEY) {
    headers['x-api-key'] = API_KEY;
  }
  return headers;
}

export async function apiFetch(path, options = {}) {
  const { headers, ...rest } = options;
  return fetch(apiUrl(path), {
    ...rest,
    headers: apiHeaders(headers || {}),
  });
}
