// Centralized API URL configuration

// 1. If VITE_API_URL is set (e.g. Netlify), use it.
// 2. If we are in PROD (Render), use the current domain (window.location.origin).
// 3. If we are in DEV (Localhost), use http://localhost:8000.
const API_BASE = import.meta.env.VITE_API_URL
    || (import.meta.env.MODE === 'production' ? window.location.origin : 'http://localhost:8000');

export const API_URL = API_BASE;
export const WS_URL = API_BASE.replace(/^http/, 'ws').replace(/^https/, 'wss');

console.log(`Using API Backend: ${API_BASE} (Mode: ${import.meta.env.MODE})`);

export default { API_URL, WS_URL };
