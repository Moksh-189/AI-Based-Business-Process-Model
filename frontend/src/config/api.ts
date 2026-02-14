// Centralized API URL configuration
// Uses VITE_API_URL env variable in production, falls back to local dev server
const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export const API_URL = API_BASE;
export const WS_URL = API_BASE.replace(/^http/, 'ws');

export default { API_URL, WS_URL };
