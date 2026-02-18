// Centralized API URL configuration
// Points to local backend for demo (laptop runs server.py on port 8000)
const API_BASE = 'http://localhost:8000';

export const API_URL = API_BASE;
export const WS_URL = API_BASE.replace(/^http/, 'ws');

export default { API_URL, WS_URL };
