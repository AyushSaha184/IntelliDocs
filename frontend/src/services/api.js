import { getAccessToken } from './supabase';

// In production (Vercel), set VITE_API_URL to your backend API URL.
// In local development, this falls back to localhost:8000.
const BASE_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000');

async function buildAuthHeaders(extra = {}) {
    const token = await getAccessToken();
    const headers = { ...extra };
    if (token) {
        headers.Authorization = `Bearer ${token}`;
    }
    return headers;
}

export async function uploadDocument(file, sessionId = null) {
    const formData = new FormData();
    formData.append('file', file);

    // Add session_id if provided (to add file to existing session)
    if (sessionId) {
        formData.append('session_id', sessionId);
    }

    const headers = await buildAuthHeaders();
    const response = await fetch(`${BASE_URL}/api/upload`, {
        method: 'POST',
        headers,
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Upload failed (${response.status})`);
    }

    return response.json(); // Returns { session_id, status, filename, message }
}

export async function checkStatus(sessionId) {
    const headers = await buildAuthHeaders();
    const response = await fetch(`${BASE_URL}/api/status/${sessionId}`, { headers });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Status check failed (${response.status})`);
    }

    return response.json(); // Returns { session_id, status, filename, chunks_count }
}

export async function processSession(sessionId) {
    const headers = await buildAuthHeaders();
    const response = await fetch(`${BASE_URL}/api/process/${sessionId}`, {
        method: 'POST',
        headers,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Process failed (${response.status})`);
    }

    return response.json();
}

export async function askQuestion(sessionId, question, opts = {}) {
    const headers = await buildAuthHeaders({ 'Content-Type': 'application/json' });
    const response = await fetch(`${BASE_URL}/api/ask`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
            session_id: sessionId,
            question,
            top_k: opts.top_k || 5,
            ...opts
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Query failed (${response.status})`);
    }

    return response.json(); // Returns { answer, query, retrieved_chunks, ... }
}

export async function healthCheck() {
    const headers = await buildAuthHeaders();
    const response = await fetch(`${BASE_URL}/api/health`, { headers });

    if (!response.ok) {
        throw new Error(`Health check failed (${response.status})`);
    }

    return response.json();
}

/**
 * Stream an answer from the backend via SSE.
 *
 * @param {string} sessionId
 * @param {string} question
 * @param {object} opts  - Optional top_k, system_prompt, temperature, max_tokens
 * @param {function} onChunk    - Called with each token string as it arrives
 * @param {function} onMeta     - Called once with {sources, retrieval_scores, metadata}
 * @param {function} onSuccess  - Called once with {grounded, confidence}
 * @param {function} onWarning  - Called once with {grounded, confidence, message} if validation fails
 * @param {function} onError    - Called with an error message string
 */
export async function askQuestionStream(
    sessionId,
    question,
    opts = {},
    onChunk = () => { },
    onMeta = () => { },
    onSuccess = () => { },
    onWarning = () => { },
    onError = () => { },
) {
    const headers = await buildAuthHeaders({ 'Content-Type': 'application/json' });
    const response = await fetch(`${BASE_URL}/api/ask/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
            session_id: sessionId,
            question,
            top_k: opts.top_k || 5,
            ...opts,
        }),
    });

    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        onError(err.detail || `Stream failed (${response.status})`);
        return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete last line

        for (const line of lines) {
            if (!line.startsWith('data:')) continue;
            const raw = line.slice('data:'.length).trim();
            if (!raw) continue;
            try {
                const evt = JSON.parse(raw);
                if (evt.event === 'chunk') onChunk(evt.data);
                else if (evt.event === 'metadata') onMeta(evt.data);
                else if (evt.event === 'success') onSuccess(evt.data);
                else if (evt.event === 'warning') onWarning(evt.data);
                else if (evt.event === 'error') onError(evt.data);
            } catch (_) { /* ignore malformed lines */ }
        }
    }
}

