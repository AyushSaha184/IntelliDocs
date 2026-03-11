// In production (Vercel), set VITE_API_URL to your backend API URL.
// In local development, this falls back to localhost:8000.
const BASE_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000');

export async function createSession() {
    const response = await fetch(`${BASE_URL}/api/session`, { method: 'POST' });
    if (!response.ok) throw new Error('Failed to create session');
    return response.json();
}

export function deleteSession(sessionId) {
    const url = `${BASE_URL}/api/session/${sessionId}`;
    fetch(url, { method: 'DELETE', keepalive: true }).catch(() => {});
}

export async function uploadDocument(file, sessionId, options = {}) {
    const formData = new FormData();
    formData.append('file', file);
    if (sessionId) formData.append('session_id', sessionId);

    const response = await fetch(`${BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
        signal: options.signal,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        const err = new Error(error.detail || `Upload failed (${response.status})`);
        err.error_code = error.error_code;
        throw err;
    }

    return response.json();
}

export async function listDocuments(sessionId) {
    const response = await fetch(`${BASE_URL}/api/documents/${sessionId}`);
    if (!response.ok) throw new Error('Failed to list documents');
    return response.json();
}

export async function checkStatus(sessionId) {
    const response = await fetch(`${BASE_URL}/api/status/${sessionId}`);
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        const err = new Error(error.detail || `Status check failed (${response.status})`);
        err.status = response.status;
        err.error_code = error.error_code;
        throw err;
    }
    return response.json();
}

export async function processSession(sessionId) {
    const response = await fetch(`${BASE_URL}/api/process/${sessionId}`, { method: 'POST' });
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Process failed (${response.status})`);
    }
    return response.json();
}

export async function healthCheck() {
    const response = await fetch(`${BASE_URL}/api/health`);
    if (!response.ok) throw new Error(`Health check failed (${response.status})`);
    return response.json();
}

export async function askQuestionStream(
    sessionId,
    question,
    opts = {},
    onChunk = () => {},
    onMeta = () => {},
    onSuccess = () => {},
    onWarning = () => {},
    onError = () => {},
) {
    const response = await fetch(`${BASE_URL}/api/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: sessionId,
            question,
            top_k: opts.top_k || 5,
            chat_history: opts.chat_history || [],
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
        buffer = lines.pop();

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
            } catch (_) {}
        }
    }
}