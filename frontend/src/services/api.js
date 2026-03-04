// In production (Render), frontend is served by the same server — use relative URLs.
// In local dev, proxy to the backend on localhost:8000.
const BASE_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000');

export async function uploadDocument(file, sessionId = null) {
    const formData = new FormData();
    formData.append('file', file);

    // Add session_id if provided (to add file to existing session)
    if (sessionId) {
        formData.append('session_id', sessionId);
    }

    const response = await fetch(`${BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Upload failed (${response.status})`);
    }

    return response.json(); // Returns { session_id, status, filename, message }
}

export async function checkStatus(sessionId) {
    const response = await fetch(`${BASE_URL}/api/status/${sessionId}`);

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Status check failed (${response.status})`);
    }

    return response.json(); // Returns { session_id, status, filename, chunks_count }
}

export async function processSession(sessionId) {
    const response = await fetch(`${BASE_URL}/api/process/${sessionId}`, {
        method: 'POST',
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Process failed (${response.status})`);
    }

    return response.json();
}

export async function askQuestion(sessionId, question, opts = {}) {
    const response = await fetch(`${BASE_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
    const response = await fetch(`${BASE_URL}/api/health`);

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
    const response = await fetch(`${BASE_URL}/api/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
