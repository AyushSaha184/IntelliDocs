const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

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
