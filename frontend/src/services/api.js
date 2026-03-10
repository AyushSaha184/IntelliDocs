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

export async function createGuestSession() {
    const headers = await buildAuthHeaders({ 'Content-Type': 'application/json' });
    const response = await fetch(`${BASE_URL}/api/guest/session`, {
        method: 'POST',
        headers,
    });
    if (!response.ok) throw new Error('Failed to create guest session');
    return response.json();
}

export function deleteGuestSession(sessionId) {
    const url = `${BASE_URL}/api/guest/session/${sessionId}`;
    fetch(url, { method: 'DELETE', keepalive: true }).catch(() => {});
}

export async function createChat(title = 'New Chat', sessionId = null) {
    const headers = await buildAuthHeaders({ 'Content-Type': 'application/json' });
    const body = { title };
    if (sessionId) body.session_id = sessionId;

    const response = await fetch(`${BASE_URL}/api/chats`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to create chat');
    }
    return response.json();
}

export async function listChats(limit = 50, offset = 0) {
    const headers = await buildAuthHeaders();
    const response = await fetch(`${BASE_URL}/api/chats?limit=${limit}&offset=${offset}`, { headers });
    if (!response.ok) throw new Error('Failed to list chats');
    return response.json();
}

export async function renameChat(chatId, title, version) {
    const headers = await buildAuthHeaders({ 'Content-Type': 'application/json' });
    const response = await fetch(`${BASE_URL}/api/chats/${chatId}`, {
        method: 'PATCH',
        headers,
        body: JSON.stringify({ title, version }),
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || err.error_code || 'Rename failed');
    }
    return response.json();
}

export async function deleteChat(chatId, sessionId = null) {
    const headers = await buildAuthHeaders();
    const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    const response = await fetch(`${BASE_URL}/api/chats/${chatId}${qs}`, {
        method: 'DELETE',
        headers,
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Delete failed');
    }
    return response.json();
}

export async function listMessages(chatId, limit = 200, offset = 0, sessionId = null) {
    const headers = await buildAuthHeaders();
    const sessionParam = sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : '';
    const response = await fetch(
        `${BASE_URL}/api/chats/${chatId}/messages?limit=${limit}&offset=${offset}${sessionParam}`,
        { headers }
    );
    if (!response.ok) throw new Error('Failed to list messages');
    return response.json();
}

export async function uploadDocument(file, chatId = null, sessionId = null, options = {}) {
    const formData = new FormData();
    formData.append('file', file);
    if (chatId) formData.append('chat_id', chatId);
    if (sessionId) formData.append('session_id', sessionId);

    const headers = await buildAuthHeaders();
    const response = await fetch(`${BASE_URL}/api/upload`, {
        method: 'POST',
        headers,
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

export async function listChatDocuments(chatId, sessionId = null) {
    const headers = await buildAuthHeaders();
    const sessionParam = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    const response = await fetch(`${BASE_URL}/api/chats/${chatId}/documents${sessionParam}`, { headers });
    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to list chat documents');
    }
    return response.json();
}

export async function clearChatFiles(chatId, sessionId = null) {
    const headers = await buildAuthHeaders();
    const sessionParam = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    const response = await fetch(`${BASE_URL}/api/chats/${chatId}/clear-files${sessionParam}`, {
        method: 'POST',
        headers,
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to clear chat files');
    }
    return response.json();
}

export async function checkStatus(chatId, sessionId = null) {
    const headers = await buildAuthHeaders();
    const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    const response = await fetch(`${BASE_URL}/api/status/${chatId}${qs}`, { headers });
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        const err = new Error(error.detail || `Status check failed (${response.status})`);
        err.status = response.status;
        err.error_code = error.error_code;
        throw err;
    }
    return response.json();
}

export async function processSession(chatId, sessionId = null) {
    const headers = await buildAuthHeaders();
    const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    const response = await fetch(`${BASE_URL}/api/process/${chatId}${qs}`, {
        method: 'POST',
        headers,
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Process failed (${response.status})`);
    }
    return response.json();
}

export async function askQuestion(chatId, question, opts = {}) {
    const headers = await buildAuthHeaders({ 'Content-Type': 'application/json' });
    const response = await fetch(`${BASE_URL}/api/ask`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
            chat_id: chatId,
            session_id: opts.session_id,
            question,
            top_k: opts.top_k || 5,
            ...opts,
        }),
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Query failed (${response.status})`);
    }
    return response.json();
}

export async function healthCheck() {
    const headers = await buildAuthHeaders();
    const response = await fetch(`${BASE_URL}/api/health`, { headers });
    if (!response.ok) throw new Error(`Health check failed (${response.status})`);
    return response.json();
}

export async function askQuestionStream(
    chatId,
    question,
    opts = {},
    onChunk = () => {},
    onMeta = () => {},
    onSuccess = () => {},
    onWarning = () => {},
    onError = () => {},
) {
    const headers = await buildAuthHeaders({ 'Content-Type': 'application/json' });
    const response = await fetch(`${BASE_URL}/api/ask/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
            chat_id: chatId,
            session_id: opts.session_id,
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