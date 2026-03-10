import { useState, useRef, useEffect, useCallback } from 'react';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import {
    uploadDocument, checkStatus, askQuestionStream, processSession,
    createGuestSession, deleteGuestSession,
    createChat, listChats, deleteChat, listMessages,
} from './services/api';
import { supabase } from './services/supabase';

export default function App() {
    const [messages, setMessages] = useState([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isProcessed, setIsProcessed] = useState(false);

    // Chat state
    const [activeChatId, setActiveChatId] = useState(null);
    const [chatList, setChatList] = useState([]);
    const [showHistory, setShowHistory] = useState(false);

    // Auth / guest state
    const [user, setUser] = useState(null);
    const [guestSessionId, setGuestSessionId] = useState(null);
    const [showAuthModal, setShowAuthModal] = useState(true);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [authError, setAuthError] = useState('');
    const [authLoading, setAuthLoading] = useState(false);
    const [showEmailAuth, setShowEmailAuth] = useState(false);

    // Upload error state (only shown when limit hit)
    const [uploadError, setUploadError] = useState(null);

    const messagesEndRef = useRef(null);
    const guestSessionRef = useRef(null);

    const isGuest = !user;

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Auth listener
    useEffect(() => {
        if (!supabase) return;

        let mounted = true;
        supabase.auth.getSession().then(({ data }) => {
            if (!mounted) return;
            const u = data?.session?.user || null;
            setUser(u);
            if (u) setShowAuthModal(false);
        });

        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
            const u = session?.user || null;
            setUser(u);
            if (u) setShowAuthModal(false);
        });

        return () => {
            mounted = false;
            subscription.unsubscribe();
        };
    }, []);

    const initGuestSession = useCallback(async () => {
        if (guestSessionId) return guestSessionId;
        const data = await createGuestSession();
        const sid = data.session_id;
        setGuestSessionId(sid);
        guestSessionRef.current = sid;
        return sid;
    }, [guestSessionId]);

    // Cleanup guest session on tab close
    useEffect(() => {
        const handleBeforeUnload = () => {
            const sid = guestSessionRef.current;
            if (sid) deleteGuestSession(sid);
        };
        window.addEventListener('beforeunload', handleBeforeUnload);
        return () => window.removeEventListener('beforeunload', handleBeforeUnload);
    }, []);

    // Load chat list for logged-in users
    const refreshChats = useCallback(async () => {
        if (isGuest) return;
        try {
            const data = await listChats();
            setChatList(data.chats || []);
        } catch { /* ignore */ }
    }, [isGuest]);

    useEffect(() => {
        if (!isGuest) refreshChats();
    }, [isGuest, refreshChats]);

    // Load messages when switching chats
    useEffect(() => {
        if (!activeChatId) return;
        let cancelled = false;
        (async () => {
            try {
                const data = await listMessages(activeChatId, 200, 0, isGuest ? guestSessionId : null);
                if (cancelled) return;
                setMessages((data.messages || []).map(m => ({
                    role: m.role,
                    content: m.content,
                    timestamp: new Date(m.created_at).getTime(),
                })));
            } catch { /* ignore */ }
        })();
        return () => { cancelled = true; };
    }, [activeChatId, isGuest, guestSessionId]);

    const handleSignIn = async () => {
        if (!supabase) return;
        setAuthError('');
        setAuthLoading(true);
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) setAuthError(error.message);
        setAuthLoading(false);
    };

    const handleSignUp = async () => {
        if (!supabase) return;
        setAuthError('');
        setAuthLoading(true);
        const { error } = await supabase.auth.signUp({ email, password });
        if (error) setAuthError(error.message);
        setAuthLoading(false);
    };

    const handleSignOut = async () => {
        if (!supabase) return;
        await supabase.auth.signOut();
        handleClear();
    };

    const handleGoogleSignIn = async () => {
        if (!supabase) return;
        setAuthError('');
        setAuthLoading(true);
        const { error } = await supabase.auth.signInWithOAuth({
            provider: 'google',
            options: {
                redirectTo: window.location.origin,
            },
        });
        if (error) {
            setAuthError(error.message);
            setAuthLoading(false);
        }
    };

    const handleSend = async (text) => {
        if (!text.trim() || isGenerating || !isProcessed || !activeChatId) return;

        const userMessage = { role: 'user', content: text, timestamp: Date.now() };
        setMessages(prev => [...prev.filter(m => !(m.role === 'system' && m.isSuccess)), userMessage]);
        setIsGenerating(true);

        const msgId = Date.now();
        setMessages(prev => [...prev, {
            role: 'assistant',
            id: msgId,
            content: '',
            timestamp: msgId,
            isStreaming: true,
            verifying: true,
            sources: [],
        }]);

        const updateMsg = (patch) =>
            setMessages(prev => prev.map(m => m.id === msgId ? { ...m, ...patch } : m));

        try {
            await askQuestionStream(
                activeChatId,
                text,
                { session_id: isGuest ? guestSessionId : null },
                (token) => {
                    setMessages(prev => prev.map(m =>
                        m.id === msgId ? { ...m, content: m.content + token } : m
                    ));
                },
                (meta) => updateMsg({ sources: meta.sources || [] }),
                (result) => {
                    updateMsg({
                        isStreaming: false,
                        verifying: false,
                        grounded: result.grounded,
                        confidence: result.confidence,
                    });
                    setIsGenerating(false);
                },
                () => {
                    updateMsg({
                        content: 'Verification failed. This response was removed for safety.',
                        isStreaming: false,
                        verifying: false,
                        isError: true,
                        grounded: false,
                    });
                    setIsGenerating(false);
                },
                (err) => {
                    updateMsg({
                        content: `Error: ${err}`,
                        isStreaming: false,
                        verifying: false,
                        isError: true,
                    });
                    setIsGenerating(false);
                },
            );
        } catch (error) {
            updateMsg({
                content: `Error: ${error.message}`,
                isStreaming: false,
                verifying: false,
                isError: true,
            });
            setIsGenerating(false);
        }
    };

    const handleUpload = async (file, existingChatId = null) => {
        setUploadError(null);
        try {
            // For guests, ensure a session exists
            let sessionId = guestSessionId;
            if (isGuest && !sessionId) {
                sessionId = await initGuestSession();
            }

            const chatId = existingChatId || activeChatId;
            const result = await uploadDocument(file, chatId, sessionId);

            // Set active chat from response
            if (result.chat_id) {
                setActiveChatId(result.chat_id);
                if (!isGuest) refreshChats();
            }

            setUploadedFiles(prev => [...prev, {
                fileName: file.name,
                fileSize: file.size,
                uploadedAt: Date.now(),
                chatId: result.chat_id,
                isProcessed: false
            }]);

            setIsProcessed(false);
            return result;
        } catch (error) {
            // Show upload error only when limit is hit
            if (error.error_code) {
                const messages = {
                    GUEST_LIMIT_REACHED: 'Guest upload limit reached (3 documents). Sign in for more.',
                    PER_CHAT_LIMIT_REACHED: 'This chat has reached the document limit (15).',
                    ACCOUNT_LIMIT_REACHED: 'Account document limit reached (40). Delete a chat to free space.',
                    DUPLICATE_DOCUMENT: 'This file has already been uploaded to this chat.',
                };
                setUploadError(messages[error.error_code] || error.detail);
            } else {
                console.error('Upload error:', error);
            }
            throw error;
        }
    };

    const handleProcess = async () => {
        if (!activeChatId) return;

        setIsProcessing(true);

        try {
            await processSession(activeChatId, isGuest ? guestSessionId : null);

            let status = 'processing';
            let attempts = 0;
            const maxAttempts = 120;

            while (status === 'processing' && attempts < maxAttempts) {
                await new Promise(resolve => setTimeout(resolve, 1000));

                const statusResult = await checkStatus(activeChatId, isGuest ? guestSessionId : null);
                status = statusResult.status;

                if (status === 'ready') {
                    setIsProcessed(true);

                    setUploadedFiles(prev => prev.map(f => ({ ...f, isProcessed: true })));

                    const successMessage = {
                        role: 'system',
                        content: `Successfully processed ${uploadedFiles.length} file${uploadedFiles.length !== 1 ? 's' : ''}. You can now ask questions!`,
                        timestamp: Date.now(),
                        isSuccess: true
                    };
                    setMessages(prev => [...prev, successMessage]);
                    break;
                }

                if (status === 'error') {
                    throw new Error(statusResult.error_message || 'Processing failed');
                }

                attempts++;
            }

            if (attempts >= maxAttempts) {
                throw new Error('Processing timeout');
            }

        } catch (error) {
            console.error('Processing error:', error);
            const errorMessage = {
                role: 'system',
                content: `Processing failed: ${error.message}`,
                timestamp: Date.now(),
                isError: true
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleClear = () => {
        setMessages([]);
        setUploadedFiles([]);
        setIsProcessing(false);
        setIsProcessed(false);
        setActiveChatId(null);
        setUploadError(null);
    };

    const handleDeleteChat = async (chatId) => {
        try {
            await deleteChat(chatId, isGuest ? guestSessionId : null);
            if (activeChatId === chatId) handleClear();
            refreshChats();
        } catch { /* ignore */ }
    };

    const handleSelectChat = async (chat) => {
        setActiveChatId(chat.id);
        setUploadedFiles([]);
        setUploadError(null);
        setIsProcessed(true); // assume previously processed chats are ready
    };

    const handleNewChat = () => {
        handleClear();
    };

    const handleContinueAsGuest = () => {
        setShowAuthModal(false);
    };

    if (supabase && showAuthModal && !user) {
        return (
            <div className="relative flex h-screen items-center justify-center overflow-hidden text-[#f5efff]">
                <div className="glass-panel w-full max-w-md m-3 p-6 rounded-3xl relative">
                    <button
                        onClick={handleContinueAsGuest}
                        className="absolute top-4 right-4 text-white/60 hover:text-white transition-colors"
                        aria-label="Close"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                    <h1 className="text-white font-semibold text-xl mb-4">Sign in to IntelliDocs</h1>
                    <div className="space-y-3">
                        <button
                            onClick={handleGoogleSignIn}
                            disabled={authLoading}
                            className="w-full rounded-xl border border-blue-400 bg-white text-black px-4 py-3 text-base font-medium hover:bg-gray-100 transition-colors"
                        >
                            Continue with Google
                        </button>

                        <div className="flex items-center gap-3 py-1">
                            <div className="h-px flex-1 bg-white/20" />
                            <span className="text-xs uppercase tracking-wide text-white/70">or</span>
                            <div className="h-px flex-1 bg-white/20" />
                        </div>

                        {!showEmailAuth ? (
                            <>
                                <button
                                    onClick={() => setShowEmailAuth(true)}
                                    className="w-full rounded-xl bg-indigo-500 px-4 py-3 text-base font-medium text-white hover:bg-indigo-400 transition-colors"
                                >
                                    Continue with Email
                                </button>
                                <button
                                    onClick={handleContinueAsGuest}
                                    className="w-full rounded-xl border border-white/20 px-4 py-3 text-base font-medium text-white/80 hover:bg-white/10 transition-colors"
                                >
                                    Continue as Guest
                                </button>
                            </>
                        ) : (
                            <>
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="Email"
                                    className="w-full rounded-xl bg-white/10 border border-white/20 px-3 py-2 text-white"
                                />
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="Password"
                                    className="w-full rounded-xl bg-white/10 border border-white/20 px-3 py-2 text-white"
                                />
                                <div className="flex gap-2">
                                    <button
                                        onClick={handleSignIn}
                                        disabled={authLoading}
                                        className="flex-1 rounded-xl bg-white/20 px-3 py-2 text-sm text-white hover:bg-white/30"
                                    >
                                        Sign In
                                    </button>
                                    <button
                                        onClick={handleSignUp}
                                        disabled={authLoading}
                                        className="flex-1 rounded-xl bg-white/10 px-3 py-2 text-sm text-white hover:bg-white/20"
                                    >
                                        Sign Up
                                    </button>
                                </div>
                                <button
                                    onClick={() => setShowEmailAuth(false)}
                                    className="w-full rounded-xl border border-white/20 px-3 py-2 text-xs text-white/80 hover:bg-white/10 transition-colors"
                                >
                                    Back
                                </button>
                            </>
                        )}
                        {authError && <p className="text-red-300 text-xs">{authError}</p>}
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="relative flex h-screen overflow-hidden text-[#f5efff]">
            <div className="pointer-events-none absolute -top-24 -left-16 h-72 w-72 rounded-full bg-fuchsia-400/20 blur-3xl" />
            <div className="pointer-events-none absolute top-10 -right-20 h-96 w-96 rounded-full bg-violet-400/20 blur-3xl" />

            <div className="glass-panel hidden md:flex flex-col w-64 lg:w-80 m-3 p-4 lg:p-6 overflow-y-auto flex-shrink-0 text-[#d8cbe9] transition-all duration-300 rounded-3xl">
                <div className="flex items-center gap-2 mb-6">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-fuchsia-400 to-violet-600 flex items-center justify-center shadow-lg shadow-violet-900/50">
                        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                    </div>
                    <h1 className="text-white font-semibold text-sm sm:text-base">IntelliDocs</h1>
                </div>

                {/* Tab toggle: Info / History */}
                {!isGuest && (
                    <div className="flex gap-1 mb-4 rounded-xl bg-white/5 p-1">
                        <button
                            onClick={() => setShowHistory(false)}
                            className={`flex-1 rounded-lg px-2 py-1.5 text-xs font-medium transition-colors ${!showHistory ? 'bg-white/15 text-white' : 'text-white/60 hover:text-white/80'}`}
                        >
                            About
                        </button>
                        <button
                            onClick={() => setShowHistory(true)}
                            className={`flex-1 rounded-lg px-2 py-1.5 text-xs font-medium transition-colors ${showHistory ? 'bg-white/15 text-white' : 'text-white/60 hover:text-white/80'}`}
                        >
                            History
                        </button>
                    </div>
                )}

                {showHistory && !isGuest ? (
                    <div className="flex flex-col flex-1 overflow-hidden">
                        <button
                            onClick={handleNewChat}
                            className="w-full mb-3 rounded-xl border border-white/20 px-3 py-2 text-sm text-white hover:bg-white/10 transition-colors flex items-center gap-2 justify-center"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                            </svg>
                            New Chat
                        </button>
                        <div className="flex-1 overflow-y-auto space-y-1">
                            {chatList.length === 0 ? (
                                <p className="text-xs text-white/40 text-center mt-4">No chats yet</p>
                            ) : chatList.map(chat => (
                                <div
                                    key={chat.id}
                                    onClick={() => handleSelectChat(chat)}
                                    className={`group flex items-center justify-between rounded-xl px-3 py-2 cursor-pointer transition-colors ${activeChatId === chat.id ? 'bg-white/15 text-white' : 'text-white/70 hover:bg-white/8 hover:text-white'}`}
                                >
                                    <span className="truncate text-sm">{chat.title || 'Untitled'}</span>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); handleDeleteChat(chat.id); }}
                                        className="opacity-0 group-hover:opacity-100 text-white/40 hover:text-red-300 transition-all ml-2 flex-shrink-0"
                                        title="Delete chat"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                        </svg>
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                ) : (
                    <>
                        <h2 className="text-white text-base lg:text-lg font-semibold mb-3">About IntelliDocs</h2>
                <p className="text-xs lg:text-sm leading-relaxed mb-6 text-[#d8cbe9]">
                    This RAG-powered AI assistant enables organizations to instantly unlock insights from their internal knowledge base. It intelligently processes diverse data sources and delivers precise, context-aware answers in real time. Built for seamless integration into existing applications and workflows, the system supports both long-form analytical queries and quick factual lookups, helping teams research faster and make better-informed decisions.
                </p>

                <h3 className="text-white font-medium mb-3 text-sm lg:text-base">Supported Documents:</h3>
                <ul className="text-xs lg:text-sm space-y-2 list-disc pl-4 text-[#baa6d6]">
                    <li>Company policies</li>
                    <li>HR documents</li>
                    <li>FAQs</li>
                    <li>Financial summaries</li>
                    <li>Product documentation</li>
                    <li>Website content</li>
                </ul>
                    </>
                )}
            </div>

            <div className="flex flex-col flex-1 overflow-hidden relative z-10">
                <div className="flex justify-between md:justify-end items-center p-4 flex-shrink-0 z-10">
                    <div className="flex items-center gap-2 md:hidden">
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-fuchsia-400 to-violet-600 flex items-center justify-center">
                            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                            </svg>
                        </div>
                        <h1 className="text-white font-semibold text-sm">IntelliDocs</h1>
                    </div>

                    <div className="flex items-center gap-2">
                        <button
                            onClick={handleClear}
                            className="glass-card inline-flex h-8 items-center justify-center rounded-full px-3 text-white text-xs sm:text-sm leading-none transition-colors hover:bg-white/20"
                            disabled={messages.length === 0 && uploadedFiles.length === 0}
                        >
                            Clear
                        </button>
                        {supabase && user && (
                            <button
                                onClick={handleSignOut}
                                className="glass-card inline-flex h-8 items-center justify-center rounded-full px-3 text-white text-xs sm:text-sm leading-none transition-colors hover:bg-white/20"
                            >
                                Sign out
                            </button>
                        )}
                        {supabase && isGuest && (
                            <button
                                onClick={() => setShowAuthModal(true)}
                                className="glass-card inline-flex h-8 items-center justify-center rounded-full px-3 text-white text-xs sm:text-sm leading-none transition-colors hover:bg-white/20"
                            >
                                Sign in
                            </button>
                        )}
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto">
                    {messages.length === 0 && uploadedFiles.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full px-4 py-6 text-center">
                            <div className="max-w-md w-full mb-6 sm:mb-8 px-5 py-4 rounded-3xl border border-white/20 bg-gradient-to-b from-white/18 to-white/8 shadow-lg shadow-violet-950/30 backdrop-blur-md">
                                <p className="text-[#d4c6e8] text-sm sm:text-base">
                                    Upload documents and ask questions. I'll search through your files and provide accurate answers.
                                </p>
                            </div>

                            <div className="max-w-3xl w-full p-0 sm:p-0">
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6">
                                    <div className="rounded-3xl border border-white/20 bg-gradient-to-b from-white/14 to-white/8 p-4 text-left shadow-lg shadow-violet-950/20 backdrop-blur-md">
                                        <div className="mb-3 flex items-center gap-2">
                                            <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-cyan-400/15 text-cyan-300">
                                                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h10M7 11h10M7 15h6M6 3h9l5 5v13a1 1 0 01-1 1H6a1 1 0 01-1-1V4a1 1 0 011-1z" />
                                                </svg>
                                            </span>
                                            <h3 className="text-white font-medium text-sm">Documents</h3>
                                        </div>
                                        <p className="text-xs sm:text-sm text-[#c6b5de]">PDF, Word, PowerPoint</p>
                                        <p className="text-xs sm:text-sm text-[#c6b5de]">Excel</p>
                                        <p className="text-xs sm:text-sm text-[#c6b5de]" title="Smart Detection for Q&A, Documents, and Bulk data">CSV</p>
                                    </div>
                                    <div className="rounded-3xl border border-white/20 bg-gradient-to-b from-white/14 to-white/8 p-4 text-left shadow-lg shadow-violet-950/20 backdrop-blur-md">
                                        <div className="mb-3 flex items-center gap-2">
                                            <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-amber-300/15 text-amber-200">
                                                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h10" />
                                                </svg>
                                            </span>
                                            <h3 className="text-white font-medium text-sm">Text/Markup</h3>
                                        </div>
                                        <p className="text-xs sm:text-sm text-[#c6b5de]">TXT, Markdown, RST</p>
                                        <p className="text-xs sm:text-sm text-[#c6b5de]">JSON, XML, HTML</p>
                                    </div>
                                    <div className="rounded-3xl border border-white/20 bg-gradient-to-b from-white/14 to-white/8 p-4 text-left shadow-lg shadow-violet-950/20 backdrop-blur-md">
                                        <div className="mb-3 flex items-center gap-2">
                                            <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-fuchsia-300/15 text-fuchsia-200">
                                                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h8M9 11h6M10 15h4M5 4h14a1 1 0 011 1v14a1 1 0 01-1 1H5a1 1 0 01-1-1V5a1 1 0 011-1z" />
                                                </svg>
                                            </span>
                                            <h3 className="text-white font-medium text-sm">Code Files</h3>
                                        </div>
                                        <p className="text-xs sm:text-sm text-[#c6b5de]">Python, JavaScript</p>
                                        <p className="text-xs sm:text-sm text-[#c6b5de]">Java, C/C++, Shell, YAML</p>
                                    </div>
                                </div>
                            </div>

                        </div>
                    ) : (
                        <div className="max-w-3xl mx-auto w-full px-3 sm:px-4 py-4 sm:py-6">
                            {messages.map((msg, idx) => (
                                <ChatMessage key={idx} message={msg} />
                            ))}
                            {isGenerating && (
                                <ChatMessage
                                    message={{
                                        role: 'assistant',
                                        content: '',
                                        timestamp: Date.now(),
                                        isTyping: true
                                    }}
                                />
                            )}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                <div className="flex-shrink-0 pb-3 px-3 sm:px-4">
                    {uploadError && (
                        <div className="max-w-3xl mx-auto w-full mb-2">
                            <div className="flex items-center gap-2 rounded-xl bg-red-500/15 border border-red-400/30 px-3 py-2 text-red-200 text-sm">
                                <span className="flex-1">{uploadError}</span>
                                <button onClick={() => setUploadError(null)} className="text-red-300 hover:text-white">
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    )}
                    <div className="glass-panel max-w-3xl mx-auto w-full rounded-2xl px-3 sm:px-4 py-3 sm:py-4">
                        <ChatInput
                            onSend={handleSend}
                            onUpload={handleUpload}
                            onProcess={handleProcess}
                            disabled={isGenerating}
                            placeholder={isGenerating ? 'Waiting for response...' : 'Send a message...'}
                            uploadedFiles={uploadedFiles}
                            isProcessing={isProcessing}
                            isProcessed={isProcessed}
                        />
                        <div className="text-center text-xs text-[#bfaed9] mt-2 flex flex-col items-center gap-1">
                            <span>
                                {isProcessing ? 'Processing documents...' : !isProcessed && uploadedFiles.length > 0 ? 'Click Process to start asking questions' : 'AI can make mistakes. Please double check important info.'}
                            </span>
                            <span className="text-white/70 flex items-center gap-1">
                                <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                Sessions and documents automatically expire after 30 minutes of inactivity
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
