import { useState, useRef, useEffect, useCallback } from 'react';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import {
    uploadDocument, checkStatus, askQuestionStream, processSession,
    createSession, deleteSession, listDocuments,
} from './services/api';

export default function App() {
    const [messages, setMessages] = useState([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isProcessed, setIsProcessed] = useState(false);
    const [processingProgress, setProcessingProgress] = useState(null);
    const [pendingFileCount, setPendingFileCount] = useState(0);
    const [clearSignal, setClearSignal] = useState(0);
    const [sessionId, setSessionId] = useState(null);
    const [uploadError, setUploadError] = useState(null);
    const [viewportHeight, setViewportHeight] = useState(window.innerHeight);
    const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
    const [showBugReport, setShowBugReport] = useState(false);
    const [chatHistory, setChatHistory] = useState([]);

    const messagesEndRef = useRef(null);
    const messagesContainerRef = useRef(null);
    const sessionRef = useRef(null);
    const cleanupSentRef = useRef(false);
    const processingCancelledRef = useRef(false);
    const prevMessageCountRef = useRef(0);
    const bugReportRef = useRef(null);

    // Create session on mount
    useEffect(() => {
        (async () => {
            try {
                const data = await createSession();
                setSessionId(data.session_id);
                sessionRef.current = data.session_id;
            } catch (e) {
                console.error('Failed to create session:', e);
            }
        })();
    }, []);

    // Cleanup session on tab close / refresh
    useEffect(() => {
        const fireCleanup = () => {
            const sid = sessionRef.current;
            if (!sid || cleanupSentRef.current) return;
            cleanupSentRef.current = true;
            deleteSession(sid);
        };

        const handleBeforeUnload = () => fireCleanup();
        const handlePageHide = () => fireCleanup();
        const handleVisibilityChange = () => {
            if (document.visibilityState === 'hidden') fireCleanup();
        };

        window.addEventListener('beforeunload', handleBeforeUnload);
        window.addEventListener('pagehide', handlePageHide);
        document.addEventListener('visibilitychange', handleVisibilityChange);

        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
            window.removeEventListener('pagehide', handlePageHide);
            document.removeEventListener('visibilitychange', handleVisibilityChange);
        };
    }, []);

    useEffect(() => {
        const messageCount = messages.length;
        if (messageCount > prevMessageCountRef.current) {
            setShouldAutoScroll(true);
        }
        prevMessageCountRef.current = messageCount;
    }, [messages.length]);

    useEffect(() => {
        if (!shouldAutoScroll) return;
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, shouldAutoScroll]);

    useEffect(() => {
        const viewport = window.visualViewport;
        if (!viewport) return;
        const updateHeight = () => setViewportHeight(Math.round(viewport.height));
        updateHeight();
        viewport.addEventListener('resize', updateHeight);
        viewport.addEventListener('scroll', updateHeight);
        return () => {
            viewport.removeEventListener('resize', updateHeight);
            viewport.removeEventListener('scroll', updateHeight);
        };
    }, []);

    const handleMessageScroll = () => {
        const container = messagesContainerRef.current;
        if (!container) return;
        const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
        setShouldAutoScroll(distanceFromBottom <= 50);
    };

    const handleSend = async (text) => {
        if (!text.trim() || isGenerating || !isProcessed || !sessionId) return;

        const userMessageTs = Date.now();
        const userMessage = {
            id: `u-${userMessageTs}-${Math.random().toString(16).slice(2)}`,
            role: 'user',
            content: text,
            timestamp: userMessageTs,
        };
        setMessages(prev => [...prev.filter(m => !(m.role === 'system' && m.isSuccess)), userMessage]);
        setIsGenerating(true);

        const newHistory = [...chatHistory, { role: 'user', content: text }];

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

        let assistantAnswer = '';
        try {
            await askQuestionStream(
                sessionId,
                text,
                { chat_history: newHistory },
                (token) => {
                    assistantAnswer += token;
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
                    setChatHistory(prev => [
                        ...prev,
                        { role: 'user', content: text },
                        { role: 'assistant', content: assistantAnswer },
                    ]);
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

    const handleUpload = async (file, _unusedChatId = null, options = {}) => {
        setUploadError(null);
        try {
            const result = await uploadDocument(file, sessionId, options);

            if (result.session_id && !sessionId) {
                setSessionId(result.session_id);
                sessionRef.current = result.session_id;
            }

            setUploadedFiles(prev => [...prev, {
                fileName: file.name,
                fileSize: file.size,
                uploadedAt: Date.now(),
                isProcessed: false,
                isFailed: false,
                isExpired: false,
            }]);

            setIsProcessed(false);
            return result;
        } catch (error) {
            if (error?.name === 'AbortError') throw error;
            if (error.error_code) {
                const errorMessages = {
                    SESSION_LIMIT_REACHED: 'Document limit reached (15 per session).',
                    DUPLICATE_DOCUMENT: 'This file is already uploaded in this session.',
                };
                setUploadError(errorMessages[error.error_code] || error.message);
            } else {
                console.error('Upload error:', error);
            }
            throw error;
        }
    };

    const handleProcess = async () => {
        if (!sessionId) return;

        processingCancelledRef.current = false;
        setIsProcessing(true);
        setProcessingProgress(null);

        try {
            await processSession(sessionId);

            let status = 'processing';
            let attempts = 0;
            const maxAttempts = 120;

            while (status === 'processing' && attempts < maxAttempts) {
                if (processingCancelledRef.current) throw new Error('Processing cancelled');
                await new Promise(resolve => setTimeout(resolve, 1000));

                const statusResult = await checkStatus(sessionId);
                status = statusResult.status;
                if (statusResult.document_progress) {
                    setProcessingProgress(statusResult.document_progress);
                }

                if (status === 'ready') {
                    const docsData = await listDocuments(sessionId);
                    const nextFiles = (docsData.documents || []).map(d => ({
                        fileName: d.filename,
                        fileSize: d.file_size,
                        uploadedAt: new Date(d.created_at).getTime(),
                        isProcessed: d.status === 'ready',
                        isFailed: d.status === 'failed',
                        isExpired: false,
                    }));
                    setUploadedFiles(nextFiles);
                    setIsProcessed(nextFiles.length > 0 && nextFiles.some(f => f.isProcessed));
                    setProcessingProgress(null);

                    setMessages(prev => [...prev, {
                        role: 'system',
                        content: `Successfully processed ${nextFiles.length} file${nextFiles.length !== 1 ? 's' : ''}. You can now ask questions!`,
                        timestamp: Date.now(),
                        isSuccess: true,
                    }]);
                    break;
                }

                if (status === 'error') {
                    throw new Error(statusResult.error_message || 'Processing failed');
                }
                attempts++;
            }

            if (attempts >= maxAttempts) throw new Error('Processing timeout');
        } catch (error) {
            console.error('Processing error:', error);
            setMessages(prev => [...prev, {
                id: `sys-${Date.now()}-${Math.random().toString(16).slice(2)}`,
                role: 'system',
                content: `Processing failed: ${error.message}`,
                timestamp: Date.now(),
                isError: true,
            }]);
        } finally {
            setIsProcessing(false);
            setProcessingProgress(null);
        }
    };

    const handleClear = () => {
        processingCancelledRef.current = true;
        setClearSignal(prev => prev + 1);
        setPendingFileCount(0);
        setIsProcessing(false);
        setProcessingProgress(null);
        setUploadError(null);
        setUploadedFiles([]);
        setIsProcessed(false);
        setMessages([]);
        setChatHistory([]);

        // Delete old session and create fresh one
        const oldSid = sessionRef.current;
        if (oldSid) deleteSession(oldSid);
        (async () => {
            try {
                const data = await createSession();
                setSessionId(data.session_id);
                sessionRef.current = data.session_id;
                cleanupSentRef.current = false;
            } catch (e) {
                console.error('Failed to create new session:', e);
            }
        })();
    };

    const orderedMessages = [...messages].sort((a, b) => {
        const ta = a.timestamp || 0;
        const tb = b.timestamp || 0;
        if (ta !== tb) return ta - tb;
        return String(a.id || '').localeCompare(String(b.id || ''));
    });

    return (
        <div className="relative flex overflow-hidden text-[#f5efff]" style={{ height: `${viewportHeight}px` }}>
            <div className="pointer-events-none absolute -top-24 -left-16 h-72 w-72 rounded-full bg-fuchsia-400/20 blur-3xl" />
            <div className="pointer-events-none absolute top-10 -right-20 h-96 w-96 rounded-full bg-violet-400/20 blur-3xl" />

            {/* Sidebar */}
            <div className="glass-panel hidden md:flex flex-col w-64 lg:w-80 m-3 p-4 lg:p-6 overflow-y-auto flex-shrink-0 text-[#d8cbe9] transition-all duration-300 rounded-3xl">
                <div className="flex items-center gap-2 mb-6">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-fuchsia-400 to-violet-600 flex items-center justify-center shadow-lg shadow-violet-900/50">
                        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                    </div>
                    <h1 className="text-white font-semibold text-sm sm:text-base">IntelliDocs</h1>
                </div>

                <h2 className="text-white text-base lg:text-lg font-semibold mb-3">About</h2>
                <p className="text-xs lg:text-sm leading-relaxed mb-6 text-[#d8cbe9]">
                    This RAG-powered AI assistant enables organizations to instantly unlock insights from their internal knowledge base. It intelligently processes diverse data sources and delivers precise, context-aware answers in real time.
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
            </div>

            {/* Main area */}
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
                        <div className="relative" ref={bugReportRef}>
                            <button
                                onClick={() => setShowBugReport(prev => !prev)}
                                className="glass-card inline-flex h-8 items-center justify-center gap-1 rounded-full px-3 text-white text-xs sm:text-sm leading-none transition-colors hover:bg-white/20"
                            >
                                <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M10.29 3.86l-8 14A1 1 0 003.15 19h17.7a1 1 0 00.86-1.5l-8-14a1 1 0 00-1.72 0z" />
                                </svg>
                                <span>Report Bug</span>
                            </button>
                            {showBugReport && (
                                <div
                                    className="absolute right-0 top-10 z-50 w-64 rounded-2xl border border-white/20 bg-[#1a1030]/90 backdrop-blur-md p-4 text-xs text-white/80 shadow-xl"
                                    onMouseLeave={() => setShowBugReport(false)}
                                >
                                    Please report the bug{' '}
                                    <a
                                        href="mailto:ayushsaha1834@gmail.com"
                                        className="text-fuchsia-300 underline underline-offset-2 hover:text-fuchsia-200 transition-colors"
                                    >
                                        here
                                    </a>
                                    {' '}with screenshots.
                                </div>
                            )}
                        </div>

                        <button
                            onClick={handleClear}
                            className="glass-card inline-flex h-8 items-center justify-center rounded-full px-3 text-white text-xs sm:text-sm leading-none transition-colors hover:bg-white/20"
                            disabled={uploadedFiles.length === 0 && pendingFileCount === 0 && !isProcessing && messages.length === 0}
                        >
                            Clear
                        </button>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto" ref={messagesContainerRef} onScroll={handleMessageScroll}>
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
                            {orderedMessages.map((msg, idx) => (
                                <ChatMessage key={msg.id || `${msg.timestamp || 0}-${idx}`} message={msg} />
                            ))}
                            {isGenerating && (
                                <ChatMessage
                                    message={{
                                        role: 'assistant',
                                        content: '',
                                        timestamp: Date.now(),
                                        isTyping: true,
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
                            processingProgress={processingProgress}
                            clearSignal={clearSignal}
                            onPendingCountChange={setPendingFileCount}
                        />
                        <div className="text-center text-xs text-[#bfaed9] mt-2 flex flex-col items-center gap-1">
                            <span>
                                {isProcessing ? 'Processing documents...' : !isProcessed && uploadedFiles.length > 0 ? 'Click Process to start asking questions' : 'AI can make mistakes. Please double check important info.'}
                            </span>
                            <span className="text-white/70 flex items-center gap-1">
                                <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                Sessions expire after 30 minutes of inactivity
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
