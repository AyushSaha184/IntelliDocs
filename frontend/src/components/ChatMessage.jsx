import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

function normalizeStreamingMarkdown(content, isStreaming) {
    if (!isStreaming || !content) return content || '';
    let safe = content;

    const fenceCount = (safe.match(/```/g) || []).length;
    if (fenceCount % 2 !== 0) {
        safe += '\n```';
    }

    const inlineMathCount = (safe.match(/\$/g) || []).length;
    if (inlineMathCount % 2 !== 0) {
        safe += '$';
    }

    return safe;
}

export default function ChatMessage({ message }) {
    const [isCopied, setIsCopied] = useState(false);

    const isUser = message.role === 'user';
    const isSystem = message.role === 'system';

    const handleCopy = () => {
        navigator.clipboard.writeText(message.content);
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
    };

    const showCopyTextButton = !isSystem && !message.isTyping && !message.isStreaming;

    // Render success message (processing complete)
    if (isSystem && message.isSuccess) {
        return (
            <div className="flex justify-center mb-4 sm:mb-6">
                <div className="inline-flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2 sm:py-3 bg-gradient-to-r from-white/12 to-white/8 border border-violet-300/35 rounded-xl max-w-full backdrop-blur-md">
                    <div className="flex-shrink-0 w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-fuchsia-300/15 flex items-center justify-center">
                        <svg className="w-4 h-4 sm:w-5 sm:h-5 text-fuchsia-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-xs sm:text-sm text-[#e9ddfb] font-medium break-words">{message.content}</p>
                    </div>
                </div>
            </div>
        );
    }

    // Render error message
    if (isSystem && message.isError) {
        return (
            <div className="flex justify-center mb-4 sm:mb-6">
                <div className="inline-flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2 sm:py-3 bg-red-500/10 border border-red-400/40 rounded-xl max-w-full backdrop-blur-md">
                    <div className="flex-shrink-0 w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-red-500/20 flex items-center justify-center">
                        <svg className="w-4 h-4 sm:w-5 sm:h-5 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-xs sm:text-sm text-red-400 font-medium break-words">{message.content}</p>
                    </div>
                </div>
            </div>
        );
    }

    // Verification badge shown below the message bubble content
    const VerificationBadge = () => {
        if (message.isTyping) return null;
        if (message.verifying) {
            return (
                <div className="flex items-center gap-1.5 mt-2 text-yellow-400/80 text-xs animate-pulse">
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                    Verifying against sources...
                </div>
            );
        }
        if (message.grounded === true) {
            return (
                <div className="flex items-center gap-1.5 mt-2 text-fuchsia-200/90 text-xs">
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                    Fact-checked & grounded {message.confidence != null ? `(${Math.round(message.confidence * 100)}%)` : ''}
                </div>
            );
        }
        return null;
    };

    const bubbleBorder = message.isError && !isSystem ? 'border border-red-400/60' : '';

    return (
        <div className={`flex gap-2 sm:gap-4 mb-4 sm:mb-6 ${isUser ? 'justify-end' : ''}`}>
            {!isUser && (
                <div className="flex-shrink-0 w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-gradient-to-br from-fuchsia-500 to-violet-600 flex items-center justify-center shadow-lg shadow-violet-900/40">
                    <svg className="w-3 h-3 sm:w-5 sm:h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                </div>
            )}

            <div className={`flex-1 flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
                <div
                    className={`group rounded-2xl px-3 sm:px-4 py-2.5 sm:py-3.5 max-w-[85%] relative ${isUser
                            ? 'bg-gradient-to-br from-fuchsia-500 to-violet-600 text-white shadow-lg shadow-violet-900/30'
                            : message.isError
                                ? `bg-red-900/20 ${bubbleBorder} text-red-200 backdrop-blur-md`
                                : 'bg-gradient-to-b from-white/16 to-white/10 border border-white/25 text-white backdrop-blur-md shadow-lg shadow-black/20'
                        }`}
                >
                    {message.isTyping ? (
                        <div className="flex gap-1">
                            <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                            <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                            <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                    ) : (
                        <>
                            <div className="prose prose-invert max-w-none text-xs sm:text-sm pr-1">
                                <ReactMarkdown
                                    remarkPlugins={[remarkGfm, remarkMath]}
                                    rehypePlugins={[rehypeKatex]}
                                    skipHtml
                                >
                                    {normalizeStreamingMarkdown(message.content, message.isStreaming)}
                                </ReactMarkdown>
                                {message.isStreaming && (
                                    <span className="inline-block w-0.5 h-4 bg-white/70 ml-0.5 animate-pulse align-middle" />
                                )}
                            </div>

                            <VerificationBadge />
                        </>
                    )}
                </div>

                {showCopyTextButton && (
                    <div className="mt-2">
                        <button
                            onClick={handleCopy}
                            className="inline-flex items-center gap-1.5 rounded-full border border-white/25 bg-white/10 px-2.5 py-1 text-[11px] sm:text-xs text-[#e7ddf8] hover:bg-white/15 hover:text-white transition-colors"
                            title="Copy text"
                            aria-label="Copy text"
                        >
                            {isCopied ? (
                                <>
                                    <svg className="w-3.5 h-3.5 text-green-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                    </svg>
                                    Copied
                                </>
                            ) : (
                                <>
                                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                    </svg>
                                    Copy text
                                </>
                            )}
                        </button>
                    </div>
                )}
            </div>

            {isUser && (
                <div className="flex-shrink-0 w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-white/15 border border-white/30 flex items-center justify-center">
                    <svg className="w-3 h-3 sm:w-5 sm:h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                </div>
            )}
        </div>
    );
}
