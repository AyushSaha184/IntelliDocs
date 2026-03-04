import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

export default function ChatMessage({ message }) {
    const [isCopied, setIsCopied] = useState(false);

    const isUser = message.role === 'user';
    const isSystem = message.role === 'system';

    const handleCopy = () => {
        navigator.clipboard.writeText(message.content);
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
    };

    // Render success message (processing complete)
    if (isSystem && message.isSuccess) {
        return (
            <div className="flex justify-center mb-4 sm:mb-6">
                <div className="inline-flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2 sm:py-3 bg-green-500/10 border border-green-500/30 rounded-lg max-w-full">
                    <div className="flex-shrink-0 w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-green-500/20 flex items-center justify-center">
                        <svg className="w-4 h-4 sm:w-5 sm:h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-xs sm:text-sm text-green-400 font-medium break-words">{message.content}</p>
                    </div>
                </div>
            </div>
        );
    }

    // Render error message
    if (isSystem && message.isError) {
        return (
            <div className="flex justify-center mb-4 sm:mb-6">
                <div className="inline-flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2 sm:py-3 bg-red-500/10 border border-red-500/30 rounded-lg max-w-full">
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
                <div className="flex items-center gap-1.5 mt-2 text-green-400/80 text-xs">
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                    ✅ Fact-checked & grounded {message.confidence != null ? `(${Math.round(message.confidence * 100)}%)` : ''}
                </div>
            );
        }
        return null;
    };

    const bubbleBorder = message.isError && !isSystem ? 'border border-red-500/50' : '';

    return (
        <div className={`flex gap-2 sm:gap-4 mb-4 sm:mb-6 ${isUser ? 'justify-end' : ''}`}>
            {!isUser && (
                <div className="flex-shrink-0 w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-[#10a37f] flex items-center justify-center">
                    <svg className="w-3 h-3 sm:w-5 sm:h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                </div>
            )}

            <div className={`flex-1 ${isUser ? 'flex justify-end' : ''}`}>
                <div
                    className={`group rounded-lg px-3 sm:px-4 py-2 sm:py-3 max-w-[85%] relative ${isUser
                            ? 'bg-[#10a37f] text-white'
                            : message.isError
                                ? `bg-red-900/20 ${bubbleBorder} text-red-300`
                                : 'bg-[#444654] text-white'
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
                            {/* Copy button — hide while streaming */}
                            {!message.isStreaming && (
                                <div className="absolute top-2 right-2 opacity-100 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity duration-200 z-10">
                                    <button
                                        onClick={handleCopy}
                                        className="p-1 rounded text-gray-300 hover:text-white hover:bg-black/20 transition-colors flex items-center justify-center bg-black/10 sm:bg-transparent shadow-sm sm:shadow-none backdrop-blur-sm sm:backdrop-blur-none"
                                        title="Copy message"
                                        aria-label="Copy message"
                                    >
                                        {isCopied ? (
                                            <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                            </svg>
                                        ) : (
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                            </svg>
                                        )}
                                    </button>
                                </div>
                            )}

                            <div className="prose prose-invert max-w-none text-xs sm:text-sm pr-6">
                                <ReactMarkdown>{message.content}</ReactMarkdown>
                                {/* Blinking cursor while streaming */}
                                {message.isStreaming && (
                                    <span className="inline-block w-0.5 h-4 bg-white/70 ml-0.5 animate-pulse align-middle" />
                                )}
                            </div>

                            <VerificationBadge />
                        </>
                    )}
                </div>
            </div>

            {isUser && (
                <div className="flex-shrink-0 w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-[#5436DA] flex items-center justify-center">
                    <svg className="w-3 h-3 sm:w-5 sm:h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                </div>
            )}
        </div>
    );
}
