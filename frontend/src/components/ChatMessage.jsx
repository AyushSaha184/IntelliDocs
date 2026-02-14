import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

export default function ChatMessage({ message }) {
    const [showSources, setShowSources] = useState(false);
    const isUser = message.role === 'user';
    const isSystem = message.role === 'system';

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
                    className={`rounded-lg px-3 sm:px-4 py-2 sm:py-3 max-w-[85%] ${
                        isUser
                            ? 'bg-[#10a37f] text-white'
                            : message.isError
                            ? 'bg-red-900/20 border border-red-500/30 text-red-300'
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
                            <div className="prose prose-invert max-w-none text-xs sm:text-sm">
                                <ReactMarkdown>{message.content}</ReactMarkdown>
                            </div>

                            {message.sources && message.sources.length > 0 && (
                                <div className="mt-2 sm:mt-3 pt-2 sm:pt-3 border-t border-white/20">
                                    <button
                                        onClick={() => setShowSources(!showSources)}
                                        className="flex items-center gap-2 text-xs text-gray-400 hover:text-white transition-colors"
                                    >
                                        <svg className={`w-3 h-3 sm:w-4 sm:h-4 transition-transform ${showSources ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                        </svg>
                                        <span>{message.sources.length} source{message.sources.length !== 1 ? 's' : ''}</span>
                                    </button>

                                    {showSources && (
                                        <div className="mt-2 space-y-2">
                                            {message.sources.slice(0, 3).map((source, idx) => (
                                                <div key={idx} className="text-xs bg-black/20 rounded p-2 border border-white/20">
                                                    <p className="text-gray-300 line-clamp-2">{source}</p>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}
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
