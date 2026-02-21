import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

export default function ChatMessage({ message }) {
    const [showSources, setShowSources] = useState(false);
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
                            {/* Action Bar (Copy) - Top Right */}
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

                            <div className="prose prose-invert max-w-none text-xs sm:text-sm pr-6">
                                <ReactMarkdown>{message.content}</ReactMarkdown>
                            </div>

                            {message.sources && message.sources.length > 0 && (
                                <div className="mt-4 pt-4 border-t border-white/20">
                                    <ul className="text-xs sm:text-sm text-gray-400 list-disc pl-4 space-y-1">
                                        {(() => {
                                            // Deduplicate sources based on DocName + Page/Row
                                            const uniqueSources = new Set();
                                            const formattedSources = [];

                                            message.sources.forEach((source) => {
                                                // Matches "[Filename.pdf, Page X]" or "[Filename.csv, Row X]" 
                                                const match = source.match(/^\[(.*?), (Page \d+|Row \d+)\]/i);

                                                let docName = "Unknown Document";
                                                let locationDesc = "";

                                                if (match) {
                                                    docName = match[1].trim();
                                                    // Extract just the number from "Page X" or "Row X"
                                                    const locMatch = match[2].match(/\d+/);
                                                    if (locMatch) {
                                                        locationDesc = ` (Page no.: ${locMatch[0]})`;
                                                    }
                                                } else {
                                                    // Fallback if regex doesn't match perfectly
                                                    const fallbackMatch = source.match(/^\[(.*?)\]/);
                                                    if (fallbackMatch) {
                                                        docName = fallbackMatch[1].trim();
                                                    }
                                                }

                                                const key = `${docName}${locationDesc}`;

                                                if (!uniqueSources.has(key)) {
                                                    uniqueSources.add(key);
                                                    formattedSources.push(`${docName}${locationDesc}`);
                                                }
                                            });

                                            return formattedSources.map((src, idx) => (
                                                <li key={idx} className="leading-relaxed">
                                                    {src}
                                                </li>
                                            ));
                                        })()}
                                    </ul>
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
