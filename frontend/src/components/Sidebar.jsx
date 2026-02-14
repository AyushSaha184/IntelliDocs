export default function Sidebar({ conversations, currentConvId, isOpen, onToggle, onSelectConv, onDeleteConv }) {
    return (
        <>
            <div
                className={`flex-shrink-0 bg-[#202123] border-r border-white/10 transition-all duration-300 ${
                    isOpen ? 'w-64' : 'w-0'
                } overflow-hidden`}
            >
                <div className="flex flex-col h-full">
                    {/* Header */}
                    <div className="p-3 border-b border-white/10">
                        <div className="flex items-center gap-2 text-white">
                            <svg className="w-5 h-5 text-[#10a37f]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                            </svg>
                            <span className="font-semibold">Conversations</span>
                        </div>
                    </div>

                    {/* Conversations List */}
                    <div className="flex-1 overflow-y-auto p-2">
                        {conversations.map((conv) => (
                            <div
                                key={conv.id}
                                className={`group relative flex items-center gap-3 px-3 py-3 rounded-md mb-1 cursor-pointer transition-colors ${
                                    conv.id === currentConvId
                                        ? 'bg-white/10'
                                        : 'hover:bg-white/10'
                                }`}
                                onClick={() => onSelectConv(conv.id)}
                            >
                                <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                                </svg>
                                <span className="text-sm text-white flex-1 truncate">{conv.title}</span>
                                {conversations.length > 1 && (
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onDeleteConv(conv.id);
                                        }}
                                        className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-white/20 transition-opacity"
                                    >
                                        <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                        </svg>
                                    </button>
                                )}
                            </div>
                        ))}
                    </div>

                    {/* Footer */}
                    <div className="border-t border-white/10 p-3">
                        <div className="flex items-center gap-2 text-xs text-gray-400">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            <span>RAG Assistant</span>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
