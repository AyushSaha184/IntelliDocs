import { useState, useEffect } from 'react';

export default function UploadedFiles({ files, onProcess, isProcessing, isProcessed }) {
    const [collapsed, setCollapsed] = useState(false);

    // Auto-collapse 2 seconds after processing completes
    useEffect(() => {
        if (isProcessed && !collapsed) {
            const timer = setTimeout(() => setCollapsed(true), 2000);
            return () => clearTimeout(timer);
        }
    }, [isProcessed]);

    if (files.length === 0) return null;

    const getFileIcon = (fileName) => {
        const ext = fileName.toLowerCase().split('.').pop();

        // Document icons
        if (['pdf'].includes(ext)) return '📄';
        if (['docx', 'doc'].includes(ext)) return '📝';
        if (['pptx', 'ppt'].includes(ext)) return '📊';
        if (['xlsx', 'xls', 'csv'].includes(ext)) return '📈';

        // Text/Markup icons
        if (['txt', 'md', 'rst'].includes(ext)) return '📃';
        if (['json', 'xml', 'html', 'htm'].includes(ext)) return '🌐';

        // Code icons
        if (['py', 'js', 'java', 'cpp', 'c', 'h', 'sh', 'yml', 'yaml'].includes(ext)) return '💻';

        return '📎';
    };

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    };

    return (
        <div className="bg-[#40414F] border border-white/20 rounded-lg mb-3 transition-all duration-300 overflow-hidden">
            {/* Header - always visible, clickable to expand/collapse when processed */}
            <div
                className={`flex items-center justify-between p-3 sm:p-4 gap-2 ${isProcessed ? 'cursor-pointer hover:bg-white/5' : ''}`}
                onClick={isProcessed ? () => setCollapsed(prev => !prev) : undefined}
            >
                <div className="flex items-center gap-2">
                    {isProcessed && collapsed && (
                        <svg className="w-3 h-3 text-gray-400 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                    )}
                    {isProcessed && !collapsed && (
                        <svg className="w-3 h-3 text-gray-400 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                    )}
                    <span className="text-xs sm:text-sm text-white font-medium">
                        📁 {collapsed ? `${files.length} file${files.length !== 1 ? 's' : ''} processed` : `Uploaded Files (${files.length})`}
                    </span>
                </div>
                {isProcessed ? (
                    <div className="flex items-center gap-1 sm:gap-2 text-green-400 text-xs sm:text-sm flex-shrink-0">
                        <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        <span className="hidden sm:inline">Processed</span>
                        <span className="sm:hidden">✓</span>
                    </div>
                ) : (
                    <button
                        type="button"
                        onClick={onProcess}
                        disabled={isProcessing}
                        className="px-3 sm:px-4 py-1 sm:py-1.5 bg-[#10a37f] text-white text-xs sm:text-sm rounded-md hover:bg-[#0d8c6f] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1 sm:gap-2 flex-shrink-0"
                    >
                        {isProcessing ? (
                            <>
                                <svg className="w-3 h-3 sm:w-4 sm:h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                </svg>
                                <span className="hidden sm:inline">Processing...</span>
                            </>
                        ) : (
                            'Process'
                        )}
                    </button>
                )}
            </div>

            {/* Collapsible content */}
            {!collapsed && (
                <div className="px-3 sm:px-4 pb-3 sm:pb-4">
                    {isProcessing && (
                        <div className="mb-2 sm:mb-3 px-2 sm:px-3 py-1.5 sm:py-2 bg-blue-500/10 border border-blue-500/30 rounded-md">
                            <p className="text-xs sm:text-sm text-blue-300">⏳ Please wait for a while...</p>
                        </div>
                    )}

                    {/* File list with max 3 visible, scroll for more */}
                    <div className="space-y-1.5 sm:space-y-2 max-h-[120px] sm:max-h-[150px] overflow-y-auto pr-1 sm:pr-2 custom-scrollbar">
                        {files.map((file, index) => (
                            <div
                                key={index}
                                className="flex items-center gap-2 sm:gap-3 p-1.5 sm:p-2 bg-[#343541] rounded-md"
                            >
                                <span className="text-base sm:text-xl flex-shrink-0">{getFileIcon(file.fileName)}</span>
                                <div className="flex-1 min-w-0">
                                    <p className="text-xs sm:text-sm text-white truncate">{file.fileName}</p>
                                    <p className="text-[10px] sm:text-xs text-gray-400">{formatFileSize(file.fileSize)}</p>
                                </div>
                                <div className="flex-shrink-0">
                                    <svg className="w-4 h-4 sm:w-5 sm:h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                    </svg>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
