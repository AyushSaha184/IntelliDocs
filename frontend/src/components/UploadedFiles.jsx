import { useState, useEffect } from 'react';

export default function UploadedFiles({ files, onProcess, isProcessing, isProcessed, processingProgress = null }) {
    const [collapsed, setCollapsed] = useState(false);

    // Auto-collapse 2 seconds after processing completes
    useEffect(() => {
        if (isProcessed && !collapsed) {
            const timer = setTimeout(() => setCollapsed(true), 2000);
            return () => clearTimeout(timer);
        }
    }, [isProcessed]);

    if (files.length === 0) return null;
    const hasExpired = files.some(f => f.isExpired);
    const hasFailed = files.some(f => f.isFailed);
    const progressTotal = processingProgress?.total || 0;
    const progressProcessed = processingProgress?.processed || 0;
    const progressPercent = progressTotal > 0 ? Math.min(100, Math.round((progressProcessed / progressTotal) * 100)) : 0;

    const getFileIcon = (fileName) => {
        const ext = fileName.toLowerCase().split('.').pop();
        const badge = (bg, label, textColor = '#fff') => (
            <svg width="32" height="20" viewBox="0 0 32 20" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ display: 'inline-block', flexShrink: 0 }}>
                <rect width="32" height="20" rx="4" fill={bg} />
                <text x="16" y="14" textAnchor="middle" fill={textColor} fontSize="8" fontWeight="700" fontFamily="'Segoe UI',system-ui,sans-serif">{label}</text>
            </svg>
        );
        if (ext === 'pdf')                               return badge('#CC2936', 'PDF');
        if (['docx', 'doc'].includes(ext))               return badge('#2B579A', 'DOCX');
        if (['pptx', 'ppt'].includes(ext))               return badge('#C43E1C', 'PPTX');
        if (['xlsx', 'xls'].includes(ext))               return badge('#217346', 'XLSX');
        if (ext === 'csv')                               return badge('#1A7A4A', 'CSV');
        if (ext === 'txt')                               return badge('#6B7280', 'TXT');
        if (ext === 'md')                                return badge('#24292F', 'MD');
        if (ext === 'rst')                               return badge('#6B7280', 'RST');
        if (ext === 'json')                              return badge('#F59E0B', 'JSON');
        if (ext === 'xml')                               return badge('#E44D26', 'XML');
        if (['html', 'htm'].includes(ext))               return badge('#E44D26', 'HTML');
        if (ext === 'py')                                return badge('#3776AB', 'PY');
        if (ext === 'js')                                return badge('#F7DF1E', 'JS', '#1A1A1A');
        if (ext === 'java')                              return badge('#E76F00', 'JAVA');
        if (['cpp', 'c', 'h'].includes(ext))             return badge('#00599C', 'C++');
        if (['sh', 'bash'].includes(ext))                return badge('#1E1E1E', 'SH');
        if (['yml', 'yaml'].includes(ext))               return badge('#CB171E', 'YAML');
        return badge('#6B7280', ext.toUpperCase().slice(0, 4));
    };

    const formatFileSize = (bytes) => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    };

    return (
        <div className="bg-white/10 border border-white/20 rounded-xl mb-3 transition-all duration-300 overflow-hidden backdrop-blur-md">
            {/* Header - always visible, clickable to expand/collapse when processed */}
            <div
                className={`flex items-center justify-between p-3 sm:p-4 gap-2 ${isProcessed ? 'cursor-pointer hover:bg-white/10' : ''}`}
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
                        📁 {collapsed ? `${files.length} file${files.length !== 1 ? 's' : ''} ${hasExpired ? 'needs re-upload' : 'processed'}` : `Uploaded Files (${files.length})`}
                    </span>
                </div>
                {isProcessed ? (
                    <div className="flex items-center gap-1 sm:gap-2 text-fuchsia-200 text-xs sm:text-sm flex-shrink-0">
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
                        disabled={isProcessing || hasExpired}
                        className="px-3 sm:px-4 py-1 sm:py-1.5 bg-gradient-to-r from-fuchsia-500 to-violet-500 text-white text-xs sm:text-sm rounded-md hover:from-fuchsia-400 hover:to-violet-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1 sm:gap-2 flex-shrink-0"
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
                        <div className="mb-2 sm:mb-3 px-2 sm:px-3 py-1.5 sm:py-2 rounded-lg border border-violet-300/35 bg-gradient-to-r from-white/12 to-white/8 backdrop-blur-sm">
                            <p className="text-xs sm:text-sm text-[#ddd1ef]">
                                {progressTotal > 0 ? `Processing ${progressProcessed}/${progressTotal} documents...` : 'Please wait for a while...'}
                            </p>
                            {progressTotal > 0 && (
                                <div className="mt-2 h-2 w-full rounded-full bg-white/10 overflow-hidden">
                                    <div className="h-full bg-gradient-to-r from-fuchsia-400 to-violet-400 transition-all duration-300" style={{ width: `${progressPercent}%` }} />
                                </div>
                            )}
                        </div>
                    )}

                    {hasExpired && (
                        <div className="mb-2 sm:mb-3 px-2 sm:px-3 py-1.5 sm:py-2 rounded-lg border border-amber-300/35 bg-amber-500/10 backdrop-blur-sm">
                            <p className="text-xs sm:text-sm text-amber-200">Some files expired. Re-upload them before processing.</p>
                        </div>
                    )}

                    {hasFailed && (
                        <div className="mb-2 sm:mb-3 px-2 sm:px-3 py-1.5 sm:py-2 rounded-lg border border-red-300/35 bg-red-500/10 backdrop-blur-sm">
                            <p className="text-xs sm:text-sm text-red-200">Some documents failed to process. Re-upload failed files and process again.</p>
                        </div>
                    )}

                    {/* File list with max 3 visible, scroll for more */}
                    <div className="space-y-1.5 sm:space-y-2 max-h-[120px] sm:max-h-[150px] overflow-y-auto pr-1 sm:pr-2 custom-scrollbar">
                        {files.map((file, index) => (
                            <div
                                key={index}
                                className="flex items-center gap-2 sm:gap-3 p-1.5 sm:p-2 bg-black/20 border border-white/10 rounded-md"
                            >
                                <span className="text-base sm:text-xl flex-shrink-0">{getFileIcon(file.fileName)}</span>
                                <div className="flex-1 min-w-0">
                                    <p className="text-xs sm:text-sm text-white truncate">{file.fileName}</p>
                                    <p className="text-[10px] sm:text-xs text-[#bdaad9]">{formatFileSize(file.fileSize)}</p>
                                    {file.isExpired && <p className="text-[10px] sm:text-xs text-amber-300">Expired - re-upload needed</p>}
                                    {file.isFailed && <p className="text-[10px] sm:text-xs text-red-300">Failed to process</p>}
                                </div>
                                <div className="flex-shrink-0">
                                    {file.isExpired ? (
                                        <svg className="w-4 h-4 sm:w-5 sm:h-5 text-amber-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M10.29 3.86l-8 14A1 1 0 003.15 19h17.7a1 1 0 00.86-1.5l-8-14a1 1 0 00-1.72 0z" />
                                        </svg>
                                    ) : file.isFailed ? (
                                        <svg className="w-4 h-4 sm:w-5 sm:h-5 text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                        </svg>
                                    ) : (
                                        <svg className="w-4 h-4 sm:w-5 sm:h-5 text-fuchsia-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                        </svg>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

