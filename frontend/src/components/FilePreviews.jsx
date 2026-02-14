export default function FilePreviews({ files, onRemove, onUploadAll, uploading }) {
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
        <div className="bg-[#40414F] border border-white/20 rounded-lg p-3 sm:p-4 mb-3">
            <div className="flex items-center justify-between mb-2 gap-2">
                <span className="text-xs sm:text-sm text-gray-400">
                    {files.length} file{files.length !== 1 ? 's' : ''} selected
                </span>
                <button
                    onClick={onUploadAll}
                    disabled={uploading}
                    className="px-3 sm:px-4 py-1 sm:py-1.5 bg-[#10a37f] text-white text-xs sm:text-sm rounded-md hover:bg-[#0d8c6f] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
                >
                    {uploading ? 'Uploading...' : 'Upload All'}
                </button>
            </div>
            
            {/* File list with max 2 visible, scroll for more */}
            <div className="space-y-1.5 sm:space-y-2 max-h-[100px] sm:max-h-[120px] overflow-y-auto pr-1 sm:pr-2 custom-scrollbar">
                {files.map((file, index) => (
                    <div
                        key={index}
                        className="flex items-center gap-2 sm:gap-3 p-1.5 sm:p-2 bg-[#343541] rounded-md group"
                    >
                        <span className="text-base sm:text-xl flex-shrink-0">{getFileIcon(file.name)}</span>
                        <div className="flex-1 min-w-0">
                            <p className="text-xs sm:text-sm text-white truncate">{file.name}</p>
                            <p className="text-[10px] sm:text-xs text-gray-400">{formatFileSize(file.size)}</p>
                        </div>
                        <button
                            onClick={() => onRemove(index)}
                            disabled={uploading}
                            className="opacity-100 sm:opacity-0 sm:group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition-all disabled:cursor-not-allowed flex-shrink-0"
                            title="Remove file"
                        >
                            <svg className="w-3 h-3 sm:w-4 sm:h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
}
