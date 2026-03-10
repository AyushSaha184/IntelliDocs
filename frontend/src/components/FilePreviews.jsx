export default function FilePreviews({ files, onRemove, onUploadAll, uploading }) {
    if (files.length === 0) return null;

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
        <div className="bg-white/10 border border-white/20 rounded-xl p-3 sm:p-4 mb-3 backdrop-blur-md">
            <div className="flex items-center justify-between mb-2 gap-2">
                <span className="text-xs sm:text-sm text-[#d0c0e8]">
                    {files.length} file{files.length !== 1 ? 's' : ''} selected
                </span>
                <button
                    type="button"
                    onClick={onUploadAll}
                    disabled={uploading}
                    className="px-3 sm:px-4 py-1 sm:py-1.5 bg-gradient-to-r from-fuchsia-500 to-violet-500 text-white text-xs sm:text-sm rounded-lg hover:from-fuchsia-400 hover:to-violet-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
                >
                    {uploading ? 'Uploading...' : 'Upload All'}
                </button>
            </div>

            {/* File list with max 2 visible, scroll for more */}
            <div className="space-y-1.5 sm:space-y-2 max-h-[100px] sm:max-h-[120px] overflow-y-auto pr-1 sm:pr-2 custom-scrollbar">
                {files.map((entry, index) => (
                    <div
                        key={entry.id || index}
                        className="flex items-center gap-2 sm:gap-3 p-1.5 sm:p-2 bg-black/20 border border-white/10 rounded-lg group"
                    >
                        <span className="text-base sm:text-xl flex-shrink-0">{getFileIcon(entry.file.name)}</span>
                        <div className="flex-1 min-w-0">
                            <p className="text-xs sm:text-sm text-white truncate">{entry.file.name}</p>
                            <div className="flex items-center gap-2">
                                <p className="text-[10px] sm:text-xs text-[#bdaad9]">{formatFileSize(entry.file.size)}</p>
                                {entry.status === 'uploading' && <p className="text-[10px] sm:text-xs text-cyan-300">Uploading...</p>}
                                {entry.status === 'failed' && <p className="text-[10px] sm:text-xs text-red-300">Failed</p>}
                                {entry.status === 'queued' && <p className="text-[10px] sm:text-xs text-white/60">Queued</p>}
                            </div>
                            {entry.error && <p className="text-[10px] sm:text-xs text-red-300 truncate">{entry.error}</p>}
                        </div>
                        <button
                            type="button"
                            onClick={() => onRemove(index)}
                            disabled={uploading}
                            className={`p-1 rounded transition-all disabled:cursor-not-allowed flex-shrink-0 ${entry.status === 'failed' ? 'opacity-100 bg-red-500/20 hover:bg-red-500/30' : 'opacity-100 sm:opacity-0 sm:group-hover:opacity-100 hover:bg-red-500/20'}`}
                            title={entry.status === 'failed' ? 'Delete failed file' : 'Remove file'}
                        >
                            {entry.status === 'failed' ? (
                                <span className="inline-flex items-center gap-1 text-[10px] sm:text-xs text-red-300 font-medium px-1">
                                    <svg className="w-3 h-3 sm:w-4 sm:h-4 text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                    Delete
                                </span>
                            ) : (
                                <svg className="w-3 h-3 sm:w-4 sm:h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            )}
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
}
