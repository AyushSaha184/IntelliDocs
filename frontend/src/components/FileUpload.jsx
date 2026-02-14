import { useState, useRef } from 'react';

const FILE_EXTENSIONS = [
    '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.csv',
    '.txt', '.md', '.rst', '.json', '.xml', '.html', '.htm',
    '.py', '.js', '.java', '.cpp', '.c', '.h', '.sh', '.yml', '.yaml'
];

const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB in bytes

export default function FileUpload({ onFilesSelected, disabled, pendingFiles = [] }) {
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState('');
    const fileInputRef = useRef(null);

    const validateFile = (file) => {
        // Check file size
        if (file.size > MAX_FILE_SIZE) {
            return `File size exceeds 20MB limit (${(file.size / 1024 / 1024).toFixed(2)}MB)`;
        }

        // Check file extension
        const fileName = file.name.toLowerCase();
        const hasValidExtension = FILE_EXTENSIONS.some(ext => fileName.endsWith(ext));
        
        if (!hasValidExtension) {
            return 'Unsupported file type. Please upload a valid document, text, or code file.';
        }

        return null;
    };

    const handleFileSelect = async (event) => {
        const files = Array.from(event.target.files || []);
        if (files.length === 0) return;

        // Validate all files
        const validFiles = [];
        for (const file of files) {
            const validationError = validateFile(file);
            if (validationError) {
                setError(validationError);
                setTimeout(() => setError(''), 5000);
                event.target.value = ''; // Reset input
                return;
            }
            validFiles.push(file);
        }

        setError('');
        
        // Pass files to parent component
        if (onFilesSelected) {
            onFilesSelected(validFiles);
        }

        event.target.value = ''; // Reset input
    };

    const handleButtonClick = () => {
        fileInputRef.current?.click();
    };

    return (
        <div className="relative">
            <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileSelect}
                accept={FILE_EXTENSIONS.join(',')}
                className="hidden"
                disabled={disabled || uploading}
                multiple
            />
            
            <button
                onClick={handleButtonClick}
                disabled={disabled || uploading}
                className="p-1.5 sm:p-2 rounded-md hover:bg-white/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed relative"
                title={pendingFiles.length > 0 ? "Add more files" : "Upload files"}
            >
                <svg 
                    className="w-4 h-4 sm:w-5 sm:h-5 text-white" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                >
                    <path 
                        strokeLinecap="round" 
                        strokeLinejoin="round" 
                        strokeWidth={2} 
                        d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" 
                    />
                </svg>
            </button>

            {/* Error message */}
            {error && (
                <div className="absolute bottom-full left-0 mb-2 p-2 sm:p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-xs sm:text-sm max-w-[250px] sm:max-w-xs z-50">
                    {error}
                </div>
            )}
        </div>
    );
}
