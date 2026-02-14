import { useState, useRef, useEffect } from 'react';
import FileUpload from './FileUpload';
import FilePreviews from './FilePreviews';
import UploadedFiles from './UploadedFiles';

export default function ChatInput({ onSend, onUpload, onProcess, disabled, placeholder, uploadedFiles = [], isProcessing = false, isProcessed = false }) {
    const [input, setInput] = useState('');
    const [pendingFiles, setPendingFiles] = useState([]);
    const [uploading, setUploading] = useState(false);
    const textareaRef = useRef(null);

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
        }
    }, [input]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.trim() && !disabled && !uploading && !isProcessing) {
            onSend(input);
            setInput('');
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    const handleFilesSelected = (files) => {
        setPendingFiles(prev => [...prev, ...files]);
    };

    const handleRemoveFile = (index) => {
        setPendingFiles(prev => prev.filter((_, i) => i !== index));
    };

    const handleUploadAll = async () => {
        if (pendingFiles.length === 0) return;
        
        setUploading(true);
        
        try {
            // Upload all files
            for (const file of pendingFiles) {
                await onUpload(file);
            }
            
            // Clear pending files after successful upload
            setPendingFiles([]);
        } catch (error) {
            console.error('Upload error:', error);
        } finally {
            setUploading(false);
        }
    };

    const isChatDisabled = disabled || uploading || isProcessing || !isProcessed;

    return (
        <div>
            {/* Uploaded files box (shown after upload, before processing) */}
            <UploadedFiles
                files={uploadedFiles}
                onProcess={onProcess}
                isProcessing={isProcessing}
                isProcessed={isProcessed}
            />

            {/* File previews box (for files selected but not uploaded yet) */}
            <FilePreviews 
                files={pendingFiles}
                onRemove={handleRemoveFile}
                onUploadAll={handleUploadAll}
                uploading={uploading}
            />

            {/* Input form */}
            <form onSubmit={handleSubmit} className="relative">
                <div className="flex items-end gap-1.5 sm:gap-2">
                    {/* File Upload Button */}
                    <div className="pb-1.5 sm:pb-2">
                        <FileUpload 
                            onFilesSelected={handleFilesSelected}
                            disabled={disabled || uploading || isProcessing}
                            pendingFiles={pendingFiles}
                        />
                    </div>

                    {/* Text Input */}
                    <div className="flex-1 relative">
                        <textarea
                            ref={textareaRef}
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder={isProcessing ? "Processing files..." : !isProcessed && uploadedFiles.length > 0 ? "Click Process to continue" : placeholder}
                            disabled={isChatDisabled}
                            rows={1}
                            className="w-full resize-none rounded-xl bg-[#40414F] border border-white/20 px-3 sm:px-4 py-2 sm:py-3 pr-10 sm:pr-12 text-white placeholder-gray-400 focus:outline-none focus:border-white/30 disabled:opacity-50 disabled:cursor-not-allowed text-xs sm:text-sm"
                            style={{ maxHeight: '200px' }}
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isChatDisabled}
                            className="absolute right-1.5 sm:right-2 bottom-1.5 sm:bottom-2 p-1.5 sm:p-2 rounded-lg bg-[#10a37f] text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#0d8c6f] transition-colors"
                            title={!isProcessed && uploadedFiles.length > 0 ? "Process files first" : "Send message"}
                        >
                            <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                            </svg>
                        </button>
                    </div>
                </div>
            </form>
        </div>
    );
}
