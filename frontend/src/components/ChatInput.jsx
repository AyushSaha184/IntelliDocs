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

        console.log('Starting upload of', pendingFiles.length, 'files');
        setUploading(true);

        try {
            // Upload all files to the SAME session
            // Track session_id locally since React state updates are async
            let currentSessionId = null;
            for (const file of pendingFiles) {
                console.log('Uploading file:', file.name, 'to session:', currentSessionId);
                const result = await onUpload(file, currentSessionId);
                console.log('Upload result for', file.name, ':', result);
                if (result?.session_id) {
                    currentSessionId = result.session_id;
                }
            }

            // Clear pending files after successful upload
            console.log('Uploads complete, clearing pending files');
            setPendingFiles([]);
        } catch (error) {
            console.error('Upload error in handleUploadAll:', error);
        } finally {
            setUploading(false);
            console.log('Uploading state set to false');
        }
    };

    const isChatDisabled = disabled || uploading || isProcessing || !isProcessed;

    const processedUploadedFiles = uploadedFiles.filter(f => f.isProcessed);
    const unprocessedUploadedFiles = uploadedFiles.filter(f => !f.isProcessed);

    return (
        <div>
            {/* Processed files box */}
            {processedUploadedFiles.length > 0 && (
                <UploadedFiles
                    files={processedUploadedFiles}
                    onProcess={onProcess}
                    isProcessing={false}
                    isProcessed={true}
                />
            )}

            {/* Unprocessed uploaded files box (newly added files waiting to be processed) */}
            {unprocessedUploadedFiles.length > 0 && (
                <UploadedFiles
                    files={unprocessedUploadedFiles}
                    onProcess={onProcess}
                    isProcessing={isProcessing}
                    isProcessed={false}
                />
            )}

            {/* File previews box (for files selected but not uploaded yet) */}
            <FilePreviews
                files={pendingFiles}
                onRemove={handleRemoveFile}
                onUploadAll={handleUploadAll}
                uploading={uploading}
            />

            {/* Input form */}
            <form onSubmit={handleSubmit} className="relative">
                <div className="flex items-center gap-1.5 sm:gap-2">
                    {/* File Upload Button */}
                    <div className="self-center">
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
                            className="chat-input-no-scrollbar w-full resize-none overflow-y-hidden rounded-2xl bg-white/10 border border-white/20 px-3 sm:px-4 py-2 sm:py-3 pr-10 sm:pr-12 text-white placeholder-[#c8b9de] focus:outline-none focus:border-white/40 focus:bg-white/15 disabled:opacity-50 disabled:cursor-not-allowed text-xs sm:text-sm"
                            style={{ maxHeight: '200px', scrollbarWidth: 'none', msOverflowStyle: 'none' }}
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isChatDisabled}
                            className="absolute right-1.5 sm:right-2 top-1/2 -translate-y-1/2 p-1.5 sm:p-2 rounded-lg bg-gradient-to-r from-fuchsia-500 to-violet-500 text-white disabled:opacity-50 disabled:cursor-not-allowed hover:from-fuchsia-400 hover:to-violet-400 transition-colors"
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
