import { useState, useRef, useEffect } from 'react';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import { uploadDocument, checkStatus, askQuestion } from './services/api';

export default function App() {
    const [messages, setMessages] = useState([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isProcessed, setIsProcessed] = useState(false);
    const [sessionId, setSessionId] = useState(null);
    const messagesEndRef = useRef(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = async (text) => {
        if (!text.trim() || isGenerating || !isProcessed || !sessionId) return;

        const userMessage = { role: 'user', content: text, timestamp: Date.now() };
        setMessages(prev => [...prev, userMessage]);

        setIsGenerating(true);

        try {
            const data = await askQuestion(sessionId, text);
            
            const assistantMessage = {
                role: 'assistant',
                content: data.answer,
                timestamp: Date.now(),
                sources: data.retrieved_chunks || []
            };
            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            const errorMessage = {
                role: 'assistant',
                content: `Error: ${error.message}`,
                timestamp: Date.now(),
                isError: true
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsGenerating(false);
        }
    };

    const handleUpload = async (file) => {
        try {
            const result = await uploadDocument(file);
            
            // Store session ID
            setSessionId(result.session_id);
            
            // Add to uploaded files list
            setUploadedFiles(prev => [...prev, {
                fileName: file.name,
                fileSize: file.size,
                uploadedAt: Date.now(),
                sessionId: result.session_id
            }]);

            return result;
        } catch (error) {
            console.error('Upload error:', error);
            throw error;
        }
    };

    const handleProcess = async () => {
        if (!sessionId) return;
        
        setIsProcessing(true);
        
        try {
            // Poll status until ready
            let status = 'processing';
            let attempts = 0;
            const maxAttempts = 60; // 60 seconds timeout
            
            while (status === 'processing' && attempts < maxAttempts) {
                await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
                
                const statusResult = await checkStatus(sessionId);
                status = statusResult.status;
                
                if (status === 'ready') {
                    setIsProcessed(true);
                    
                    // Add success message
                    const successMessage = {
                        role: 'system',
                        content: `Successfully processed ${uploadedFiles.length} file${uploadedFiles.length !== 1 ? 's' : ''}. You can now ask questions!`,
                        timestamp: Date.now(),
                        isSuccess: true
                    };
                    setMessages(prev => [...prev, successMessage]);
                    break;
                }
                
                if (status === 'error') {
                    throw new Error(statusResult.error_message || 'Processing failed');
                }
                
                attempts++;
            }
            
            if (attempts >= maxAttempts) {
                throw new Error('Processing timeout');
            }
            
        } catch (error) {
            console.error('Processing error:', error);
            const errorMessage = {
                role: 'system',
                content: `Processing failed: ${error.message}`,
                timestamp: Date.now(),
                isError: true
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleClear = () => {
        setMessages([]);
        setUploadedFiles([]);
        setIsProcessing(false);
        setIsProcessed(false);
        setSessionId(null);
    };

    return (
        <div className="flex flex-col h-screen bg-[#343541]">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/20 flex-shrink-0">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-[#10a37f] flex items-center justify-center">
                        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                    </div>
                    <h1 className="text-white font-semibold text-sm sm:text-base">RAG Assistant</h1>
                </div>
                <button
                    onClick={handleClear}
                    className="px-3 py-1.5 rounded-md hover:bg-white/10 transition-colors text-white text-xs sm:text-sm"
                    disabled={messages.length === 0 && uploadedFiles.length === 0}
                >
                    Clear
                </button>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto">
                {messages.length === 0 && uploadedFiles.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full px-4 py-6 text-center">
                        <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-full bg-[#10a37f] flex items-center justify-center mb-3 sm:mb-4">
                            <svg className="w-6 h-6 sm:w-8 sm:h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                            </svg>
                        </div>
                        <h2 className="text-xl sm:text-2xl font-semibold text-white mb-2">
                            Document AI Assistant
                        </h2>
                        <p className="text-gray-400 max-w-md mb-6 sm:mb-8 text-sm sm:text-base px-4">
                            Upload documents and ask questions. I'll search through your files and provide accurate answers.
                        </p>

                        {/* Supported File Types Box */}
                        <div className="max-w-3xl w-full bg-[#40414F] border border-white/20 rounded-lg p-4 sm:p-6">
                            <div className="flex items-center gap-2 text-white mb-3 sm:mb-4">
                                <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                <span className="font-semibold text-sm sm:text-base">Supported File Types (20MB max)</span>
                            </div>
                            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6">
                                <div>
                                    <h3 className="text-white font-medium mb-2 text-sm">Documents</h3>
                                    <p className="text-xs sm:text-sm text-gray-400">PDF, Word, PowerPoint</p>
                                    <p className="text-xs sm:text-sm text-gray-400">Excel, CSV</p>
                                </div>
                                <div>
                                    <h3 className="text-white font-medium mb-2 text-sm">Text/Markup</h3>
                                    <p className="text-xs sm:text-sm text-gray-400">TXT, Markdown, RST</p>
                                    <p className="text-xs sm:text-sm text-gray-400">JSON, XML, HTML</p>
                                </div>
                                <div>
                                    <h3 className="text-white font-medium mb-2 text-sm">Code Files</h3>
                                    <p className="text-xs sm:text-sm text-gray-400">Python, JavaScript</p>
                                    <p className="text-xs sm:text-sm text-gray-400">Java, C/C++, Shell, YAML</p>
                                </div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="max-w-3xl mx-auto w-full px-3 sm:px-4 py-4 sm:py-6">
                        {messages.map((msg, idx) => (
                            <ChatMessage key={idx} message={msg} />
                        ))}
                        {isGenerating && (
                            <ChatMessage
                                message={{
                                    role: 'assistant',
                                    content: '',
                                    timestamp: Date.now(),
                                    isTyping: true
                                }}
                            />
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className="border-t border-white/20 bg-[#343541] flex-shrink-0">
                <div className="max-w-3xl mx-auto w-full px-3 sm:px-4 py-3 sm:py-4">
                    <ChatInput
                        onSend={handleSend}
                        onUpload={handleUpload}
                        onProcess={handleProcess}
                        disabled={isGenerating}
                        placeholder={isGenerating ? "Waiting for response..." : "Send a message..."}
                        uploadedFiles={uploadedFiles}
                        isProcessing={isProcessing}
                        isProcessed={isProcessed}
                    />
                    <p className="text-center text-xs text-gray-500 mt-2">
                        {isProcessing ? "Processing documents..." : !isProcessed && uploadedFiles.length > 0 ? "Click Process to start asking questions" : "RAG Assistant • Upload documents to get started"}
                    </p>
                </div>
            </div>
        </div>
    );
}
