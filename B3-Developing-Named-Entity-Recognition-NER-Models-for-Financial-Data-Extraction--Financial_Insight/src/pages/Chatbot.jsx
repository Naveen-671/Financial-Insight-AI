import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, Send, Trash2, Upload, FileText, CheckCircle2, Loader2, Eye, X } from 'lucide-react';
import { cn } from '../lib/utils';
import { uploadRAGDocument, queryRAG } from '../services/api';

export default function ChatbotPage() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm RAG (Contextual AI Guide), your financial document assistant. Upload a document first, then ask me questions about it!",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedDocs, setUploadedDocs] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [viewingDoc, setViewingDoc] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    const botMessage = {
      id: messages.length + 1,
      text: `Uploading "${file.name}"...`,
      sender: 'bot',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, botMessage]);

    try {
      const result = await uploadRAGDocument(file);

      const newDoc = {
        name: file.name,
        size: result.size_bytes,
        url: `http://localhost:8001/uploads/${result.filename}`
      };

      setUploadedDocs(prev => [...prev, newDoc]);

      const successMessage = {
        id: messages.length + 2,
        text: `✅ Successfully uploaded and processed "${file.name}"! Ask me anything about this document.`,
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, successMessage]);
    } catch (error) {
      const errorMessage = {
        id: messages.length + 2,
        text: `❌ Failed to upload document: ${error.message}`,
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    // Add user message
    const userMessage = {
      id: messages.length + 1,
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages([...messages, userMessage]);
    setInputValue('');

    // Check if documents uploaded
    if (uploadedDocs.length === 0) {
      setTimeout(() => {
        const botMessage = {
          id: messages.length + 2,
          text: "Please upload a document first so I can answer questions about it. Click the upload button below!",
          sender: 'bot',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      }, 500);
      return;
    }

    // Query RAG
    setIsLoading(true);
    try {
      const response = await queryRAG(inputValue);

      const botMessage = {
        id: messages.length + 2,
        text: response.success ? response.answer : "I encountered an error processing your question.",
        sender: 'bot',
        timestamp: new Date(),
        sources: response.metadata?.sources_used
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: messages.length + 2,
        text: `Error: ${error.message}. Make sure you've uploaded a document first!`,
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        id: 1,
        text: "Chat cleared. Hello! I'm RAG, your financial document assistant. Upload a document to get started!",
        sender: 'bot',
        timestamp: new Date()
      }
    ]);
    setUploadedDocs([]);
  };

  return (
    <div className="min-h-screen bg-horizon-light dark:bg-horizon-dark transition-colors duration-300 flex flex-col relative">

      {/* PDF Preview Modal */}
      <AnimatePresence>
        {viewingDoc && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm"
            onClick={() => setViewingDoc(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white dark:bg-horizon-secondary w-full h-[90vh] max-w-6xl rounded-2xl overflow-hidden flex flex-col shadow-2xl"
              onClick={e => e.stopPropagation()}
            >
              <div className="p-4 border-b border-gray-200 dark:border-white/10 flex justify-between items-center bg-gray-50 dark:bg-horizon-secondary/50">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-horizon-primary" />
                  <h3 className="font-bold text-lg text-gray-800 dark:text-white truncate max-w-md">{viewingDoc.name}</h3>
                </div>
                <button
                  onClick={() => setViewingDoc(null)}
                  className="p-2 hover:bg-gray-200 dark:hover:bg-white/10 rounded-full transition-colors"
                >
                  <X className="w-6 h-6 text-gray-600 dark:text-gray-300" />
                </button>
              </div>
              <div className="flex-grow bg-gray-100 dark:bg-black/50 relative">
                {viewingDoc.name.toLowerCase().endsWith('.pdf') ? (
                  <iframe
                    src={viewingDoc.url}
                    className="w-full h-full"
                    title="Document Preview"
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-gray-500 dark:text-gray-400 p-8 text-center">
                    <FileText className="w-16 h-16 mb-4 opacity-50" />
                    <p className="text-xl font-medium">Preview not available for this file type.</p>
                    <p className="mt-2 text-sm">Only PDF files can be previewed directly. You can still ask questions about it!</p>
                    <a
                      href={viewingDoc.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-6 px-6 py-2 bg-horizon-primary text-white rounded-full hover:bg-horizon-primary/90 transition-colors"
                    >
                      Download File
                    </a>
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <motion.div
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="bg-gradient-to-r from-horizon-primary to-horizon-accent.blue text-white py-6 px-6 shadow-xl"
      >
        <div className="max-w-4xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center border border-white/30">
                <MessageCircle className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">RAG - Contextual AI Guide</h1>
                <p className="text-white/80">Your intelligent financial document assistant</p>
              </div>
            </div>

            {/* Uploaded Docs List */}
            {uploadedDocs.length > 0 && (
              <div className="flex gap-2 overflow-x-auto max-w-xs md:max-w-md pb-2 md:pb-0 scrollbar-hide">
                {uploadedDocs.map((doc, idx) => (
                  <motion.button
                    key={idx}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    whileHover={{ scale: 1.05 }}
                    onClick={() => setViewingDoc(doc)}
                    className="flex items-center gap-2 bg-white/10 hover:bg-white/20 px-3 py-2 rounded-lg border border-white/10 transition-all flex-shrink-0"
                    title="Click to view"
                  >
                    <FileText className="w-3 h-3" />
                    <span className="text-xs font-medium truncate max-w-[80px]">{doc.name}</span>
                    <Eye className="w-3 h-3 opacity-70" />
                  </motion.button>
                ))}
              </div>
            )}
          </div>
        </div>
      </motion.div>

      {/* Chat Area */}
      <div className="flex-grow flex flex-col max-w-4xl mx-auto w-full">
        {/* Messages Container */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="flex-grow overflow-y-auto p-6 space-y-4"
        >
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, x: message.sender === 'user' ? 20 : -20, y: 10 }}
              animate={{ opacity: 1, x: 0, y: 0 }}
              transition={{ duration: 0.4 }}
              className={cn(
                "flex",
                message.sender === 'user' ? "justify-end" : "justify-start"
              )}
            >
              <div
                className={cn(
                  "max-w-md px-6 py-4 rounded-2xl shadow-lg backdrop-blur-sm",
                  message.sender === 'user'
                    ? "bg-gradient-to-br from-horizon-primary to-horizon-accent.blue text-white"
                    : "bg-white dark:bg-horizon-secondary/50 border border-gray-200 dark:border-white/10"
                )}
              >
                <div className={cn(
                  "leading-relaxed font-medium whitespace-pre-wrap prose dark:prose-invert max-w-none",
                  message.sender === 'user' ? "text-white" : "text-slate-700 dark:text-slate-300"
                )}>
                  {/* Improved markdown rendering */}
                  {message.text.split('\n').map((line, i) => {
                    // Check for bullet points (- item or * item)
                    const bulletMatch = line.match(/^[-*]\s+(.*)/);
                    if (bulletMatch) {
                      return (
                        <div key={i} className="flex gap-2 ml-2 mb-1">
                          <span className="text-horizon-primary dark:text-horizon-accent.blue">•</span>
                          <span>
                            {bulletMatch[1].split(/(\*\*.*?\*\*)/).map((part, j) => {
                              if (part.startsWith('**') && part.endsWith('**')) {
                                return <strong key={j}>{part.slice(2, -2)}</strong>;
                              }
                              return part;
                            })}
                          </span>
                        </div>
                      );
                    }

                    // Check for headers (starts with ** and ends with **)
                    const headerMatch = line.match(/^\*\*(.*)\*\*$/);
                    if (headerMatch) {
                      return <h4 key={i} className="font-bold text-lg mt-4 mb-2">{headerMatch[1]}</h4>;
                    }

                    // Regular paragraph with inline bolding
                    return (
                      <p key={i} className="mb-1">
                        {line.split(/(\*\*.*?\*\*)/).map((part, j) => {
                          if (part.startsWith('**') && part.endsWith('**')) {
                            return <strong key={j}>{part.slice(2, -2)}</strong>;
                          }
                          return part;
                        })}
                      </p>
                    );
                  })}
                </div>

                {message.sources && (
                  <div className="mt-3 pt-2 border-t border-white/20 text-xs opacity-70">
                    <p className="font-semibold mb-1">Sources:</p>
                    <ul className="list-disc list-inside">
                      {message.sources.map((source, idx) => (
                        <li key={idx} className="truncate">{source.split('\\').pop().split('/').pop()}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <span className={cn(
                  "text-xs mt-2 block",
                  message.sender === 'user'
                    ? "text-white/70"
                    : "text-gray-600 dark:text-gray-400"
                )}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            </motion.div>
          ))}

          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start"
            >
              <div className="bg-white dark:bg-horizon-secondary/50 border border-gray-200 dark:border-white/10 rounded-2xl px-6 py-4 shadow-lg">
                <div className="flex gap-2">
                  <motion.div
                    animate={{ y: [0, -8, 0] }}
                    transition={{ duration: 1, repeat: Infinity }}
                    className="w-2 h-2 bg-horizon-primary rounded-full"
                  />
                  <motion.div
                    animate={{ y: [0, -8, 0] }}
                    transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                    className="w-2 h-2 bg-horizon-primary rounded-full"
                  />
                  <motion.div
                    animate={{ y: [0, -8, 0] }}
                    transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                    className="w-2 h-2 bg-horizon-primary rounded-full"
                  />
                </div>
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Input Area */}
        <motion.div
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="border-t border-gray-200 dark:border-white/10 p-6 bg-white/50 dark:bg-horizon-secondary/20 backdrop-blur-sm"
        >
          <div className="max-w-4xl mx-auto">
            {/* Upload Section */}
            <div className="mb-4 flex items-center gap-3">
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileUpload}
                accept=".pdf,.docx,.doc,.txt"
                className="hidden"
                id="doc-upload"
              />
              <label
                htmlFor="doc-upload"
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-xl border-2 border-dashed cursor-pointer transition-all",
                  isUploading
                    ? "border-gray-300 bg-gray-100 cursor-not-allowed"
                    : "border-horizon-primary/50 hover:border-horizon-primary hover:bg-horizon-primary/5"
                )}
              >
                {isUploading ? (
                  <Loader2 className="w-5 h-5 animate-spin text-gray-500" />
                ) : (
                  <Upload className="w-5 h-5 text-horizon-primary" />
                )}
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {isUploading ? 'Uploading...' : 'Upload Document'}
                </span>
              </label>
              {uploadedDocs.length > 0 && (
                <div className="flex items-center gap-2 text-sm text-green-600">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>{uploadedDocs[uploadedDocs.length - 1].name}</span>
                </div>
              )}
            </div>

            {/* Message Input */}
            <div className="flex gap-3">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder="Ask me anything about your documents... (Shift+Enter for new line)"
                className="flex-grow px-4 py-3 rounded-2xl border border-gray-300 dark:border-white/20 bg-white dark:bg-horizon-dark text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-horizon-primary dark:focus:ring-horizon-accent.blue resize-none"
                rows="3"
              />
              <div className="flex flex-col gap-2">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleSendMessage}
                  disabled={!inputValue.trim() || isLoading}
                  className="p-3 rounded-full bg-gradient-horizon text-white shadow-lg hover:shadow-horizon-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Send message"
                >
                  <Send className="w-5 h-5" />
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleClearChat}
                  className="p-3 rounded-full bg-gray-200 dark:bg-horizon-secondary text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-horizon-secondary/70 transition-colors"
                  title="Clear chat"
                >
                  <Trash2 className="w-5 h-5" />
                </motion.button>
              </div>
            </div>

            {/* Quick Actions */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="mt-4 flex flex-wrap gap-2"
            >
              <p className="w-full text-sm text-gray-600 dark:text-gray-400 mb-2">Quick actions:</p>
              {[
                "What is this document about?",
                "Summarize the key points",
                "Extract important entities",
                "What are the main clauses?"
              ].map((action, idx) => (
                <motion.button
                  key={idx}
                  whileHover={{ scale: 1.05 }}
                  onClick={() => setInputValue(action)}
                  disabled={uploadedDocs.length === 0}
                  className="px-3 py-1 rounded-full bg-horizon-primary/10 dark:bg-horizon-primary/20 text-horizon-primary dark:text-horizon-accent.blue border border-horizon-primary/30 dark:border-horizon-primary/40 hover:bg-horizon-primary/20 dark:hover:bg-horizon-primary/30 transition-colors text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {action}
                </motion.button>
              ))}
            </motion.div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
