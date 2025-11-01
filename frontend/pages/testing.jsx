import React, { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { motion } from 'framer-motion';

const TestingPage = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('deepseek-v3');
  const [selectedEndpoint, setSelectedEndpoint] = useState('chat');
  const [showSettings, setShowSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(2000);
  const messagesEndRef = useRef(null);

  const models = [
    { id: 'deepseek-v3', name: 'DeepSeek V3 0324', context: '128K' },
    { id: 'gpt-oss-120b', name: 'OpenAI GPT OSS 120B', context: '128K' },
    { id: 'gpt-oss-20b', name: 'OpenAI GPT OSS 20B', context: '128K' },
    { id: 'gemma-3-27b', name: 'Google Gemma 3 27B', context: '128K' },
    { id: 'qwen-2.5-7b', name: 'Qwen 2.5 7B Instruct', context: '128K' },
    { id: 'qwen-2.5-vl-72b', name: 'Qwen 2.5 VL 72B (Vision)', context: '128K' },
    { id: 'qwen3-vl-235b', name: 'Qwen3 VL 235B (Vision)', context: '128K' },
  ];

  const endpoints = [
    { id: 'chat', name: 'Chat', description: 'Multi-turn conversation' },
    { id: 'inference', name: 'Inference', description: 'Single prompt' },
  ];

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      let requestBody;

      if (selectedEndpoint === 'inference') {
        requestBody = {
          endpoint: 'inference',
          prompt: input,
          model: selectedModel,
          temperature,
          max_tokens: maxTokens,
        };
      } else {
        // Build message history for chat endpoint
        const chatMessages = [
          ...messages
            .filter((m) => m.role === 'user' || m.role === 'assistant')
            .map((m) => ({
              role: m.role,
              content: m.content,
            })),
          { role: 'user', content: input },
        ];

        requestBody = {
          endpoint: 'chat',
          messages: chatMessages,
          model: selectedModel,
          temperature,
          max_tokens: maxTokens,
        };
      }

      const response = await fetch('/api/llm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to get response');
      }

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        metadata: {
          model: data.model,
          usage: data.usage,
        },
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        role: 'error',
        content: `Error: ${error.message}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const getModelName = (id) => {
    return models.find((m) => m.id === id)?.name || id;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center gap-3">
              <Image
                src="/images/TeeTee.png"
                alt="TeeTee Logo"
                width={40}
                height={40}
              />
              <div>
                <h1
                  className="text-2xl font-bold bg-gradient-to-r from-violet-400 to-purple-300 text-transparent bg-clip-text"
                  style={{ fontFamily: 'var(--font-pacifico)' }}
                >
                  TeeTee
                </h1>
                <p className="text-xs text-gray-500">LLM Testing Console</p>
              </div>
            </Link>

            <nav className="flex items-center gap-4">
              <Link
                href="/chat"
                className="text-sm text-gray-600 hover:text-violet-600 transition-colors"
              >
                Chat
              </Link>
              <Link
                href="/models"
                className="text-sm text-gray-600 hover:text-violet-600 transition-colors"
              >
                Models
              </Link>
              <span className="text-sm font-medium text-violet-600">Testing</span>
            </nav>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Settings Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 sticky top-24">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Settings
              </h2>

              {/* Endpoint Selection */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Endpoint
                </label>
                <div className="space-y-2">
                  {endpoints.map((endpoint) => (
                    <button
                      key={endpoint.id}
                      onClick={() => setSelectedEndpoint(endpoint.id)}
                      className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                        selectedEndpoint === endpoint.id
                          ? 'bg-violet-100 text-violet-900 border border-violet-300'
                          : 'bg-gray-50 text-gray-700 hover:bg-gray-100 border border-gray-200'
                      }`}
                    >
                      <div className="font-medium">{endpoint.name}</div>
                      <div className="text-xs opacity-70">
                        {endpoint.description}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Model Selection */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model
                </label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-violet-400 focus:border-transparent"
                >
                  {models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name} ({model.context})
                    </option>
                  ))}
                </select>
              </div>

              {/* Temperature */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Temperature: {temperature}
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Focused</span>
                  <span>Creative</span>
                </div>
              </div>

              {/* Max Tokens */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Tokens
                </label>
                <input
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-violet-400 focus:border-transparent"
                  min="1"
                  max="4000"
                />
              </div>

              {/* Clear Chat */}
              <button
                onClick={clearChat}
                className="w-full px-4 py-2 bg-red-50 text-red-600 rounded-lg text-sm font-medium hover:bg-red-100 transition-colors"
              >
                Clear Chat
              </button>

              {/* Stats */}
              <div className="mt-6 pt-6 border-t border-gray-200">
                <div className="text-xs text-gray-500 space-y-1">
                  <div className="flex justify-between">
                    <span>Messages:</span>
                    <span className="font-medium text-gray-700">
                      {messages.length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Endpoint:</span>
                    <span className="font-medium text-gray-700">
                      /{selectedEndpoint}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Chat Panel */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 flex flex-col h-[calc(100vh-200px)]">
              {/* Chat Header */}
              <div className="border-b border-gray-200 px-6 py-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900">
                      {getModelName(selectedModel)}
                    </h2>
                    <p className="text-xs text-gray-500">
                      Running via Phala Cloud Confidential AI
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
                        isLoading
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-green-100 text-green-800'
                      }`}
                    >
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          isLoading ? 'bg-yellow-500' : 'bg-green-500'
                        }`}
                      ></span>
                      {isLoading ? 'Generating...' : 'Ready'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4">
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <div className="bg-gradient-to-br from-violet-100 to-purple-100 rounded-full p-6 mb-4">
                      <svg
                        className="w-12 h-12 text-violet-600"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                        />
                      </svg>
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      Start Testing
                    </h3>
                    <p className="text-gray-600 max-w-sm">
                      Send a message to test the LLM server. Responses are
                      generated securely using Phala Cloud's GPU TEE.
                    </p>
                  </div>
                ) : (
                  messages.map((msg) => (
                    <motion.div
                      key={msg.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex ${
                        msg.role === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      <div
                        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                          msg.role === 'user'
                            ? 'bg-gradient-to-r from-violet-400 to-purple-400 text-white rounded-tr-none'
                            : msg.role === 'error'
                            ? 'bg-red-50 border border-red-200 text-red-800 rounded-tl-none'
                            : 'bg-gray-50 border border-gray-200 text-gray-900 rounded-tl-none'
                        }`}
                      >
                        <div className="text-sm whitespace-pre-wrap break-words">
                          {msg.content}
                        </div>
                        <div className="flex items-center gap-2 mt-2 text-xs opacity-70">
                          <span>
                            {msg.timestamp.toLocaleTimeString([], {
                              hour: '2-digit',
                              minute: '2-digit',
                            })}
                          </span>
                          {msg.metadata?.usage && (
                            <span>
                              • {msg.metadata.usage.total_tokens} tokens
                            </span>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div className="border-t border-gray-200 p-4">
                <div className="flex items-end gap-3">
                  <div className="flex-1">
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSend();
                        }
                      }}
                      placeholder="Type your message... (Shift+Enter for new line)"
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-violet-400 focus:border-transparent resize-none"
                      rows="3"
                      disabled={isLoading}
                    />
                  </div>
                  <button
                    onClick={handleSend}
                    disabled={!input.trim() || isLoading}
                    className="px-6 py-3 bg-gradient-to-r from-violet-400 to-purple-400 text-white rounded-lg font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity flex items-center gap-2"
                  >
                    {isLoading ? (
                      <>
                        <svg
                          className="animate-spin h-5 w-5"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          ></circle>
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          ></path>
                        </svg>
                        <span>Sending...</span>
                      </>
                    ) : (
                      <>
                        <svg
                          className="w-5 h-5"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                          />
                        </svg>
                        <span>Send</span>
                      </>
                    )}
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Connected to LLM server at localhost:3001
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TestingPage;

