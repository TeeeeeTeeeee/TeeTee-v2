"use client";

import React, { useEffect, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { useAccount } from 'wagmi';
import {
  useCheckUserCredits,
  useCheckBundlePrice,
  useBuyCredits,
  useUsePrompt,
} from '@/lib/contracts/creditUse';

interface Conversation {
  id: number;
  title: string;
}

interface Message {
  id: number;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

const ChatPage = () => {
  const [isOpen, setIsOpen] = useState(true);
  const [selectedModel, setSelectedModel] = useState('gpt-5 nano');
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const OPENAI_MODEL = 'gpt-5-nano-2025-08-07';
  const [activeConversation, setActiveConversation] = useState<number | null>(null);

  const { address, isConnected } = useAccount();

  // Contract: credits and buy
  const { data: myCredits, refetch: refetchMyCredits } = useCheckUserCredits(address);
  const { data: bundlePrice } = useCheckBundlePrice();
  const { buyCredits, isWriting: isBuying, isConfirmed: isBuyConfirmed, resetWrite: resetBuy } = useBuyCredits();
  const { usePrompt, isWriting: isUsingPrompt, resetWrite: resetUsePrompt } = useUsePrompt();

  useEffect(() => {
    if (isBuyConfirmed) {
      refetchMyCredits?.();
    }
  }, [isBuyConfirmed, refetchMyCredits]);

  const models = [
    'gpt-5 nano',
    'gpt-5 micro',
    'gpt-5 small',
  ];

  // Sample conversations data (empty for now)
  const conversations: Conversation[] = [];

  const handleBuyBundle = async () => {
    if (!isConnected || !bundlePrice) return;
    try {
      resetBuy();
      await buyCredits(bundlePrice as bigint);
    } catch (e) {
      // no-op, surfaced via wallet UI
    }
  };

  // Handle sending a message
  const handleSendMessage = async () => {
    if (!message.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now(),
      text: message,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setMessage('');

    // Require wallet connection and signature first, then call model
    if (!isConnected) {
      setMessages(prev => prev.concat({ id: Date.now() + 1, text: 'Please connect your wallet to continue.', isUser: false, timestamp: new Date() }));
      return;
    }

    try {
      resetUsePrompt();
      // Await here so user signs/submits the tx before we proceed
      await usePrompt(0n);
      // Optionally refresh credits after submitting tx
      refetchMyCredits?.();
    } catch (e) {
      setMessages(prev => prev.concat({ id: Date.now() + 1, text: 'Transaction was rejected or failed. No credits consumed.', isUser: false, timestamp: new Date() }));
      return;
    }

    // After signing/submitting the tx, call the OpenAI API and return the response
    setIsGenerating(true);
    try {
      const history = messages.concat(userMessage).map((m) => ({ role: m.isUser ? 'user' : 'assistant', content: m.text }));
      const resp = await fetch('/api/chat-openai', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history, model: OPENAI_MODEL }),
      });
      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(errText || 'OpenAI request failed');
      }
      const data = await resp.json();
      const text = data?.text || 'No response';
      const aiMessage: Message = { id: Date.now() + 1, text, isUser: false, timestamp: new Date() };
      setMessages(prev => [...prev, aiMessage]);
    } catch (err: any) {
      setMessages(prev => prev.concat({ id: Date.now() + 1, text: `Error from model: ${err?.message || 'Unknown error'}`, isUser: false, timestamp: new Date() }));
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="relative flex h-screen font-inter">
      {/* Sidebar */}
      <motion.div 
        className="fixed left-0 top-0 bg-white flex flex-col h-screen border-r border-gray-200"
        animate={{
          width: isOpen ? "260px" : "60px"
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
      >
        {/* Logo and Title with Toggle */}
        <div className="p-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3 min-w-0">
            <Image 
              src="/images/TeeTee.png" 
              alt="TeeTee Logo" 
              width={32} 
              height={32}
              className="flex-shrink-0"
            />
            {isOpen && (
              <span className="text-xl" style={{ fontFamily: 'var(--font-pacifico)' }}>
                <span className="bg-gradient-to-r from-violet-400 to-purple-300 text-transparent bg-clip-text truncate">
                  TeeTee
                </span>
              </span>
            )}
          </Link>
          {/* Sidebar Toggle Button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="p-1.5 hover:bg-gray-100 rounded-md transition-colors flex-shrink-0"
          >
            <svg className="w-4 h-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>

        {/* Navigation Links */}
        <div className="px-2 py-2">
          <nav className="flex flex-col gap-1">
            <Link 
              href="/"
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-700 hover:bg-violet-200/50 cursor-pointer transition-colors"
              title={!isOpen ? "Home" : ""}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
              {isOpen && <span>Home</span>}
            </Link>
            <Link 
              href="/chat"
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm bg-gradient-to-r from-violet-400 to-violet-400 text-white transition-colors"
              title={!isOpen ? "Chat" : ""}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              {isOpen && <span>Chat</span>}
            </Link>
            <Link 
              href="/models"
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-700 hover:bg-violet-200/50 cursor-pointer transition-colors"
              title={!isOpen ? "Models" : ""}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              {isOpen && <span>Models</span>}
            </Link>
            <Link 
              href="/storage"
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-700 hover:bg-violet-200/50 cursor-pointer transition-colors"
              title={!isOpen ? "Storage" : ""}
            >
              <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m-9 0h10M5 4h14l-1 13a2 2 0 01-2 2H8a2 2 0 01-2-2L5 4z" />
              </svg>
              {isOpen && <span>Storage</span>}
            </Link>
          </nav>
        </div>

        {/* Tokens + Buy */}
        {isOpen && (
          <div className="px-4 py-2">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-gradient-to-r from-violet-200 to-violet-200 text-black rounded-full text-sm">
              <span>Tokens: {myCredits?.toString?.() ?? '-'}</span>
              <button
                onClick={handleBuyBundle}
                disabled={!isConnected || !bundlePrice || isBuying}
                className="ml-2 px-2 py-0.5 bg-violet-500 text-white rounded-full hover:bg-violet-600 disabled:opacity-50"
                title="Buy 1 bundle"
              >
                {isBuying ? 'Buying…' : 'Buy'}
              </button>
            </div>
          </div>
        )}

        {/* New Chat Button */}
        <div className="p-2">
          <button 
            className="w-full bg-white hover:bg-gray-50 text-gray-700 border border-gray-200 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-3"
            title={!isOpen ? "New chat" : ""}
          >
            <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            {isOpen && <span>New chat</span>}
          </button>
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto px-2">
          <div className="flex flex-col gap-1 min-h-0">
            {conversations.length > 0 ? (
              conversations.map((chat) => (
                <button
                  key={chat.id}
                  className="w-full text-left text-gray-700 hover:bg-gray-50 px-3 py-2.5 rounded-lg text-sm flex items-center gap-3"
                  title={!isOpen ? chat.title : ""}
                >
                  <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                  {isOpen && <span className="truncate">{chat.title}</span>}
                </button>
              ))
            ) : (
              isOpen && (
                <div className="flex flex-col items-center justify-center py-8 text-gray-500">
                  <svg className="w-8 h-8 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                  <p className="text-sm">No conversations yet.</p>
                </div>
              )
            )}
          </div>
        </div>

        {/* Connect Wallet Button */}
        <div className="border-t border-gray-200 p-2 mt-auto">
          <ConnectButton.Custom>
            {({
              account,
              chain,
              openAccountModal,
              openChainModal,
              openConnectModal,
              mounted,
            }) => {
              const ready = mounted;
              const connected = ready && account && chain;

              return (
                <div
                  {...(!ready && {
                    'aria-hidden': true,
                    style: { opacity: 0, pointerEvents: 'none', userSelect: 'none' },
                  })}
                >
                  {(() => {
                    if (!connected) {
                      return (
                        <button 
                          onClick={openConnectModal}
                          className="w-full bg-gradient-to-r from-violet-400 to-violet-400 text-white px-3 py-2.5 rounded-lg text-sm font-medium hover:opacity-90 transition-opacity flex items-center justify-center gap-3"
                          title={!isOpen ? "Connect Wallet" : ""}
                        >
                          <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z" />
                          </svg>
                          {isOpen && <span>Connect Wallet</span>}
                        </button>
                      );
                    }

                    return (
                      <div className="flex flex-col gap-2">
                        <button
                          onClick={openAccountModal}
                          className="w-full text-left text-gray-700 hover:bg-gray-50 px-3 py-2.5 rounded-lg text-sm flex items-center gap-3"
                          title={!isOpen ? account.displayName : ""}
                        >
                          <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z" />
                          </svg>
                          {isOpen && <span>{account.displayName}</span>}
                        </button>
                      </div>
                    );
                  })()}
                </div>
              );
            }}
          </ConnectButton.Custom>
        </div>
      </motion.div>

      {/* Main Chat Area */}
      <motion.main
        className="flex-1 h-screen overflow-hidden flex flex-col bg-gradient-to-l from-violet-400/20 via-white to-purple-300/20"
        animate={{
          marginLeft: isOpen ? "260px" : "60px",
          width: `calc(100% - ${isOpen ? "260px" : "60px"})`
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
      >
        {/* Top Bar with Model Selection */}
        <div className="p-4 flex-shrink-0">
          <div className="flex items-center gap-4">
            {/* Model Selection Dropdown */}
            <div className="relative">
              <button
                onClick={() => setShowModelDropdown(!showModelDropdown)}
                className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
              >
                <span>{selectedModel}</span>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {showModelDropdown && (
                <div className="absolute top-full left-0 mt-1 w-48 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
                  {models.map((model) => (
                    <button
                      key={model}
                      onClick={() => {
                        setSelectedModel(model);
                        setShowModelDropdown(false);
                      }}
                      className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 first:rounded-t-lg last:rounded-b-lg"
                    >
                      {model}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex-1 flex flex-col overflow-auto">
          <div className="max-w-4xl mx-auto px-4 py-8 flex-1 flex flex-col w-full">
            {/* Chat messages */}
            {messages.length > 0 ? (
              <div className="flex-1 flex flex-col space-y-4 overflow-y-auto pb-4">
                {messages.map((msg) => (
                  <div 
                    key={msg.id}
                    className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}
                  >
                    <div 
                      className={`max-w-[70%] px-4 py-3 rounded-2xl ${
                        msg.isUser 
                          ? 'bg-violet-400 text-white rounded-tr-none' 
                          : 'bg-white border border-gray-200 text-gray-800 rounded-tl-none'
                      }`}
                    >
                      <p className="text-sm">{msg.text}</p>
                      <div className="text-xs mt-1 opacity-70 text-right">
                        {new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center">
                <h1 className="text-3xl font-bold text-gray-900 flex items-baseline">
                  <span className="text-2xl mr-2">Welcome to</span> 
                  <span className="text-4xl bg-gradient-to-r from-violet-400 to-purple-300 text-transparent bg-clip-text" style={{ fontFamily: 'var(--font-pacifico)' }}>
                    TeeTee
                  </span>
                </h1>
                <p className="mt-2 text-gray-600">This is a secure AI assistant running in a TEE.</p>
              </div>
            )}
          
            {/* Input Area */}
            <div className="mt-auto pb-6">
              <div className="relative w-full max-w-3xl mx-auto">
                <div className="flex items-center bg-white border border-gray-200 rounded-full px-4 py-3 shadow-sm hover:shadow-md transition-shadow">
                  {/* File Attach Icon */}
                  <button className="flex-shrink-0 p-1 text-gray-500 hover:text-gray-700 transition-colors">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                    </svg>
                  </button>

                  {/* Input Field */}
                  <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        handleSendMessage();
                      }
                    }}
                    placeholder="Ask anything"
                    className="flex-1 px-3 py-1 text-gray-900 placeholder-gray-500 bg-transparent border-none outline-none resize-none font-inter"
                  />

                  {/* Send Icon */}
                  <button 
                    onClick={handleSendMessage}
                    className="flex-shrink-0 p-1 text-gray-500 hover:text-gray-700 transition-colors"
                    disabled={isUsingPrompt}
                    title={isUsingPrompt ? 'Waiting for transaction…' : 'Send'}
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  </button>
                </div>

                {/* Footer Text */}
                <div className="text-center mt-3">
                  <p className="text-xs text-gray-500">
                    All responses are verified through decentralized TEE computation.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.main>
    </div>
  );
};

export default ChatPage;
