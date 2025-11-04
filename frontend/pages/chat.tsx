"use client";

import React, { useEffect, useRef, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { useAccount, useConfig } from 'wagmi';
import { readContract } from '@wagmi/core';
import { encode, isWithinTokenLimit } from 'gpt-tokenizer';
import { MessageSquare } from 'lucide-react';
import {
  useCheckUserCredits,
  useCheckBundlePrice,
  useBuyCredits,
  useUsePrompt,
} from '@/lib/contracts/creditUse';
import { useGetTotalLLMs } from '@/lib/contracts/creditUse/reads/useGetTotalLLMs';
import { useInference } from '../hooks/useInference';
import { useCheckINFTAuthorization } from '../hooks/useINFT';
import { Navbar } from '@/components/Navbar';
import ABI from '../utils/abi.json';
import { CONTRACT_ADDRESS } from '../utils/address';

interface Conversation {
  _id: string;
  walletAddress: string;
  filename: string;
  preview?: string; // First user message preview
  rootHash: string | null;
  txHash: string;
  messageCount: number;
  createdAt: string;
  updatedAt: string;
}

interface Message {
  id: number;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

interface HostedModel {
  id: number;
  modelName: string;
  host1: string;
  host2: string;
  shardUrl1: string;
  shardUrl2: string;
  isComplete: boolean;
}

const ChatPage = () => {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeConversation, setActiveConversation] = useState<string | null>(null);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isSavingToStorage, setIsSavingToStorage] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 0G storage: auto-save transcript after each message exchange
  const [chatFilename, setChatFilename] = useState<string>(`chat_${Date.now()}.txt`);
  
  // Delete modal state
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [chatToDelete, setChatToDelete] = useState<Conversation | null>(null);

  // Hosted models from blockchain
  const [hostedModels, setHostedModels] = useState<HostedModel[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  const { address, isConnected } = useAccount();
  const config = useConfig();
  
  // Get total LLMs from contract
  const { totalLLMs } = useGetTotalLLMs();

  // Contract: credits and buy
  const { data: myCredits, refetch: refetchMyCredits } = useCheckUserCredits(address);
  const { data: bundlePrice } = useCheckBundlePrice();
  const { buyCredits, isWriting: isBuying, isConfirmed: isBuyConfirmed, resetWrite: resetBuy } = useBuyCredits();
  const { usePrompt, isWriting: isUsingPrompt, resetWrite: resetUsePrompt, isConfirming, isConfirmed } = useUsePrompt();
  
  // INFT inference hook
  const { infer: runINFTInference, isInferring: isINFTInferring } = useInference();
  
  // Check if user has INFT authorization
  // Only recognize INFTs issued by the specified address (set in env)
  const ALLOWED_INFT_ISSUER = process.env.NEXT_PUBLIC_INFT_ISSUER_ADDRESS; // Your wallet address that mints INFTs
  const { isAuthorized: hasINFT, refetch: refetchINFT } = useCheckINFTAuthorization(
    1, 
    address,
    ALLOWED_INFT_ISSUER // Only accept INFTs from this issuer
  );
  
  // Toggle for using INFT vs token-based inference
  const [useINFTInference, setUseINFTInference] = useState(true);

  // Track latest confirmation state to avoid stale closures while waiting
  const isConfirmedRef = useRef<boolean>(false);
  useEffect(() => {
    isConfirmedRef.current = !!isConfirmed;
  }, [isConfirmed]);

  useEffect(() => {
    if (isBuyConfirmed) {
      refetchMyCredits?.();
    }
  }, [isBuyConfirmed, refetchMyCredits]);
  
  // Refetch INFT status when address changes
  useEffect(() => {
    if (address) {
      refetchINFT?.();
    }
  }, [address, refetchINFT]);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Fetch all complete hosted models from blockchain
  useEffect(() => {
    const fetchHostedModels = async () => {
      if (totalLLMs === undefined) {
        setHostedModels([]);
        return;
      }

      setIsLoadingModels(true);
      
      try {
        const models: HostedModel[] = [];
        
        // Fetch all LLMs and filter for complete ones
        for (let i = 0; i < Number(totalLLMs); i++) {
          try {
            const data = await readContract(config, {
              address: CONTRACT_ADDRESS as `0x${string}`,
              abi: ABI,
              functionName: 'getHostedLLM',
              args: [BigInt(i)]
            }) as any;
            
            if (data) {
              const host1 = data.host1 || data[0] || '';
              const host2 = data.host2 || data[1] || '';
              const isComplete = data.isComplete !== undefined ? data.isComplete : (data[10] !== undefined ? data[10] : false);
              
              // Only add complete models to the list
              if (isComplete || (host1 !== '0x0000000000000000000000000000000000000000' && host2 !== '0x0000000000000000000000000000000000000000')) {
                models.push({
                  id: i,
                  modelName: data.modelName || data[4] || 'Unknown Model',
                  host1,
                  host2,
                  shardUrl1: data.shardUrl1 || data[2] || '',
                  shardUrl2: data.shardUrl2 || data[3] || '',
                  isComplete: true
                });
              }
            }
          } catch (error) {
            console.error(`Failed to fetch LLM ${i}:`, error);
          }
        }
        
        // Deduplicate models by name - keep the first occurrence
        const uniqueModels: HostedModel[] = [];
        const seenNames = new Set<string>();
        
        for (const model of models) {
          if (!seenNames.has(model.modelName)) {
            seenNames.add(model.modelName);
            uniqueModels.push(model);
          }
        }
        
        setHostedModels(uniqueModels);
        
        // Auto-select first model if none selected
        if (uniqueModels.length > 0 && !selectedModel) {
          setSelectedModel(uniqueModels[0].modelName);
          setSelectedModelId(uniqueModels[0].id);
        }
      } catch (error) {
        console.error('Error fetching hosted models:', error);
      } finally {
        setIsLoadingModels(false);
      }
    };

    fetchHostedModels();
  }, [totalLLMs, config]);

  // Prevent browser close/refresh during save
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isSavingToStorage) {
        e.preventDefault();
        e.returnValue = 'Your chat is still being saved to 0G Storage. Are you sure you want to leave?';
        return e.returnValue;
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isSavingToStorage]);

  // Prevent navigation to other pages during save
  useEffect(() => {
    const handleRouteChange = (url: string) => {
      if (isSavingToStorage) {
        const confirmed = window.confirm(
          'Your chat is still being saved to 0G Storage. ' +
          'Leaving this page now may interrupt the save. Continue anyway?'
        );
        if (!confirmed) {
          router.events.emit('routeChangeError');
          throw 'Route change aborted by user';
        }
      }
    };

    router.events.on('routeChangeStart', handleRouteChange);
    return () => {
      router.events.off('routeChangeStart', handleRouteChange);
    };
  }, [isSavingToStorage, router]);

  // Chat sessions from local JSON storage
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);

  // Fetch chat sessions for the connected wallet from JSON storage
  // Each entry in JSON represents a conversation with its LATEST root hash
  const fetchChatSessions = async () => {
    if (!address) return;
    
    setIsLoadingSessions(true);
    try {
      const res = await fetch(`/api/get-chat-sessions?walletAddress=${address}`);
      if (!res.ok) {
        throw new Error('Failed to fetch chat sessions');
      }
      const data = await res.json();
      // These sessions contain the latest root hash for each conversation
      setConversations(data.sessions || []);
      console.log(`Loaded ${data.sessions?.length || 0} chat sessions from JSON storage`);
    } catch (err) {
      console.error('Error fetching chat sessions:', err);
    } finally {
      setIsLoadingSessions(false);
    }
  };

  // Fetch sessions when wallet connects
  useEffect(() => {
    if (isConnected && address) {
      fetchChatSessions();
    } else {
      setConversations([]);
    }
  }, [isConnected, address]);

  // Auto-save to local JSON and MongoDB after each message exchange
  const autoSaveToStorage = async (messagesToSave: Message[]) => {
    if (!isConnected || !address || messagesToSave.length === 0) return;
    
    setIsSavingToStorage(true);
    try {
      // Convert all messages to the format expected by the API
      const payloadMessages = messagesToSave.map((m) => ({
        role: m.isUser ? 'user' : 'assistant',
        content: m.text,
        timestamp: m.timestamp instanceof Date ? m.timestamp.getTime() : new Date(m.timestamp).getTime(),
      }));

      // Save to local JSON and MongoDB through the API
      const res = await fetch('/api/chat-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          messages: payloadMessages, 
          filename: chatFilename,
          walletAddress: address,
          sessionId: activeConversation,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || 'Failed to save chat session');
      
      if (!activeConversation && data.sessionId) {
        setActiveConversation(data.sessionId);
        console.log('New chat session created:', data.sessionId);
      } else {
        console.log('Updated existing session:', activeConversation, 'with new root hash:', data.rootHash);
      }
      
      await fetchChatSessions();
      
    } catch (err: any) {
      console.error('Auto-save error:', err);
    } finally {
      setIsSavingToStorage(false);
    }
  };

  // Load messages from a saved chat session
  // Uses the LATEST root hash from JSON to download full conversation from 0G
  const loadChatSession = async (session: Conversation) => {
    // Check if currently saving - warn user
    if (isSavingToStorage) {
      const confirmed = window.confirm(
        'Your current chat is still being saved to 0G Storage. ' +
        'Loading another chat now may interrupt the save. Continue anyway?'
      );
      if (!confirmed) return;
    }

    if (!session.rootHash) {
      console.error('No root hash available for this session');
      return;
    }

    setIsLoadingMessages(true);
    try {
      console.log(`Loading chat session ${session._id} from 0G with root hash: ${session.rootHash}`);
      
      // Download the ENTIRE conversation from 0G using the latest root hash
      const res = await fetch(`/api/get-chat-messages?rootHash=${session.rootHash}`);
      if (!res.ok) {
        throw new Error('Failed to load chat messages');
      }
      const data = await res.json();
      
      // Convert the API response to Message format
      const loadedMessages: Message[] = data.messages.map((msg: any, index: number) => ({
        id: Date.now() + index,
        text: msg.content,
        isUser: msg.role === 'user',
        timestamp: new Date(msg.timestamp),
      }));

      setMessages(loadedMessages);
      setActiveConversation(session._id);
      setChatFilename(session.filename);
      
      console.log(`Loaded ${loadedMessages.length} messages from 0G Storage`);
    } catch (err: any) {
      console.error('Error loading chat session:', err);
      setMessages([]);
    } finally {
      setIsLoadingMessages(false);
    }
  };

  // Start a new chat
  const startNewChat = () => {
    // Check if currently saving - warn user
    if (isSavingToStorage) {
      const confirmed = window.confirm(
        'Your current chat is still being saved to 0G Storage. ' +
        'Starting a new chat now may interrupt the save. Continue anyway?'
      );
      if (!confirmed) return;
    }

    setMessages([]);
    setActiveConversation(null);
    setChatFilename(`chat_${Date.now()}.txt`);
  };

  // Handle delete button click
  const handleDeleteClick = (e: React.MouseEvent, chat: Conversation) => {
    e.stopPropagation(); // Prevent loading the chat
    setChatToDelete(chat);
    setShowDeleteModal(true);
  };

  // Confirm deletion
  const confirmDelete = async () => {
    if (!chatToDelete) return;

    try {
      const res = await fetch('/api/delete-chat-session', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: chatToDelete._id }),
      });

      if (!res.ok) {
        throw new Error('Failed to delete chat session');
      }

      // If the deleted chat was active, clear the messages
      if (activeConversation === chatToDelete._id) {
        setMessages([]);
        setActiveConversation(null);
        setChatFilename(`chat_${Date.now()}.txt`);
      }

      // Refresh the chat list
      await fetchChatSessions();
      
      console.log('Chat session deleted:', chatToDelete._id);
    } catch (err) {
      console.error('Error deleting chat session:', err);
      alert('Failed to delete chat session. Please try again.');
    } finally {
      setShowDeleteModal(false);
      setChatToDelete(null);
    }
  };

  // Cancel deletion
  const cancelDelete = () => {
    setShowDeleteModal(false);
    setChatToDelete(null);
  };

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

    // Check if currently saving - prevent new message
    if (isSavingToStorage) {
      alert('Please wait while your previous message is being saved to 0G Storage...');
      return;
    }

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

    // Only deduct tokens if user doesn't have INFT authorization OR user chose not to use INFT
    if (!hasINFT || !useINFTInference) {
      try {
        resetUsePrompt();
        // User signs/submits the tx
        // Compute token usage for this user message (OpenAI-style tokenizer)
        const tokenIds = encode(message);
        const tokensUsed = BigInt(tokenIds.length);
        await usePrompt(0n, tokensUsed);
        // Wait until the transaction is actually confirmed on-chain before proceeding
        const waitForConfirmation = async (timeoutMs = 120000) => {
          const start = Date.now();
          while (!isConfirmedRef.current) {
            if (Date.now() - start > timeoutMs) {
              throw new Error('Timed out waiting for on-chain confirmation');
            }
            await new Promise((r) => setTimeout(r, 500));
          }
        };

        await waitForConfirmation();
        // Refresh credits after confirmation
        refetchMyCredits?.();
      } catch (e: any) {
        const errMsg = e?.message || 'Transaction was rejected or failed. No credits consumed.';
        setMessages(prev => prev.concat({ id: Date.now() + 1, text: errMsg, isUser: false, timestamp: new Date() }));
        return;
      }
    } else {
      console.log('User has INFT authorization and chose to use it - skipping token deduction');
    }

    // After signing/submitting the tx, call INFT inference
    setIsGenerating(true);
    try {
      // Use token ID 1 by default for testing (can be changed to selectedModelId later)
      const tokenId = 1;
      
      // Use INFT inference with token ID 1
      const inferenceResult = await runINFTInference(tokenId, message, address);
      
      const text = inferenceResult?.output || 'No response from INFT';
      const aiMessage: Message = { id: Date.now() + 1, text, isUser: false, timestamp: new Date() };
      const updatedMessages = [...messages, userMessage, aiMessage];
      setMessages(updatedMessages);
      
      // Log inference metadata
      if (inferenceResult?.metadata) {
        console.log('INFT Inference:', {
          provider: inferenceResult.metadata.provider,
          model: inferenceResult.metadata.model,
          proofHash: inferenceResult.metadata.proofHash
        });
      }
      
      // Auto-save to 0G storage after AI response
      await autoSaveToStorage(updatedMessages);
    } catch (err: any) {
      console.error('INFT Inference error:', err);
      setMessages(prev => prev.concat({ 
        id: Date.now() + 1, 
        text: `Error from INFT: ${err?.message || 'Unknown error'}. Please ensure you are authorized for the INFT token.`, 
        isUser: false, 
        timestamp: new Date() 
      }));
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="relative flex h-screen font-inter">
      {/* New Navbar */}
      <Navbar hideLogo={true} hasSidebar={true} />
      
      {/* Left Sidebar - ChatGPT Style */}
      <div className="fixed left-0 top-0 h-screen w-56 bg-white border-r border-gray-200 flex flex-col z-[1001]">
        {/* TeeTee Logo */}
        <div className="px-4 pt-6 pb-4 flex justify-center">
          <span className="text-4xl bg-gradient-to-r from-violet-400 to-purple-300 text-transparent bg-clip-text" style={{ fontFamily: 'var(--font-pacifico)' }}>
            TeeTee
          </span>
        </div>
        
        {/* Model Selection at the top */}
        <div className="p-4 border-b border-gray-200">
          <div className="relative">
            <button
              onClick={() => setShowModelDropdown(!showModelDropdown)}
              disabled={isLoadingModels || hostedModels.length === 0}
              className="w-full flex items-center justify-between gap-2 px-4 py-2.5 text-sm font-medium text-gray-700 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="flex items-center gap-2 min-w-0">
                {isLoadingModels ? (
                  <>
                    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Loading models...</span>
                  </>
                ) : hostedModels.length === 0 ? (
                  <span className="text-gray-500">No models available</span>
                ) : (
                  <span className="truncate">{selectedModel || 'Select a model'}</span>
                )}
              </div>
              <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {showModelDropdown && hostedModels.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-2 bg-white border border-gray-200 rounded-lg shadow-xl z-50 max-h-64 overflow-y-auto">
                {hostedModels.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => {
                      setSelectedModel(model.modelName);
                      setSelectedModelId(model.id);
                      setShowModelDropdown(false);
                    }}
                    className={`w-full text-left px-4 py-2.5 text-sm font-medium transition-colors first:rounded-t-lg last:rounded-b-lg ${
                      selectedModelId === model.id
                        ? 'bg-violet-50 text-violet-900'
                        : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    {model.modelName}
                  </button>
                ))}
              </div>
            )}
          </div>
          
          {/* No Models Available Message */}
          {!isLoadingModels && hostedModels.length === 0 && (
            <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-xs text-yellow-800 font-medium mb-1">No models available</p>
              <p className="text-xs text-yellow-700 mb-2">There are no complete hosted models yet.</p>
              <Link 
                href="/models" 
                className="text-xs text-violet-600 hover:text-violet-700 font-medium underline"
              >
                Go to Models page →
              </Link>
            </div>
          )}

          {/* Token Display and Buy Button */}
          <div className="mt-3 space-y-2">
            <div className="flex items-center justify-between text-xs px-2">
              <span className="text-gray-600">Tokens Left:</span>
              <span className="font-semibold text-gray-900">
                {isConnected ? (myCredits?.toString() || '0') : '-'}
              </span>
            </div>
            <button
              onClick={handleBuyBundle}
              disabled={!isConnected || isBuying}
              className="w-full px-3 py-2 text-xs font-medium text-violet-600 bg-violet-50 hover:bg-violet-100 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors"
            >
              {isBuying ? 'Buying...' : 'Buy Tokens (0.001 OG)'}
            </button>
          </div>
          
          {/* INFT Toggle */}
          <div className="mt-3 px-2 py-2 bg-gray-50 rounded-lg">
            <label htmlFor="inft-toggle" className={`flex items-center gap-2 ${hasINFT ? 'cursor-pointer' : 'cursor-not-allowed opacity-60'}`}>
              <input
                id="inft-toggle"
                type="checkbox"
                checked={useINFTInference}
                onChange={(e) => setUseINFTInference(e.target.checked)}
                disabled={!hasINFT}
                className={`w-4 h-4 text-violet-600 bg-white border-gray-300 rounded focus:ring-violet-500 focus:ring-2 ${hasINFT ? 'cursor-pointer' : 'cursor-not-allowed'}`}
              />
              <span className="text-xs text-gray-700 font-medium flex-1">
                Use hosted INFT
                <span className="block text-[10px] text-gray-500 font-normal">
                  {hasINFT ? '(No Tokens Used)' : '(You don\'t have a hosted model)'}
                </span>
              </span>
            </label>
          </div>
          
          {/* New Chat Button */}
          <button
            onClick={startNewChat}
            className="w-full mt-3 flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium text-white bg-gradient-to-r from-violet-400 to-purple-300 hover:opacity-90 rounded-lg transition-opacity"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>New Chat</span>
          </button>
        </div>
        
        {/* Chat History */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-2">
            <h3 className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              History
            </h3>
            {isLoadingSessions ? (
              <div className="p-4 text-center text-gray-500 text-sm">
                Loading...
              </div>
            ) : conversations.length > 0 ? (
              <div className="space-y-1">
                {conversations.map((chat) => (
                  <div key={chat._id} className="relative group">
                    <button
                      onClick={() => loadChatSession(chat)}
                      disabled={isLoadingMessages}
                      className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-colors ${
                        activeConversation === chat._id
                          ? 'bg-violet-100 text-violet-900'
                          : 'text-gray-700 hover:bg-gray-50'
                      } disabled:opacity-50`}
                    >
                      <div className="flex items-start gap-2">
                        <MessageSquare size={16} className="mt-0.5 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="truncate font-medium">
                            {chat.preview 
                              ? (chat.preview.length > 30 ? chat.preview.substring(0, 30) + '...' : chat.preview)
                              : chat.filename
                            }
                          </div>
                          <div className="text-xs text-gray-500 mt-0.5 truncate">
                            {chat.messageCount} msgs • {new Date(chat.createdAt).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setChatToDelete(chat);
                        setShowDeleteModal(true);
                      }}
                      className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-md text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors opacity-0 group-hover:opacity-100"
                      title="Delete chat"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 text-center text-gray-500">
                <p className="text-sm">No conversations yet</p>
                <p className="text-xs mt-1">Start chatting!</p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1001]">
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Delete Chat?</h3>
            <p className="text-gray-600 mb-6">
              Are you sure you want to delete "{chatToDelete?.filename}"? This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={cancelDelete}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="px-4 py-2 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-lg transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Chat Area - With left sidebar offset */}
      <main className="ml-56 w-[calc(100%-224px)] h-screen overflow-hidden flex flex-col bg-gradient-to-l from-violet-400/20 via-white to-purple-300/20 pt-20">

        {/* Centered Chat Container - Aligned with Navbar center */}
        <div className="flex-1 flex flex-col overflow-auto">
          <div className="fixed left-1/2 -translate-x-1/2 top-20 bottom-0 w-full max-w-4xl px-8 py-8 flex flex-col overflow-y-auto">
            {/* Loading state */}
            {isLoadingMessages ? (
              <div className="flex-1 flex flex-col items-center justify-center">
                <div className="animate-pulse text-gray-600">Loading messages...</div>
              </div>
            ) : messages.length > 0 ? (
              <div className="flex-1 flex flex-col space-y-4 pb-4">
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
                <div ref={messagesEndRef} />
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center">
                <h1 className="text-5xl font-bold text-gray-900 mb-4">
                  <span className="text-3xl mr-2">Welcome to</span> 
                  <span className="text-6xl bg-gradient-to-r from-violet-400 to-purple-300 text-transparent bg-clip-text" style={{ fontFamily: 'var(--font-pacifico)' }}>
                    TeeTee
                  </span>
                </h1>
                <p className="text-lg text-gray-600">This is a secure AI assistant running in a TEE.</p>
              </div>
            )}
          
            {/* Input Area */}
            <div className="mt-auto pb-6">
              <div className="relative w-full mx-auto">
                <div className="flex items-center bg-white border border-gray-200 rounded-full px-4 py-3 shadow-lg hover:shadow-xl transition-shadow">
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
                    className="flex-shrink-0 p-1 text-gray-500 hover:text-gray-700 transition-colors disabled:opacity-50"
                    disabled={isUsingPrompt || isGenerating || isINFTInferring || !message.trim()}
                    title={
                      isUsingPrompt ? 'Waiting for transaction…' : 
                      isINFTInferring || isGenerating ? 'INFT is processing...' : 
                      'Send'
                    }
                  >
                    {isINFTInferring || isGenerating ? (
                      <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    ) : (
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                      </svg>
                    )}
                  </button>
                </div>

                 {/* Auto-save indicator */}
                 {isSavingToStorage && (
                   <div className="mt-2 flex items-center justify-center gap-2 text-xs text-gray-500">
                     <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                       <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                       <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                     </svg>
                     <span>Saving to 0G Storage</span>
                   </div>
                 )}
                 {!isSavingToStorage && messages.length > 0 && isConnected && (
                   <div className="mt-2 flex items-center justify-center gap-1 text-xs text-gray-400">
                     <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                       <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                     </svg>
                     <span>Auto-saved to 0G Storage</span>
                   </div>
                 )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ChatPage;
