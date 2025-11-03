"use client";

import React, { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useAccount, useConfig, useReadContract } from 'wagmi';
import { readContract } from '@wagmi/core';
import { encode } from 'gpt-tokenizer';
import { MessageSquare } from 'lucide-react';
import {
  useCheckUserCredits,
  useCheckBundlePrice,
  useBuyCredits,
  useUsePrompt,
} from '@/lib/contracts/creditUse';
import { useGetTotalLLMs } from '@/lib/contracts/creditUse/reads/useGetTotalLLMs';
import { useUserSessionCount } from '@/lib/contracts/creditUse/reads/useChatReads';
import { 
  useCreateChatSession, 
  useStoreMessageExchange, 
  useDeleteChatSession 
} from '@/lib/contracts/creditUse/writes/useChatFunctions';
import { useInference } from '../hooks/useInference';
import { useCheckINFTAuthorization } from '../hooks/useINFT';
import { Navbar } from '@/components/Navbar';
import { encryptMessage, decryptMessage } from '@/lib/encryption';
import ABI from '../utils/abi.json';
import { CONTRACT_ADDRESS } from '../utils/address';

interface Message {
  id: number;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

interface ChatSession {
  id: number;
  messageCount: number;
  preview: string;
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

const ChatPageOnChain = () => {
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeSession, setActiveSession] = useState<number | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [isSavingToChain, setIsSavingToChain] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<ChatSession | null>(null);

  const [hostedModels, setHostedModels] = useState<HostedModel[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  const { address, isConnected } = useAccount();
  const config = useConfig();
  
  const { totalLLMs } = useGetTotalLLMs();
  const { sessionCount, refetch: refetchSessionCount } = useUserSessionCount(address);

  // Contract hooks
  const { data: myCredits, refetch: refetchMyCredits } = useCheckUserCredits(address);
  const { data: bundlePrice } = useCheckBundlePrice();
  const { buyCredits, isWriting: isBuying, isConfirmed: isBuyConfirmed, resetWrite: resetBuy } = useBuyCredits();
  const { usePrompt, isWriting: isUsingPrompt, resetWrite: resetUsePrompt, isConfirming, isConfirmed } = useUsePrompt();
  const { createSession, isWriting: isCreating, isConfirmed: isCreateConfirmed } = useCreateChatSession();
  const { storeMessages, isWriting: isStoring, isConfirmed: isStoreConfirmed } = useStoreMessageExchange();
  const { deleteSession: deleteSessionContract, isConfirmed: isDeleteConfirmed } = useDeleteChatSession();
  
  const { infer: runINFTInference, isInferring: isINFTInferring } = useInference();
  const { isAuthorized: hasINFT, refetch: refetchINFT } = useCheckINFTAuthorization(1, address);
  const [useINFTInference, setUseINFTInference] = useState(true);

  const isConfirmedRef = useRef<boolean>(false);
  useEffect(() => {
    isConfirmedRef.current = !!isConfirmed;
  }, [isConfirmed]);

  useEffect(() => {
    if (isBuyConfirmed) {
      refetchMyCredits?.();
    }
  }, [isBuyConfirmed, refetchMyCredits]);
  
  useEffect(() => {
    if (address) {
      refetchINFT?.();
    }
  }, [address, refetchINFT]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Fetch hosted models
  useEffect(() => {
    const fetchHostedModels = async () => {
      if (totalLLMs === undefined) {
        setHostedModels([]);
        return;
      }

      setIsLoadingModels(true);
      
      try {
        const models: HostedModel[] = [];
        
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
        
        const uniqueModels: HostedModel[] = [];
        const seenNames = new Set<string>();
        
        for (const model of models) {
          if (!seenNames.has(model.modelName)) {
            seenNames.add(model.modelName);
            uniqueModels.push(model);
          }
        }
        
        setHostedModels(uniqueModels);
        
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

  // Load user's chat sessions from blockchain
  useEffect(() => {
    const loadSessions = async () => {
      if (!isConnected || !address || sessionCount === undefined) {
        setSessions([]);
        return;
      }

      setIsLoadingSessions(true);
      try {
        const count = Number(sessionCount);
        const loadedSessions: ChatSession[] = [];

        for (let i = 0; i < count; i++) {
          try {
            // Get message count for this session
            const messageCount = await readContract(config, {
              address: CONTRACT_ADDRESS as `0x${string}`,
              abi: ABI,
              functionName: 'getSessionMessageCount',
              args: [BigInt(i)],
              account: address,
            }) as bigint;

            if (messageCount && Number(messageCount) > 0) {
              // Get first message as preview
              const sessionMessages = await readContract(config, {
                address: CONTRACT_ADDRESS as `0x${string}`,
                abi: ABI,
                functionName: 'getSessionMessages',
                args: [BigInt(i)],
                account: address,
              }) as any[];

              let preview = `Session ${i + 1}`;
              if (sessionMessages && sessionMessages.length > 0) {
                try {
                  const firstMsg = sessionMessages[0];
                  const decrypted = decryptMessage(firstMsg.encryptedContent || firstMsg[0], address);
                  preview = decrypted.substring(0, 30) + (decrypted.length > 30 ? '...' : '');
                } catch (e) {
                  console.error('Failed to decrypt preview:', e);
                }
              }

              loadedSessions.push({
                id: i,
                messageCount: Number(messageCount),
                preview,
              });
            }
          } catch (error) {
            console.error(`Failed to load session ${i}:`, error);
          }
        }

        setSessions(loadedSessions);
      } catch (error) {
        console.error('Error loading sessions:', error);
      } finally {
        setIsLoadingSessions(false);
      }
    };

    loadSessions();
  }, [isConnected, address, sessionCount, config, isStoreConfirmed, isCreateConfirmed, isDeleteConfirmed]);

  // Load messages from a session
  const loadSession = async (session: ChatSession) => {
    if (!address) return;

    setIsLoadingMessages(true);
    try {
      const sessionMessages = await readContract(config, {
        address: CONTRACT_ADDRESS as `0x${string}`,
        abi: ABI,
        functionName: 'getSessionMessages',
        args: [BigInt(session.id)],
        account: address,
      }) as any[];

      if (sessionMessages) {
        const decryptedMessages: Message[] = sessionMessages.map((msg: any, index: number) => {
          const encryptedContent = msg.encryptedContent || msg[0];
          const timestamp = msg.timestamp || msg[1];
          const isUser = msg.isUser !== undefined ? msg.isUser : msg[2];

          try {
            const decrypted = decryptMessage(encryptedContent, address);
            return {
              id: Date.now() + index,
              text: decrypted,
              isUser,
              timestamp: new Date(Number(timestamp) * 1000),
            };
          } catch (e) {
            console.error('Failed to decrypt message:', e);
            return {
              id: Date.now() + index,
              text: '[Encrypted message]',
              isUser,
              timestamp: new Date(Number(timestamp) * 1000),
            };
          }
        });

        setMessages(decryptedMessages);
        setActiveSession(session.id);
      }
    } catch (error) {
      console.error('Error loading session:', error);
      setMessages([]);
    } finally {
      setIsLoadingMessages(false);
    }
  };

  // Start new chat
  const startNewChat = async () => {
    if (!isConnected || !address) {
      alert('Please connect your wallet first');
      return;
    }

    try {
      await createSession();
      // Wait for confirmation
      await new Promise((resolve) => {
        const checkInterval = setInterval(() => {
          if (isCreateConfirmed) {
            clearInterval(checkInterval);
            resolve(true);
          }
        }, 500);
      });

      refetchSessionCount?.();
      setMessages([]);
      setActiveSession(null);
    } catch (error) {
      console.error('Error creating session:', error);
      alert('Failed to create new chat session');
    }
  };

  // Delete session
  const confirmDelete = async () => {
    if (!sessionToDelete) return;

    try {
      await deleteSessionContract(BigInt(sessionToDelete.id));
      
      // Wait for confirmation
      await new Promise((resolve) => {
        const checkInterval = setInterval(() => {
          if (isDeleteConfirmed) {
            clearInterval(checkInterval);
            resolve(true);
          }
        }, 500);
      });

      if (activeSession === sessionToDelete.id) {
        setMessages([]);
        setActiveSession(null);
      }

      refetchSessionCount?.();
    } catch (error) {
      console.error('Error deleting session:', error);
      alert('Failed to delete session');
    } finally {
      setShowDeleteModal(false);
      setSessionToDelete(null);
    }
  };

  const handleBuyBundle = async () => {
    if (!isConnected || !bundlePrice) return;
    try {
      resetBuy();
      await buyCredits(bundlePrice as bigint);
    } catch (e) {
      // no-op
    }
  };

  // Handle sending message
  const handleSendMessage = async () => {
    if (!message.trim() || !address) return;

    if (isSavingToChain) {
      alert('Please wait while your previous message is being saved...');
      return;
    }

    const userMessage: Message = {
      id: Date.now(),
      text: message,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const userMessageText = message;
    setMessage('');

    if (!isConnected) {
      setMessages(prev => prev.concat({ 
        id: Date.now() + 1, 
        text: 'Please connect your wallet to continue.', 
        isUser: false, 
        timestamp: new Date() 
      }));
      return;
    }

    // Deduct tokens if needed
    if (!hasINFT || !useINFTInference) {
      try {
        resetUsePrompt();
        const tokenIds = encode(userMessageText);
        const tokensUsed = BigInt(tokenIds.length);
        await usePrompt(BigInt(selectedModelId || 0), tokensUsed);
        
        const waitForConfirmation = async (timeoutMs = 120000) => {
          const start = Date.now();
          while (!isConfirmedRef.current) {
            if (Date.now() - start > timeoutMs) {
              throw new Error('Timed out waiting for confirmation');
            }
            await new Promise((r) => setTimeout(r, 500));
          }
        };

        await waitForConfirmation();
        refetchMyCredits?.();
      } catch (e: any) {
        const errMsg = e?.message || 'Transaction failed';
        setMessages(prev => prev.concat({ 
          id: Date.now() + 1, 
          text: errMsg, 
          isUser: false, 
          timestamp: new Date() 
        }));
        return;
      }
    }

    // Get AI response
    setIsGenerating(true);
    try {
      if (selectedModelId === null) {
        throw new Error('Please select a model first');
      }
      
      const inferenceResult = await runINFTInference(selectedModelId, userMessageText, address);
      const aiText = inferenceResult?.output || 'No response';
      const aiMessage: Message = { 
        id: Date.now() + 1, 
        text: aiText, 
        isUser: false, 
        timestamp: new Date() 
      };
      
      setMessages(prev => [...prev, aiMessage]);

      // Store on blockchain
      setIsSavingToChain(true);
      try {
        // Encrypt messages
        const encryptedUser = encryptMessage(userMessageText, address);
        const encryptedAI = encryptMessage(aiText, address);

        // Get or create session
        let sessionId = activeSession;
        if (sessionId === null) {
          await createSession();
          const count = await refetchSessionCount();
          sessionId = Number(count?.data || 0) - 1;
          setActiveSession(sessionId);
        }

        // Store encrypted messages
        await storeMessages(BigInt(sessionId), encryptedUser, encryptedAI);
        
        console.log('Messages stored on-chain successfully');
      } catch (storageError) {
        console.error('Failed to store on chain:', storageError);
      } finally {
        setIsSavingToChain(false);
      }
    } catch (err: any) {
      console.error('Inference error:', err);
      setMessages(prev => prev.concat({ 
        id: Date.now() + 1, 
        text: `Error: ${err?.message || 'Unknown error'}`, 
        isUser: false, 
        timestamp: new Date() 
      }));
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="relative flex h-screen font-inter">
      <Navbar hideLogo={true} hasSidebar={true} />
      
      {/* Left Sidebar */}
      <div className="fixed left-0 top-0 h-screen w-56 bg-white border-r border-gray-200 flex flex-col z-[1001]">
        <div className="px-4 pt-6 pb-4 flex justify-center">
          <span className="text-4xl bg-gradient-to-r from-violet-400 to-purple-300 text-transparent bg-clip-text" style={{ fontFamily: 'var(--font-pacifico)' }}>
            TeeTee
          </span>
        </div>
        
        {/* Model Selection */}
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

          {/* Tokens and Buy */}
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
          <div className="mt-3 flex items-center justify-between px-2 py-2 bg-gray-50 rounded-lg">
            <label htmlFor="inft-toggle" className="flex items-center gap-2 cursor-pointer flex-1">
              <input
                id="inft-toggle"
                type="checkbox"
                checked={useINFTInference}
                onChange={(e) => setUseINFTInference(e.target.checked)}
                className="w-4 h-4 text-violet-600 bg-white border-gray-300 rounded focus:ring-violet-500 focus:ring-2 cursor-pointer"
              />
              <span className="text-xs text-gray-700 font-medium">
                Use hosted INFT
                <span className="block text-[10px] text-gray-500 font-normal">(No Tokens Used)</span>
              </span>
            </label>
          </div>
          
          {/* New Chat */}
          <button
            onClick={startNewChat}
            disabled={isCreating}
            className="w-full mt-3 flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium text-white bg-gradient-to-r from-violet-400 to-purple-300 hover:opacity-90 rounded-lg transition-opacity disabled:opacity-50"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>{isCreating ? 'Creating...' : 'New Chat'}</span>
          </button>
        </div>
        
        {/* Chat History */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-2">
            <h3 className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              On-Chain History
            </h3>
            {isLoadingSessions ? (
              <div className="p-4 text-center text-gray-500 text-sm">Loading...</div>
            ) : sessions.length > 0 ? (
              <div className="space-y-1">
                {sessions.map((session) => (
                  <div key={session.id} className="relative group">
                    <button
                      onClick={() => loadSession(session)}
                      disabled={isLoadingMessages}
                      className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-colors ${
                        activeSession === session.id
                          ? 'bg-violet-100 text-violet-900'
                          : 'text-gray-700 hover:bg-gray-50'
                      } disabled:opacity-50`}
                    >
                      <div className="flex items-start gap-2">
                        <MessageSquare size={16} className="mt-0.5 flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <div className="truncate font-medium">{session.preview}</div>
                          <div className="text-xs text-gray-500 mt-0.5 truncate">
                            {session.messageCount} msgs
                          </div>
                        </div>
                      </div>
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSessionToDelete(session);
                        setShowDeleteModal(true);
                      }}
                      className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-md text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors opacity-0 group-hover:opacity-100"
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
                <p className="text-sm">No chats yet</p>
                <p className="text-xs mt-1">Start chatting!</p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Delete Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1001]">
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Delete Chat?</h3>
            <p className="text-gray-600 mb-6">
              This will delete session #{sessionToDelete?.id}. This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowDeleteModal(false)}
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

      {/* Main Chat Area */}
      <main className="ml-56 w-[calc(100%-224px)] h-screen overflow-hidden flex flex-col bg-gradient-to-l from-violet-400/20 via-white to-purple-300/20 pt-20">
        <div className="flex-1 flex flex-col overflow-auto">
          <div className="fixed left-1/2 -translate-x-1/2 top-20 bottom-0 w-full max-w-4xl px-8 py-8 flex flex-col overflow-y-auto">
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
                      <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
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
                <p className="text-lg text-gray-600">Fully on-chain encrypted chat storage</p>
              </div>
            )}
          
            {/* Input */}
            <div className="mt-auto pb-6">
              <div className="relative w-full mx-auto">
                <div className="flex items-center bg-white border border-gray-200 rounded-full px-4 py-3 shadow-lg hover:shadow-xl transition-shadow">
                  <button className="flex-shrink-0 p-1 text-gray-500 hover:text-gray-700 transition-colors">
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                    </svg>
                  </button>

                  <input
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        handleSendMessage();
                      }
                    }}
                    placeholder="Ask anything (stored on-chain)"
                    className="flex-1 px-3 py-1 text-gray-900 placeholder-gray-500 bg-transparent border-none outline-none resize-none font-inter"
                  />

                  <button 
                    onClick={handleSendMessage}
                    className="flex-shrink-0 p-1 text-gray-500 hover:text-gray-700 transition-colors disabled:opacity-50"
                    disabled={isUsingPrompt || isGenerating || isINFTInferring || isSavingToChain || selectedModelId === null || !message.trim()}
                  >
                    {isINFTInferring || isGenerating || isSavingToChain ? (
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

                {isSavingToChain && (
                  <div className="mt-2 flex items-center justify-center gap-2 text-xs text-gray-500">
                    <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Saving to blockchain...</span>
                  </div>
                )}
                {!isSavingToChain && messages.length > 0 && isConnected && (
                  <div className="mt-2 flex items-center justify-center gap-1 text-xs text-gray-400">
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <span>Stored on blockchain</span>
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

export default ChatPageOnChain;

