"use client";

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { Navbar } from '../components/Navbar';
import { useGetIncompleteLLMs } from '../lib/contracts/creditUse/reads/useGetIncompleteLLMs';
import { useGetTotalLLMs } from '../lib/contracts/creditUse/reads/useGetTotalLLMs';
import { useAccount, useConfig } from 'wagmi';
import { readContract } from '@wagmi/core';
import ABI from '../utils/abi.json';
import { CONTRACT_ADDRESS } from '../utils/address';
import { ModelFilters } from '../components/ModelFilters';

interface IncompleteLLM {
  id: number;
  modelName: string;
  host1: string;
  host2?: string;
  shardUrl1: string;
  shardUrl2?: string;
  poolBalance: bigint;
  totalTimeHost1: bigint;
  totalTimeHost2?: bigint;
  isComplete?: boolean;
}

// LLM Icon mapping
const LLM_ICONS: Record<string, string> = {
  'TinyLlama-1.1B-Chat-v1.0': '/images/tinyllama.png',
  'Mistral-7B-Instruct-v0.3': 'https://vectorseek.com/wp-content/uploads/2023/12/Mistral-AI-Icon-Logo-Vector.svg-.png',
  'Qwen2.5-7B-Instruct': 'https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/qwen-color.png',
  'Phi-3-Mini-4K-Instruct': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Microsoft_icon.svg/2048px-Microsoft_icon.svg.png',
  'Gemma-2-2B-Instruct': 'https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/gemma-color.png',
  'GPT-3.5-Turbo': 'https://uxwing.com/wp-content/themes/uxwing/download/brands-and-social-media/openai-icon.png',
  'Claude-3-Haiku': 'https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/light/anthropic.png',
  'DeepSeek-V2-Lite': 'https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/deepseek-color.png'
};

const DEFAULT_LLM_ICON = 'https://www.redpill.ai/_next/image?url=https%3A%2F%2Ft0.gstatic.com%2FfaviconV2%3Fclient%3DSOCIAL%26type%3DFAVICON%26fallback_opts%3DTYPE%2CSIZE%2CURL%26url%3Dhttps%3A%2F%2Fhuggingface.co%2F%26size%3D32&w=48&q=75';

const getModelIcon = (modelName: string): string => {
  return LLM_ICONS[modelName] || DEFAULT_LLM_ICON;
};

const ModelsPage = () => {
  const router = useRouter();
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<'all' | 'available' | 'incomplete' | 'mymodels'>('all');
  const [allModels, setAllModels] = useState<IncompleteLLM[]>([]);
  const [isLoadingAllModels, setIsLoadingAllModels] = useState(false);
  const [incompleteLLMDetails, setIncompleteLLMDetails] = useState<IncompleteLLM[]>([]);
  const [isLoadingIncomplete, setIsLoadingIncomplete] = useState(false);
  const [myHostedModels, setMyHostedModels] = useState<IncompleteLLM[]>([]);
  const [isLoadingMyModels, setIsLoadingMyModels] = useState(false);

  // Smart contract hooks
  const { incompleteLLMs } = useGetIncompleteLLMs();
  const { totalLLMs } = useGetTotalLLMs();
  const { address: connectedAddress } = useAccount();
  const config = useConfig();

  // Fetch ALL models (complete + incomplete)
  useEffect(() => {
    const fetchAllModels = async () => {
      if (totalLLMs === undefined) {
        setAllModels([]);
        return;
      }

      setIsLoadingAllModels(true);
      
      try {
        const models: IncompleteLLM[] = [];
        
        for (let i = 0; i < Number(totalLLMs); i++) {
          try {
            const data = await readContract(config, {
              address: CONTRACT_ADDRESS as `0x${string}`,
              abi: ABI,
              functionName: 'getHostedLLM',
              args: [BigInt(i)]
            }) as any;
            
            if (data) {
              models.push({
                id: i,
                modelName: data.modelName || data[4] || 'Unknown Model',
                host1: data.host1 || data[0] || '0x0000000000000000000000000000000000000000',
                host2: data.host2 || data[1] || '0x0000000000000000000000000000000000000000',
                shardUrl1: data.shardUrl1 || data[2] || '',
                shardUrl2: data.shardUrl2 || data[3] || '',
                poolBalance: data.poolBalance !== undefined ? data.poolBalance : (data[5] !== undefined ? data[5] : 0n),
                totalTimeHost1: data.totalTimeHost1 !== undefined ? data.totalTimeHost1 : (data[6] !== undefined ? data[6] : 0n),
                totalTimeHost2: data.totalTimeHost2 !== undefined ? data.totalTimeHost2 : (data[7] !== undefined ? data[7] : 0n),
                isComplete: data.isComplete !== undefined ? data.isComplete : (data[10] !== undefined ? data[10] : false)
              });
            }
          } catch (error) {
            console.error(`Failed to fetch LLM ${i}:`, error);
          }
        }
        
        setAllModels(models);
      } catch (error) {
        console.error('Error fetching all models:', error);
      } finally {
        setIsLoadingAllModels(false);
      }
    };

    fetchAllModels();
  }, [totalLLMs, config]);

  // Fetch incomplete LLM details
  useEffect(() => {
    const fetchIncompleteLLMDetails = async () => {
      if (!incompleteLLMs || incompleteLLMs.length === 0) {
        setIncompleteLLMDetails([]);
        return;
      }

      setIsLoadingIncomplete(true);
      
      try {
        const details: IncompleteLLM[] = [];
        
        for (const llmId of incompleteLLMs) {
          try {
            const data = await readContract(config, {
              address: CONTRACT_ADDRESS as `0x${string}`,
              abi: ABI,
              functionName: 'getHostedLLM',
              args: [llmId]
            }) as any;
            
            if (data) {
              details.push({
                id: Number(llmId),
                modelName: data.modelName || data[4] || 'Unknown Model',
                host1: data.host1 || data[0] || '0x0000000000000000000000000000000000000000',
                shardUrl1: data.shardUrl1 || data[2] || '',
                poolBalance: data.poolBalance !== undefined ? data.poolBalance : (data[5] !== undefined ? data[5] : 0n),
                totalTimeHost1: data.totalTimeHost1 !== undefined ? data.totalTimeHost1 : (data[6] !== undefined ? data[6] : 0n)
              });
            }
          } catch (error) {
            console.error(`Failed to fetch LLM ${llmId}:`, error);
          }
        }
        
        setIncompleteLLMDetails(details);
      } catch (error) {
        console.error('Error fetching incomplete LLM details:', error);
      } finally {
        setIsLoadingIncomplete(false);
      }
    };

    fetchIncompleteLLMDetails();
  }, [incompleteLLMs, config]);

  // Fetch models hosted by the current user
  useEffect(() => {
    const fetchMyHostedModels = async () => {
      if (!connectedAddress || totalLLMs === undefined) {
        setMyHostedModels([]);
        return;
      }

      setIsLoadingMyModels(true);
      
      try {
        const myModels: IncompleteLLM[] = [];
        
        for (let i = 0; i < Number(totalLLMs); i++) {
          try {
            const data = await readContract(config, {
              address: CONTRACT_ADDRESS as `0x${string}`,
              abi: ABI,
              functionName: 'getHostedLLM',
              args: [BigInt(i)]
            }) as any;
            
            if (data) {
              const host1 = (data.host1 || data[0] || '').toLowerCase();
              const host2 = (data.host2 || data[1] || '').toLowerCase();
              const userAddress = connectedAddress.toLowerCase();
              
              if (host1 === userAddress || host2 === userAddress) {
                myModels.push({
                  id: i,
                  modelName: data.modelName || data[4] || 'Unknown Model',
                  host1: data.host1 || data[0] || '0x0000000000000000000000000000000000000000',
                  host2: data.host2 || data[1] || '0x0000000000000000000000000000000000000000',
                  shardUrl1: data.shardUrl1 || data[2] || '',
                  shardUrl2: data.shardUrl2 || data[3] || '',
                  poolBalance: data.poolBalance !== undefined ? data.poolBalance : (data[5] !== undefined ? data[5] : 0n),
                  totalTimeHost1: data.totalTimeHost1 !== undefined ? data.totalTimeHost1 : (data[6] !== undefined ? data[6] : 0n),
                  totalTimeHost2: data.totalTimeHost2 !== undefined ? data.totalTimeHost2 : (data[7] !== undefined ? data[7] : 0n),
                  isComplete: data.isComplete !== undefined ? data.isComplete : (data[10] !== undefined ? data[10] : false)
                });
              }
            }
          } catch (error) {
            console.error(`Failed to fetch LLM ${i}:`, error);
          }
        }
        
        setMyHostedModels(myModels);
      } catch (error) {
        console.error('Error fetching my hosted models:', error);
      } finally {
        setIsLoadingMyModels(false);
      }
    };

    fetchMyHostedModels();
  }, [connectedAddress, totalLLMs, config]);


  return (
    <div className="min-h-screen bg-transparent font-inter">
      {/* Navbar */}
      <Navbar />

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-8 py-12 mt-24">
        {/* Models Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
            <div className="mb-6 flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">
                  {filterStatus === 'mymodels' ? 'My Hosted Models' : 
                   filterStatus === 'all' ? 'All Models' :
                   filterStatus === 'available' ? 'Available Models' :
                   'Models Needing Host'}
                </h2>
                <p className="text-gray-600">
                  {filterStatus === 'mymodels' 
                    ? 'Models you are currently hosting' 
                    : filterStatus === 'all'
                    ? 'All registered models on the network'
                    : filterStatus === 'available'
                    ? 'Complete models ready to use'
                    : 'Incomplete models waiting for a second host'}
                </p>
              </div>
              
              {/* Want to host button */}
              <button
                onClick={() => router.push('/console')}
                className="px-6 py-3 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-full hover:opacity-90 transition-opacity font-medium flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Want to host your own model?
              </button>
            </div>

            {/* Search and Filter Controls */}
            <div className="bg-white rounded-lg mb-8">
              <div className="flex flex-col lg:flex-row gap-4 items-center justify-between">
                {/* Search */}
                <div className="max-w-xs w-full">
                  <div className="relative">
                    <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <input
                      type="text"
                      placeholder="Search models..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-10 pr-4 py-1.5 text-sm border border-gray-300 rounded-lg bg-white text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-violet-500"
                    />
                  </div>
                </div>

                {/* Filter Buttons */}
                <div className="flex items-center flex-1 gap-6 justify-between">
                  <div className="flex items-center gap-6">
                  <button
                    onClick={() => setFilterStatus('all')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 transform ${
                      filterStatus === 'all'
                        ? 'text-violet-600 bg-violet-100 scale-105'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100 hover:scale-102'
                    }`}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 4.5v15m6-15v15m-10.875 0h15.75c.621 0 1.125-.504 1.125-1.125V5.625c0-.621-.504-1.125-1.125-1.125H4.125C3.504 4.5 3 5.004 3 5.625v12.75c0 .621.504 1.125 1.125 1.125Z" />
                    </svg>
                    All ({allModels.length})
                  </button>
                  
                  <div className="h-6 w-px bg-gray-300"></div>
                  
                  <button
                    onClick={() => setFilterStatus('available')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 transform ${
                      filterStatus === 'available'
                        ? 'text-violet-600 bg-violet-100 scale-105'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100 hover:scale-102'
                    }`}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Available ({allModels.filter(m => m.isComplete).length})
                  </button>
                  
                  <div className="h-6 w-px bg-gray-300"></div>
                  
                  <button
                    onClick={() => setFilterStatus('incomplete')}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 transform ${
                      filterStatus === 'incomplete'
                        ? 'text-violet-600 bg-violet-100 scale-105'
                        : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100 hover:scale-102'
                    }`}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Incomplete ({incompleteLLMDetails.length})
                  </button>
                  
                  {connectedAddress && (
                    <>
                      <div className="h-6 w-px bg-gray-300"></div>
                      <button
                        onClick={() => setFilterStatus('mymodels')}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 transform ${
                          filterStatus === 'mymodels'
                            ? 'text-violet-600 bg-violet-100 scale-105'
                            : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100 hover:scale-102'
                        }`}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                        </svg>
                        My Models ({myHostedModels.length})
                      </button>
                    </>
                  )}
                  </div>
                  
                  {/* Filter Icon */}
                  <ModelFilters />
                </div>
              </div>
            </div>

            {(isLoadingAllModels || isLoadingIncomplete || (filterStatus === 'mymodels' && isLoadingMyModels)) ? (
              <div className="flex items-center justify-center py-10">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-400"></div>
              </div>
            ) : (() => {
              // Determine which models to show based on filter
              let modelsToShow: IncompleteLLM[] = [];
              if (filterStatus === 'all') {
                modelsToShow = allModels;
              } else if (filterStatus === 'available') {
                modelsToShow = allModels.filter(m => m.isComplete);
              } else if (filterStatus === 'incomplete') {
                modelsToShow = incompleteLLMDetails;
              } else if (filterStatus === 'mymodels') {
                modelsToShow = myHostedModels;
              }
              
              return modelsToShow.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20">
                  <div className="text-center max-w-md">
                    <h3 className="text-xl font-bold text-gray-900 mb-2">
                      {filterStatus === 'all' ? 'No models found' :
                       filterStatus === 'available' ? 'No available models' :
                       filterStatus === 'incomplete' ? 'No incomplete models' :
                       'No models hosted yet'}
                    </h3>
                    <p className="text-gray-600 mb-6">
                      {filterStatus === 'all' ? 'There are no models registered on the network yet. Be the first to add one!' :
                       filterStatus === 'available' ? 'There are no complete models ready to use yet' :
                       filterStatus === 'incomplete' ? 'There are currently no hosting slots waiting for a second host' :
                       'You\'re not currently hosting any models. Start by adding a new model or joining an existing hosting slot!'}
                    </p>
                    {filterStatus === 'all' && (
                      <button
                        onClick={() => router.push('/console')}
                        className="px-6 py-3 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity font-medium inline-flex items-center gap-2"
                      >
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        Add Your First Model
                      </button>
                    )}
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {modelsToShow
                    .filter(llm => {
                      if (searchTerm) {
                        return llm.modelName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                               llm.host1.toLowerCase().includes(searchTerm.toLowerCase()) ||
                               llm.shardUrl1.toLowerCase().includes(searchTerm.toLowerCase());
                      }
                      return true;
                    })
                    .map((llm) => {
                    const isComplete = llm.isComplete || (llm.host2 && llm.host2 !== '0x0000000000000000000000000000000000000000');
                    
                    return (
                <motion.div
                  key={llm.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="bg-white rounded-xl border border-gray-200 p-6 hover:border-violet-400 hover:shadow-md transition-all"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-start gap-3 flex-1">
                      <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center flex-shrink-0 p-1.5">
                        <img 
                          src={getModelIcon(llm.modelName)} 
                          alt={llm.modelName}
                          className="w-full h-full object-contain"
                          onError={(e) => {
                            e.currentTarget.src = DEFAULT_LLM_ICON;
                          }}
                        />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-bold text-lg text-gray-900">{llm.modelName}</h3>
                      </div>
                    </div>
                    <div className={`px-2.5 py-1 rounded-full text-xs font-medium flex-shrink-0 ${
                      isComplete 
                        ? 'bg-green-100 text-green-700' 
                        : 'bg-yellow-100 text-yellow-700'
                    }`}>
                      {isComplete ? '✓ Complete' : '⏳ Pending'}
                    </div>
                  </div>

                  <div className="space-y-2 text-xs mb-4">
                    <div className="flex items-start gap-2">
                      <span className="text-gray-500">Model ID:</span>
                      <span className="text-gray-700">#{llm.id}</span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-gray-500">Host 1:</span>
                      <span className="font-mono text-gray-700">{llm.host1.slice(0, 10)}...</span>
                    </div>
                    {isComplete && llm.host2 && llm.host2 !== '0x0000000000000000000000000000000000000000' && (
                      <div className="flex items-start gap-2">
                        <span className="text-gray-500">Host 2:</span>
                        <span className="font-mono text-gray-700">{llm.host2.slice(0, 10)}...</span>
                      </div>
                    )}
                    {!isComplete && (
                      <div className="flex items-start gap-2">
                        <span className="text-gray-500">Shard 1:</span>
                        <span className="text-gray-700">Lower Layers (1-50)</span>
                      </div>
                    )}
                    <div className="flex items-start gap-2">
                      <span className="text-gray-500 flex-shrink-0">URL:</span>
                      <span className="font-mono text-gray-700 break-all">{llm.shardUrl1.slice(0, 60)}...</span>
                    </div>
                  </div>

                  {isComplete ? (
                    <button
                      onClick={() => router.push('/chat')}
                      className="w-full py-2 px-4 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors font-medium text-sm flex items-center justify-center gap-2"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                      </svg>
                      Use in Chat
                    </button>
                  ) : (
                    <button
                      onClick={() => router.push('/console')}
                      className="w-full py-2 px-4 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity font-medium text-sm"
                    >
                      Join as Host 2
                    </button>
                  )}
                </motion.div>
                    );
                  })}
              </div>
              );
            })()}
        </motion.div>
      </div>
    </div>
  );
};

export default ModelsPage;
