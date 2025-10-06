"use client";

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { Navbar } from '../components/Navbar';
import { useRegisterLLM } from '../lib/contracts/creditUse/writes/useRegisterLLM';
import { useGetIncompleteLLMs } from '../lib/contracts/creditUse/reads/useGetIncompleteLLMs';
import { useGetTotalLLMs } from '../lib/contracts/creditUse/reads/useGetTotalLLMs';
import { useCheckHostedLLM } from '../lib/contracts/creditUse/reads/useCheckHostedLLM';
import { useAccount, useReadContract, useConfig } from 'wagmi';
import { readContract } from '@wagmi/core';
import ABI from '../utils/abi.json';
import { CONTRACT_ADDRESS } from '../utils/address';

interface Model {
  id: number;
  name: string;
  shard: string;
  walletAddress: string;
  status: 'hosting' | 'inactive';
  dateAdded: Date;
}

interface AddModelForm {
  modelName: string;
  shard: string;
  walletAddress: string;
}

interface JoinHostForm {
  llmId: number;
  shard: string;
  walletAddress: string;
}

interface IncompleteLLM {
  id: number;
  modelName: string;
  host1: string;
  shardUrl1: string;
  poolBalance: bigint;
  totalTimeHost1: bigint;
}

const AVAILABLE_MODELS = [
  'TinyLlama-1.1B-Chat-v1.0',
  'Llama-2-7B-Chat',
  'Mistral-7B-Instruct',
  'CodeLlama-7B-Python',
  'Phi-2-2.7B',
  'Gemma-2B-Instruct'
];

const AVAILABLE_SHARDS = [
  { id: 'shard-1', name: 'Shard 1', region: 'US-East', capacity: '75%' },
  { id: 'shard-2', name: 'Shard 2', region: 'US-West', capacity: '60%' },
  { id: 'shard-3', name: 'Shard 3', region: 'EU-Central', capacity: '45%' },
  { id: 'shard-4', name: 'Shard 4', region: 'Asia-Pacific', capacity: '30%' },
  { id: 'shard-5', name: 'Shard 5', region: 'US-Central', capacity: '90%' },
  { id: 'shard-6', name: 'Shard 6', region: 'EU-West', capacity: '55%' }
];

const ModelsPage = () => {
  const router = useRouter();
  const [models, setModels] = useState<Model[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'status'>('date');
  const [filterStatus, setFilterStatus] = useState<'all' | 'hosting' | 'inactive'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [formData, setFormData] = useState<AddModelForm>({
    modelName: '',
    shard: '',
    walletAddress: ''
  });
  const [joinFormData, setJoinFormData] = useState<JoinHostForm>({
    llmId: 0,
    shard: '',
    walletAddress: ''
  });
  const [showJoinForm, setShowJoinForm] = useState(false);
  const [selectedLLMId, setSelectedLLMId] = useState<number | null>(null);
  const [incompleteLLMDetails, setIncompleteLLMDetails] = useState<IncompleteLLM[]>([]);
  const [isLoadingIncomplete, setIsLoadingIncomplete] = useState(false);

  // Smart contract hooks
  const { registerLLM, txHash, isWriting, writeError, resetWrite, isConfirming, isConfirmed } = useRegisterLLM();
  const { incompleteLLMs, refetch: refetchIncomplete } = useGetIncompleteLLMs();
  const { totalLLMs, refetch: refetchTotal } = useGetTotalLLMs();
  const { address: connectedAddress } = useAccount();
  const config = useConfig();

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
        
        // Fetch details for each incomplete LLM using the contract
        for (const llmId of incompleteLLMs) {
          try {
            const data = await readContract(config, {
              address: CONTRACT_ADDRESS as `0x${string}`,
              abi: ABI,
              functionName: 'getHostedLLM',
              args: [llmId]
            }) as any;
            
            console.log('LLM Data for ID', llmId, ':', data);
            
            if (data) {
              // Access struct fields by name (wagmi returns structs as objects)
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

  // Simulate loading models from storage/API
  useEffect(() => {
    const loadModels = async () => {
      setIsLoading(true);
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Load from localStorage or start with empty array
      const savedModels = localStorage.getItem('teetee-models');
      if (savedModels) {
        const parsedModels = JSON.parse(savedModels);
        setModels(parsedModels.map((model: any) => ({
          ...model,
          dateAdded: new Date(model.dateAdded)
        })));
      }
      setIsLoading(false);
    };

    loadModels();
  }, []);

  // Save models to localStorage
  const saveModels = (newModels: Model[]) => {
    localStorage.setItem('teetee-models', JSON.stringify(newModels));
    setModels(newModels);
  };

  // Reset form state
  const resetForm = () => {
    setShowAddForm(false);
    setFormData({
      modelName: '',
      shard: '',
      walletAddress: ''
    });
    resetWrite();
  };

  // Reset join form
  const resetJoinForm = () => {
    setShowJoinForm(false);
    setSelectedLLMId(null);
    setJoinFormData({
      llmId: 0,
      shard: '',
      walletAddress: ''
    });
  };

  // Handle form submission for registering first host
  const handleAddModel = async () => {
    if (!formData.modelName || !formData.shard || !formData.walletAddress || totalLLMs === undefined) {
      return;
    }

    if (!connectedAddress) {
      alert('Please connect your wallet first');
      return;
    }

    try {
      resetWrite();
      
      const selectedShard = AVAILABLE_SHARDS.find(s => s.id === formData.shard);
      
      // Register first host on blockchain (creates slot)
      // Pass totalLLMs as llmId to create new entry
      // Use address(0) for host2 and empty string for shardUrl2 to leave them empty
      // Pass 0 for time fields - these will be set by oracle
      await registerLLM(
        Number(totalLLMs),                              // llmId - array length for new entry
        formData.walletAddress,                         // host1
        '0x0000000000000000000000000000000000000000',  // host2 - empty (address zero)
        selectedShard?.id || formData.shard,           // shardUrl1
        '',                                             // shardUrl2 - empty
        formData.modelName,                            // modelName
        0,                                              // totalTimeHost1 - will be set by oracle
        0                                               // totalTimeHost2 - empty
      );
    } catch (error) {
      // Error is surfaced via wallet UI
    }
  };

  // Handle joining as second host
  const handleJoinAsSecondHost = async () => {
    if (selectedLLMId === null || !joinFormData.shard || !joinFormData.walletAddress) return;

    if (!connectedAddress) {
      alert('Please connect your wallet first');
      return;
    }

    try {
      resetWrite();
      
      const selectedShard = AVAILABLE_SHARDS.find(s => s.id === joinFormData.shard);
      
      // Join as second host on blockchain
      // Pass existing llmId, leave host1 fields as 0/empty to keep them, fill in host2 fields
      // Pass 0 for time - will be set by oracle
      await registerLLM(
        selectedLLMId,                                  // llmId - existing entry
        '0x0000000000000000000000000000000000000000',  // host1 - empty (keep existing)
        joinFormData.walletAddress,                     // host2 - new host
        '',                                             // shardUrl1 - empty (keep existing)
        selectedShard?.id || joinFormData.shard,       // shardUrl2
        '',                                             // modelName - empty (keep existing)
        0,                                              // totalTimeHost1 - 0 (keep existing)
        0                                               // totalTimeHost2 - will be set by oracle
      );
    } catch (error) {
      console.error('Failed to join as second host:', error);
      // Error is surfaced via wallet UI
    }
  };

  // Effect to handle successful transaction confirmation
  useEffect(() => {
    if (isConfirmed) {
      refetchIncomplete();
      refetchTotal();
      
      // If it was creating a new model (not joining)
      if (formData.modelName && !showJoinForm) {
    const selectedShard = AVAILABLE_SHARDS.find(s => s.id === formData.shard);
    
    const newModel: Model = {
      id: Date.now(),
      name: formData.modelName,
          shard: `${selectedShard?.name || formData.shard} (Waiting for 2nd host)`,
      walletAddress: formData.walletAddress,
          status: 'inactive',
      dateAdded: new Date()
    };

    const updatedModels = [newModel, ...models];
    saveModels(updatedModels);
    resetForm();
      } 
      // If it was joining as second host
      else if (showJoinForm) {
        resetJoinForm();
        // Optionally reload models or update the specific model status
      }
    }
  }, [isConfirmed]);

  // Handle model status toggle
  const handleToggleModel = (modelId: number) => {
    const updatedModels = models.map(model => 
      model.id === modelId 
        ? { ...model, status: (model.status === 'hosting' ? 'inactive' : 'hosting') as 'hosting' | 'inactive' }
        : model
    );
    saveModels(updatedModels);
  };

  // Handle deleting a model
  const handleDeleteModel = (modelId: number) => {
    const updatedModels = models.filter(model => model.id !== modelId);
    saveModels(updatedModels);
  };

  // Filter and sort models
  const getFilteredAndSortedModels = () => {
    let filteredModels = models;

    // Apply search filter
    if (searchTerm) {
      filteredModels = filteredModels.filter(model => 
        model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        model.shard.toLowerCase().includes(searchTerm.toLowerCase()) ||
        model.walletAddress.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply status filter
    if (filterStatus !== 'all') {
      filteredModels = filteredModels.filter(model => model.status === filterStatus);
    }

    // Apply sorting
    const sortedModels = [...filteredModels].sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'date':
          return new Date(b.dateAdded).getTime() - new Date(a.dateAdded).getTime();
        case 'status':
          return a.status.localeCompare(b.status);
        default:
          return 0;
      }
    });

    return sortedModels;
  };

  return (
    <div className="min-h-screen bg-transparent font-inter">
      {/* Navbar */}
      <Navbar />

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-8 py-12 mt-24">
        {/* Header */}
        <div className="mb-10">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Models</h1>
              <p className="text-gray-600 mt-1">Manage your AI models across distributed shards</p>
            </div>
            
            {/* Add Model Button - Always visible */}
            <button
              onClick={() => setShowAddForm(!showAddForm)}
              className={`px-6 py-3 rounded-full text-sm font-medium transition-all flex items-center gap-2 ${
                showAddForm 
                  ? 'bg-gray-200 text-gray-700 hover:bg-gray-300' 
                  : 'bg-gradient-to-r from-violet-400 to-purple-300 text-white hover:opacity-90'
              }`}
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {showAddForm ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                )}
              </svg>
              {showAddForm ? 'Cancel' : 'Add Model'}
            </button>
          </div>
        </div>

        {/* Add Model Form - Expands when button is clicked */}
        {showAddForm && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="bg-white rounded-xl border border-gray-200 shadow-sm mb-12 overflow-hidden"
          >
            {/* Form Header */}
            <div className="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-violet-50 to-purple-50">
              <div>
                <h2 className="text-xl font-bold text-gray-900">Add New Model</h2>
                <p className="text-sm text-gray-600 mt-1">Complete all fields to add your model</p>
              </div>
            </div>

            {/* All Steps Content */}
            <div className="p-6 space-y-8">
              {/* Step 1: Model Selection */}
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-violet-400 text-white text-sm flex items-center justify-center">1</span>
                    Select Model
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">Choose from available AI models</p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {AVAILABLE_MODELS.map((model) => (
                    <button
                      key={model}
                      onClick={() => setFormData({ ...formData, modelName: model })}
                      className={`text-left p-4 rounded-lg border transition-colors ${
                        formData.modelName === model
                          ? 'border-violet-400 bg-violet-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="font-medium text-gray-900">{model}</div>
                      <div className="text-sm text-gray-500">AI Language Model</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Divider */}
              <div className="border-t border-gray-200"></div>

              {/* Step 2: Shard Selection */}
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-violet-400 text-white text-sm flex items-center justify-center">2</span>
                    Select Your Shard
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">Choose where to host your part of the model</p>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {AVAILABLE_SHARDS.map((shard) => (
                    <button
                      key={shard.id}
                      onClick={() => setFormData({ ...formData, shard: shard.id })}
                      className={`text-left p-4 rounded-lg border transition-colors ${
                        formData.shard === shard.id
                          ? 'border-violet-400 bg-violet-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-medium text-gray-900">{shard.name}</div>
                        <div className={`text-xs px-2 py-1 rounded-full ${
                          parseInt(shard.capacity) > 80 
                            ? 'bg-red-100 text-red-800'
                            : parseInt(shard.capacity) > 60
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-green-100 text-green-800'
                        }`}>
                          {shard.capacity}
                        </div>
                      </div>
                      <div className="text-sm text-gray-500">{shard.region}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Divider */}
              <div className="border-t border-gray-200"></div>

              {/* Step 3: Wallet Address */}
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-violet-400 text-white text-sm flex items-center justify-center">3</span>
                    Your Wallet Address
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">Enter your wallet address to receive hosting rewards</p>
                </div>
                
                <div className="max-w-2xl">
                  <label htmlFor="walletAddress" className="block text-sm font-medium text-gray-700 mb-2">
                    Wallet Address *
                  </label>
                  <div className="flex gap-2">
                  <input
                    id="walletAddress"
                    type="text"
                    value={formData.walletAddress}
                    onChange={(e) => setFormData({ ...formData, walletAddress: e.target.value })}
                    placeholder="0x1234567890abcdef..."
                      className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent outline-none"
                    />
                    <button
                      type="button"
                      onClick={() => connectedAddress && setFormData({ ...formData, walletAddress: connectedAddress })}
                      disabled={!connectedAddress}
                      className="px-4 py-3 bg-violet-100 text-violet-700 rounded-lg hover:bg-violet-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap text-sm font-medium"
                      title={connectedAddress ? 'Use connected wallet' : 'No wallet connected'}
                    >
                      Use Connected
                    </button>
                  </div>
                </div>
              </div>

              {/* Summary - Show when all fields are filled */}
              {formData.modelName && formData.shard && formData.walletAddress && (
                <div className="mt-6 p-4 bg-gradient-to-r from-violet-50 to-purple-50 rounded-lg border border-violet-200">
                  <h4 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
                    <svg className="w-5 h-5 text-violet-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Summary
                  </h4>
                  <div className="space-y-2 text-sm text-gray-700">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Model:</span> 
                      <span className="text-violet-700">{formData.modelName}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Your Shard:</span> 
                      <span className="text-violet-700">{AVAILABLE_SHARDS.find(s => s.id === formData.shard)?.name} ({AVAILABLE_SHARDS.find(s => s.id === formData.shard)?.region})</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Your Wallet:</span> 
                      <span className="text-violet-700 font-mono text-xs">{formData.walletAddress.slice(0, 10)}...</span>
                    </div>
                    <div className="mt-3 pt-3 border-t border-violet-200">
                      <p className="text-xs text-gray-600">⚠️ This will create a slot waiting for a second host to complete the registration</p>
                      <p className="text-xs text-gray-600 mt-1">ℹ️ Hosting duration will be set by the oracle</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Form Footer */}
            <div className="flex items-center justify-between p-6 border-t border-gray-200 bg-gray-50">
              <div className="flex items-center gap-4">
              <button
                onClick={resetForm}
                  disabled={isWriting || isConfirming}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
                
                {/* Transaction Status */}
                {(isWriting || isConfirming) && (
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-violet-400"></div>
                    <span>{isWriting ? 'Submitting transaction...' : 'Confirming...'}</span>
                  </div>
                )}
                
                {writeError && (
                  <div className="text-sm text-red-600">
                    Error: {writeError.message.slice(0, 50)}...
                  </div>
                )}
              </div>

              <button
                onClick={handleAddModel}
                disabled={!formData.modelName || !formData.shard || !formData.walletAddress || isWriting || isConfirming}
                className="px-6 py-2 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {(isWriting || isConfirming) ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    {isWriting ? 'Submitting...' : 'Confirming...'}
                  </>
                ) : (
                  <>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                    Create Hosting Slot
                  </>
                )}
              </button>
            </div>
          </motion.div>
        )}

        {/* Available Slots Section - Show incomplete LLMs waiting for second host */}
        {!showAddForm && incompleteLLMDetails.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-12"
          >
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Available Hosting Slots</h2>
              <p className="text-gray-600">Join as the second host for these models</p>
            </div>

            {isLoadingIncomplete ? (
              <div className="flex items-center justify-center py-10">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-400"></div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {incompleteLLMDetails.map((llm) => (
                  <motion.div
                    key={llm.id}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="bg-gradient-to-br from-violet-50 to-purple-50 rounded-xl border-2 border-violet-200 p-6 hover:border-violet-400 transition-all"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <h3 className="font-bold text-lg text-gray-900 mb-1">{llm.modelName}</h3>
                        <div className="inline-flex items-center gap-2 px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs font-medium">
                          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                          </svg>
                          Waiting for 2nd host
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2 text-sm mb-4">
                      <div className="flex items-center gap-2">
                        <span className="text-gray-500">Host 1:</span>
                        <span className="font-mono text-xs text-gray-700">{llm.host1.slice(0, 10)}...</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-gray-500">Shard 1:</span>
                        <span className="text-gray-700">{AVAILABLE_SHARDS.find(s => s.id === llm.shardUrl1)?.name || llm.shardUrl1}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-gray-500">Pool Balance:</span>
                        <span className="text-violet-700 font-semibold">{llm.poolBalance?.toString() || '0'} credits</span>
                      </div>
                    </div>

                    <button
                      onClick={() => {
                        setSelectedLLMId(llm.id);
                        setJoinFormData({
                          llmId: llm.id,
                          shard: '',
                          walletAddress: connectedAddress || ''
                        });
                        setShowJoinForm(true);
                      }}
                      className="w-full py-2 px-4 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity font-medium text-sm"
                    >
                      Join as Host 2
                    </button>
                  </motion.div>
                ))}
              </div>
            )}
          </motion.div>
        )}

        {/* Join Form Modal */}
        {showJoinForm && selectedLLMId !== null && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={() => setShowJoinForm(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-white rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Form Header */}
              <div className="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-violet-50 to-purple-50">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-bold text-gray-900">Join as Second Host</h2>
                    <p className="text-sm text-gray-600 mt-1">
                      Model: {incompleteLLMDetails.find(llm => llm.id === selectedLLMId)?.modelName}
                    </p>
                  </div>
                  <button
                    onClick={() => setShowJoinForm(false)}
                    className="text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>

              {/* Form Content */}
              <div className="p-6 space-y-6">
                {/* Shard Selection */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-violet-400 text-white text-sm flex items-center justify-center">1</span>
                    Select Your Shard
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">Choose where to host your part of the model</p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {AVAILABLE_SHARDS.map((shard) => (
                      <button
                        key={shard.id}
                        onClick={() => setJoinFormData({ ...joinFormData, shard: shard.id })}
                        className={`text-left p-4 rounded-lg border transition-colors ${
                          joinFormData.shard === shard.id
                            ? 'border-violet-400 bg-violet-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="font-medium text-gray-900">{shard.name}</div>
                          <div className={`text-xs px-2 py-1 rounded-full ${
                            parseInt(shard.capacity) > 80 
                              ? 'bg-red-100 text-red-800'
                              : parseInt(shard.capacity) > 60
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-green-100 text-green-800'
                          }`}>
                            {shard.capacity}
                          </div>
                        </div>
                        <div className="text-sm text-gray-500">{shard.region}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Wallet Address */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center gap-2">
                    <span className="w-6 h-6 rounded-full bg-violet-400 text-white text-sm flex items-center justify-center">2</span>
                    Your Wallet Address
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">Enter your wallet address to receive hosting rewards</p>
                  
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={joinFormData.walletAddress}
                      onChange={(e) => setJoinFormData({ ...joinFormData, walletAddress: e.target.value })}
                      placeholder="0x1234567890abcdef..."
                      className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent outline-none"
                    />
                    <button
                      type="button"
                      onClick={() => connectedAddress && setJoinFormData({ ...joinFormData, walletAddress: connectedAddress })}
                      disabled={!connectedAddress}
                      className="px-4 py-3 bg-violet-100 text-violet-700 rounded-lg hover:bg-violet-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap text-sm font-medium"
                      title={connectedAddress ? 'Use connected wallet' : 'No wallet connected'}
                    >
                      Use Connected
                    </button>
                  </div>
                </div>

                {/* Summary */}
                {joinFormData.shard && joinFormData.walletAddress && (
                  <div className="p-4 bg-gradient-to-r from-violet-50 to-purple-50 rounded-lg border border-violet-200">
                    <h4 className="font-medium text-gray-900 mb-3 flex items-center gap-2">
                      <svg className="w-5 h-5 text-violet-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Summary
                    </h4>
                    <div className="space-y-2 text-sm text-gray-700">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Your Shard:</span> 
                        <span className="text-violet-700">{AVAILABLE_SHARDS.find(s => s.id === joinFormData.shard)?.name} ({AVAILABLE_SHARDS.find(s => s.id === joinFormData.shard)?.region})</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Your Wallet:</span> 
                        <span className="text-violet-700 font-mono text-xs">{joinFormData.walletAddress.slice(0, 10)}...</span>
                      </div>
                      <div className="mt-3 pt-3 border-t border-violet-200">
                        <p className="text-xs text-gray-600">ℹ️ Hosting duration will be set by the oracle</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Form Footer */}
              <div className="flex items-center justify-between p-6 border-t border-gray-200 bg-gray-50">
                <div className="flex items-center gap-4">
                  <button
                    onClick={resetJoinForm}
                    disabled={isWriting || isConfirming}
                    className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  
                  {/* Transaction Status */}
                  {(isWriting || isConfirming) && (
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-violet-400"></div>
                      <span>{isWriting ? 'Submitting transaction...' : 'Confirming...'}</span>
                    </div>
                  )}
                  
                  {writeError && (
                    <div className="text-sm text-red-600">
                      Error: {writeError.message.slice(0, 50)}...
                    </div>
                  )}
                </div>

                <button
                  onClick={handleJoinAsSecondHost}
                  disabled={!joinFormData.shard || !joinFormData.walletAddress || isWriting || isConfirming}
                  className="px-6 py-2 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {(isWriting || isConfirming) ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      {isWriting ? 'Submitting...' : 'Confirming...'}
                    </>
                  ) : (
                    <>
                      Join as Host 2
                    </>
                  )}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}

        {/* Models Content - Hidden when form is open */}
        {!showAddForm && (
          isLoading ? (
            // Loading State
            <div className="flex items-center justify-center py-20">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-violet-400 mx-auto mb-4"></div>
                <p className="text-gray-600">Loading models...</p>
              </div>
            </div>
          ) : models.length === 0 ? (
            // Empty State
            <div className="flex flex-col items-center justify-center py-20">
              <div className="text-center max-w-md">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">No models found</h2>
                <p className="text-gray-600 mb-8">Add your first model to get started with decentralized AI hosting</p>
              </div>
            </div>
          ) : (
            <>
              {/* Sort and Filter Controls */}
              <div className="bg-white rounded-lg mb-12">
                <div className="flex flex-col lg:flex-row gap-6 items-center justify-between">
                  {/* Search */}
                  <div className="flex-1 max-w-md">
                    <div className="relative">
                      <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                      <input
                        type="text"
                        placeholder="Search models by name, shard, or wallet..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full pl-12 pr-12 py-2 border border-gray-300 rounded-lg bg-white text-gray-900 placeholder-gray-500 focus:outline-none"
                      />
                    </div>
                  </div>

                  {/* Filter Buttons */}
                  <div className="flex items-center flex-1 gap-6">
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
                      All ({models.length})
                    </button>
                    
                    <div className="h-6 w-px bg-gray-300"></div>
                    
                    <button
                      onClick={() => setFilterStatus('hosting')}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 transform ${
                        filterStatus === 'hosting'
                          ? 'text-violet-600 bg-violet-100 scale-105'
                          : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100 hover:scale-102'
                      }`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
                      </svg>
                      Hosting ({models.filter(m => m.status === 'hosting').length})
                    </button>
                    
                    <div className="h-6 w-px bg-gray-300"></div>
                    
                    <button
                      onClick={() => setFilterStatus('inactive')}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 transform ${
                        filterStatus === 'inactive'
                          ? 'text-violet-600 bg-violet-100 scale-105'
                          : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100 hover:scale-102'
                      }`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25v13.5m-7.5-13.5v13.5" />
                      </svg>
                      Inactive ({models.filter(m => m.status === 'inactive').length})
                    </button>
                  </div>

                  {/* Sort Button */}
                  <div className="relative">
                    <button className="flex items-center justify-center w-10 h-10 border border-gray-300 rounded-lg bg-white text-gray-700 hover:bg-gray-50 transition-colors focus:outline-none">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c2.755 0 5.455.232 8.083.678.533.09.917.556.917 1.096v1.044a2.25 2.25 0 01-.659 1.591l-5.432 5.432a2.25 2.25 0 00-.659 1.591v2.927a2.25 2.25 0 01-1.244 2.013L9.75 21v-6.568a2.25 2.25 0 00-.659-1.591L3.659 7.409A2.25 2.25 0 013 5.818V4.774c0-.54.384-1.006.917-1.096A48.32 48.32 0 0112 3z" />
                      </svg>
                    </button>
                    
                    {/* Hidden Sort Dropdown for now - can be expanded later */}
                    <select
                      value={sortBy}
                      onChange={(e) => setSortBy(e.target.value as 'name' | 'date' | 'status')}
                      className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer"
                    >
                      <option value="date">Date Added</option>
                      <option value="name">Name</option>
                      <option value="status">Status</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Models Grid - Folder Design */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-12">
                {getFilteredAndSortedModels().map((model) => (
                <motion.div
                  key={model.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="relative group cursor-pointer"
                >
                  {/* Folder Container */}
                  <div className="relative bg-white hover:bg-gray-50 transition-colors duration-200">
                    {/* Folder Tab */}
                    <div className="absolute -top-6 bg-white w-30 h-8 rounded-t-lg border-l-2 border-t-2 border-r-2 border-black"></div>
                    
                    {/* Folder Body */}
                    <div className="bg-white border-2 border-black rounded-lg pt-8 pb-6 px-8 min-h-[200px] flex flex-col">
                      {/* Header with Model Name and Status */}
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="font-bold text-lg text-black truncate pr-2">
                          {model.name}
                        </h3>
                        {/* Status Indicator */}
                        <div className={`w-3 h-3 rounded-full flex-shrink-0 ${
                          model.status === 'hosting' ? 'bg-green-500' : 'bg-gray-500'
                        }`}></div>
                      </div>

                      {/* Model Details */}
                      <div className="space-y-2 text-xs font-mono text-gray-700 flex-1">
                        <div className="flex items-center gap-1">
                          <span className="text-gray-500">Shard:</span>
                          <span className="truncate">{model.shard}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="text-gray-500">Wallet:</span>
                          <span className="truncate">{model.walletAddress.slice(0, 10)}...</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="text-gray-500">Added:</span>
                          <span>{model.dateAdded.toLocaleDateString('en-US', { 
                            month: 'short', 
                            day: 'numeric' 
                          })}</span>
                        </div>
                      </div>

                      {/* Horizontal Line Separator */}
                      <div className="border-t-2 border-black my-6"></div>

                      {/* Action Buttons - Left and Right Aligned */}
                      <div className="flex justify-between items-center">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteModel(model.id);
                          }}
                          className="px-3 py-2 text-xs font-mono font-medium bg-gray-200 text-gray-700 hover:bg-gray-300 transition-colors rounded"
                          title="Delete model"
                        >
                          DELETE
                        </button>
                        
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleToggleModel(model.id);
                          }}
                          className={`px-4 py-2 text-xs font-mono font-medium transition-colors rounded ${
                            model.status === 'hosting'
                              ? 'bg-red-500 text-white hover:bg-red-600'
                              : 'bg-gradient-to-r from-violet-400 to-purple-400 text-white hover:from-violet-500 hover:to-purple-500'
                          }`}
                        >
                          {model.status === 'hosting' ? 'STOP' : 'HOST'}
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
            </>
          )
        )}
      </div>
    </div>
  );
};

export default ModelsPage;
