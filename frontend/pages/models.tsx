"use client";

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { Navbar } from '../components/Navbar';
import { AvailableHostingSlots } from '../components/AvailableHostingSlots';
import { MyHostedModels } from '../components/MyHostedModels';
import { ModelFilters } from '../components/ModelFilters';
import { AddModelForm } from '../components/AddModelForm';
import { useRegisterLLM } from '../lib/contracts/creditUse/writes/useRegisterLLM';
import { useGetIncompleteLLMs } from '../lib/contracts/creditUse/reads/useGetIncompleteLLMs';
import { useGetTotalLLMs } from '../lib/contracts/creditUse/reads/useGetTotalLLMs';
import { useCheckHostedLLM } from '../lib/contracts/creditUse/reads/useCheckHostedLLM';
import { useAccount, useReadContract, useConfig } from 'wagmi';
import { readContract } from '@wagmi/core';
import ABI from '../utils/abi.json';
import { CONTRACT_ADDRESS } from '../utils/address';
import { useMintINFT, useAuthorizeINFT } from '../hooks/useINFT';

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
  host2?: string;
  shardUrl1: string;
  shardUrl2?: string;
  poolBalance: bigint;
  totalTimeHost1: bigint;
  totalTimeHost2?: bigint;
  isComplete?: boolean;
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
  const [showAddForm, setShowAddForm] = useState(false);
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'status'>('date');
  const [filterStatus, setFilterStatus] = useState<'all' | 'hosting' | 'inactive' | 'mymodels'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [joinFormData, setJoinFormData] = useState<JoinHostForm>({
    llmId: 0,
    shard: '',
    walletAddress: ''
  });
  const [showJoinForm, setShowJoinForm] = useState(false);
  const [selectedLLMId, setSelectedLLMId] = useState<number | null>(null);
  const [incompleteLLMDetails, setIncompleteLLMDetails] = useState<IncompleteLLM[]>([]);
  const [isLoadingIncomplete, setIsLoadingIncomplete] = useState(false);
  const [showClaimINFTModal, setShowClaimINFTModal] = useState(false);
  const [registeredModelName, setRegisteredModelName] = useState<string>('');
  const [isSecondHost, setIsSecondHost] = useState(false);
  const [myHostedModels, setMyHostedModels] = useState<IncompleteLLM[]>([]);
  const [isLoadingMyModels, setIsLoadingMyModels] = useState(false);
  const [pausingModelId, setPausingModelId] = useState<number | null>(null);
  const [stoppingModelId, setStoppingModelId] = useState<number | null>(null);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const [modelToStop, setModelToStop] = useState<IncompleteLLM | null>(null);
  const [submittedFormData, setSubmittedFormData] = useState<AddModelForm | null>(null);

  // Smart contract hooks
  const { registerLLM, txHash, isWriting, writeError, resetWrite, isConfirming, isConfirmed } = useRegisterLLM();
  const { incompleteLLMs, refetch: refetchIncomplete } = useGetIncompleteLLMs();
  const { totalLLMs, refetch: refetchTotal } = useGetTotalLLMs();
  const { address: connectedAddress } = useAccount();
  const config = useConfig();
  
  // INFT hooks for minting and authorization
  const { mint: mintINFT, isPending: isMinting, isConfirmed: isMintConfirmed } = useMintINFT();
  const { authorize: authorizeINFT, isPending: isAuthorizing, isConfirmed: isAuthConfirmed } = useAuthorizeINFT();

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

  // Fetch models hosted by the current user
  useEffect(() => {
    const fetchMyHostedModels = async () => {
      if (!connectedAddress) {
        setMyHostedModels([]);
        setIsLoadingMyModels(false);
        return;
      }

      if (totalLLMs === undefined) {
        setMyHostedModels([]);
        setIsLoadingMyModels(false);
        return;
      }

      setIsLoadingMyModels(true);
      
      try {
        const myModels: IncompleteLLM[] = [];
        
        // Iterate through all LLMs and find ones where user is host1 or host2
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
              
              // Check if user is either host1 or host2
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
  }, [connectedAddress, totalLLMs, config, isConfirmed]);

  // Reset form state
  const resetForm = () => {
    setShowAddForm(false);
    setSubmittedFormData(null);
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
  const handleAddModel = async (formData: AddModelForm) => {
    if (!formData.modelName || !formData.shard || !formData.walletAddress || totalLLMs === undefined) {
      return;
    }

    if (!connectedAddress) {
      alert('Please connect your wallet first');
      return;
    }

    try {
      resetWrite();
      setSubmittedFormData(formData);
      
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

  // Effect to handle successful model registration - show claim modal instead of auto-minting
  useEffect(() => {
    if (isConfirmed && connectedAddress && !showJoinForm && submittedFormData?.modelName) {
      console.log('Model registered successfully! Showing claim INFT modal...');
      
      // Store the model name for the modal
      setRegisteredModelName(submittedFormData.modelName);
      setIsSecondHost(false);
      
      // Show claim modal
      setShowClaimINFTModal(true);
      
      // Reset the registration form
      resetForm();
      
      // Refetch data
      refetchIncomplete();
      refetchTotal();
    } else if (isConfirmed && showJoinForm && selectedLLMId !== null) {
      console.log('Joined as second host successfully! Showing claim INFT modal...');
      
      // Get the model name from incompleteLLMDetails
      const llmDetails = incompleteLLMDetails.find(llm => llm.id === selectedLLMId);
      setRegisteredModelName(llmDetails?.modelName || 'Model');
      setIsSecondHost(true);
      
      // Show claim modal
      setShowClaimINFTModal(true);
      
      // Reset the join form
      resetJoinForm();
      
      // Refetch data
      refetchIncomplete();
      refetchTotal();
    }
  }, [isConfirmed, connectedAddress, showJoinForm, submittedFormData?.modelName, selectedLLMId, incompleteLLMDetails]);
  
  // Effect to handle successful INFT mint - then authorize the user
  useEffect(() => {
    const handleMintSuccess = async () => {
      if (isMintConfirmed && connectedAddress) {
        console.log('INFT minted, authorizing user...');
        
        try {
          // Authorize the hoster to use the INFT (assuming token ID 1 for now)
          // In production, you'd track the actual token ID from the mint transaction
          const tokenId = 1; // This should be retrieved from mint transaction event
          
          await authorizeINFT(tokenId, connectedAddress);
        } catch (error) {
          console.error('Failed to authorize user:', error);
        }
      }
    };
    
    handleMintSuccess();
  }, [isMintConfirmed, connectedAddress]);
  
  // Effect to auto-close modal after successful authorization
  useEffect(() => {
    if (isAuthConfirmed) {
      console.log('Authorization confirmed, closing modal in 2 seconds...');
      const timer = setTimeout(() => {
        setShowClaimINFTModal(false);
        setRegisteredModelName('');
        setIsSecondHost(false);
      }, 2000); // Close after 2 seconds to show success message
      
      return () => clearTimeout(timer);
    }
  }, [isAuthConfirmed]);
  
  // Handle claiming INFT from modal
  const handleClaimINFT = async () => {
    if (!connectedAddress) {
      alert('Please connect your wallet first');
      return;
    }
    
    try {
      // Mint INFT for the model hoster
      const encryptedURI = '0g://storage/model-data-' + Date.now();
      const metadataHash = '0x' + Array(64).fill('0').join(''); // Placeholder hash
      
      const mintSuccess = await mintINFT(connectedAddress, encryptedURI, metadataHash);
      
      if (mintSuccess) {
        console.log('INFT claim initiated...');
      }
    } catch (error) {
      console.error('Failed to claim INFT:', error);
    }
  };

  // Handle pausing a model
  const handlePauseModel = async (modelId: number) => {
    if (!connectedAddress) {
      alert('Please connect your wallet first');
      return;
    }

    try {
      setPausingModelId(modelId);
      
      // TODO: Implement pause functionality with smart contract
      // For now, just show a message
      alert('Pause functionality will report downtime to the contract. This feature is coming soon!');
      
      console.log(`Pausing model ${modelId}`);
      
      // In production, you would call a contract function here
      // Example: await reportDowntime(modelId, downtime_minutes);
      
    } catch (error) {
      console.error('Failed to pause model:', error);
      alert('Failed to pause model. Please try again.');
    } finally {
      setPausingModelId(null);
    }
  };

  // Handle stopping a model (show confirmation first)
  const handleStopModel = (model: IncompleteLLM) => {
    setModelToStop(model);
    setShowStopConfirm(true);
  };

  // Confirm stopping a model
  const confirmStopModel = async () => {
    if (!connectedAddress || !modelToStop) return;

    try {
      setStoppingModelId(modelToStop.id);
      
      const userIsHost1 = modelToStop.host1.toLowerCase() === connectedAddress.toLowerCase();
      
      // Remove the user from the hosting slot by setting their address to 0x0
      await registerLLM(
        modelToStop.id,                                   // llmId - existing entry
        userIsHost1 ? '0x0000000000000000000000000000000000000000' : modelToStop.host1,  // host1
        userIsHost1 ? modelToStop.host2 || '0x0000000000000000000000000000000000000000' : '0x0000000000000000000000000000000000000000',  // host2
        userIsHost1 ? '' : modelToStop.shardUrl1,        // shardUrl1
        userIsHost1 ? modelToStop.shardUrl2 || '' : '',  // shardUrl2
        '',                                               // modelName - keep existing
        0,                                                // totalTimeHost1
        0                                                 // totalTimeHost2
      );
      
      console.log(`Stopped hosting model ${modelToStop.id}`);
      
      // Close modal and reset
      setShowStopConfirm(false);
      setModelToStop(null);
      
      // Refresh data after confirmation
      setTimeout(() => {
        refetchTotal();
      }, 2000);
      
    } catch (error) {
      console.error('Failed to stop hosting:', error);
      alert('Failed to stop hosting. Please try again.');
    } finally {
      setStoppingModelId(null);
    }
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
          <AddModelForm
            availableModels={AVAILABLE_MODELS}
            availableShards={AVAILABLE_SHARDS}
            connectedAddress={connectedAddress}
            onSubmit={handleAddModel}
            onCancel={resetForm}
            isWriting={isWriting}
            isConfirming={isConfirming}
            isMinting={isMinting}
            isAuthorizing={isAuthorizing}
            writeError={writeError}
          />
        )}

        {/* Available Hosting Slots Section - Show incomplete LLMs waiting for second host or user's hosted models */}
        {!showAddForm && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-12"
          >
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                {filterStatus === 'mymodels' ? 'My Hosted Models' : 'Available Hosting Slots'}
              </h2>
              <p className="text-gray-600">
                {filterStatus === 'mymodels' 
                  ? 'Models you are currently hosting' 
                  : 'Join as the second host for these models'}
              </p>
            </div>

            {/* Search and Filter Controls */}
            <ModelFilters
              searchTerm={searchTerm}
              setSearchTerm={setSearchTerm}
              filterStatus={filterStatus}
              setFilterStatus={setFilterStatus}
              sortBy={sortBy}
              setSortBy={setSortBy}
              incompleteLLMCount={incompleteLLMDetails.length}
              myModelsCount={myHostedModels.length}
              connectedAddress={connectedAddress}
              isLoadingMyModels={isLoadingMyModels}
            />

            {filterStatus === 'mymodels' ? (
              <MyHostedModels
                myHostedModels={myHostedModels}
                isLoadingMyModels={isLoadingMyModels}
                searchTerm={searchTerm}
                availableShards={AVAILABLE_SHARDS}
                connectedAddress={connectedAddress!}
                pausingModelId={pausingModelId}
                stoppingModelId={stoppingModelId}
                onShowAddForm={() => setShowAddForm(true)}
                onPauseModel={handlePauseModel}
                onStopModel={handleStopModel}
              />
            ) : (
              <AvailableHostingSlots
                incompleteLLMDetails={incompleteLLMDetails}
                isLoadingIncomplete={isLoadingIncomplete}
                searchTerm={searchTerm}
                availableShards={AVAILABLE_SHARDS}
                connectedAddress={connectedAddress}
                onJoinAsHost={(llmId) => {
                  setSelectedLLMId(llmId);
                        setJoinFormData({
                    llmId: llmId,
                          shard: '',
                          walletAddress: connectedAddress || ''
                        });
                        setShowJoinForm(true);
                      }}
              />
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
              className="bg-white rounded-xl max-w-2xl w-full max-h-[70vh] overflow-y-auto"
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
        
        {/* Claim INFT Modal */}
        {showClaimINFTModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1100] p-4"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-white rounded-2xl max-w-lg w-full max-h-[85vh] overflow-y-auto shadow-2xl"
            >
              {/* Header with gradient - Sticky */}
              <div className="bg-gradient-to-r from-violet-400 to-purple-300 px-6 py-4 text-white sticky top-0 z-10">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <h2 className="text-xl font-bold">
                      {isSecondHost ? 'Joined Successfully!' : 'Registration Successful!'}
                    </h2>
                  </div>
                  <button
                    onClick={() => {
                      setShowClaimINFTModal(false);
                      setRegisteredModelName('');
                      setIsSecondHost(false);
                    }}
                    disabled={isMinting || isAuthorizing}
                    className="text-white hover:text-violet-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <p className="text-violet-100 text-sm">
                  {isSecondHost 
                    ? 'You are now the second host for this model'
                    : 'Your model is now registered on the network'
                  }
                </p>
              </div>

              {/* Content */}
              <div className="p-6">
                {/* Success Message */}
                <div className="mb-4">
                  <div className="flex items-start gap-2 p-3 bg-green-50 border border-green-200 rounded-lg">
                    <svg className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <div>
                      <h3 className="font-semibold text-green-900 mb-1">
                        {isSecondHost ? 'Joined as Second Host' : 'Model Registered'}
                      </h3>
                      <p className="text-sm text-green-700">
                        {isSecondHost ? (
                          <>You have successfully joined <strong>{registeredModelName}</strong> as the second host. The model is now complete!</>
                        ) : (
                          <><strong>{registeredModelName}</strong> has been successfully registered as a hosting slot.</>
                        )}
                      </p>
                    </div>
                  </div>
                </div>

                {/* INFT Explanation */}
                <div className="mb-4">
                  <h3 className="text-base font-bold text-gray-900 mb-2 flex items-center gap-2">
                    <svg className="w-5 h-5 text-violet-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Claim Your INFT Token
                  </h3>
                  <p className="text-sm text-gray-700 mb-3">
                    As a {isSecondHost ? 'co-host' : 'model hoster'}, you're eligible for an <strong>Intelligent NFT (INFT)</strong> token. This token grants you:
                  </p>
                  <ul className="space-y-1.5 mb-3">
                    <li className="flex items-start gap-2">
                      <svg className="w-4 h-4 text-violet-500 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      <span className="text-sm text-gray-700"><strong>AI Inference Access</strong> - Run AI queries</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <svg className="w-4 h-4 text-violet-500 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                      </svg>
                      <span className="text-sm text-gray-700"><strong>Verified Responses</strong> - Cryptographic proof</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <svg className="w-4 h-4 text-violet-500 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                      </svg>
                      <span className="text-sm text-gray-700"><strong>Ownership Rights</strong> - Full control</span>
                    </li>
                  </ul>
                </div>

                {/* Action Section */}
                <div className="bg-gradient-to-r from-violet-50 to-purple-50 border border-violet-200 rounded-lg p-3 mb-4">
                  <p className="text-xs font-semibold text-gray-700 mb-1.5">
                    Next Steps:
                  </p>
                  <ol className="text-xs text-gray-600 space-y-0.5 ml-4 list-decimal">
                    <li>Click "Claim My INFT" to mint your token</li>
                    <li>Approve the transaction in your wallet</li>
                    <li>You'll be automatically authorized</li>
                    <li>Start using AI inference in Chat!</li>
                  </ol>
                </div>

                {/* Buttons - Sticky Bottom */}
                <div className="flex gap-2 sticky bottom-0 bg-white pt-3 pb-1 -mx-6 px-6 border-t border-gray-100">
                  <button
                  onClick={() => {
                    setShowClaimINFTModal(false);
                    setRegisteredModelName('');
                    setIsSecondHost(false);
                  }}
                    disabled={isMinting || isAuthorizing}
                    className="flex-1 px-4 py-2.5 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50 font-medium text-sm"
                  >
                    Claim Later
                  </button>
                  <button
                    onClick={handleClaimINFT}
                    disabled={isMinting || isAuthorizing}
                    className="flex-1 px-4 py-2.5 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 font-medium flex items-center justify-center gap-2 text-sm"
                  >
                    {isMinting ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                        Minting...
                      </>
                    ) : isAuthorizing ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                        Authorizing...
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Claim My INFT
                      </>
                    )}
                  </button>
                </div>

                {/* Success state after authorization */}
                {isAuthConfirmed && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg"
                  >
                    <div className="flex items-center gap-2 text-green-800">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <span className="font-semibold text-sm">INFT Claimed Successfully!</span>
                    </div>
                    <p className="text-xs text-green-700 mt-1">
                      You can now use AI inference in Chat. Closing...
                    </p>
                  </motion.div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
        
        {/* Stop Hosting Confirmation Modal */}
        {showStopConfirm && modelToStop && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1100] p-4"
            onClick={() => !stoppingModelId && setShowStopConfirm(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-white rounded-2xl max-w-md w-full shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="bg-gradient-to-r from-red-500 to-red-600 px-6 py-4 text-white rounded-t-2xl">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-white bg-opacity-20 flex items-center justify-center">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">Stop Hosting?</h2>
                    <p className="text-red-100 text-sm">This action cannot be undone</p>
                  </div>
                </div>
              </div>

              {/* Content */}
              <div className="p-6">
                <div className="mb-4">
                  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <h3 className="font-semibold text-gray-900 mb-2">You are about to stop hosting:</h3>
                    <p className="text-lg font-bold text-red-700 mb-1">{modelToStop.modelName}</p>
                    <p className="text-sm text-gray-600">Model ID: #{modelToStop.id}</p>
                  </div>
                </div>

                <div className="space-y-3 mb-6">
                  <div className="flex items-start gap-2">
                    <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    <p className="text-sm text-gray-700">You will no longer be hosting this model</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-sm text-gray-700">Your accumulated rewards will be distributed</p>
                  </div>
                  <div className="flex items-start gap-2">
                    <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-sm text-gray-700">The model may become incomplete if you're the only host</p>
                  </div>
                </div>

                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-6">
                  <p className="text-xs text-yellow-800">
                    <strong>Note:</strong> If you want to host this model again in the future, you'll need to register from scratch.
                  </p>
                </div>

                {/* Buttons */}
                <div className="flex gap-3">
                  <button
                    onClick={() => setShowStopConfirm(false)}
                    disabled={stoppingModelId !== null}
                    className="flex-1 px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={confirmStopModel}
                    disabled={stoppingModelId !== null}
                    className="flex-1 px-4 py-3 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
                  >
                    {stoppingModelId !== null ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                        Stopping...
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                        </svg>
                        Yes, Stop Hosting
                      </>
                    )}
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default ModelsPage;
