"use client";

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { Navbar } from '../components/Navbar';
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
import { areUrlsSame } from '../utils/shardUtils';

interface AddModelFormData {
  modelName: string;
  walletAddress: string;
  shardSelection: string;
  shardUrl: string;
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
  'Mistral-7B-Instruct-v0.3',
  'Qwen2.5-7B-Instruct',
  'Phi-3-Mini-4K-Instruct',
  'Gemma-2-2B-Instruct',
  'GPT-3.5-Turbo',
  'Claude-3-Haiku',
  'DeepSeek-V2-Lite'
];

// LLM Icon mapping - Maps model names to image URLs
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

// Default icon URL for models not in the mapping
const DEFAULT_LLM_ICON = 'https://www.redpill.ai/_next/image?url=https%3A%2F%2Ft0.gstatic.com%2FfaviconV2%3Fclient%3DSOCIAL%26type%3DFAVICON%26fallback_opts%3DTYPE%2CSIZE%2CURL%26url%3Dhttps%3A%2F%2Fhuggingface.co%2F%26size%3D32&w=48&q=75';

// Helper function to get icon URL for a model
const getModelIcon = (modelName: string): string => {
  return LLM_ICONS[modelName] || DEFAULT_LLM_ICON;
};

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
  const [filterStatus, setFilterStatus] = useState<'all' | 'available' | 'incomplete' | 'mymodels'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [showJoinForm, setShowJoinForm] = useState(false);
  const [selectedLLMId, setSelectedLLMId] = useState<number | null>(null);
  const [selectedModelName, setSelectedModelName] = useState<string>('');
  const [existingShardUrl, setExistingShardUrl] = useState<string>('');
  const [allModels, setAllModels] = useState<IncompleteLLM[]>([]);
  const [isLoadingAllModels, setIsLoadingAllModels] = useState(false);
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

  // Smart contract hooks
  const { registerLLM, txHash, isWriting, writeError, resetWrite, isConfirming, isConfirmed } = useRegisterLLM();
  const { incompleteLLMs, refetch: refetchIncomplete } = useGetIncompleteLLMs();
  const { totalLLMs, refetch: refetchTotal } = useGetTotalLLMs();
  const { address: connectedAddress } = useAccount();
  const config = useConfig();
  
  // INFT hooks for minting and authorization
  const { mint: mintINFT, isPending: isMinting, isConfirmed: isMintConfirmed } = useMintINFT();
  const { authorize: authorizeINFT, isPending: isAuthorizing, isConfirmed: isAuthConfirmed } = useAuthorizeINFT();

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
        
        // Fetch all LLMs
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
  }, [totalLLMs, config, isConfirmed]);

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
      if (!connectedAddress || totalLLMs === undefined) {
        setMyHostedModels([]);
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
    resetWrite();
  };

  // Reset join form
  const resetJoinForm = () => {
    setShowJoinForm(false);
    setSelectedLLMId(null);
    setSelectedModelName('');
    setExistingShardUrl('');
    resetWrite();
  };

  // Handle form submission for registering first host
  const handleAddModel = async (formData: AddModelFormData) => {
    if (!formData.modelName || !formData.shardUrl || !formData.walletAddress || totalLLMs === undefined) {
      return;
    }

    if (!connectedAddress) {
      alert('Please connect your wallet first');
      return;
    }

    try {
      resetWrite();
      
      // Store model name for the claim modal later
      setRegisteredModelName(formData.modelName);
      
      // Register first host on blockchain (creates slot)
      // Pass totalLLMs as llmId to create new entry
      // Use address(0) for host2 and empty string for shardUrl2 to leave them empty
      // Pass 0 for time fields - these will be set by oracle
      await registerLLM(
        Number(totalLLMs),                              // llmId - array length for new entry
        formData.walletAddress,                         // host1
        '0x0000000000000000000000000000000000000000',  // host2 - empty (address zero)
        formData.shardUrl,                              // shardUrl1 - TEE endpoint URL
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
  const handleJoinAsSecondHost = async (formData: AddModelFormData) => {
    if (selectedLLMId === null || !formData.walletAddress || !formData.shardUrl) return;

    if (!connectedAddress) {
      alert('Please connect your wallet first');
      return;
    }

    try {
      resetWrite();
      
      // Store model name for the claim modal later
      const llmDetails = incompleteLLMDetails.find(llm => llm.id === selectedLLMId);
      setRegisteredModelName(llmDetails?.modelName || 'Model');
      
      // Double-check for URL duplication (final validation)
      if (llmDetails?.shardUrl1 && areUrlsSame(llmDetails.shardUrl1, formData.shardUrl)) {
        alert('Error: Cannot use the same TEE endpoint URL as the first host. Please use a different endpoint.');
        return;
      }
      
      // Join as second host on blockchain
      // Pass existing llmId, leave host1 fields as 0/empty to keep them, fill in host2 fields
      // Pass 0 for time - will be set by oracle
      await registerLLM(
        selectedLLMId,                                  // llmId - existing entry
        '0x0000000000000000000000000000000000000000',  // host1 - empty (keep existing)
        formData.walletAddress,                         // host2 - new host
        '',                                             // shardUrl1 - empty (keep existing)
        formData.shardUrl,                              // shardUrl2 - TEE endpoint URL
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
    if (isConfirmed && connectedAddress && !showJoinForm && registeredModelName) {
      console.log('Model registered successfully! Showing claim INFT modal...');
      
      setIsSecondHost(false);
      
      // Show claim modal
      setShowClaimINFTModal(true);
      
      // Reset the registration form
      resetForm();
      
      // Refetch data
      refetchIncomplete();
      refetchTotal();
    } else if (isConfirmed && showJoinForm && registeredModelName) {
      console.log('Joined as second host successfully! Showing claim INFT modal...');
      
      setIsSecondHost(true);
      
      // Show claim modal
      setShowClaimINFTModal(true);
      
      // Reset the join form
      resetJoinForm();
      
      // Refetch data
      refetchIncomplete();
      refetchTotal();
    }
  }, [isConfirmed, connectedAddress, showJoinForm, registeredModelName]);
  
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
        {!showAddForm && !showJoinForm && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-12"
          >
            <div className="mb-6">
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

                {/* Sort Button */}
                <div className="relative">
                  <button className="flex items-center justify-center w-10 h-10 border border-gray-300 rounded-lg bg-white text-gray-700 hover:bg-gray-50 transition-colors focus:outline-none">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c2.755 0 5.455.232 8.083.678.533.09.917.556.917 1.096v1.044a2.25 2.25 0 01-.659 1.591l-5.432 5.432a2.25 2.25 0 00-.659 1.591v2.927a2.25 2.25 0 01-1.244 2.013L9.75 21v-6.568a2.25 2.25 0 00-.659-1.591L3.659 7.409A2.25 2.25 0 013 5.818V4.774c0-.54.384-1.006.917-1.096A48.32 48.32 0 0112 3z" />
                    </svg>
                  </button>
                  
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

            {(isLoadingAllModels || isLoadingIncomplete || (filterStatus === 'mymodels' && isLoadingMyModels)) ? (
              <div className="flex items-center justify-center py-10">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-400"></div>
              </div>
            ) : filterStatus === 'mymodels' ? (
              // Show user's hosted models
              myHostedModels.length === 0 ? (
                <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl border-2 border-gray-200 p-8">
                  <div className="text-center max-w-md mx-auto">
                    <div className="mb-4">
                      <svg className="w-16 h-16 mx-auto text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 mb-2">No models hosted yet</h3>
                    <p className="text-gray-600 mb-4">You're not currently hosting any models. Start by adding a new model or joining an existing hosting slot!</p>
                    <button
                      onClick={() => setShowAddForm(true)}
                      className="px-6 py-3 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity font-medium"
                    >
                      Add Your First Model
                    </button>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {myHostedModels
                    .filter(model => {
                      // Apply search filter
                      if (searchTerm) {
                        return model.modelName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                               model.host1.toLowerCase().includes(searchTerm.toLowerCase()) ||
                               (model.host2 && model.host2.toLowerCase().includes(searchTerm.toLowerCase())) ||
                               model.shardUrl1.toLowerCase().includes(searchTerm.toLowerCase()) ||
                               (model.shardUrl2 && model.shardUrl2.toLowerCase().includes(searchTerm.toLowerCase()));
                      }
                      return true;
                    })
                    .map((model) => {
                      const userIsHost1 = model.host1.toLowerCase() === connectedAddress!.toLowerCase();
                      const userIsHost2 = model.host2?.toLowerCase() === connectedAddress!.toLowerCase();
                      const userRole = userIsHost1 ? 'Host 1' : 'Host 2';
                      const userShard = userIsHost1 ? model.shardUrl1 : model.shardUrl2;
                      const partnerAddress = userIsHost1 ? model.host2 : model.host1;
                      const partnerShard = userIsHost1 ? model.shardUrl2 : model.shardUrl1;

                      return (
                        <motion.div
                          key={model.id}
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className="bg-white rounded-xl border border-gray-200 p-6 hover:border-violet-400 hover:shadow-md transition-all"
                        >
                          <div className="flex items-start justify-between mb-4">
                            <div className="flex items-start gap-3 flex-1">
                              {/* LLM Icon */}
                              <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center flex-shrink-0 p-1.5">
                                <img 
                                  src={getModelIcon(model.modelName)} 
                                  alt={model.modelName}
                                  className="w-full h-full object-contain"
                                  onError={(e) => {
                                    e.currentTarget.src = DEFAULT_LLM_ICON;
                                  }}
                                />
                              </div>
                              <div className="flex-1">
                                <h3 className="font-bold text-lg text-gray-900">{model.modelName}</h3>
                              </div>
                            </div>
                            {/* Status Badge - Top Right */}
                            <div className="px-2.5 py-1 rounded-full text-xs font-medium flex-shrink-0 bg-green-100 text-green-700">
                              ✓ Hosting
                            </div>
                          </div>

                          <div className="space-y-3 mb-4">
                            {/* Model Info */}
                            <div className="space-y-2 text-xs">
                              <div className="flex items-start gap-2">
                                <span className="text-gray-500">Model ID:</span>
                                <span className="text-gray-900">#{model.id}</span>
                              </div>
                            </div>

                            {/* Your Role */}
                            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                              <div className="flex items-center gap-2 mb-2">
                                <svg className="w-4 h-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                </svg>
                                <span className="text-xs font-semibold text-gray-900">Your Role:</span>
                              </div>
                              <div className="text-sm font-medium text-gray-900 mb-2">{userRole}</div>
                              <div className="space-y-1.5 text-xs text-gray-600">
                                {!model.isComplete && (
                                  <div className="flex items-start gap-2">
                                    <span className="text-gray-500">Shard:</span>
                                    <span className="text-gray-900">{userIsHost1 ? 'Shard 1' : 'Shard 2'} (Lower Layers)</span>
                                  </div>
                                )}
                                <div className="flex items-start gap-2">
                                  <span className="text-gray-500 flex-shrink-0">URL:</span>
                                  <span className="font-mono text-gray-900 break-all">{userShard ? userShard.slice(0, 60) + '...' : 'N/A'}</span>
                                </div>
                              </div>
                            </div>

                            {/* Partner Info */}
                            {partnerAddress && partnerAddress !== '0x0000000000000000000000000000000000000000' && (
                              <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                                <div className="flex items-center gap-2 mb-2">
                                  <svg className="w-4 h-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                                  </svg>
                                  <span className="text-xs font-semibold text-gray-900">Partner:</span>
                                </div>
                                <div className="space-y-1.5 text-xs">
                                  <div className="flex items-start gap-2">
                                    <span className="text-gray-500">Address:</span>
                                    <span className="font-mono text-gray-900">{partnerAddress.slice(0, 10)}...{partnerAddress.slice(-8)}</span>
                                  </div>
                                  {!model.isComplete && (
                                    <div className="flex items-start gap-2">
                                      <span className="text-gray-500">Shard:</span>
                                      <span className="text-gray-900">{userIsHost1 ? 'Shard 2' : 'Shard 1'} (Upper Layers)</span>
                                    </div>
                                  )}
                                  <div className="flex items-start gap-2">
                                    <span className="text-gray-500 flex-shrink-0">URL:</span>
                                    <span className="font-mono text-gray-900 break-all">{partnerShard ? partnerShard.slice(0, 60) + '...' : 'N/A'}</span>
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* Pool Balance */}
                            <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <svg className="w-4 h-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                  </svg>
                                  <span className="text-xs font-semibold text-gray-900">Rewards Pool:</span>
                                </div>
                                <span className="text-sm font-bold text-gray-900">{model.poolBalance?.toString() || '0'}</span>
                              </div>
                            </div>
                          </div>

                          {/* Action Buttons */}
                          <div className="space-y-2">
                            <button
                              onClick={() => router.push('/chat')}
                              className="w-full py-2 px-4 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors font-medium text-sm flex items-center justify-center gap-2"
                            >
                              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                              </svg>
                              Use in Chat
                            </button>
                            
                            <div className="grid grid-cols-2 gap-2">
                              <button
                                onClick={() => handlePauseModel(model.id)}
                                disabled={pausingModelId === model.id || stoppingModelId === model.id}
                                className="py-2 px-3 bg-yellow-100 text-yellow-700 rounded-lg hover:bg-yellow-200 transition-colors font-medium text-xs flex items-center justify-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                                title="Report downtime for this model"
                              >
                                {pausingModelId === model.id ? (
                                  <>
                                    <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-yellow-700"></div>
                                    Pausing...
                                  </>
                                ) : (
                                  <>
                                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    Pause
                                  </>
                                )}
                              </button>
                              
                              <button
                                onClick={() => handleStopModel(model)}
                                disabled={pausingModelId === model.id || stoppingModelId === model.id}
                                className="py-2 px-3 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors font-medium text-xs flex items-center justify-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                                title="Stop hosting this model"
                              >
                                {stoppingModelId === model.id ? (
                                  <>
                                    <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-red-700"></div>
                                    Stopping...
                                  </>
                                ) : (
                                  <>
                                    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                                    </svg>
                                    Stop
                                  </>
                                )}
                              </button>
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                </div>
              )
            ) : (() => {
              // Determine which models to show based on filter
              let modelsToShow: IncompleteLLM[] = [];
              if (filterStatus === 'all') {
                modelsToShow = allModels;
              } else if (filterStatus === 'available') {
                modelsToShow = allModels.filter(m => m.isComplete);
              } else if (filterStatus === 'incomplete') {
                modelsToShow = incompleteLLMDetails;
              }
              
              return modelsToShow.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20">
                <div className="text-center max-w-md">
                    <h3 className="text-xl font-bold text-gray-900 mb-2">
                      {filterStatus === 'all' ? 'No models found' :
                       filterStatus === 'available' ? 'No available models' :
                       'No incomplete models'}
                    </h3>
                    <p className="text-gray-600">
                      {filterStatus === 'all' ? 'There are no models registered on the network yet' :
                       filterStatus === 'available' ? 'There are no complete models ready to use' :
                       'There are currently no hosting slots waiting for a second host'}
                    </p>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {modelsToShow
                  .filter(llm => {
                    // Apply search filter
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
                            {/* LLM Icon */}
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
                          {/* Status Badge - Top Right */}
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
                          <div className="flex items-start gap-2">
                        <span className="text-gray-500">Pool Balance:</span>
                            <span className="text-gray-700 font-semibold">{llm.poolBalance?.toString() || '0'} credits</span>
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
                      onClick={() => {
                        setSelectedLLMId(llm.id);
                              setSelectedModelName(llm.modelName);
                        setExistingShardUrl(llm.shardUrl1 || '');
                        setShowJoinForm(true);
                      }}
                            className="w-full py-2 px-4 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors font-medium text-sm"
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
        )}

        {/* Join as Second Host Form - Embedded */}
        {showJoinForm && selectedLLMId !== null && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mb-12"
          >
            <div className="bg-gradient-to-r from-violet-50 to-purple-50 rounded-xl border-2 border-violet-300 p-6 mb-6">
              <div className="flex items-center justify-between mb-2">
                  <div className="flex-1">
                  <h2 className="text-2xl font-bold text-gray-900">Join as Second Host</h2>
                    <p className="text-sm text-gray-600 mt-1">
                    Model: <strong className="text-violet-700">{selectedModelName}</strong>
                    </p>
                    <div className="mt-2 flex items-center gap-3 text-xs">
                      <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-red-500"></div>
                        <span className="text-gray-700">Shard 1: Taken</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-green-500"></div>
                        <span className="text-gray-700 font-medium">Shard 2: Available</span>
                      </div>
                    </div>
                    {existingShardUrl && (
                      <div className="mt-2 p-2 bg-white rounded border border-violet-200">
                        <p className="text-xs font-medium text-gray-700 mb-1">
                          <svg className="w-3 h-3 inline mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          First host's TEE URL (you must use a different one):
                        </p>
                        <code className="text-xs font-mono text-violet-700 break-all">{existingShardUrl}</code>
                      </div>
                    )}
                  </div>
                  <button
                  onClick={resetJoinForm}
                  disabled={isWriting || isConfirming || isMinting || isAuthorizing}
                  className="text-gray-400 hover:text-gray-600 transition-colors disabled:opacity-50 p-2 hover:bg-white rounded-lg"
                  title="Cancel"
                  >
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>

            <AddModelForm
              availableModels={[selectedModelName]}
              availableShards={AVAILABLE_SHARDS}
              connectedAddress={connectedAddress}
              onSubmit={handleJoinAsSecondHost}
              onCancel={resetJoinForm}
              isWriting={isWriting}
              isConfirming={isConfirming}
              isMinting={isMinting}
              isAuthorizing={isAuthorizing}
              writeError={writeError}
              availableShard="shard2"
              existingShardUrl={existingShardUrl}
            />
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
