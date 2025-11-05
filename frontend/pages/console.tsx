"use client";

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Navbar } from '../components/Navbar';
import { AddModelForm } from '../components/AddModelForm';
import MagicBento from '../components/MagicBento';
import { useRegisterLLM } from '../lib/contracts/creditUse/writes/useRegisterLLM';
import { useGetTotalLLMs } from '../lib/contracts/creditUse/reads/useGetTotalLLMs';
import { useStopLLM } from '../lib/contracts/creditUse/writes/useStopLLM';
import { useWithdrawAll } from '../lib/contracts/creditUse/writes/useWithdrawAll';
import { useGetLifetimeEarnings } from '../lib/contracts/creditUse/reads/useGetLifetimeEarnings';
import { useGetTotalUnclaimedEarnings } from '../lib/contracts/creditUse/reads/useGetTotalUnclaimedEarnings';
import { useAccount, useConfig } from 'wagmi';
import { readContract } from '@wagmi/core';
import ABI from '../utils/abi.json';
import { CONTRACT_ADDRESS } from '../utils/address';
// INFT minting and authorization is now handled by backend
import { useRouter } from 'next/router';

interface AddModelFormData {
  modelName: string;
  walletAddress: string;
  shardSelection: string;
  shardUrl: string;
}

interface HostedLLM {
  id: number;
  modelName: string;
  host1: string;
  host2?: string;
  shardUrl1: string;
  shardUrl2?: string;
  poolBalance: bigint;
  registeredAtHost1: bigint;
  registeredAtHost2?: bigint;
  usageCount: bigint;
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

// Docker compose content for hosting guide
const DOCKER_COMPOSE_CONTENT = `version: '3.8'

services:
  llm-server:
    image: derek2403/teetee-llm-server:latest
    container_name: teetee-llm-server
    ports:
      - "3001:3001"
    environment:
      - PHALA_API_KEY=\${PHALA_API_KEY}
      - PORT=3001
      - NODE_ENV=production
    restart: unless-stopped
    volumes:
      - /var/run/tappd.sock:/var/run/tappd.sock`;

// Slideshow images (16:9 ratio)
const GUIDE_SLIDES = [
  '/images/guide/1.png',
  '/images/guide/2.png',
  '/images/guide/3.png',
  '/images/guide/4.png',
];

const AVAILABLE_SHARDS: any[] = [];

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

// Helper function to format pool balance from Wei (21 decimals) to OG
const formatPoolBalance = (balance: bigint | undefined): string => {
  if (!balance) return '0';
  const weiValue = Number(balance);
  const ogValue = weiValue / 1e15;
  return ogValue.toFixed(3); // Show 9 decimal places
};

// Helper function to format time in seconds to readable format
const formatHostingTime = (seconds: number): string => {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86400)}d`;
};

const DashboardPage = () => {
  const router = useRouter();
  const [showAddForm, setShowAddForm] = useState(false);
  const [showHostingGuide, setShowHostingGuide] = useState(false);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isCopied, setIsCopied] = useState(false);
  const [showClaimINFTModal, setShowClaimINFTModal] = useState(false);
  const [registeredModelName, setRegisteredModelName] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'dashboard' | 'mymodels' | 'incomplete'>('dashboard');
  const [myHostedModels, setMyHostedModels] = useState<HostedLLM[]>([]);
  const [isLoadingMyModels, setIsLoadingMyModels] = useState(false);
  const [incompleteModels, setIncompleteModels] = useState<HostedLLM[]>([]);
  const [isLoadingIncomplete, setIsLoadingIncomplete] = useState(false);
  const [showJoinForm, setShowJoinForm] = useState(false);
  const [selectedLLMId, setSelectedLLMId] = useState<number | null>(null);
  const [selectedModelName, setSelectedModelName] = useState<string>('');
  const [existingShardUrl, setExistingShardUrl] = useState<string>('');
  const [isSecondHost, setIsSecondHost] = useState(false);

  // Stop LLM modal state
  const [showStopModal, setShowStopModal] = useState(false);
  const [stopModelId, setStopModelId] = useState<number | null>(null);
  const [stopModelName, setStopModelName] = useState<string>('');
  const [stopHostNumber, setStopHostNumber] = useState<number | null>(null);
  const [isBothHostsForStop, setIsBothHostsForStop] = useState(false);

  // Withdraw earnings modal state
  const [showWithdrawModal, setShowWithdrawModal] = useState(false);
  const [totalEarnings, setTotalEarnings] = useState<string>('0');

  // Real-time ticker for hosting time
  const [currentTime, setCurrentTime] = useState(Math.floor(Date.now() / 1000));

  // Smart contract hooks
  const { registerLLM, txHash, isWriting, writeError, resetWrite, isConfirming, isConfirmed } = useRegisterLLM();
  const { totalLLMs, refetch: refetchTotal } = useGetTotalLLMs();
  const { address: connectedAddress } = useAccount();
  const config = useConfig();
  
  // Stop and Withdraw hooks
  const { stopLLM, isWriting: isStopping, isConfirming: isStopConfirming, isConfirmed: isStopConfirmed, resetWrite: resetStopWrite } = useStopLLM();
  const { withdrawAll, isWriting: isWithdrawWriting, isConfirming: isWithdrawConfirming, isConfirmed: isWithdrawConfirmed, resetWrite: resetWithdrawWrite } = useWithdrawAll();
  
  // Earnings hooks
  const { lifetimeEarnings, refetch: refetchLifetime } = useGetLifetimeEarnings(connectedAddress as `0x${string}`);
  const { unclaimedEarnings, refetch: refetchUnclaimed } = useGetTotalUnclaimedEarnings(connectedAddress as `0x${string}`);
  
  // Backend mint & authorization state
  const [isMintingAndAuthorizing, setIsMintingAndAuthorizing] = useState(false);
  const [isMintAuthConfirmed, setIsMintAuthConfirmed] = useState(false);
  const [mintAuthError, setMintAuthError] = useState<string | null>(null);
  const [mintedTokenId, setMintedTokenId] = useState<number | null>(null);
  
  // Track if the claiming process has started (after user clicks button)
  const [hasStartedClaiming, setHasStartedClaiming] = useState(false);

  // Real-time ticker - updates every second
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(Math.floor(Date.now() / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Fetch models hosted by the current user
  useEffect(() => {
    const fetchMyHostedModels = async () => {
      if (!connectedAddress || totalLLMs === undefined) {
        setMyHostedModels([]);
        return;
      }

      setIsLoadingMyModels(true);
      
      try {
        const myModels: HostedLLM[] = [];
        
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
                  registeredAtHost1: data.registeredAtHost1 !== undefined ? data.registeredAtHost1 : (data[6] !== undefined ? data[6] : 0n),
                  registeredAtHost2: data.registeredAtHost2 !== undefined ? data.registeredAtHost2 : (data[7] !== undefined ? data[7] : 0n),
                  usageCount: data.usageCount !== undefined ? data.usageCount : (data[10] !== undefined ? data[10] : 0n),
                  isComplete: data.isComplete !== undefined ? data.isComplete : (data[11] !== undefined ? data[11] : false)
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
  }, [connectedAddress, totalLLMs, config, isConfirmed, isStopConfirmed]);

  // Fetch incomplete models (models waiting for second host)
  useEffect(() => {
    const fetchIncompleteModels = async () => {
      if (totalLLMs === undefined) {
        setIncompleteModels([]);
        return;
      }

      setIsLoadingIncomplete(true);
      
      try {
        const incomplete: HostedLLM[] = [];
        
        for (let i = 0; i < Number(totalLLMs); i++) {
          try {
            const data = await readContract(config, {
              address: CONTRACT_ADDRESS as `0x${string}`,
              abi: ABI,
              functionName: 'getHostedLLM',
              args: [BigInt(i)]
            }) as any;
            
            if (data) {
              const host1 = data.host1 || data[0] || '0x0000000000000000000000000000000000000000';
              const host2 = data.host2 || data[1] || '0x0000000000000000000000000000000000000000';
              const isComplete = data.isComplete !== undefined ? data.isComplete : (data[11] !== undefined ? data[11] : false);
              
              // Only include models that are incomplete (missing either host1 or host2)
              // Trust the contract's isComplete flag
              if (!isComplete) {
                incomplete.push({
                  id: i,
                  modelName: data.modelName || data[4] || 'Unknown Model',
                  host1: host1,
                  host2: host2,
                  shardUrl1: data.shardUrl1 || data[2] || '',
                  shardUrl2: data.shardUrl2 || data[3] || '',
                  poolBalance: data.poolBalance !== undefined ? data.poolBalance : (data[5] !== undefined ? data[5] : 0n),
                  registeredAtHost1: data.registeredAtHost1 !== undefined ? data.registeredAtHost1 : (data[6] !== undefined ? data[6] : 0n),
                  registeredAtHost2: data.registeredAtHost2 !== undefined ? data.registeredAtHost2 : (data[7] !== undefined ? data[7] : 0n),
                  usageCount: data.usageCount !== undefined ? data.usageCount : (data[10] !== undefined ? data[10] : 0n),
                  isComplete: false
                });
              }
            }
          } catch (error) {
            console.error(`Failed to fetch LLM ${i}:`, error);
          }
        }
        
        setIncompleteModels(incomplete);
      } catch (error) {
        console.error('Error fetching incomplete models:', error);
      } finally {
        setIsLoadingIncomplete(false);
      }
    };

    fetchIncompleteModels();
  }, [totalLLMs, config, isConfirmed, isStopConfirmed]);

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

  // Handle joining as second host
  const handleJoinAsSecondHost = async (formData: AddModelFormData) => {
    if (selectedLLMId === null || !formData.walletAddress || !formData.shardUrl) return;

    if (!connectedAddress) {
      alert('Please connect your wallet first');
      return;
    }

    try {
      resetWrite();
      
      setRegisteredModelName(selectedModelName);
      setIsSecondHost(true);
      
      await registerLLM(
        selectedLLMId,
        '0x0000000000000000000000000000000000000000',
        formData.walletAddress,
        '',
        formData.shardUrl,
        ''
      );
    } catch (error) {
      console.error('Failed to join as second host:', error);
    }
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
      await registerLLM(
        Number(totalLLMs),                              // llmId - array length for new entry
        formData.walletAddress,                         // host1
        '0x0000000000000000000000000000000000000000',  // host2 - empty (address zero)
        formData.shardUrl,                              // shardUrl1 - TEE endpoint URL
        '',                                             // shardUrl2 - empty
        formData.modelName                             // modelName
      );
    } catch (error) {
      console.error('Failed to register model:', error);
    }
  };

  // Effect to handle successful model registration - show claim modal
  React.useEffect(() => {
    if (isConfirmed && connectedAddress && registeredModelName) {
      console.log('Model registered successfully! Showing claim INFT modal...');
      
      // Reset claiming state
      setHasStartedClaiming(false);
      setIsMintAuthConfirmed(false);
      setIsMintingAndAuthorizing(false);
      setMintAuthError(null);
      setMintedTokenId(null);
      
      // Show claim modal
      setShowClaimINFTModal(true);
      
      // Reset the appropriate form
      if (showJoinForm) {
        resetJoinForm();
      } else {
        resetForm();
      }
      
      // Refetch data
      refetchTotal();
      refetchLifetime();
      refetchUnclaimed();
    }
  }, [isConfirmed, connectedAddress, registeredModelName]);
  
  // Handle claiming INFT from modal - calls backend to mint then authorize in 2 steps
  const handleClaimINFT = async () => {
    if (!connectedAddress) {
      alert('Please connect your wallet first');
      return;
    }
    
    // Mark that the claiming process has started
    setHasStartedClaiming(true);
    setIsMintingAndAuthorizing(true);
    setMintAuthError(null);
    
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';
    
    try {
      // Step 1: Mint the INFT
      console.log('Step 1: Minting INFT...');
      
      // Generate a proper metadata hash (not all zeros)
      const encryptedURI = '0g://storage/model-data-' + Date.now();
      const hashInput = `${connectedAddress}-${encryptedURI}-${Date.now()}`;
      const encoder = new TextEncoder();
      const data = encoder.encode(hashInput);
      const hashBuffer = await crypto.subtle.digest('SHA-256', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const metadataHash = '0x' + hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      
      const mintBody = {
        userAddress: connectedAddress,
        encryptedURI: encryptedURI,
        metadataHash: metadataHash
      };
      
      const mintResponse = await fetch(`${backendUrl}/mint-inft`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(mintBody)
      });
      
      const mintData = await mintResponse.json();
      
      if (!mintResponse.ok || !mintData.success) {
        throw new Error(mintData.error || 'Mint failed');
      }
      
      const tokenId = mintData.tokenId;
      
      // Check if user already owned an INFT
      if (mintData.alreadyOwned) {
        console.log('✅ User already has INFT #' + tokenId);
        console.log('   No minting needed - proceeding with authorization');
      } else {
        console.log('✅ INFT minted!', mintData);
        console.log('   Token ID:', tokenId);
        console.log('   Mint Tx:', mintData.txHash);
      }
      
      // Safety check: Token IDs must be >= 1
      if (!tokenId || tokenId < 1) {
        console.error('Invalid token ID received from mint:', tokenId);
        throw new Error(`Invalid token ID received: ${tokenId}. Expected >= 1`);
      }
      
      setMintedTokenId(tokenId);
      
      // Wait for the transaction to settle on the blockchain (only if newly minted)
      if (!mintData.alreadyOwned) {
        console.log('Waiting 5 seconds before authorization...');
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
      
      // Step 2: Authorize the user
      console.log('Step 2: Authorizing user...');
      
      const authBody = {
        tokenId: tokenId,
        userAddress: connectedAddress
      };
          
      const authResponse = await fetch(`${backendUrl}/authorize-inft`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
        body: JSON.stringify(authBody)
      });
      
      const authData = await authResponse.json();
      
      if (!authResponse.ok || !authData.success) {
        throw new Error(authData.error || 'Authorization failed');
      }
      
      console.log('✅ User authorized!', authData);
      console.log('   Auth Tx:', authData.txHash);
      
      // Success - both operations completed
      setIsMintAuthConfirmed(true);
      setIsMintingAndAuthorizing(false);
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('Failed to mint/authorize INFT:', errorMessage);
      setMintAuthError(errorMessage);
      setIsMintingAndAuthorizing(false);
      setHasStartedClaiming(false); // Reset so user can try again
    }
  };

  // Handle copying docker compose content
  const handleCopyDockerCompose = async () => {
    try {
      await navigator.clipboard.writeText(DOCKER_COMPOSE_CONTENT);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // Handle slideshow navigation
  const handleNextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % GUIDE_SLIDES.length);
  };

  const handlePrevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + GUIDE_SLIDES.length) % GUIDE_SLIDES.length);
  };

  // Handle stop LLM button click
  const handleStopLLMClick = (model: HostedLLM) => {
    const isHost1 = model.host1.toLowerCase() === connectedAddress?.toLowerCase();
    const isHost2 = model.host2?.toLowerCase() === connectedAddress?.toLowerCase();
    const isBothHosts = isHost1 && isHost2;
    
    setStopModelId(model.id);
    setStopModelName(model.modelName);
    setIsBothHostsForStop(isBothHosts);
    
    // If user only hosts one shard, auto-select which one
    if (!isBothHosts) {
      setStopHostNumber(isHost1 ? 1 : 2);
    } else {
      setStopHostNumber(null); // User needs to choose
    }
    
    setShowStopModal(true);
  };

  // Handle stop LLM confirmation
  const handleStopLLMConfirm = async () => {
    if (stopModelId === null || stopHostNumber === null) return;
    
    try {
      await stopLLM(stopModelId, stopHostNumber);
    } catch (error) {
      console.error('Failed to stop LLM:', error);
    }
  };

  // Effect to handle successful stop - deauthorize INFT
  React.useEffect(() => {
    const handleDeauthorization = async () => {
      if (!isStopConfirmed || !connectedAddress) return;
      
      console.log('Successfully stopped hosting LLM - deauthorizing INFT...');
      
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';
      
      try {
        // Backend will automatically find the user's INFT from this contract
        const deauthResponse = await fetch(`${backendUrl}/deauthorize-inft`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            userAddress: connectedAddress
          })
        });
        
        const deauthData = await deauthResponse.json();
        
        if (deauthResponse.ok && deauthData.success) {
          console.log('✅ INFT deauthorized successfully!', deauthData);
        } else {
          console.warn('⚠️ INFT deauthorization failed:', deauthData.error);
        }
      } catch (error) {
        console.error('❌ Failed to deauthorize INFT:', error);
      }
      
      // Close modal and reset state
      setShowStopModal(false);
      setStopModelId(null);
      setStopModelName('');
      setStopHostNumber(null);
      setIsBothHostsForStop(false);
      resetStopWrite();
      refetchTotal();
      refetchLifetime();
      refetchUnclaimed();
    };
    
    handleDeauthorization();
  }, [isStopConfirmed, connectedAddress]);

  // Handle withdraw all earnings - open modal
  const handleWithdrawAllClick = () => {
    if (!connectedAddress || myHostedModels.length === 0) return;
    
    // Get unclaimed earnings from smart contract (correct Wei conversion)
    const unclaimedWei = unclaimedEarnings ? Number(unclaimedEarnings) : 0;
    const unclaimedOG = unclaimedWei / 1e18; // Wei to 0G
    const formattedEarnings = unclaimedOG.toFixed(5);
    
    // Always open modal - user can see lifetime earnings even if unclaimed is 0
    setTotalEarnings(formattedEarnings);
    setShowWithdrawModal(true);
  };

  // Handle the actual withdrawal transaction
  const handleWithdrawConfirm = async () => {
    if (!connectedAddress || myHostedModels.length === 0 || isWithdrawWriting) return;
    
    try {
      // Withdraw from ALL models in one transaction
      await withdrawAll();
    } catch (error) {
      console.error('Withdrawal error:', error);
    }
  };
  
  // Auto-refetch data when withdrawal is confirmed and close modal
  useEffect(() => {
    if (isWithdrawConfirmed) {
      refetchTotal();
      refetchLifetime();
      refetchUnclaimed();
      // Close the modal after successful withdrawal
      setTimeout(() => {
        setShowWithdrawModal(false);
        resetWithdrawWrite();
      }, 2000); // Show success state for 2 seconds before closing
    }
  }, [isWithdrawConfirmed, refetchTotal, refetchLifetime, refetchUnclaimed]);

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
              <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
              <p className="text-gray-600 mt-1">Add and manage your model hosting configurations</p>
            </div>
            
            {/* Action Buttons */}
            <div className="flex items-center gap-3">
              {myHostedModels.length > 0 && !showAddForm && (
                <button
                  onClick={() => setShowAddForm(true)}
                  className="px-6 py-2 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-full hover:opacity-90 transition-opacity font-medium flex items-center gap-2 text-sm"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  Add New Model
                </button>
              )}
            <button
              onClick={() => setShowHostingGuide(true)}
              className="p-3 rounded-full bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-800 transition-all"
              title="How to host a model on Phala Cloud"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </button>
          </div>
          </div>
        </div>

        {/* Tabs - Always show */}
        {!showAddForm && !showJoinForm && (
          <div className="mb-8 border-b border-gray-200">
            <div className="flex gap-8">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`pb-4 px-2 font-semibold transition-all ${
                  activeTab === 'dashboard'
                    ? 'text-violet-600 border-b-2 border-violet-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab('mymodels')}
                className={`pb-4 px-2 font-semibold transition-all ${
                  activeTab === 'mymodels'
                    ? 'text-violet-600 border-b-2 border-violet-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                My Models ({myHostedModels.length})
              </button>
              <button
                onClick={() => setActiveTab('incomplete')}
                className={`pb-4 px-2 font-semibold transition-all ${
                  activeTab === 'incomplete'
                    ? 'text-violet-600 border-b-2 border-violet-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Incomplete Models ({incompleteModels.length})
              </button>
            </div>
          </div>
        )}

        {/* Dashboard Content */}
        {!showAddForm && !showJoinForm && activeTab === 'dashboard' && myHostedModels.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-12"
          >
            {/* Magic Bento Dashboard */}
            <div className="flex justify-center">
              <MagicBento 
                textAutoHide={true}
                enableStars={true}
                enableSpotlight={true}
                enableBorderGlow={true}
                enableTilt={true}
                enableMagnetism={true}
                clickEffect={true}
                spotlightRadius={300}
                particleCount={12}
                glowColor="132, 0, 255"
                cardData={[
                  {
                    color: '#FFFFFF',
                    title: (() => {
                      // Count actual shards: if user hosts both host1 and host2, count as 2
                      return myHostedModels.reduce((total, model) => {
                        const isHost1 = model.host1.toLowerCase() === connectedAddress?.toLowerCase();
                        const isHost2 = model.host2?.toLowerCase() === connectedAddress?.toLowerCase();
                        return total + (isHost1 ? 1 : 0) + (isHost2 ? 1 : 0);
                      }, 0).toString();
                    })(),
                    description: 'Shard(s) currently active',
                    label: 'Total Shards Hosted',
                    onClick: () => setActiveTab('mymodels')
                  },
                  (() => {
                    // Find most hosted model
                    const modelCounts: { [key: string]: number } = {};
                    myHostedModels.forEach(m => {
                      modelCounts[m.modelName] = (modelCounts[m.modelName] || 0) + 1;
                    });
                    const mostHosted = Object.entries(modelCounts).sort((a, b) => b[1] - a[1])[0];
                    const mostHostedName = mostHosted ? mostHosted[0] : 'None';
                    
                    return {
                      color: '#FFFFFF',
                      title: mostHostedName,
                      description: 'Your most hosted model',
                      label: 'Most Hosted Shards',
                      icon: mostHostedName !== 'None' ? getModelIcon(mostHostedName) : undefined,
                      modelName: mostHostedName !== 'None' ? mostHostedName : undefined,
                      onClick: () => setActiveTab('mymodels')
                    };
                  })(),
                  {
                    color: '#FFFFFF',
                    title: (() => {
                      const totalTime = myHostedModels.reduce((sum, m) => {
                        const time1 = Number(m.registeredAtHost1) > 0 ? currentTime - Number(m.registeredAtHost1) : 0;
                        const time2 = Number(m.registeredAtHost2 || 0n) > 0 ? currentTime - Number(m.registeredAtHost2 || 0n) : 0;
                        return sum + time1 + time2;
                      }, 0);
                      return formatHostingTime(totalTime);
                    })(),
                    description: 'Total time hosting models',
                    label: 'Total Time Hosted',
                    onClick: () => setActiveTab('mymodels')
                  },
                  {
                    color: '#FFFFFF',
                    title: myHostedModels.reduce((sum, m) => sum + Number(m.usageCount), 0).toString(),
                    description: 'Total times models were used',
                    label: 'Usage Count',
                    onClick: () => {}
                  },
                  {
                    color: '#FFFFFF',
                    title: (() => {
                      // Show unclaimed earnings - number only (unit in label)
                      const unclaimedWei = unclaimedEarnings ? Number(unclaimedEarnings) : 0;
                      const unclaimedOG = unclaimedWei / 1e18; // Wei to 0G
                      const unclaimedGwei = unclaimedWei / 1e9; // Wei to Gwei
                      
                      // If less than 0.001 0G, show in Gwei (no decimals)
                      if (unclaimedOG < 0.001) {
                        return Math.floor(unclaimedGwei).toLocaleString();
                      }
                      return unclaimedOG.toFixed(5);
                    })(),
                    description: (() => {
                      // Show credit equivalent (0.001 0G = 200 credits)
                      const unclaimedWei = unclaimedEarnings ? Number(unclaimedEarnings) : 0;
                      const unclaimedOG = unclaimedWei / 1e18;
                      const credits = Math.floor(unclaimedOG * 200000); // 0.001 0G = 200 credits
                      return `${credits.toLocaleString()} credits`;
                    })(),
                    label: (() => {
                      // Add unit to label
                      const unclaimedWei = unclaimedEarnings ? Number(unclaimedEarnings) : 0;
                      const unclaimedOG = unclaimedWei / 1e18;
                      
                      if (unclaimedOG < 0.001) {
                        return 'Total Unclaimed Earnings (Gwei)';
                      }
                      return 'Total Unclaimed Earnings (0G)';
                    })(),
                    onClick: () => handleWithdrawAllClick()
                  },
                  {
                    color: '#FFFFFF',
                    title: '+',
                    description: 'Host a new model',
                    label: 'Add Model',
                    onClick: () => setShowAddForm(true)
                  }
                ]}
              />
            </div>
          </motion.div>
        )}

        {/* My Models Content */}
        {!showAddForm && !showJoinForm && activeTab === 'mymodels' && myHostedModels.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-12"
          >
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-gray-900">My Hosted Models</h2>
            </div>

            {isLoadingMyModels ? (
              <div className="flex items-center justify-center py-10">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-400"></div>
              </div>
             ) : (
               <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                 {myHostedModels.map((model) => {
                   // Trust the contract's isComplete value - it's updated correctly when stopping
                   const isComplete = model.isComplete;
                   const isHost1 = model.host1.toLowerCase() === connectedAddress?.toLowerCase();
                   const isHost2 = model.host2?.toLowerCase() === connectedAddress?.toLowerCase();
                   const isBothHosts = isHost1 && isHost2;
                  
                  return (
                    <motion.div
                      key={model.id}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="bg-white rounded-xl border border-gray-200 p-6 hover:border-violet-400 hover:shadow-md transition-all"
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <h3 className="font-bold text-lg text-gray-900">{model.modelName}</h3>
                          <p className="text-xs text-gray-500 mt-1">Model ID: #{model.id}</p>
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
                        {/* Shard(s) */}
                        {isBothHosts ? (
                          <>
                            <div className="flex items-start gap-2">
                              <span className="text-gray-500">Shard 1:</span>
                              <span className="text-gray-700 text-xs">Lower Layers (1-50)</span>
                            </div>
                            <div className="flex items-start gap-2">
                              <span className="text-gray-500">URL 1:</span>
                              <span className="font-mono text-gray-700 break-all text-xs">
                                {model.shardUrl1?.slice(0, 35)}...
                              </span>
                            </div>
                            <div className="flex items-start gap-2">
                              <span className="text-gray-500">Shard 2:</span>
                              <span className="text-gray-700 text-xs">Upper Layers (51-100)</span>
                            </div>
                            <div className="flex items-start gap-2">
                              <span className="text-gray-500">URL 2:</span>
                              <span className="font-mono text-gray-700 break-all text-xs">
                                {model.shardUrl2?.slice(0, 35)}...
                              </span>
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="flex items-start gap-2">
                              <span className="text-gray-500">Shard:</span>
                              <span className="text-gray-700 text-xs">
                                {isHost1 ? 'Lower Layers (1-50)' : 'Upper Layers (51-100)'}
                              </span>
                            </div>
                            <div className="flex items-start gap-2">
                              <span className="text-gray-500">Your URL:</span>
                              <span className="font-mono text-gray-700 break-all text-xs">
                                {isHost1 ? model.shardUrl1?.slice(0, 35) : model.shardUrl2?.slice(0, 35)}...
                              </span>
                            </div>
                          </>
                        )}
                        
                        <div className="flex items-start gap-2">
                          <span className="text-gray-500">Pool Balance:</span>
                          <span className="text-gray-700 text-xs">{formatPoolBalance(model.poolBalance)} <span className="text-[12px]">0G</span></span>
                        </div>
                        
                        {/* Hosting Time */}
                        {isBothHosts ? (
                          <>
                            <div className="flex items-start gap-2">
                              <span className="text-gray-500">Hosted (Shard 1):</span>
                              <span className="text-gray-700 text-xs">
                                {(() => {
                                  const startTime = Number(model.registeredAtHost1);
                                  if (startTime === 0) return 'Not started';
                                  return formatHostingTime(currentTime - startTime);
                                })()}
                              </span>
                            </div>
                            <div className="flex items-start gap-2">
                              <span className="text-gray-500">Hosted (Shard 2):</span>
                              <span className="text-gray-700 text-xs">
                                {(() => {
                                  const startTime = Number(model.registeredAtHost2 || 0n);
                                  if (startTime === 0) return 'Not started';
                                  return formatHostingTime(currentTime - startTime);
                                })()}
                              </span>
                            </div>
                          </>
                        ) : (
                          <div className="flex items-start gap-2">
                            <span className="text-gray-500">Hosting Time:</span>
                            <span className="text-gray-700 text-xs">
                              {(() => {
                                const startTime = isHost1 
                                  ? Number(model.registeredAtHost1)
                                  : Number(model.registeredAtHost2 || 0n);
                                if (startTime === 0) return 'Not started';
                                return formatHostingTime(currentTime - startTime);
                              })()}
                            </span>
                          </div>
                        )}
                        
                        <div className="flex items-start gap-2">
                          <span className="text-gray-500">Usage Count:</span>
                          <span className="text-gray-700 text-xs font-semibold">
                            {Number(model.usageCount)} times
                          </span>
                       </div>
                     </div>

                       <div className="space-y-2">
                         {isComplete ? (
                           <button
                             onClick={() => router.push(`/chat?modelId=${model.id}`)}
                             className="w-full py-2 px-4 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors font-medium text-sm flex items-center justify-center gap-2"
                           >
                             <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                             </svg>
                             Use in Chat
                           </button>
                         ) : isHost1 ? (
                           <button
                             onClick={() => {
                               setSelectedLLMId(model.id);
                               setSelectedModelName(model.modelName);
                               setExistingShardUrl(model.shardUrl1 || '');
                               setShowJoinForm(true);
                             }}
                             className="w-full py-2 px-4 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity font-medium text-sm flex items-center justify-center gap-2"
                           >
                             <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                             </svg>
                             Complete My Model
                           </button>
                         ) : null}
                         
                         {/* Stop Hosting Button */}
                         <button
                           onClick={() => handleStopLLMClick(model)}
                           className="w-full py-2 px-4 bg-red-50 text-red-600 border border-red-200 rounded-lg hover:bg-red-100 transition-colors font-medium text-sm flex items-center justify-center gap-2"
                         >
                           <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                           </svg>
                           Stop Hosting
                         </button>
                       </div>
                     </motion.div>
                  );
                })}
              </div>
            )}
          </motion.div>
        )}

        {/* My Models Empty State */}
        {!showAddForm && !showJoinForm && activeTab === 'mymodels' && myHostedModels.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gradient-to-br from-violet-50 to-purple-50 rounded-xl border-2 border-violet-300 p-8"
          >
            <div className="text-center max-w-2xl mx-auto">
              <div className="mb-4">
                <svg className="w-16 h-16 mx-auto text-violet-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v16m8-8H4" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Ready to Host a Model?</h2>
              <p className="text-gray-600 mb-6">
                Start hosting an AI model on the TeeTee network and earn rewards. Follow our step-by-step guide to configure your TEE environment.
              </p>
              <div className="flex items-center justify-center gap-4">
                <button
                  onClick={() => setShowAddForm(true)}
                  className="px-8 py-3 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-full hover:opacity-90 transition-opacity font-medium flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  Add Model
                </button>
                <button
                  onClick={() => setShowHostingGuide(true)}
                  className="px-8 py-3 bg-white text-gray-700 rounded-full hover:bg-gray-50 transition-colors font-medium border-2 border-gray-200"
                >
                  View Guide
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Dashboard Empty State */}
        {!showAddForm && !showJoinForm && activeTab === 'dashboard' && myHostedModels.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gradient-to-br from-violet-50 to-purple-50 rounded-xl border-2 border-violet-300 p-8"
          >
            <div className="text-center max-w-2xl mx-auto">
              <div className="mb-4">
                <svg className="w-16 h-16 mx-auto text-violet-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v16m8-8H4" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Ready to Host a Model?</h2>
              <p className="text-gray-600 mb-6">
                Start hosting an AI model on the TeeTee network and earn rewards. Follow our step-by-step guide to configure your TEE environment.
              </p>
              <div className="flex items-center justify-center gap-4">
                <button
                  onClick={() => setShowAddForm(true)}
                  className="px-8 py-3 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-full hover:opacity-90 transition-opacity font-medium flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  Add Model
                </button>
                <button
                  onClick={() => setShowHostingGuide(true)}
                  className="px-8 py-3 bg-white text-gray-700 rounded-full hover:bg-gray-50 transition-colors font-medium border-2 border-gray-200"
                >
                  View Guide
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Incomplete Models Content */}
        {!showAddForm && !showJoinForm && activeTab === 'incomplete' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-12"
          >
            <div className="mb-6">
              <h2 className="text-2xl font-bold text-gray-900">Incomplete Models</h2>
              <p className="text-gray-600 mt-1">Models waiting for a host to complete. Join to earn rewards!</p>
            </div>

            {isLoadingIncomplete ? (
              <div className="flex items-center justify-center py-10">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-400"></div>
              </div>
            ) : incompleteModels.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20">
                <div className="text-center max-w-md">
                  <h3 className="text-xl font-bold text-gray-900 mb-2">No Incomplete Models</h3>
                  <p className="text-gray-600 mb-6">
                    All models on the network are complete! Check back later or add your own model.
                  </p>
                  <button
                    onClick={() => setShowAddForm(true)}
                    className="px-6 py-3 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity font-medium inline-flex items-center gap-2"
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    Add Your Model
                  </button>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {incompleteModels.map((model) => {
                  // Determine which host is missing
                  const host1Empty = !model.host1 || model.host1 === '0x0000000000000000000000000000000000000000';
                  const host2Empty = !model.host2 || model.host2 === '0x0000000000000000000000000000000000000000';
                  
                  // Determine available shard and existing host info
                  const needsHost1 = host1Empty && !host2Empty;
                  const needsHost2 = !host1Empty && host2Empty;
                  const existingHost = needsHost1 ? model.host2 : model.host1;
                  const existingShardUrl = needsHost1 ? model.shardUrl2 : model.shardUrl1;
                  const availableShard = needsHost1 ? 'Lower Layers (1-50)' : 'Upper Layers (51-100)';
                  const hostLabel = needsHost1 ? 'Host 2' : 'Host 1';
                  const joinAsHost = needsHost1 ? '1' : '2';
                  
                  return (
                    <motion.div
                      key={model.id}
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="bg-white rounded-xl border border-gray-200 p-6 hover:border-violet-400 hover:shadow-md transition-all"
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <h3 className="font-bold text-lg text-gray-900">{model.modelName}</h3>
                          <p className="text-xs text-gray-500 mt-1">Model ID: #{model.id}</p>
                        </div>
                        <div className="px-2.5 py-1 rounded-full text-xs font-medium flex-shrink-0 bg-yellow-100 text-yellow-700">
                          ⏳ Needs Host {joinAsHost}
                        </div>
                      </div>

                      <div className="space-y-2 text-xs mb-4">
                        <div className="flex items-start gap-2">
                          <span className="text-gray-500">{hostLabel}:</span>
                          <span className="font-mono text-gray-700 text-xs">{existingHost?.slice(0, 10)}...</span>
                        </div>
                        <div className="flex items-start gap-2">
                          <span className="text-gray-500">Available Shard:</span>
                          <span className="text-gray-700 font-semibold text-xs">{availableShard}</span>
                        </div>
                        <div className="flex items-start gap-2">
                          <span className="text-gray-500">Existing URL:</span>
                          <span className="font-mono text-gray-700 break-all text-xs">{existingShardUrl?.slice(0, 40)}...</span>
                        </div>
                        <div className="flex items-start gap-2">
                          <span className="text-gray-500">Pool Balance:</span>
                          <span className="text-gray-700 font-semibold text-xs">{formatPoolBalance(model.poolBalance)} <span className="text-[10px]">0G</span></span>
                        </div>
                      </div>

                      <button
                        onClick={() => {
                          setSelectedLLMId(model.id);
                          setSelectedModelName(model.modelName);
                          setExistingShardUrl(existingShardUrl || '');
                          setShowJoinForm(true);
                        }}
                        className="w-full py-2 px-4 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity font-medium text-sm flex items-center justify-center gap-2"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        Join as Host {joinAsHost}
                      </button>
                    </motion.div>
                  );
                })}
              </div>
            )}
          </motion.div>
        )}

        {/* Join as Second Host Form */}
        {showJoinForm && selectedLLMId !== null && (
          <AddModelForm
            availableModels={[selectedModelName]}
            availableShards={AVAILABLE_SHARDS}
            connectedAddress={connectedAddress}
            onSubmit={handleJoinAsSecondHost}
            onCancel={resetJoinForm}
            isWriting={isWriting}
            isConfirming={isConfirming}
            isMinting={false}
            isAuthorizing={false}
            writeError={writeError}
            availableShard="shard2"
            existingShardUrl={existingShardUrl}
          />
        )}

        {/* Add Model Form */}
        {showAddForm && !isLoadingMyModels && !showJoinForm && (
          <div className="mb-12">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Add New Model</h2>
                <p className="text-gray-600 mt-1">Configure your model hosting setup</p>
              </div>
              <button
                onClick={resetForm}
                className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                title="Cancel"
              >
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <AddModelForm
              availableModels={AVAILABLE_MODELS}
              availableShards={AVAILABLE_SHARDS}
              connectedAddress={connectedAddress}
              onSubmit={handleAddModel}
              onCancel={resetForm}
              isWriting={isWriting}
              isConfirming={isConfirming}
              isMinting={false}
              isAuthorizing={false}
              writeError={writeError}
            />
          </div>
        )}

        {/* Hosting Guide Modal */}
        {showHostingGuide && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1100] p-4"
            onClick={() => setShowHostingGuide(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl scrollbar-hide"
              style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="bg-gradient-to-r from-violet-500 to-purple-500 px-6 py-4 text-white sticky top-0 z-10">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold">How to Host a Model on Phala Cloud</h2>
                    <p className="text-violet-100 text-sm mt-1">Follow these steps to set up your TEE environment</p>
                  </div>
                  <button
                    onClick={() => setShowHostingGuide(false)}
                    className="text-white hover:text-violet-100 transition-colors"
                  >
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>

              {/* Content */}
              <div className="p-6 space-y-6">
                {/* Docker Compose Copy Section */}
                <div className="bg-gray-100 rounded-lg p-4 border border-gray-200">
                  <div className="flex items-center justify-between">
                    <h3 className="text-base font-semibold text-gray-900 flex items-center gap-2">
                      <svg className="w-5 h-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Copy this docker compose file <span className="text-red-500 text-xs font-italic mt-1">(will be used in next step)</span>
                    </h3>
                    <button
                      onClick={handleCopyDockerCompose}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                        isCopied
                          ? 'bg-green-100 text-green-700'
                          : 'bg-violet-600 text-white hover:bg-violet-700'
                      }`}
                    >
                      {isCopied ? (
                        <>
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          Copied!
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                          Copy
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Slideshow Section */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <svg className="w-5 h-5 text-violet-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    Step-by-Step Guide
                  </h3>
                  
                  {/* Slideshow */}
                  <div className="relative bg-gray-100 rounded-lg overflow-hidden" style={{ aspectRatio: '16/9' }}>
                    {/* Image */}
                    <div className="w-full h-full flex items-center justify-center">
                      <img
                        src={GUIDE_SLIDES[currentSlide]}
                        alt={`Guide step ${currentSlide + 1}`}
                        className="max-w-full max-h-full object-contain"
                        onError={(e) => {
                          e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450"%3E%3Crect fill="%23e5e7eb" width="800" height="450"/%3E%3Ctext x="50%25" y="50%25" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-size="24" fill="%236b7280"%3ESlide ' + (currentSlide + 1) + '%3C/text%3E%3C/svg%3E';
                        }}
                      />
                    </div>

                    {/* Previous Button */}
                    <button
                      onClick={handlePrevSlide}
                      className="absolute left-4 top-1/2 -translate-y-1/2 bg-white bg-opacity-90 hover:bg-opacity-100 text-gray-800 p-3 rounded-full shadow-lg transition-all"
                    >
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                      </svg>
                    </button>

                    {/* Next Button */}
                    <button
                      onClick={handleNextSlide}
                      className="absolute right-4 top-1/2 -translate-y-1/2 bg-white bg-opacity-90 hover:bg-opacity-100 text-gray-800 p-3 rounded-full shadow-lg transition-all"
                    >
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </button>

                    {/* Slide Counter */}
                    <div className="absolute top-4 right-4 bg-gray-800 bg-opacity-70 text-white px-3 py-1 rounded-full text-sm font-medium">
                      {currentSlide + 1} / {GUIDE_SLIDES.length}
                    </div>
                  </div>
                </div>

                {/* Footer with helpful links */}
                <div className="bg-violet-50 rounded-lg p-4 border border-violet-200">
                  <p className="text-sm text-violet-900 font-medium mb-2">Need more help?</p>
                  <div className="flex flex-wrap gap-3">
                    <a
                      href="https://docs.phala.network"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 text-sm text-violet-700 hover:text-violet-900 transition-colors"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                      </svg>
                      Documentation
                    </a>
                    <a
                      href="https://discord.gg/phala"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 text-sm text-violet-700 hover:text-violet-900 transition-colors"
                    >
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M20.317 4.37a19.791 19.791 0 00-4.885-1.515.074.074 0 00-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 00-5.487 0 12.64 12.64 0 00-.617-1.25.077.077 0 00-.079-.037A19.736 19.736 0 003.677 4.37a.07.07 0 00-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 00.031.057 19.9 19.9 0 005.993 3.03.078.078 0 00.084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 00-.041-.106 13.107 13.107 0 01-1.872-.892.077.077 0 01-.008-.128 10.2 10.2 0 00.372-.292.074.074 0 01.077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 01.078.01c.12.098.246.198.373.292a.077.077 0 01-.006.127 12.299 12.299 0 01-1.873.892.077.077 0 00-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 00.084.028 19.839 19.839 0 006.002-3.03.077.077 0 00.032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 00-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
                      </svg>
                      Join Discord
                    </a>
                  </div>
                </div>
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
            onClick={(e) => {
              // Prevent closing modal by clicking backdrop during minting/authorizing
              if (!isMintingAndAuthorizing) {
                e.stopPropagation();
              }
            }}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-white rounded-2xl max-w-lg w-full max-h-[85vh] overflow-y-auto shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header with gradient */}
              <div className="bg-gradient-to-r from-violet-400 to-purple-300 px-6 py-4 text-white sticky top-0 z-10">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <h2 className="text-xl font-bold">{isSecondHost ? 'Joined Successfully!' : 'Registration Successful!'}</h2>
                  </div>
                  {!hasStartedClaiming && (
                    <button
                      onClick={() => {
                        setShowClaimINFTModal(false);
                        setRegisteredModelName('');
                        setIsSecondHost(false);
                        setHasStartedClaiming(false);
                        setIsMintAuthConfirmed(false);
                        setIsMintingAndAuthorizing(false);
                        setMintAuthError(null);
                        setMintedTokenId(null);
                      }}
                      className="text-white hover:text-violet-100 transition-colors"
                    >
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                  {hasStartedClaiming && (
                    <div className="text-white opacity-50 cursor-not-allowed" title="Please wait for the process to complete">
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                      </svg>
                    </div>
                  )}
                </div>
                <p className="text-violet-100 text-sm">
                  {isSecondHost ? 'You are now the second host for this model' : 'Your model is now registered on the network'}
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
                      <h3 className="font-semibold text-green-900 mb-1">{isSecondHost ? 'Joined as Second Host' : 'Model Registered'}</h3>
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

                {/* Progress Steps - Single unified step */}
                <div className="bg-gradient-to-r from-violet-50 to-purple-50 border border-violet-200 rounded-lg p-4 mb-4">
                  <p className="text-xs font-semibold text-gray-700 mb-3">
                    {!hasStartedClaiming ? 'What happens when you claim:' : 'Progress:'}
                  </p>
                  
                  {/* Single Step: Complete Process */}
                  <div className="flex items-start gap-3">
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all ${
                      isMintAuthConfirmed
                        ? 'bg-green-500 border-green-500' 
                        : hasStartedClaiming
                        ? 'bg-violet-500 border-violet-500 animate-pulse' 
                        : 'bg-white border-violet-300'
                    }`}>
                      {isMintAuthConfirmed ? (
                        <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : hasStartedClaiming ? (
                        <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
                      ) : (
                        <svg className="w-5 h-5 text-violet-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                      )}
                    </div>
                    <div className="flex-1">
                      <p className={`text-base font-semibold mb-1 ${
                        isMintAuthConfirmed ? 'text-green-700' : hasStartedClaiming ? 'text-violet-700' : 'text-gray-800'
                      }`}>
                        {isMintAuthConfirmed ? '✓ INFT Claimed Successfully!' : hasStartedClaiming ? 'Claiming Your INFT...' : 'Mint & Authorize INFT Token'}
                      </p>
                      <p className="text-xs text-gray-600 leading-relaxed">
                        {isMintAuthConfirmed ? (
                          <>
                            Your INFT (#{mintedTokenId}) has been minted and authorized. You can now use AI inference!
                          </>
                        ) : hasStartedClaiming ? (
                          <>
                            {isMintingAndAuthorizing ? '⏳ Minting INFT and authorizing access...' : '⏳ Processing...'}
                          </>
                        ) : (
                          <>
                            • Backend mints your Intelligent NFT token<br />
                            • Transaction confirmed on blockchain<br />
                            • Automatically authorizes your access
                          </>
                        )}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Buttons */}
                {!isMintAuthConfirmed && (
                  <div className="flex gap-2">
                    <button
                      onClick={() => {
                        setShowClaimINFTModal(false);
                        setRegisteredModelName('');
                        setHasStartedClaiming(false);
                        setIsMintAuthConfirmed(false);
                        setIsMintingAndAuthorizing(false);
                        setMintAuthError(null);
                        setMintedTokenId(null);
                      }}
                      disabled={hasStartedClaiming}
                      className="flex-1 px-4 py-2.5 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
                    >
                      Claim Later
                    </button>
                    <button
                      onClick={handleClaimINFT}
                      disabled={hasStartedClaiming}
                      className="flex-1 px-4 py-2.5 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2 text-sm"
                    >
                      {hasStartedClaiming ? (
                        <>
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                          Processing...
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
                )}

                {/* Success state after authorization */}
                {isMintAuthConfirmed && (
                  <>
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="p-4 bg-green-50 border-2 border-green-300 rounded-lg"
                    >
                      <div className="flex items-center gap-2 text-green-800 mb-2">
                        <svg className="w-6 h-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span className="font-bold text-base">All Done! 🎉</span>
                      </div>
                      <p className="text-sm text-green-700 mb-3">
                        Your INFT #{mintedTokenId} has been claimed and authorized successfully! You can now use AI inference in the Chat.
                      </p>
                      <button
                        onClick={() => {
                          setShowClaimINFTModal(false);
                          setRegisteredModelName('');
                          setIsSecondHost(false);
                          setHasStartedClaiming(false);
                          setIsMintAuthConfirmed(false);
                          setIsMintingAndAuthorizing(false);
                          setMintAuthError(null);
                          setMintedTokenId(null);
                        }}
                        className="w-full px-4 py-2.5 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium text-sm flex items-center justify-center gap-2"
                      >
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        Close & Continue
                      </button>
                    </motion.div>
                  </>
                )}

                {/* Error state if mint/authorization fails */}
                {mintAuthError && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 bg-red-50 border-2 border-red-300 rounded-lg"
                  >
                    <div className="flex items-start gap-2 text-red-800 mb-3">
                      <svg className="w-5 h-5 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <div className="flex-1">
                        <span className="font-semibold text-sm">Claim Failed</span>
                        <p className="text-xs text-red-700 mt-1">
                          {mintAuthError}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        setShowClaimINFTModal(false);
                        setRegisteredModelName('');
                        setIsSecondHost(false);
                        setHasStartedClaiming(false);
                        setIsMintAuthConfirmed(false);
                        setIsMintingAndAuthorizing(false);
                        setMintAuthError(null);
                        setMintedTokenId(null);
                      }}
                      className="w-full px-4 py-2.5 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium text-sm flex items-center justify-center gap-2"
                    >
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                      Close
                    </button>
                  </motion.div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}

        {/* Stop LLM Modal */}
        {showStopModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1100] p-4"
            onClick={(e) => {
              if (!isStopping && !isStopConfirming) {
                setShowStopModal(false);
                setStopModelId(null);
                setStopModelName('');
                setStopHostNumber(null);
                setIsBothHostsForStop(false);
              }
            }}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-white rounded-2xl max-w-md w-full shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="bg-gradient-to-r from-red-500 to-red-400 px-6 py-4 text-white">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <h2 className="text-xl font-bold">Stop Hosting</h2>
                  </div>
                  {!isStopping && !isStopConfirming && (
                    <button
                      onClick={() => {
                        setShowStopModal(false);
                        setStopModelId(null);
                        setStopModelName('');
                        setStopHostNumber(null);
                        setIsBothHostsForStop(false);
                      }}
                      className="text-white hover:text-red-100 transition-colors"
                    >
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>

              {/* Content */}
              <div className="p-6">
                <div className="mb-4">
                  <p className="text-gray-700 mb-2">
                    You are about to stop hosting <strong>{stopModelName}</strong>.
                  </p>
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
                    <div className="flex items-start gap-2">
                      <svg className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                      <div className="text-sm text-yellow-800">
                        <p className="font-semibold mb-1">Important:</p>
                        <ul className="list-disc list-inside space-y-1">
                          <li>Any pending rewards will be paid out</li>
                          <li>Your hosting slot will be cleared</li>
                          <li>The model may become incomplete if you're the only host</li>
                          <li><strong>Your INFT access will be revoked - you'll need to host again to regain access</strong></li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>

                {/* If user hosts both shards, ask which one to stop */}
                {isBothHostsForStop && (
                  <div className="mb-4">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Which shard do you want to stop hosting?
                    </label>
                    <div className="space-y-2">
                      <button
                        onClick={() => setStopHostNumber(1)}
                        className={`w-full p-3 rounded-lg border-2 transition-all text-left ${
                          stopHostNumber === 1
                            ? 'border-red-500 bg-red-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                      >
                        <div className="font-semibold text-gray-900">Shard 1</div>
                        <div className="text-xs text-gray-600">Lower Layers (1-50)</div>
                      </button>
                      <button
                        onClick={() => setStopHostNumber(2)}
                        className={`w-full p-3 rounded-lg border-2 transition-all text-left ${
                          stopHostNumber === 2
                            ? 'border-red-500 bg-red-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                      >
                        <div className="font-semibold text-gray-900">Shard 2</div>
                        <div className="text-xs text-gray-600">Upper Layers (51-100)</div>
                      </button>
                    </div>
                  </div>
                )}

                {/* If user only hosts one shard, show which one */}
                {!isBothHostsForStop && stopHostNumber && (
                  <div className="mb-4">
                    <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                      <div className="text-sm text-gray-700">
                        <span className="font-semibold">Stopping: </span>
                        Shard {stopHostNumber} - {stopHostNumber === 1 ? 'Lower Layers (1-50)' : 'Upper Layers (51-100)'}
                      </div>
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      setShowStopModal(false);
                      setStopModelId(null);
                      setStopModelName('');
                      setStopHostNumber(null);
                      setIsBothHostsForStop(false);
                    }}
                    disabled={isStopping || isStopConfirming}
                    className="flex-1 px-4 py-2.5 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleStopLLMConfirm}
                    disabled={isStopping || isStopConfirming || stopHostNumber === null}
                    className="flex-1 px-4 py-2.5 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm flex items-center justify-center gap-2"
                  >
                    {isStopping || isStopConfirming ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        {isStopping ? 'Confirming...' : 'Processing...'}
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                        Stop Hosting
                      </>
                    )}
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}

        {/* Withdraw Earnings Modal */}
        {showWithdrawModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1100] p-4"
            onClick={(e) => {
              // Prevent closing modal by clicking backdrop during withdrawal
              if (!isWithdrawWriting && !isWithdrawConfirming) {
                e.stopPropagation();
              }
            }}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-white rounded-2xl max-w-lg w-full max-h-[85vh] overflow-y-auto shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header with gradient */}
              <div className="bg-gradient-to-r from-violet-400 to-purple-300 px-6 py-4 text-white sticky top-0 z-10">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <h2 className="text-xl font-bold">Withdraw Earnings</h2>
                  </div>
                  {!isWithdrawWriting && !isWithdrawConfirming && !isWithdrawConfirmed && (
                    <button
                      onClick={() => {
                        setShowWithdrawModal(false);
                        setTotalEarnings('0');
                      }}
                      className="text-white hover:text-violet-100 transition-colors"
                    >
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                  {(isWithdrawWriting || isWithdrawConfirming) && (
                    <div className="text-white opacity-50 cursor-not-allowed" title="Please wait for the transaction to complete">
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                      </svg>
                    </div>
                  )}
                </div>
                <p className="text-violet-100 text-sm">
                  Claim your earnings from hosting models
                </p>
              </div>

              {/* Content */}
              <div className="p-6">
                {/* Earnings Display */}
                <div className="mb-4">
                  <div className="flex items-center justify-center p-6 bg-gradient-to-r from-violet-50 to-purple-50 border border-violet-200 rounded-lg">
                    <div className="text-center">
                      <p className="text-sm text-gray-600 mb-1">Total Available Earnings</p>
                      <p className="text-4xl font-bold text-violet-600">
                        {(() => {
                          const ogAmount = parseFloat(totalEarnings);
                          const gwei = ogAmount * 1e9; // 0G to Gwei
                          
                          // If less than 0.001 0G, show in Gwei (no decimals)
                          if (ogAmount < 0.001 && ogAmount > 0) {
                            return `${Math.floor(gwei).toLocaleString()} Gwei`;
                          }
                          return `${totalEarnings} 0G`;
                        })()}
                      </p>
                      <p className="text-xs text-gray-500 mt-2">
                        {(() => {
                          const credits = Math.floor(parseFloat(totalEarnings) * 200000);
                          return `≈ ${credits.toLocaleString()} credits`;
                        })()}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Lifetime Earnings Display */}
                <div className="mb-4">
                  <div className="flex items-center justify-between p-3 bg-gray-50 border border-gray-200 rounded-lg">
                    <div className="flex-1">
                      <p className="text-xs text-gray-500">Lifetime Earnings (All Time)</p>
                      <p className="text-lg font-semibold text-gray-800">
                        {(() => {
                          // Lifetime = already withdrawn + unclaimed (correct Wei conversion)
                          const withdrawnWei = lifetimeEarnings ? Number(lifetimeEarnings) : 0;
                          const unclaimedWei = unclaimedEarnings ? Number(unclaimedEarnings) : 0;
                          const withdrawnOG = withdrawnWei / 1e18;
                          const unclaimedOG = unclaimedWei / 1e18;
                          const totalLifetimeOG = withdrawnOG + unclaimedOG;
                          const totalLifetimeGwei = (withdrawnWei + unclaimedWei) / 1e9;
                          
                          // If less than 0.001 0G, show in Gwei (no decimals)
                          if (totalLifetimeOG < 0.001 && totalLifetimeOG > 0) {
                            return `${Math.floor(totalLifetimeGwei).toLocaleString()} Gwei`;
                          }
                          return `${totalLifetimeOG.toFixed(5)} 0G`;
                        })()}
                      </p>
                      <p className="text-xs text-gray-400 mt-1">
                        {(() => {
                          const withdrawnWei = lifetimeEarnings ? Number(lifetimeEarnings) : 0;
                          const unclaimedWei = unclaimedEarnings ? Number(unclaimedEarnings) : 0;
                          const withdrawnOG = withdrawnWei / 1e18;
                          const unclaimedOG = unclaimedWei / 1e18;
                          const totalLifetimeOG = withdrawnOG + unclaimedOG;
                          const credits = Math.floor(totalLifetimeOG * 200000);
                          return `≈ ${credits.toLocaleString()} credits`;
                        })()}
                      </p>
                    </div>
                    <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                </div>

                {/* Info */}
                <div className="mb-4">
                  <div className="flex items-start gap-2 p-3 bg-violet-50 border border-violet-200 rounded-lg">
                    <svg className="w-5 h-5 text-violet-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div>
                      <h3 className="font-semibold text-violet-900 mb-1">About Withdrawals</h3>
                      <p className="text-sm text-violet-700">
                        This will withdraw all your unclaimed earnings from hosting models in a single transaction. The funds will be sent to your connected wallet.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Progress Steps */}
                {!isWithdrawConfirmed && (
                  <div className="bg-gradient-to-r from-violet-50 to-purple-50 border border-violet-200 rounded-lg p-4 mb-4">
                    <p className="text-xs font-semibold text-gray-700 mb-3">
                      {!isWithdrawWriting && !isWithdrawConfirming ? 'What happens when you withdraw:' : 'Progress:'}
                    </p>
                    
                    <div className="flex items-start gap-3">
                      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all ${
                        isWithdrawWriting || isWithdrawConfirming
                          ? 'bg-violet-500 border-violet-500 animate-pulse' 
                          : 'bg-white border-violet-300'
                      }`}>
                        {(isWithdrawWriting || isWithdrawConfirming) ? (
                          <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
                        ) : (
                          <svg className="w-5 h-5 text-violet-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        )}
                      </div>
                      <div className="flex-1">
                        <p className={`text-base font-semibold mb-1 ${
                          (isWithdrawWriting || isWithdrawConfirming) ? 'text-violet-700' : 'text-gray-800'
                        }`}>
                          {isWithdrawConfirming ? 'Confirming Transaction...' : isWithdrawWriting ? 'Waiting for Signature...' : 'Sign Transaction to Withdraw'}
                        </p>
                        <p className="text-xs text-gray-600 leading-relaxed">
                          {(isWithdrawWriting || isWithdrawConfirming) ? (
                            <>⏳ Processing withdrawal from all hosted models...</>
                          ) : (
                            <>
                              • You'll be prompted to sign the transaction<br />
                              • All earnings will be withdrawn at once<br />
                              • Funds will be sent to your wallet
                            </>
                          )}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Buttons */}
                {!isWithdrawConfirmed && (
                  <>
                    {/* Show message if no unclaimed earnings */}
                    {parseFloat(totalEarnings) === 0 && (
                      <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                        <div className="flex items-start gap-2">
                          <svg className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <p className="text-sm text-yellow-800">
                            No unclaimed earnings available to withdraw at this time. Keep hosting to earn more!
                          </p>
                        </div>
                      </div>
                    )}
                    
                    <div className="flex gap-2">
                      <button
                        onClick={() => {
                          setShowWithdrawModal(false);
                          setTotalEarnings('0');
                        }}
                        disabled={isWithdrawWriting || isWithdrawConfirming}
                        className="flex-1 px-4 py-2.5 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium text-sm"
                      >
                        Close
                      </button>
                      <button
                        onClick={handleWithdrawConfirm}
                        disabled={isWithdrawWriting || isWithdrawConfirming || parseFloat(totalEarnings) === 0}
                        className="flex-1 px-4 py-2.5 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2 text-sm"
                        title={parseFloat(totalEarnings) === 0 ? 'No unclaimed earnings to withdraw' : ''}
                      >
                        {(isWithdrawWriting || isWithdrawConfirming) ? (
                          <>
                            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                            {isWithdrawWriting ? 'Signing...' : 'Confirming...'}
                          </>
                        ) : (
                          <>
                            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            Withdraw Now
                          </>
                        )}
                      </button>
                    </div>
                  </>
                )}

                {/* Success state after withdrawal */}
                {isWithdrawConfirmed && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-4 bg-green-50 border-2 border-green-300 rounded-lg"
                  >
                    <div className="flex items-center gap-2 text-green-800 mb-2">
                      <svg className="w-6 h-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <span className="font-bold text-base">Withdrawal Successful! 🎉</span>
                    </div>
                    <p className="text-sm text-green-700 mb-3">
                      Your unclaimed earnings of {totalEarnings} 0G have been successfully withdrawn to your wallet!
                    </p>
                    <p className="text-xs text-gray-600">
                      This window will close automatically...
                    </p>
                  </motion.div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}

      </div>
    </div>
  );
};

export default DashboardPage;

