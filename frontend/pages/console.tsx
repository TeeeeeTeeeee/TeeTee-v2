"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Navbar } from '../components/Navbar';
import { AddModelForm } from '../components/AddModelForm';
import { useRegisterLLM } from '../lib/contracts/creditUse/writes/useRegisterLLM';
import { useGetTotalLLMs } from '../lib/contracts/creditUse/reads/useGetTotalLLMs';
import { useAccount } from 'wagmi';
import { useMintINFT, useAuthorizeINFT } from '../hooks/useINFT';
import { useRouter } from 'next/router';

interface AddModelFormData {
  modelName: string;
  walletAddress: string;
  shardSelection: string;
  shardUrl: string;
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

const AVAILABLE_SHARDS = [
  { id: 'shard-1', name: 'Shard 1', region: 'US-East', capacity: '75%' },
  { id: 'shard-2', name: 'Shard 2', region: 'US-West', capacity: '60%' },
  { id: 'shard-3', name: 'Shard 3', region: 'EU-Central', capacity: '45%' },
  { id: 'shard-4', name: 'Shard 4', region: 'Asia-Pacific', capacity: '30%' },
  { id: 'shard-5', name: 'Shard 5', region: 'US-Central', capacity: '90%' },
  { id: 'shard-6', name: 'Shard 6', region: 'EU-West', capacity: '55%' }
];

const DashboardPage = () => {
  const router = useRouter();
  const [showAddForm, setShowAddForm] = useState(false);
  const [showHostingGuide, setShowHostingGuide] = useState(false);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isCopied, setIsCopied] = useState(false);
  const [showClaimINFTModal, setShowClaimINFTModal] = useState(false);
  const [registeredModelName, setRegisteredModelName] = useState<string>('');

  // Smart contract hooks
  const { registerLLM, txHash, isWriting, writeError, resetWrite, isConfirming, isConfirmed } = useRegisterLLM();
  const { totalLLMs, refetch: refetchTotal } = useGetTotalLLMs();
  const { address: connectedAddress } = useAccount();
  
  // INFT hooks for minting and authorization
  const { mint: mintINFT, isPending: isMinting, isConfirmed: isMintConfirmed } = useMintINFT();
  const { authorize: authorizeINFT, isPending: isAuthorizing, isConfirmed: isAuthConfirmed } = useAuthorizeINFT();

  // Reset form state
  const resetForm = () => {
    setShowAddForm(false);
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
      console.error('Failed to register model:', error);
    }
  };

  // Effect to handle successful model registration - show claim modal
  React.useEffect(() => {
    if (isConfirmed && connectedAddress && registeredModelName) {
      console.log('Model registered successfully! Showing claim INFT modal...');
      
      // Show claim modal
      setShowClaimINFTModal(true);
      
      // Reset the registration form
      resetForm();
      
      // Refetch data
      refetchTotal();
    }
  }, [isConfirmed, connectedAddress, registeredModelName]);
  
  // Effect to handle successful INFT mint - then authorize the user
  React.useEffect(() => {
    const handleMintSuccess = async () => {
      if (isMintConfirmed && connectedAddress) {
        console.log('INFT minted, authorizing user...');
        
        try {
          const tokenId = 1;
          await authorizeINFT(tokenId, connectedAddress);
        } catch (error) {
          console.error('Failed to authorize user:', error);
        }
      }
    };
    
    handleMintSuccess();
  }, [isMintConfirmed, connectedAddress]);
  
  // Effect to auto-close modal after successful authorization
  React.useEffect(() => {
    if (isAuthConfirmed) {
      console.log('Authorization confirmed, closing modal in 2 seconds...');
      const timer = setTimeout(() => {
        setShowClaimINFTModal(false);
        setRegisteredModelName('');
      }, 2000);
      
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
      const encryptedURI = '0g://storage/model-data-' + Date.now();
      const metadataHash = '0x' + Array(64).fill('0').join('');
      
      const mintSuccess = await mintINFT(connectedAddress, encryptedURI, metadataHash);
      
      if (mintSuccess) {
        console.log('INFT claim initiated...');
      }
    } catch (error) {
      console.error('Failed to claim INFT:', error);
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
            
            {/* Info Button */}
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

        {/* Add Model Section */}
        <div className="mb-12">
          {!showAddForm ? (
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
          ) : (
            <div>
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
                isMinting={isMinting}
                isAuthorizing={isAuthorizing}
                writeError={writeError}
              />
            </div>
          )}
        </div>

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
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="bg-white rounded-2xl max-w-lg w-full max-h-[85vh] overflow-y-auto shadow-2xl"
            >
              {/* Header with gradient */}
              <div className="bg-gradient-to-r from-violet-400 to-purple-300 px-6 py-4 text-white sticky top-0 z-10">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <h2 className="text-xl font-bold">Registration Successful!</h2>
                  </div>
                  <button
                    onClick={() => {
                      setShowClaimINFTModal(false);
                      setRegisteredModelName('');
                    }}
                    disabled={isMinting || isAuthorizing}
                    className="text-white hover:text-violet-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <p className="text-violet-100 text-sm">Your model is now registered on the network</p>
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
                      <h3 className="font-semibold text-green-900 mb-1">Model Registered</h3>
                      <p className="text-sm text-green-700">
                        <strong>{registeredModelName}</strong> has been successfully registered as a hosting slot.
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
                    As a model hoster, you're eligible for an <strong>Intelligent NFT (INFT)</strong> token. This token grants you:
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
                  <p className="text-xs font-semibold text-gray-700 mb-1.5">Next Steps:</p>
                  <ol className="text-xs text-gray-600 space-y-0.5 ml-4 list-decimal">
                    <li>Click "Claim My INFT" to mint your token</li>
                    <li>Approve the transaction in your wallet</li>
                    <li>You'll be automatically authorized</li>
                    <li>Start using AI inference in Chat!</li>
                  </ol>
                </div>

                {/* Buttons */}
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      setShowClaimINFTModal(false);
                      setRegisteredModelName('');
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
      </div>
    </div>
  );
};

export default DashboardPage;

