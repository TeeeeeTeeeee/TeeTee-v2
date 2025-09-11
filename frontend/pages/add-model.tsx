"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { Navbar } from '../components/Navbar';

interface AddModelForm {
  modelName: string;
  shard: string;
  walletAddress: string;
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

const AddModelPage = () => {
  const router = useRouter();
  const [formData, setFormData] = useState<AddModelForm>({
    modelName: '',
    shard: '',
    walletAddress: ''
  });

  // Handle form submission
  const handleAddModel = () => {
    if (!formData.modelName || !formData.shard || !formData.walletAddress) return;

    const selectedShard = AVAILABLE_SHARDS.find(s => s.id === formData.shard);
    
    const newModel = {
      id: Date.now(),
      name: formData.modelName,
      shard: selectedShard?.name || formData.shard,
      walletAddress: formData.walletAddress,
      status: 'hosting',
      dateAdded: new Date()
    };

    // Save to localStorage
    const savedModels = localStorage.getItem('teetee-models');
    const models = savedModels ? JSON.parse(savedModels) : [];
    const updatedModels = [newModel, ...models];
    localStorage.setItem('teetee-models', JSON.stringify(updatedModels));

    // Navigate back to models page
    router.push('/models');
  };

  return (
    <div className="min-h-screen bg-gradient-to-l from-violet-200/20 to-white font-inter">
      {/* Navbar */}
      <Navbar />

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-6 py-8 mt-20">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-4">
            <button
              onClick={() => router.back()}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <svg className="w-5 h-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </button>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Add New Model</h1>
              <p className="text-gray-600 mt-1">Complete all fields to add your model to the network</p>
            </div>
          </div>
        </div>

        {/* Add Model Form */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden"
        >
          {/* All Steps Content */}
          <div className="p-8 space-y-8">
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
                  Select Shard
                </h3>
                <p className="text-sm text-gray-600 mb-4">Choose where to host your model</p>
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
                  Wallet Address
                </h3>
                <p className="text-sm text-gray-600 mb-4">Enter the wallet address for hosting rewards</p>
              </div>
              
              <div className="max-w-2xl">
                <label htmlFor="walletAddress" className="block text-sm font-medium text-gray-700 mb-2">
                  Wallet Address *
                </label>
                <input
                  id="walletAddress"
                  type="text"
                  value={formData.walletAddress}
                  onChange={(e) => setFormData({ ...formData, walletAddress: e.target.value })}
                  placeholder="0x1234567890abcdef..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent outline-none"
                />
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
                    <span className="font-medium">Shard:</span> 
                    <span className="text-violet-700">{AVAILABLE_SHARDS.find(s => s.id === formData.shard)?.name}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Region:</span> 
                    <span className="text-violet-700">{AVAILABLE_SHARDS.find(s => s.id === formData.shard)?.region}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Wallet:</span> 
                    <span className="text-violet-700 font-mono text-xs">{formData.walletAddress}</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Form Footer */}
          <div className="flex items-center justify-between p-6 border-t border-gray-200 bg-gray-50">
            <button
              onClick={() => router.back()}
              className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              Cancel
            </button>

            <button
              onClick={handleAddModel}
              disabled={!formData.modelName || !formData.shard || !formData.walletAddress}
              className="px-6 py-2 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add Model
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default AddModelPage;
