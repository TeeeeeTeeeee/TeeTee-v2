import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Stepper, { Step } from './Stepper';
import { ModelCard } from './ModelCard';
import { ShardCard } from './ShardCard';
import { ConfigurationSummary } from './ConfigurationSummary';

interface AddModelFormData {
  modelName: string;
  shard: string;
  walletAddress: string;
}

interface Shard {
  id: string;
  name: string;
  region: string;
  capacity: string;
}

interface AddModelFormProps {
  availableModels: string[];
  availableShards: Shard[];
  connectedAddress?: string;
  onSubmit: (formData: AddModelFormData) => Promise<void>;
  onCancel: () => void;
  isWriting: boolean;
  isConfirming: boolean;
  isMinting: boolean;
  isAuthorizing: boolean;
  writeError: Error | null;
}

export const AddModelForm: React.FC<AddModelFormProps> = ({
  availableModels,
  availableShards,
  connectedAddress,
  onSubmit,
  onCancel,
  isWriting,
  isConfirming,
  isMinting,
  isAuthorizing,
  writeError,
}) => {
  const [formData, setFormData] = useState<AddModelFormData>({
    modelName: '',
    shard: '',
    walletAddress: ''
  });

  const handleSubmit = async () => {
    await onSubmit(formData);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="mb-12"
    >
      <Stepper
        initialStep={1}
        onStepChange={(step) => {
          console.log('Current step:', step);
        }}
        onFinalStepCompleted={handleSubmit}
        backButtonText="Previous"
        nextButtonText="Next"
        cancelButtonText="Cancel"
        onCancel={onCancel}
        nextButtonProps={{
          disabled: isWriting || isConfirming || isMinting || isAuthorizing
        }}
        backButtonProps={{
          disabled: isWriting || isConfirming || isMinting || isAuthorizing
        }}
        cancelButtonProps={{
          disabled: isWriting || isConfirming || isMinting || isAuthorizing
        }}
      >
        {/* Step 1: Model Selection */}
        <Step>
          <div className="space-y-4">
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Select Model</h3>
              <p className="text-base text-gray-600">Choose from available AI models</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              {availableModels.map((model) => (
                <ModelCard
                  key={model}
                  modelName={model}
                  isSelected={formData.modelName === model}
                  onSelect={(modelName) => setFormData({ ...formData, modelName })}
                />
              ))}
            </div>
            
            {!formData.modelName && (
              <p className="text-xs text-amber-600">Please select a model to continue</p>
            )}
          </div>
        </Step>

        {/* Step 2: Shard Selection */}
        <Step>
          <div className="space-y-4">
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Select Your Shard</h3>
              <p className="text-base text-gray-600">Choose where to host your part of the model</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {availableShards.map((shard) => (
                <ShardCard
                  key={shard.id}
                  shard={shard}
                  isSelected={formData.shard === shard.id}
                  onSelect={(shardId) => setFormData({ ...formData, shard: shardId })}
                />
              ))}
            </div>
            
            {!formData.shard && (
              <p className="text-xs text-amber-600">Please select a shard to continue</p>
            )}
          </div>
        </Step>

        {/* Step 3: Wallet Address */}
        <Step>
          <div className="space-y-4">
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Your Wallet Address</h3>
              <p className="text-base text-gray-600">Enter your wallet address to receive hosting rewards</p>
            </div>
            
            <div>
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
                  className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent outline-none"
                />
                <button
                  type="button"
                  onClick={() => connectedAddress && setFormData({ ...formData, walletAddress: connectedAddress })}
                  disabled={!connectedAddress}
                  className="px-3 py-2 bg-violet-100 text-violet-700 rounded-lg hover:bg-violet-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap text-xs font-medium"
                  title={connectedAddress ? 'Use connected wallet' : 'No wallet connected'}
                >
                  Use Connected
                </button>
              </div>
            </div>

            {!formData.walletAddress && (
              <p className="text-xs text-amber-600">Please enter your wallet address to continue</p>
            )}
          </div>
        </Step>

        {/* Step 4: Review & Confirm */}
        <Step>
          <div className="space-y-4">
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Review & Confirm</h3>
              <p className="text-base text-gray-600">Please review your configuration before submitting</p>
            </div>

            {/* Summary Card */}
            <ConfigurationSummary
              modelName={formData.modelName}
              shard={availableShards.find(s => s.id === formData.shard)}
              walletAddress={formData.walletAddress}
            />
            
            {/* Transaction Status */}
            {(isWriting || isConfirming || isMinting || isAuthorizing) && (
              <div className="flex items-center gap-3 text-sm text-blue-700 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-violet-600"></div>
                <span>
                  {isWriting ? 'Submitting transaction...' : 
                   isConfirming ? 'Confirming registration...' : 
                   isMinting ? 'Minting INFT token...' : 
                   isAuthorizing ? 'Authorizing access...' : 'Processing...'}
                </span>
              </div>
            )}
            
            {writeError && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-start gap-2">
                  <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div>
                    <div className="font-medium text-red-900 text-sm">Transaction Error</div>
                    <div className="text-sm text-red-700 mt-0.5">
                      {writeError.message.slice(0, 80)}...
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Step>
      </Stepper>
    </motion.div>
  );
};

