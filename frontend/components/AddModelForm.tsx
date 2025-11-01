import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Stepper, { Step } from './Stepper';
import { ModelCard } from './ModelCard';
import { ShardCard } from './ShardCard';
import { ConfigurationSummary } from './ConfigurationSummary';
import { verifyTEEEndpoint } from '../utils/teeVerification';

interface AddModelFormData {
  modelName: string;
  walletAddress: string;
  shardUrl: string;
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
    walletAddress: '',
    shardUrl: ''
  });
  const [currentStep, setCurrentStep] = useState(1);
  const [isVerifying, setIsVerifying] = useState(false);
  const [verificationError, setVerificationError] = useState<string | null>(null);
  const [verificationSuccess, setVerificationSuccess] = useState(false);
  const [attestationHash, setAttestationHash] = useState<string | null>(null);

  const handleSubmit = async () => {
    await onSubmit(formData);
  };

  // Handle step change with verification on Step 3
  const handleStepChange = async (newStep: number) => {
    // If moving from step 3 to step 4, verify TEE endpoint first
    if (currentStep === 3 && newStep === 4 && formData.shardUrl) {
      setIsVerifying(true);
      setVerificationError(null);
      setVerificationSuccess(false);
      
      const result = await verifyTEEEndpoint(formData.shardUrl);
      
      if (result.success && result.attestationHash) {
        setAttestationHash(result.attestationHash);
        setVerificationSuccess(true);
        setIsVerifying(false);
        setCurrentStep(newStep);
      } else {
        setVerificationError(result.error || 'Failed to verify TEE endpoint');
        setIsVerifying(false);
        // Stay on step 3 (error message will be shown)
      }
    } else {
      setCurrentStep(newStep);
    }
  };

  // Validation logic for each step
  const isStepValid = (step: number): boolean => {
    switch (step) {
      case 1: // Model Selection
        return !!formData.modelName;
      case 2: // Wallet Address
        return !!formData.walletAddress && formData.walletAddress.length > 0;
      case 3: // Shard URL
        return !!formData.shardUrl && formData.shardUrl.length > 0;
      case 4: // Review & Confirm
        return true; // Always valid on review step
      default:
        return false;
    }
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
        onStepChange={handleStepChange}
        onFinalStepCompleted={handleSubmit}
        backButtonText="Previous"
        nextButtonText={currentStep === 3 ? 'Verify & Continue' : 'Next'}
        cancelButtonText="Cancel"
        onCancel={onCancel}
        nextButtonProps={{
          disabled: !isStepValid(currentStep) || isVerifying || isWriting || isConfirming || isMinting || isAuthorizing
        }}
        backButtonProps={{
          disabled: isVerifying || isWriting || isConfirming || isMinting || isAuthorizing
        }}
        cancelButtonProps={{
          disabled: isVerifying || isWriting || isConfirming || isMinting || isAuthorizing
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

        {/* Step 2: Wallet Address */}
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

        {/* Step 3: Shard URL */}
        <Step>
          <div className="space-y-4">
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">TEE Shard URL</h3>
              <p className="text-base text-gray-600">Enter your Phala TEE shard endpoint URL for attestation</p>
            </div>
            
            <div>
              <label htmlFor="shardUrl" className="block text-sm font-medium text-gray-700 mb-2">
                Shard URL *
              </label>
              <input
                id="shardUrl"
                type="url"
                value={formData.shardUrl}
                onChange={(e) => setFormData({ ...formData, shardUrl: e.target.value })}
                placeholder="https://1e8ddb822fabefe60399b39bbfb83478c1a12e3c-3001.dstack-pha-prod7.phala.network/"
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent outline-none"
              />
              <p className="mt-2 text-xs text-gray-500">
                Example: https://[app-id]-3001.dstack-pha-prod7.phala.network/
              </p>
            </div>

            {/* Verification Loading State */}
            {isVerifying && (
              <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-yellow-600"></div>
                  <div>
                    <h4 className="text-sm font-semibold text-gray-900">Verifying TEE Endpoint...</h4>
                    <p className="text-xs text-gray-600 mt-1">
                      Checking health and attestation endpoints
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Verification Success State */}
            {verificationSuccess && attestationHash && (
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-start gap-2">
                  <svg className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div className="flex-1">
                    <h4 className="text-sm font-semibold text-green-900">✓ TEE Verification Successful</h4>
                    <p className="text-xs text-green-700 mt-1">
                      Health check passed and attestation verified.
                    </p>
                    <div className="mt-2 p-2 bg-white rounded border border-green-200">
                      <p className="text-xs font-medium text-gray-700 mb-1">Attestation Hash (SHA-256):</p>
                      <code className="text-xs font-mono text-green-700 break-all">{attestationHash}</code>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Verification Error State */}
            {verificationError && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-start gap-2">
                  <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <div>
                    <h4 className="text-sm font-semibold text-red-900">Verification Failed</h4>
                    <p className="text-xs text-red-700 mt-1">{verificationError}</p>
                    <button
                      type="button"
                      onClick={async () => {
                        setVerificationError(null);
                        setIsVerifying(true);
                        const result = await verifyTEEEndpoint(formData.shardUrl);
                        if (result.success && result.attestationHash) {
                          setAttestationHash(result.attestationHash);
                          setVerificationSuccess(true);
                        } else {
                          setVerificationError(result.error || 'Failed to verify TEE endpoint');
                        }
                        setIsVerifying(false);
                      }}
                      className="mt-2 text-xs font-medium text-red-600 hover:text-red-700 underline"
                    >
                      Try Again
                    </button>
                  </div>
                </div>
              </div>
            )}

            {!formData.shardUrl && (
              <p className="text-xs text-amber-600">Please enter your shard URL to continue</p>
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
            <div className="bg-gradient-to-br from-violet-50 to-purple-50 rounded-xl border-2 border-violet-200 p-6">
              <h4 className="font-bold text-lg text-gray-900 mb-4">Configuration Summary</h4>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Model:</span>
                  <span className="text-sm font-semibold text-gray-900">{formData.modelName || 'Not selected'}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Wallet:</span>
                  <span className="text-xs font-mono text-gray-900">
                    {formData.walletAddress ? `${formData.walletAddress.slice(0, 10)}...${formData.walletAddress.slice(-8)}` : 'Not provided'}
                  </span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-sm text-gray-600">TEE Shard:</span>
                  <span className="text-xs font-mono text-violet-700 break-all">
                    {formData.shardUrl || 'Not provided'}
                  </span>
                </div>
                {attestationHash && (
                  <div className="flex flex-col gap-1 pt-2 border-t border-violet-200">
                    <div className="flex items-center gap-2">
                      <svg className="w-4 h-4 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                      </svg>
                      <span className="text-sm text-gray-600 font-medium">TEE Verified</span>
                    </div>
                    <div className="p-2 bg-white rounded border border-green-200">
                      <p className="text-xs font-medium text-gray-700 mb-1">Attestation Hash:</p>
                      <code className="text-xs font-mono text-green-700 break-all">{attestationHash.slice(0, 32)}...</code>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
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

