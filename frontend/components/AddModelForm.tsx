import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Stepper, { Step } from './Stepper';
import { ModelCard } from './ModelCard';
import { ShardCard } from './ShardCard';
import { ConfigurationSummary } from './ConfigurationSummary';
import { verifyTEEEndpoint } from '../utils/teeVerification';

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

const DEFAULT_LLM_ICON = 'https://w7.pngwing.com/pngs/839/288/png-transparent-hugging-face-favicon-logo-tech-companies-thumbnail.png';

const getModelIcon = (modelName: string): string => {
  return LLM_ICONS[modelName] || DEFAULT_LLM_ICON;
};

interface AddModelFormData {
  modelName: string;
  walletAddress: string;
  shardSelection: string;
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
    shardSelection: '',
    shardUrl: ''
  });
  const [currentStep, setCurrentStep] = useState(1);
  const [isVerifying, setIsVerifying] = useState(false);
  const [healthCheckStatus, setHealthCheckStatus] = useState<'idle' | 'checking' | 'success' | 'error'>('idle');
  const [healthCheckError, setHealthCheckError] = useState<string | null>(null);
  const [attestationCheckStatus, setAttestationCheckStatus] = useState<'idle' | 'checking' | 'success' | 'error'>('idle');
  const [attestationCheckError, setAttestationCheckError] = useState<string | null>(null);
  const [attestationHash, setAttestationHash] = useState<string | null>(null);

  const handleSubmit = async () => {
    await onSubmit(formData);
  };

  // Auto-verify when shard URL changes (with debounce)
  React.useEffect(() => {
    if (!formData.shardUrl || currentStep !== 4) {
      setHealthCheckStatus('idle');
      setAttestationCheckStatus('idle');
      setHealthCheckError(null);
      setAttestationCheckError(null);
      setAttestationHash(null);
      return;
    }

    const verifyEndpoint = async () => {
      setIsVerifying(true);
      setHealthCheckStatus('checking');
      setAttestationCheckStatus('idle');
      setHealthCheckError(null);
      setAttestationCheckError(null);
      setAttestationHash(null);

      try {
        const baseUrl = formData.shardUrl.endsWith('/') ? formData.shardUrl : `${formData.shardUrl}/`;
        
        // Step 1: Health Check
        const healthResponse = await fetch(`${baseUrl}health`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });

        if (!healthResponse.ok) {
          setHealthCheckStatus('error');
          setHealthCheckError('Health check failed. TEE endpoint is not accessible.');
          setIsVerifying(false);
          return;
        }

        setHealthCheckStatus('success');

        // Step 2: Attestation Check
        setAttestationCheckStatus('checking');
        
        const attestResponse = await fetch(`${baseUrl}attest/quick`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });

        if (!attestResponse.ok) {
          setAttestationCheckStatus('error');
          setAttestationCheckError('Attestation request failed.');
          setIsVerifying(false);
          return;
        }

        const attestationData = await attestResponse.json();

        if (!attestationData.success) {
          setAttestationCheckStatus('error');
          setAttestationCheckError('Attestation was not successful.');
          setIsVerifying(false);
          return;
        }

        if (!attestationData.note) {
          setAttestationCheckStatus('error');
          setAttestationCheckError('Attestation note not found.');
          setIsVerifying(false);
          return;
        }

        // Generate SHA-256 hash of the note
        const encoder = new TextEncoder();
        const data = encoder.encode(attestationData.note);
        const hashBuffer = await crypto.subtle.digest('SHA-256', data);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

        // Verify hash matches expected value
        const EXPECTED_HASH = 'df99bee2e34b5f653722df6fb654c0d906c57173b6be3fab104815e58ce064bc';
        
        if (hashHex !== EXPECTED_HASH) {
          setAttestationCheckStatus('error');
          setAttestationCheckError(`Hash mismatch. Expected: ${EXPECTED_HASH}, Got: ${hashHex}`);
          setIsVerifying(false);
          return;
        }

        setAttestationHash(hashHex);
        setAttestationCheckStatus('success');
        setIsVerifying(false);

      } catch (error: any) {
        if (healthCheckStatus === 'checking') {
          setHealthCheckStatus('error');
          setHealthCheckError(error.message || 'Failed to connect to health endpoint.');
        } else {
          setAttestationCheckStatus('error');
          setAttestationCheckError(error.message || 'Failed to verify attestation.');
        }
        setIsVerifying(false);
      }
    };

    // Debounce the verification to avoid too many requests
    const timeoutId = setTimeout(verifyEndpoint, 800);
    return () => clearTimeout(timeoutId);
  }, [formData.shardUrl, currentStep]);

  // Handle step change
  const handleStepChange = async (newStep: number) => {
    setCurrentStep(newStep);
  };

  // Validation logic for each step
  const isStepValid = (step: number): boolean => {
    switch (step) {
      case 1: // Model Selection
        return !!formData.modelName;
      case 2: // Wallet Address
        return !!formData.walletAddress && formData.walletAddress.length > 0;
      case 3: // Shard Selection
        return !!formData.shardSelection;
      case 4: // TEE Shard URL - valid ONLY if both checks pass
        return !!formData.shardUrl && 
               healthCheckStatus === 'success' && 
               attestationCheckStatus === 'success' && 
               !isVerifying;
      case 5: // Review & Confirm
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
        nextButtonText="Next"
        cancelButtonText="Cancel"
        onCancel={onCancel}
        nextButtonProps={{
          disabled: !isStepValid(currentStep) || isWriting || isConfirming || isMinting || isAuthorizing
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
                  iconUrl={getModelIcon(model)}
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

        {/* Step 3: Shard Selection */}
        <Step>
          <div className="space-y-4">
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Select Shard Layer</h3>
              <p className="text-base text-gray-600">Choose which shard layer to host your model on</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Shard 1 Option */}
              <button
                type="button"
                onClick={() => setFormData({ ...formData, shardSelection: 'shard1' })}
                className={`p-6 rounded-xl border-2 transition-all text-left ${
                  formData.shardSelection === 'shard1'
                    ? 'border-violet-400 bg-violet-50 shadow-md'
                    : 'border-gray-200 hover:border-violet-200 bg-white'
                }`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                      formData.shardSelection === 'shard1'
                        ? 'border-violet-600 bg-violet-600'
                        : 'border-gray-300'
                    }`}>
                      {formData.shardSelection === 'shard1' && (
                        <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </div>
                    <h4 className="text-lg font-bold text-gray-900">Shard 1</h4>
                  </div>
                  <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${
                    formData.shardSelection === 'shard1'
                      ? 'bg-violet-200 text-violet-800'
                      : 'bg-gray-200 text-gray-700'
                  }`}>
                    Lower Layers
                  </span>
                </div>
                <p className="text-sm text-gray-600 mb-3">Host layers 1 to 50 of the model</p>
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>Optimized for faster inference</span>
                </div>
              </button>

              {/* Shard 2 Option */}
              <button
                type="button"
                onClick={() => setFormData({ ...formData, shardSelection: 'shard2' })}
                className={`p-6 rounded-xl border-2 transition-all text-left ${
                  formData.shardSelection === 'shard2'
                    ? 'border-violet-400 bg-violet-50 shadow-md'
                    : 'border-gray-200 hover:border-violet-200 bg-white'
                }`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                      formData.shardSelection === 'shard2'
                        ? 'border-violet-600 bg-violet-600'
                        : 'border-gray-300'
                    }`}>
                      {formData.shardSelection === 'shard2' && (
                        <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      )}
                    </div>
                    <h4 className="text-lg font-bold text-gray-900">Shard 2</h4>
                  </div>
                  <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${
                    formData.shardSelection === 'shard2'
                      ? 'bg-violet-200 text-violet-800'
                      : 'bg-gray-200 text-gray-700'
                  }`}>
                    Upper Layers
                  </span>
                </div>
                <p className="text-sm text-gray-600 mb-3">Host layers 51 to 100 of the model</p>
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                  </svg>
                  <span>Balanced workload distribution</span>
                </div>
              </button>
            </div>
            
            {!formData.shardSelection && (
              <p className="text-xs text-amber-600">Please select a shard layer to continue</p>
            )}
          </div>
        </Step>

        {/* Step 4: TEE Shard URL */}
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

            {/* Verification Status - Two Separate Checks */}
            {formData.shardUrl && (
              <div className="space-y-3">
                {/* Health Check Status */}
                <div className={`p-4 rounded-lg border-2 transition-all ${
                  healthCheckStatus === 'checking'
                    ? 'bg-yellow-50 border-yellow-300'
                    : healthCheckStatus === 'success'
                    ? 'bg-green-50 border-green-300'
                    : healthCheckStatus === 'error'
                    ? 'bg-red-50 border-red-300'
                    : 'bg-gray-50 border-gray-200'
                }`}>
                  <div className="flex items-start gap-3">
                    {healthCheckStatus === 'checking' ? (
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-yellow-600 flex-shrink-0 mt-0.5"></div>
                    ) : healthCheckStatus === 'success' ? (
                      <svg className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    ) : healthCheckStatus === 'error' ? (
                      <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    ) : (
                      <svg className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    )}
                    <div className="flex-1">
                      <h4 className={`text-sm font-semibold ${
                        healthCheckStatus === 'checking'
                          ? 'text-yellow-900'
                          : healthCheckStatus === 'success'
                          ? 'text-green-900'
                          : healthCheckStatus === 'error'
                          ? 'text-red-900'
                          : 'text-gray-600'
                      }`}>
                        {healthCheckStatus === 'checking'
                          ? '1. Checking Health Endpoint...'
                          : healthCheckStatus === 'success'
                          ? '✓ 1. Health Check Passed'
                          : healthCheckStatus === 'error'
                          ? '✗ 1. Health Check Failed'
                          : '1. Health Check - Waiting'}
                      </h4>
                      {healthCheckError && (
                        <p className="text-xs text-red-700 mt-1">{healthCheckError}</p>
                      )}
                      {healthCheckStatus === 'success' && (
                        <p className="text-xs text-green-700 mt-1">Endpoint is accessible and responding</p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Attestation Check Status */}
                <div className={`p-4 rounded-lg border-2 transition-all ${
                  attestationCheckStatus === 'checking'
                    ? 'bg-yellow-50 border-yellow-300'
                    : attestationCheckStatus === 'success'
                    ? 'bg-green-50 border-green-300'
                    : attestationCheckStatus === 'error'
                    ? 'bg-red-50 border-red-300'
                    : 'bg-gray-50 border-gray-200'
                }`}>
                  <div className="flex items-start gap-3">
                    {attestationCheckStatus === 'checking' ? (
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-yellow-600 flex-shrink-0 mt-0.5"></div>
                    ) : attestationCheckStatus === 'success' ? (
                      <svg className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    ) : attestationCheckStatus === 'error' ? (
                      <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    ) : (
                      <svg className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    )}
                    <div className="flex-1">
                      <h4 className={`text-sm font-semibold ${
                        attestationCheckStatus === 'checking'
                          ? 'text-yellow-900'
                          : attestationCheckStatus === 'success'
                          ? 'text-green-900'
                          : attestationCheckStatus === 'error'
                          ? 'text-red-900'
                          : 'text-gray-600'
                      }`}>
                        {attestationCheckStatus === 'checking'
                          ? '2. Verifying Attestation Note...'
                          : attestationCheckStatus === 'success'
                          ? '✓ 2. Model Hash Verified'
                          : attestationCheckStatus === 'error'
                          ? '✗ 2. Attestation Note Failed'
                          : '2. Attestation Check - Waiting'}
                      </h4>
                      {attestationCheckError && (
                        <p className="text-xs text-red-700 mt-1">{attestationCheckError}</p>
                      )}
                      {attestationCheckStatus === 'success' && attestationHash && (
                        <div className="mt-2">
                          <p className="text-xs text-green-700 mb-1">Model Hash matches expected value</p>
                          <div className="p-2 bg-white rounded border border-green-200">
                            <p className="text-xs font-medium text-gray-700 mb-1">Attestation Hash:</p>
                            <code className="text-xs font-mono text-green-700 break-all">{attestationHash}</code>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

              </div>
            )}

            {!formData.shardUrl && (
              <p className="text-xs text-amber-600">Please enter your shard URL to continue</p>
            )}
          </div>
        </Step>

        {/* Step 5: Review & Confirm */}
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
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Shard Layer:</span>
                  <span className="text-sm font-semibold text-gray-900">
                    {formData.shardSelection === 'shard1' ? 'Shard 1 (Layers 1-50)' : 
                     formData.shardSelection === 'shard2' ? 'Shard 2 (Layers 51-100)' : 
                     'Not selected'}
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
              <div className="p-4 bg-blue-50 rounded-lg border-2 border-blue-300">
                <div className="flex items-start gap-3">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-violet-600 flex-shrink-0 mt-0.5"></div>
                  <div className="flex-1">
                    <h4 className="text-base font-semibold text-blue-900 mb-1">
                      {isWriting ? '⏳ Waiting for Transaction Signature' : 
                       isConfirming ? '⏳ Transaction Confirming...' : 
                       isMinting ? '⏳ Minting INFT Token...' : 
                       isAuthorizing ? '⏳ Authorizing Access...' : '⏳ Processing...'}
                    </h4>
                    <p className="text-sm text-blue-700">
                      {isWriting ? 'Please sign the transaction in your wallet to continue' : 
                       isConfirming ? 'Your transaction is being confirmed on the blockchain...' : 
                       isMinting ? 'Creating your INFT token...' : 
                       isAuthorizing ? 'Setting up access permissions...' : 'Please wait...'}
                    </p>
                  </div>
                </div>
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

