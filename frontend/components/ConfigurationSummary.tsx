import React from 'react';

interface Shard {
  id: string;
  name: string;
  region: string;
  capacity: string;
}

interface ConfigurationSummaryProps {
  modelName: string;
  shard: Shard | undefined;
  walletAddress: string;
}

export const ConfigurationSummary: React.FC<ConfigurationSummaryProps> = ({
  modelName,
  shard,
  walletAddress
}) => {
  return (
    <div className="p-5 bg-gray-800 bg-opacity-50 rounded-xl border border-gray-700">
      <h4 className="font-semibold text-base text-white mb-4 flex items-center gap-2">
        <svg className="w-5 h-5 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Configuration Summary
      </h4>
      
      <div className="space-y-3">
        {/* Model Info */}
        <div className="p-3 bg-gray-700 bg-opacity-50 rounded-lg border border-gray-600">
          <div className="text-xs text-gray-400 mb-1">Selected Model</div>
          <div className="font-semibold text-base text-white">{modelName}</div>
          <div className="text-xs text-gray-400">AI Language Model</div>
        </div>

        {/* Shard Info */}
        {shard && (
          <div className="p-3 bg-gray-700 bg-opacity-50 rounded-lg border border-gray-600">
            <div className="text-xs text-gray-400 mb-1">Your Shard</div>
            <div className="font-semibold text-base text-white">
              {shard.name}
            </div>
            <div className="text-xs text-gray-400">
              {shard.region} • {shard.capacity}
            </div>
          </div>
        )}

        {/* Wallet Info */}
        <div className="p-3 bg-gray-700 bg-opacity-50 rounded-lg border border-gray-600">
          <div className="text-xs text-gray-400 mb-1">Wallet Address</div>
          <div className="font-mono text-xs text-gray-200 break-all">{walletAddress}</div>
        </div>

        {/* Important Notes */}
        <div className="p-3 bg-violet-900 bg-opacity-30 rounded-lg border border-violet-700 space-y-2">
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-violet-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-violet-200">This will create a hosting slot waiting for a second host</p>
          </div>
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-violet-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm text-violet-200">Hosting duration will be tracked by the oracle</p>
          </div>
        </div>
      </div>
    </div>
  );
};

