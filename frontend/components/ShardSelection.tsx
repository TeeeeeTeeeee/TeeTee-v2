import React from 'react';

interface ShardSelectionProps {
  selectedShard: string;
  onShardSelect: (shard: 'shard1' | 'shard2') => void;
  availableShard?: 'shard1' | 'shard2' | null;
  disabled?: boolean;
}

export const ShardSelection: React.FC<ShardSelectionProps> = ({
  selectedShard,
  onShardSelect,
  availableShard,
  disabled = false,
}) => {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-2xl font-bold text-gray-900 mb-2">Select Shard Layer</h3>
        <p className="text-base text-gray-600">
          {availableShard 
            ? availableShard === 'shard1'
              ? 'You will be hosting Shard 1 (Lower Layers) - Shard 2 is already taken'
              : 'You will be hosting Shard 2 (Upper Layers) - Shard 1 is already taken'
            : 'Choose which shard layer to host - both are available'}
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Shard 1 Option */}
        <button
          type="button"
          onClick={() => !disabled && (!availableShard || availableShard === 'shard1') && onShardSelect('shard1')}
          disabled={disabled || !!(availableShard && availableShard !== 'shard1')}
          className={`p-6 rounded-xl border-2 transition-all text-left relative ${
            selectedShard === 'shard1'
              ? 'border-violet-400 bg-violet-50 shadow-md'
              : availableShard && availableShard !== 'shard1'
              ? 'border-gray-200 bg-gray-100 opacity-50 cursor-not-allowed'
              : disabled
              ? 'border-gray-200 bg-gray-100 opacity-50 cursor-not-allowed'
              : 'border-gray-200 hover:border-violet-200 bg-white'
          }`}
        >
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                selectedShard === 'shard1'
                  ? 'border-violet-600 bg-violet-600'
                  : 'border-gray-300'
              }`}>
                {selectedShard === 'shard1' && (
                  <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                )}
              </div>
              <h4 className="text-lg font-bold text-gray-900">Shard 1</h4>
            </div>
            <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${
              availableShard && availableShard !== 'shard1'
                ? 'bg-red-200 text-red-800'
                : selectedShard === 'shard1'
                ? 'bg-violet-200 text-violet-800'
                : 'bg-gray-200 text-gray-700'
            }`}>
              {availableShard && availableShard !== 'shard1' ? 'Taken' : 'Lower Layers'}
            </span>
          </div>
          <p className="text-sm text-gray-600 mb-3">Host layers 1 to 50 of the model</p>
          
        </button>

        {/* Shard 2 Option */}
        <button
          type="button"
          onClick={() => !disabled && (!availableShard || availableShard === 'shard2') && onShardSelect('shard2')}
          disabled={disabled || !!(availableShard && availableShard !== 'shard2')}
          className={`p-6 rounded-xl border-2 transition-all text-left relative ${
            selectedShard === 'shard2'
              ? 'border-violet-400 bg-violet-50 shadow-md'
              : availableShard && availableShard !== 'shard2'
              ? 'border-gray-200 bg-gray-100 opacity-50 cursor-not-allowed'
              : disabled
              ? 'border-gray-200 bg-gray-100 opacity-50 cursor-not-allowed'
              : 'border-gray-200 hover:border-violet-200 bg-white'
          }`}
        >
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                selectedShard === 'shard2'
                  ? 'border-violet-600 bg-violet-600'
                  : 'border-gray-300'
              }`}>
                {selectedShard === 'shard2' && (
                  <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                )}
              </div>
              <h4 className="text-lg font-bold text-gray-900">Shard 2</h4>
            </div>
            <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${
              availableShard && availableShard !== 'shard2'
                ? 'bg-red-200 text-red-800'
                : selectedShard === 'shard2'
                ? 'bg-violet-200 text-violet-800'
                : 'bg-gray-200 text-gray-700'
            }`}>
              {availableShard && availableShard !== 'shard2' ? 'Taken' : 'Upper Layers'}
            </span>
          </div>
          <p className="text-sm text-gray-600 mb-3">Host layers 51 to 100 of the model</p>
          
        </button>
      </div>
      
      {!selectedShard && (
        <p className="text-xs text-amber-600">Please select a shard layer to continue</p>
      )}
    </div>
  );
};

