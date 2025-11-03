import React from 'react';
import { motion } from 'framer-motion';

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

interface Shard {
  id: string;
  name: string;
  region: string;
  capacity: string;
}

interface AvailableHostingSlotsProps {
  incompleteLLMDetails: IncompleteLLM[];
  isLoadingIncomplete: boolean;
  searchTerm: string;
  availableShards: Shard[];
  connectedAddress: string | undefined;
  onJoinAsHost: (llmId: number) => void;
}

export const AvailableHostingSlots: React.FC<AvailableHostingSlotsProps> = ({
  incompleteLLMDetails,
  isLoadingIncomplete,
  searchTerm,
  availableShards,
  connectedAddress,
  onJoinAsHost
}) => {
  if (isLoadingIncomplete) {
    return (
      <div className="flex items-center justify-center py-10">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-400"></div>
      </div>
    );
  }

  if (incompleteLLMDetails.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <div className="text-center max-w-md">
          <h3 className="text-xl font-bold text-gray-900 mb-2">No available slots</h3>
          <p className="text-gray-600">There are currently no hosting slots waiting for a second host</p>
        </div>
      </div>
    );
  }

  const filteredLLMs = incompleteLLMDetails.filter(llm => {
    if (searchTerm) {
      return llm.modelName.toLowerCase().includes(searchTerm.toLowerCase()) ||
             llm.host1.toLowerCase().includes(searchTerm.toLowerCase()) ||
             llm.shardUrl1.toLowerCase().includes(searchTerm.toLowerCase());
    }
    return true;
  });

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {filteredLLMs.map((llm) => (
        <motion.div
          key={llm.id}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-gradient-to-br from-violet-50 to-purple-50 rounded-xl border-2 border-violet-200 p-6 hover:border-violet-400 transition-all"
        >
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <h3 className="font-bold text-lg text-gray-900 mb-1">{llm.modelName}</h3>
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs font-medium">
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                </svg>
                Waiting for 2nd host
              </div>
            </div>
          </div>

          <div className="space-y-2 text-sm mb-4">
            <div className="flex items-center gap-2">
              <span className="text-gray-500">Host 1:</span>
              <span className="font-mono text-xs text-gray-700">{llm.host1.slice(0, 10)}...</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-500">Shard 1:</span>
              <span className="text-gray-700">{availableShards.find(s => s.id === llm.shardUrl1)?.name || llm.shardUrl1}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-500">Pool Balance:</span>
              <span className="text-violet-700 font-semibold">{llm.poolBalance?.toString() || '0'} credits</span>
            </div>
          </div>

          <button
            onClick={() => onJoinAsHost(llm.id)}
            className="w-full py-2 px-4 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity font-medium text-sm"
          >
            Join as Host 2
          </button>
        </motion.div>
      ))}
    </div>
  );
};

