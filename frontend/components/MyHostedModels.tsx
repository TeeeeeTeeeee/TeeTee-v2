import React from 'react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/router';

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

interface MyHostedModelsProps {
  myHostedModels: IncompleteLLM[];
  isLoadingMyModels: boolean;
  searchTerm: string;
  availableShards: Shard[];
  connectedAddress: string;
  pausingModelId: number | null;
  stoppingModelId: number | null;
  onShowAddForm: () => void;
  onPauseModel: (modelId: number) => void;
  onStopModel: (model: IncompleteLLM) => void;
}

export const MyHostedModels: React.FC<MyHostedModelsProps> = ({
  myHostedModels,
  isLoadingMyModels,
  searchTerm,
  availableShards,
  connectedAddress,
  pausingModelId,
  stoppingModelId,
  onShowAddForm,
  onPauseModel,
  onStopModel
}) => {
  const router = useRouter();

  if (isLoadingMyModels) {
    return (
      <div className="flex items-center justify-center py-10">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-400"></div>
      </div>
    );
  }

  if (myHostedModels.length === 0) {
    return (
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
            onClick={onShowAddForm}
            className="px-6 py-3 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity font-medium"
          >
            Add Your First Model
          </button>
        </div>
      </div>
    );
  }

  const filteredModels = myHostedModels.filter(model => {
    if (searchTerm) {
      return model.modelName.toLowerCase().includes(searchTerm.toLowerCase()) ||
             model.host1.toLowerCase().includes(searchTerm.toLowerCase()) ||
             (model.host2 && model.host2.toLowerCase().includes(searchTerm.toLowerCase())) ||
             model.shardUrl1.toLowerCase().includes(searchTerm.toLowerCase()) ||
             (model.shardUrl2 && model.shardUrl2.toLowerCase().includes(searchTerm.toLowerCase()));
    }
    return true;
  });

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {filteredModels.map((model) => {
        const userIsHost1 = model.host1.toLowerCase() === connectedAddress.toLowerCase();
        const userIsHost2 = model.host2?.toLowerCase() === connectedAddress.toLowerCase();
        const userRole = userIsHost1 ? 'Host 1' : 'Host 2';
        const userShard = userIsHost1 ? model.shardUrl1 : model.shardUrl2;
        const partnerAddress = userIsHost1 ? model.host2 : model.host1;
        const partnerShard = userIsHost1 ? model.shardUrl2 : model.shardUrl1;

        return (
          <motion.div
            key={model.id}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl border-2 border-green-300 p-6 hover:border-green-400 transition-all shadow-sm"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <h3 className="font-bold text-lg text-gray-900 mb-2">{model.modelName}</h3>
                <div className="inline-flex items-center gap-2 px-3 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">
                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  Hosting Active
                </div>
              </div>
            </div>

            <div className="space-y-3 mb-4">
              {/* Your Role */}
              <div className="p-3 bg-white bg-opacity-60 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <svg className="w-4 h-4 text-violet-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                  <span className="text-xs font-semibold text-gray-700">Your Role:</span>
                </div>
                <div className="text-sm font-medium text-violet-700">{userRole}</div>
                <div className="text-xs text-gray-600 mt-1">
                  Shard: {availableShards.find(s => s.id === userShard)?.name || userShard}
                  {availableShards.find(s => s.id === userShard) && (
                    <span className="ml-1">({availableShards.find(s => s.id === userShard)?.region})</span>
                  )}
                </div>
              </div>

              {/* Partner Info */}
              {partnerAddress && partnerAddress !== '0x0000000000000000000000000000000000000000' && (
                <div className="p-3 bg-white bg-opacity-40 rounded-lg">
                  <div className="flex items-center gap-2 mb-2">
                    <svg className="w-4 h-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                    </svg>
                    <span className="text-xs font-semibold text-gray-700">Partner:</span>
                  </div>
                  <div className="text-xs font-mono text-gray-700">{partnerAddress.slice(0, 10)}...{partnerAddress.slice(-8)}</div>
                  <div className="text-xs text-gray-600 mt-1">
                    Shard: {availableShards.find(s => s.id === partnerShard)?.name || partnerShard}
                  </div>
                </div>
              )}

              {/* Pool Balance */}
              <div className="p-3 bg-gradient-to-r from-violet-100 to-purple-100 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4 text-violet-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="text-xs font-semibold text-gray-700">Rewards Pool:</span>
                  </div>
                  <span className="text-sm font-bold text-violet-700">{model.poolBalance?.toString() || '0'}</span>
                </div>
              </div>

              {/* Model Status */}
              <div className="flex items-center justify-between text-xs text-gray-600">
                <span>Model ID: #{model.id}</span>
                <span className={`px-2 py-1 rounded-full font-medium ${
                  model.isComplete 
                    ? 'bg-green-100 text-green-700' 
                    : 'bg-yellow-100 text-yellow-700'
                }`}>
                  {model.isComplete ? 'Complete' : 'Waiting for Partner'}
                </span>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="space-y-2">
              <button
                onClick={() => router.push('/chat')}
                className="w-full py-2 px-4 bg-gradient-to-r from-violet-400 to-purple-300 text-white rounded-lg hover:opacity-90 transition-opacity font-medium text-sm flex items-center justify-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                Use in Chat
              </button>
              
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => onPauseModel(model.id)}
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
                  onClick={() => onStopModel(model)}
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
  );
};

