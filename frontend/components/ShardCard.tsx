import React from 'react';

interface Shard {
  id: string;
  name: string;
  region: string;
  capacity: string;
}

interface ShardCardProps {
  shard: Shard;
  isSelected: boolean;
  onSelect: (shardId: string) => void;
}

export const ShardCard: React.FC<ShardCardProps> = ({ shard, isSelected, onSelect }) => {
  return (
    <button
      onClick={() => onSelect(shard.id)}
      className={`text-left p-3 rounded-lg border transition-colors ${
        isSelected
          ? 'border-violet-400 bg-violet-50'
          : 'border-gray-200 hover:border-gray-300'
      }`}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="font-medium text-sm text-gray-900">{shard.name}</div>
        <div className={`text-xs px-1.5 py-0.5 rounded-full ${
          parseInt(shard.capacity) > 80 
            ? 'bg-red-100 text-red-800'
            : parseInt(shard.capacity) > 60
            ? 'bg-yellow-100 text-yellow-800'
            : 'bg-green-100 text-green-800'
        }`}>
          {shard.capacity}
        </div>
      </div>
      <div className="text-xs text-gray-500">{shard.region}</div>
    </button>
  );
};

