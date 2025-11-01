import React from 'react';

interface ModelCardProps {
  modelName: string;
  isSelected: boolean;
  onSelect: (modelName: string) => void;
}

export const ModelCard: React.FC<ModelCardProps> = ({ modelName, isSelected, onSelect }) => {
  return (
    <button
      onClick={() => onSelect(modelName)}
      className={`text-left p-3 rounded-lg border transition-colors ${
        isSelected
          ? 'border-violet-400 bg-violet-50'
          : 'border-gray-200 hover:border-gray-300'
      }`}
    >
      <div className="font-medium text-sm text-gray-900">{modelName}</div>
      <div className="text-xs text-gray-500">AI Language Model</div>
    </button>
  );
};

