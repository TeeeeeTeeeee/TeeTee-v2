import React from 'react';

interface ModelCardProps {
  modelName: string;
  isSelected: boolean;
  onSelect: (modelName: string) => void;
  iconUrl?: string;
}

const DEFAULT_ICON = 'https://www.redpill.ai/_next/image?url=https%3A%2F%2Ft0.gstatic.com%2FfaviconV2%3Fclient%3DSOCIAL%26type%3DFAVICON%26fallback_opts%3DTYPE%2CSIZE%2CURL%26url%3Dhttps%3A%2F%2Fhuggingface.co%2F%26size%3D32&w=48&q=75';

export const ModelCard: React.FC<ModelCardProps> = ({ modelName, isSelected, onSelect, iconUrl }) => {
  return (
    <button
      onClick={() => onSelect(modelName)}
      className={`text-left p-3 rounded-lg border transition-all ${
        isSelected
          ? 'border-violet-400 bg-violet-50'
          : 'border-gray-200 hover:border-gray-300 bg-white'
      }`}
    >
      <div className="flex items-center gap-2">
        {/* LLM Icon */}
        <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 p-1 bg-gray-100">
          <img 
            src={iconUrl || DEFAULT_ICON} 
            alt={modelName}
            className="w-full h-full object-contain"
            onError={(e) => {
              e.currentTarget.src = DEFAULT_ICON;
            }}
          />
        </div>
        <div>
          <div className="font-medium text-sm text-gray-900">{modelName}</div>
          <div className="text-xs text-gray-500">AI Language Model</div>
        </div>
      </div>
    </button>
  );
};

