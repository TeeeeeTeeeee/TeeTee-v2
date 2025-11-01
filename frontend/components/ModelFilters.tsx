import React from 'react';

interface ModelFiltersProps {
  searchTerm: string;
  setSearchTerm: (value: string) => void;
  filterStatus: 'all' | 'hosting' | 'inactive' | 'mymodels';
  setFilterStatus: (status: 'all' | 'hosting' | 'inactive' | 'mymodels') => void;
  sortBy: 'name' | 'date' | 'status';
  setSortBy: (value: 'name' | 'date' | 'status') => void;
  incompleteLLMCount: number;
  myModelsCount: number;
  connectedAddress?: string;
  isLoadingMyModels?: boolean;
}

export const ModelFilters: React.FC<ModelFiltersProps> = ({
  searchTerm,
  setSearchTerm,
  filterStatus,
  setFilterStatus,
  sortBy,
  setSortBy,
  incompleteLLMCount,
  myModelsCount,
  connectedAddress,
  isLoadingMyModels = false
}) => {
  // Show 0 when loading or not ready, otherwise show actual count
  const displayMyModelsCount = isLoadingMyModels ? 0 : (myModelsCount || 0);
  return (
    <div className="bg-white rounded-lg mb-8">
      <div className="flex flex-col lg:flex-row gap-6 items-center justify-between">
        {/* Search */}
        <div className="flex-1 max-w-md w-full">
          <div className="relative">
            <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Search models by name, shard, or wallet..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-12 pr-12 py-2 border border-gray-300 rounded-lg bg-white text-gray-900 placeholder-gray-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Filter Buttons */}
        <div className="flex items-center flex-1 gap-6">
          <button
            onClick={() => setFilterStatus('all')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 transform ${
              filterStatus === 'all'
                ? 'text-violet-600 bg-violet-100 scale-105'
                : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100 hover:scale-102'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 4.5v15m6-15v15m-10.875 0h15.75c.621 0 1.125-.504 1.125-1.125V5.625c0-.621-.504-1.125-1.125-1.125H4.125C3.504 4.5 3 5.004 3 5.625v12.75c0 .621.504 1.125 1.125 1.125Z" />
            </svg>
            All ({incompleteLLMCount})
          </button>
          
          <div className="h-6 w-px bg-gray-300"></div>
          
          {connectedAddress && (
            <>
              <button
                onClick={() => setFilterStatus('mymodels')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 transform ${
                  filterStatus === 'mymodels'
                    ? 'text-violet-600 bg-violet-100 scale-105'
                    : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100 hover:scale-102'
                }`}
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
                </svg>
                My Models ({displayMyModelsCount})
              </button>
              
              <div className="h-6 w-px bg-gray-300"></div>
            </>
          )}
          
          <button
            onClick={() => setFilterStatus('inactive')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all duration-300 transform ${
              filterStatus === 'inactive'
                ? 'text-violet-600 bg-violet-100 scale-105'
                : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100 hover:scale-102'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25v13.5m-7.5-13.5v13.5" />
            </svg>
            Available ({incompleteLLMCount})
          </button>
        </div>

        {/* Sort Button */}
        <div className="relative">
          <button className="flex items-center justify-center w-10 h-10 border border-gray-300 rounded-lg bg-white text-gray-700 hover:bg-gray-50 transition-colors focus:outline-none">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" className="w-4 h-4">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c2.755 0 5.455.232 8.083.678.533.09.917.556.917 1.096v1.044a2.25 2.25 0 01-.659 1.591l-5.432 5.432a2.25 2.25 0 00-.659 1.591v2.927a2.25 2.25 0 01-1.244 2.013L9.75 21v-6.568a2.25 2.25 0 00-.659-1.591L3.659 7.409A2.25 2.25 0 013 5.818V4.774c0-.54.384-1.006.917-1.096A48.32 48.32 0 0112 3z" />
            </svg>
          </button>
          
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'name' | 'date' | 'status')}
            className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer"
          >
            <option value="date">Date Added</option>
            <option value="name">Name</option>
            <option value="status">Status</option>
          </select>
        </div>
      </div>
    </div>
  );
};

