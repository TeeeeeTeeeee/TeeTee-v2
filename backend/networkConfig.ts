/**
 * Network Configuration for Backend Service
 * Supports both Testnet (Galileo) and Mainnet
 * 
 * 🎯 TO SWITCH NETWORKS:
 * Change NETWORK_TYPE in both:
 * - backend/networkConfig.ts (this file)
 * - frontend/lib/networkConfig.ts
 */

export type NetworkType = 'testnet' | 'mainnet';

// 🎯 CHANGE THIS TO SWITCH NETWORKS (must match frontend/lib/networkConfig.ts)
export const NETWORK_TYPE: NetworkType = 'testnet' as NetworkType;

// Network configurations
export const NETWORK_CONFIG = {
  testnet: {
    chainId: 16602,
    name: '0G-Galileo-Testnet',
    rpcUrl: process.env.GALILEO_RPC_URL || 'https://evmrpc-testnet.0g.ai',
    blockExplorer: 'https://chainscan-galileo.0g.ai',
  },
  mainnet: {
    chainId: 16661,
    name: '0G Mainnet',
    rpcUrl: process.env.MAINNET_RPC_URL || 'https://evmrpc.0g.ai',
    blockExplorer: 'https://chainscan.0g.ai',
  },
} as const;

// Contract Addresses by Network
// IMPORTANT: The INFT address here defines which INFTs are authorized for inference
// Only INFTs minted from this specific contract will be validated
// Generic INFTs from other contracts are automatically excluded
export const CONTRACT_ADDRESSES = {
  testnet: {
    INFT: '0x58eA47cc625102585AE8EE0E2da262e9d7Bb4eB0', // Your specific INFT contract for free usage
    DATA_VERIFIER: '0x12deF9560Bca2daa2A7744b6B56836aB006D8C73`',
    ORACLE_STUB: '0x99056b490D1f3ab062C0BeA16135b3F7Eaa8971a',
  },
  mainnet: {
    // 0G Storage Contract Addresses on Mainnet
    FLOW: '0x62D4144dB0F0a6fBBaeb6296c785C71B3D57C526',
    MINE: '0xCd01c5Cd953971CE4C2c9bFb95610236a7F414fe',
    REWARD: '0x457aC76B58ffcDc118AABD6DbC63ff9072880870',
    // Add your deployed INFT contracts on mainnet here
    INFT: process.env.NEXT_PUBLIC_MAINNET_INFT_ADDRESS || '',
    DATA_VERIFIER: process.env.NEXT_PUBLIC_MAINNET_DATA_VERIFIER_ADDRESS || '',
    ORACLE_STUB: process.env.NEXT_PUBLIC_MAINNET_ORACLE_STUB_ADDRESS || '',
  },
};

// Storage Indexer URLs
export const STORAGE_INDEXER = {
  testnet: 'https://indexer-storage-testnet.0g.ai',
  mainnet: 'https://indexer-storage-turbo.0g.ai',
};

// Get current network configuration
export const getCurrentNetwork = () => {
  return NETWORK_CONFIG[NETWORK_TYPE as keyof typeof NETWORK_CONFIG];
};

// Get current contract addresses
export const getCurrentContractAddresses = () => {
  return CONTRACT_ADDRESSES[NETWORK_TYPE as keyof typeof CONTRACT_ADDRESSES];
};

// Get current storage indexer
export const getCurrentStorageIndexer = () => {
  return STORAGE_INDEXER[NETWORK_TYPE as keyof typeof STORAGE_INDEXER];
};

// Get network information
export const getNetworkInfo = () => {
  const network = getCurrentNetwork();
  return {
    networkType: NETWORK_TYPE,
    chainId: network.chainId,
    name: network.name,
    rpcUrl: network.rpcUrl,
    blockExplorer: network.blockExplorer,
    storageIndexer: getCurrentStorageIndexer(),
    isMainnet: NETWORK_TYPE === 'mainnet',
    isTestnet: NETWORK_TYPE === 'testnet',
  };
};

// Validation helper
export const validateNetworkConfig = (): { valid: boolean; errors: string[] } => {
  const errors: string[] = [];
  const contracts = getCurrentContractAddresses();
  
  if (NETWORK_TYPE === 'mainnet') {
    if (!contracts.INFT) {
      errors.push('MAINNET_INFT_ADDRESS not configured');
    }
    if (!contracts.DATA_VERIFIER) {
      errors.push('MAINNET_DATA_VERIFIER_ADDRESS not configured');
    }
    if (!contracts.ORACLE_STUB) {
      errors.push('MAINNET_ORACLE_STUB_ADDRESS not configured');
    }
  }
  
  return {
    valid: errors.length === 0,
    errors,
  };
};

