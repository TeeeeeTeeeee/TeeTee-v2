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
export const NETWORK_TYPE: NetworkType = 'mainnet' as NetworkType;

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
    INFT: '0x4354cdF31300877Aada37377a30eA14FE167Ec6A', // Your specific INFT contract for free usage
    DATA_VERIFIER: '0x345326bE60b700E3FB9d9Fd7De1ab46B8C80D562`',
    ORACLE_STUB: '0x2Df933ed51aa12Ac987214C32D2420dcD9627abf',
  },
  mainnet: {
    // 0G Storage Contract Addresses on Mainnet
    FLOW: '0x62D4144dB0F0a6fBBaeb6296c785C71B3D57C526',
    MINE: '0xCd01c5Cd953971CE4C2c9bFb95610236a7F414fe',
    REWARD: '0x457aC76B58ffcDc118AABD6DbC63ff9072880870',
    // Add your deployed INFT contracts on mainnet here
    INFT: process.env.NEXT_PUBLIC_MAINNET_INFT_ADDRESS || '0x56776a7878c7d4cC9943b17d91a3E098C77614da',
    DATA_VERIFIER: process.env.NEXT_PUBLIC_MAINNET_DATA_VERIFIER_ADDRESS || '0x8889106de495dC1731a9b60A58817DE6E0142ac0',
    ORACLE_STUB: process.env.NEXT_PUBLIC_MAINNET_ORACLE_STUB_ADDRESS || '0x20F8F585F8E0d3D1fCe7907A3C02AEaA5C924707',
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

