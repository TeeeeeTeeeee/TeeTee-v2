/**
 * Network Configuration for 0G Blockchain
 * Supports both Testnet (Galileo) and Mainnet
 * 
 * 🎯 TO SWITCH NETWORKS:
 * Change NETWORK_TYPE in both:
 * - frontend/lib/networkConfig.ts (this file)
 * - backend/networkConfig.ts
 */

import type { Chain } from '@rainbow-me/rainbowkit';

export type NetworkType = 'testnet' | 'mainnet';

// 🎯 CHANGE THIS TO SWITCH NETWORKS (must match backend/networkConfig.ts)
export const NETWORK_TYPE: NetworkType = 'mainnet' as NetworkType;

// 0G Galileo Testnet Configuration
export const galileoTestnet = {
  id: 16602,
  name: '0G-Galileo-Testnet',
  iconUrl: '/0g.webp',
  iconBackground: '#fff',
  nativeCurrency: { name: '0G', symbol: 'OG', decimals: 18 },
  rpcUrls: {
    default: { http: ['https://evmrpc-testnet.0g.ai/'] },
  },
  blockExplorers: {
    default: { name: '0G-Galileo-Testnet', url: 'https://chainscan-galileo.0g.ai' },
  },
} as const satisfies Chain;

// 0G Mainnet Configuration
export const zgMainnet = {
  id: 16661,
  name: '0G Mainnet',
  iconUrl: '/0g.webp',
  iconBackground: '#fff',
  nativeCurrency: { name: '0G', symbol: '0G', decimals: 18 },
  rpcUrls: {
    default: { http: ['https://evmrpc.0g.ai'] },
    quicknode: { http: ['https://your-quicknode-url.com'] }, // Add your QuickNode URL
    thirdweb: { http: ['https://your-thirdweb-url.com'] }, // Add your ThirdWeb URL
    ankr: { http: ['https://your-ankr-url.com'] }, // Add your Ankr URL
  },
  blockExplorers: {
    default: { name: '0G Mainnet Explorer', url: 'https://chainscan.0g.ai' },
  },
} as const satisfies Chain;

// Contract Addresses by Network
// IMPORTANT: The INFT address here defines which INFTs get free inference usage
// Only INFTs minted from this specific contract will qualify for free INFT-based inference
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

// Get current chain configuration
export const getCurrentChain = (): Chain => {
  return NETWORK_TYPE === 'mainnet' ? zgMainnet : galileoTestnet;
};

// Get current contract addresses
export const getCurrentContractAddresses = () => {
  return CONTRACT_ADDRESSES[NETWORK_TYPE];
};

// Get current storage indexer
export const getCurrentStorageIndexer = () => {
  return STORAGE_INDEXER[NETWORK_TYPE];
};

// Network information
export const getNetworkInfo = () => {
  const chain = getCurrentChain();
  return {
    networkType: NETWORK_TYPE,
    chainId: chain.id,
    name: chain.name,
    rpcUrl: chain.rpcUrls.default.http[0],
    blockExplorer: chain.blockExplorers?.default.url || '',
    storageIndexer: getCurrentStorageIndexer(),
    isMainnet: NETWORK_TYPE === 'mainnet',
    isTestnet: NETWORK_TYPE === 'testnet',
  };
};

