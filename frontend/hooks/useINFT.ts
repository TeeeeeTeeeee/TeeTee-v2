import { useWriteContract, useWaitForTransactionReceipt, useReadContract } from 'wagmi'
import { useState } from 'react'
import { getCurrentContractAddresses } from '../lib/networkConfig'

// Get Contract Addresses based on current network (testnet or mainnet)
export const CONTRACT_ADDRESSES = getCurrentContractAddresses()

// INFT Contract ABI
export const INFT_ABI = [
  {
    "inputs": [
      {"internalType": "address", "name": "to", "type": "address"},
      {"internalType": "string", "name": "_encryptedURI", "type": "string"},
      {"internalType": "bytes32", "name": "_metadataHash", "type": "bytes32"}
    ],
    "name": "mint",
    "outputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {"internalType": "uint256", "name": "tokenId", "type": "uint256"},
      {"internalType": "address", "name": "user", "type": "address"}
    ],
    "name": "authorizeUsage",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {"internalType": "uint256", "name": "tokenId", "type": "uint256"},
      {"internalType": "address", "name": "user", "type": "address"}
    ],
    "name": "revokeUsage",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {"internalType": "uint256", "name": "tokenId", "type": "uint256"},
      {"internalType": "address", "name": "user", "type": "address"}
    ],
    "name": "isAuthorized",
    "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
    "name": "ownerOf",
    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [{"internalType": "address", "name": "owner", "type": "address"}],
    "name": "balanceOf",
    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "getCurrentTokenId",
    "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
    "name": "encryptedURI",
    "outputs": [{"internalType": "string", "name": "", "type": "string"}],
    "stateMutability": "view",
    "type": "function"
  },
]

/**
 * Custom hook for minting INFT tokens
 */
export function useMintINFT() {
  const { writeContract, data: hash, isPending } = useWriteContract()
  const { isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash })
  const [error, setError] = useState<string | null>(null)

  const mint = async (recipient: string, encryptedURI: string, metadataHash: string) => {
    setError(null)
    
    if (!recipient || !encryptedURI || !metadataHash) {
      setError('All fields are required')
      return false
    }

    try {
      await writeContract({
        address: CONTRACT_ADDRESSES.INFT as `0x${string}`,
        abi: INFT_ABI,
        functionName: 'mint',
        args: [recipient as `0x${string}`, encryptedURI, metadataHash as `0x${string}`],
      })
      return true
    } catch (err: any) {
      const errorMessage = err?.message || 'Unknown error'
      setError(errorMessage)
      console.error('Mint failed:', err)
      return false
    }
  }

  return {
    mint,
    isPending,
    isConfirming,
    isConfirmed,
    error,
    hash,
  }
}

/**
 * Custom hook for authorizing INFT usage
 */
export function useAuthorizeINFT() {
  const { writeContract, data: hash, isPending } = useWriteContract()
  const { isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash })
  const [error, setError] = useState<string | null>(null)

  const authorize = async (tokenId: string | number, userAddress: string) => {
    setError(null)
    
    if (!tokenId || !userAddress) {
      setError('Token ID and user address are required')
      return false
    }

    try {
      await writeContract({
        address: CONTRACT_ADDRESSES.INFT as `0x${string}`,
        abi: INFT_ABI,
        functionName: 'authorizeUsage',
        args: [BigInt(tokenId), userAddress as `0x${string}`],
      })
      return true
    } catch (err: any) {
      const errorMessage = err?.message || 'Unknown error'
      setError(errorMessage)
      console.error('Authorize failed:', err)
      return false
    }
  }

  const revoke = async (tokenId: string | number, userAddress: string) => {
    setError(null)
    
    if (!tokenId || !userAddress) {
      setError('Token ID and user address are required')
      return false
    }

    try {
      await writeContract({
        address: CONTRACT_ADDRESSES.INFT as `0x${string}`,
        abi: INFT_ABI,
        functionName: 'revokeUsage',
        args: [BigInt(tokenId), userAddress as `0x${string}`],
      })
      return true
    } catch (err: any) {
      const errorMessage = err?.message || 'Unknown error'
      setError(errorMessage)
      console.error('Revoke failed:', err)
      return false
    }
  }

  return {
    authorize,
    revoke,
    isPending,
    isConfirming,
    isConfirmed,
    error,
    hash,
  }
}

/**
 * Custom hook to check if a user is authorized for an INFT
 */
export function useCheckINFTAuthorization(tokenId: number = 1, userAddress?: string) {
  const { data: isAuthorized, refetch } = useReadContract({
    address: CONTRACT_ADDRESSES.INFT as `0x${string}`,
    abi: INFT_ABI,
    functionName: 'isAuthorized',
    args: userAddress ? [BigInt(tokenId), userAddress as `0x${string}`] : undefined,
    query: {
      enabled: !!userAddress,
    },
  })

  return {
    isAuthorized: !!isAuthorized,
    refetch,
  }
}

