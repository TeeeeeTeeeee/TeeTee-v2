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
    "name": "ownerAuthorizeUsage",
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
  {
    "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
    "name": "burn",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
    "name": "ownerBurn",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [],
    "name": "owner",
    "outputs": [{"internalType": "address", "name": "", "type": "address"}],
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
 * Custom hook for burning INFT tokens (owner-only, transfers to blackhole)
 */
export function useBurnINFT() {
  const { writeContract, data: hash, isPending } = useWriteContract()
  const { isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash })
  const [error, setError] = useState<string | null>(null)

  const burn = async (tokenId: string | number) => {
    setError(null)
    
    if (!tokenId) {
      setError('Token ID is required')
      return false
    }

    try {
      // Use ownerBurn which transfers to blackhole address (simpler than _burn)
      await writeContract({
        address: CONTRACT_ADDRESSES.INFT as `0x${string}`,
        abi: INFT_ABI,
        functionName: 'ownerBurn',
        args: [BigInt(tokenId)],
      })
      return true
    } catch (err: any) {
      const errorMessage = err?.message || 'Unknown error'
      setError(errorMessage)
      console.error('Owner burn failed:', err)
      console.error('Full error details:', err)
      return false
    }
  }

  return {
    burn,
    isPending,
    isConfirming,
    isConfirmed,
    error,
    hash,
  }
}

/**
 * Custom hook to check if a user is authorized for an INFT
 * 
 * IMPORTANT: This hook only checks INFTs from the specific contract address 
 * defined in networkConfig.ts (CONTRACT_ADDRESSES.INFT).
 * Generic INFTs from other contracts are automatically excluded since we're 
 * querying a specific contract address.
 */
export function useCheckINFTAuthorization(
  tokenId: number = 1, 
  userAddress?: string
) {
  // Check if user is authorized for the INFT from our specific contract
  // Only INFTs minted from CONTRACT_ADDRESSES.INFT are valid
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
    contractAddress: CONTRACT_ADDRESSES.INFT, // The specific INFT contract address being checked
    refetch,
  }
}

