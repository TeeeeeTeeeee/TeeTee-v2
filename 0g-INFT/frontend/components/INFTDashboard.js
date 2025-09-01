'use client'

import React, { useState, useEffect, useCallback } from 'react'
import { useAccount, useConnect, useDisconnect, useReadContract } from 'wagmi'
import { 
  Wallet, 
  Coins, 
  Users, 
  ArrowUpRight, 
  Copy,
  MessageSquare,
  Settings,
  Plus,
  Shield,
  Sparkles,
  Activity,
  Play,
  Pause,
  ChevronRight,
  ExternalLink
} from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Badge } from './ui/badge'
import { Separator } from './ui/separator'
import { Switch } from './ui/switch'
import { Skeleton } from './ui/skeleton'
import { useINFT } from '../lib/useINFT'
import { addZeroGNetwork } from '../lib/wagmi'
import { CONTRACT_ADDRESSES, INFT_ABI } from '../lib/constants'

/**
 * Component to display user's owned token IDs
 */
function MyTokensList({ userAddress }) {
  const [ownedTokens, setOwnedTokens] = useState([])
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    if (!userAddress) return
    
    const fetchOwnedTokens = async () => {
      try {
        setLoading(true)
        const tokens = []
        
        // Check ownership of tokens 1 through 10 (reasonable range for testing)
        // In production, you'd use events or a more efficient method
        console.log('Checking token ownership for tokens 1-10...')
        
        for (let tokenId = 1; tokenId <= 10; tokenId++) {
          try {
            // Add delay between requests to prevent rate limiting
            if (tokenId > 1) {
              await new Promise(resolve => setTimeout(resolve, 100)) // 100ms delay
            }
            
            const response = await fetch(`https://evmrpc-testnet.0g.ai`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                jsonrpc: '2.0',
                id: tokenId, // Use tokenId as unique request ID
                method: 'eth_call',
                params: [{
                  to: CONTRACT_ADDRESSES.INFT,
                  data: `0x6352211e${tokenId.toString(16).padStart(64, '0')}`  // ownerOf(uint256)
                }, 'latest']
              })
            })
            
            const result = await response.json()
            
            // Check if the call succeeded and returned an owner
            if (result.result && result.result !== '0x' && !result.error) {
              const owner = '0x' + result.result.slice(-40)
              console.log(`Token ${tokenId}: owner = ${owner}`)
              if (owner.toLowerCase() === userAddress.toLowerCase()) {
                console.log(`✅ You own token ${tokenId}`)
                tokens.push(tokenId)
              }
            } else if (result.error) {
              // Token doesn't exist (execution reverted)
              console.log(`Token ${tokenId}: does not exist (${result.error.message})`)
            }
          } catch (error) {
            console.log(`Token ${tokenId} check failed:`, error)
            // If we hit rate limiting, break to avoid further errors
            if (error.message && error.message.includes('rate')) {
              console.log('Rate limit detected, stopping token checks')
              break
            }
          }
        }
        
        setOwnedTokens(tokens)
      } catch (error) {
        console.error('Error fetching owned tokens:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchOwnedTokens()
  }, [userAddress])
  
  if (loading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-4 w-24" />
        <div className="flex gap-2">
          <Skeleton className="h-8 w-16" />
          <Skeleton className="h-8 w-16" />
          <Skeleton className="h-8 w-16" />
        </div>
      </div>
    )
  }
  
  if (ownedTokens.length === 0) {
    return (
      <div className="text-center py-6">
        <div className="text-muted-foreground text-sm">No tokens found</div>
        <div className="text-xs text-muted-foreground mt-1">Mint your first INFT to get started</div>
      </div>
    )
  }
  
  return (
    <div className="space-y-3">
      <div className="text-sm text-muted-foreground">Your INFTs</div>
      <div className="flex flex-wrap gap-2">
        {ownedTokens.map(tokenId => (
          <Badge key={tokenId} variant="secondary" className="px-3 py-1">
            #{tokenId}
          </Badge>
        ))}
      </div>
      <div className="text-xs text-muted-foreground flex items-center gap-1">
        <Activity className="h-3 w-3" />
        Ready for AI inference
      </div>
    </div>
  )
}

/**
 * Main INFT Dashboard Component
 * Provides UI for all INFT operations: mint, authorize, infer, transfer
 */
export default function INFTDashboard() {
  const [mounted, setMounted] = useState(false)
  const { address, isConnected, chain } = useAccount()
  const { connect, connectors } = useConnect()
  const { disconnect } = useDisconnect()

  // Fix hydration error by only rendering after mount
  useEffect(() => {
    setMounted(true)
  }, [])

  // Debug logging
  useEffect(() => {
    console.log('Wallet state:', { 
      address, 
      isConnected, 
      chain: chain?.id,
      chainName: chain?.name 
    })
  }, [address, isConnected, chain])

  // Auto-populate transfer form with connected wallet address
  useEffect(() => {
    if (address && isConnected) {
      setTransferForm(prev => ({
        ...prev,
        from: address
      }))
    }
  }, [address, isConnected])
  
  const {
    currentTokenId,
    userBalance,
    mintINFT,
    authorizeUsage,
    revokeUsage,
    transferINFT,
    performInference,
    performStreamingInference,
    hash,
    isWritePending,
    isConfirming,
    isConfirmed,
    writeError,
  } = useINFT()

  // Form states
  const [mintForm, setMintForm] = useState({
    recipient: '',
    encryptedURI: '',
    metadataHash: ''
  })
  
  const [authorizeForm, setAuthorizeForm] = useState({
    tokenId: '1',
    userAddress: ''
  })
  
  const [inferForm, setInferForm] = useState({
    tokenId: '2',  // Default to token 2 which you own
    input: ''
  })
  
  const [transferForm, setTransferForm] = useState({
    from: '',
    to: '',
    tokenId: '1'
  })

  const [inferenceResult, setInferenceResult] = useState(null)
  const [isInferring, setIsInferring] = useState(false)
  const [inferenceError, setInferenceError] = useState(null)
  
  // Streaming-related state
  const [isStreamingMode, setIsStreamingMode] = useState(false)
  const [streamingTokens, setStreamingTokens] = useState([])
  const [streamingMetadata, setStreamingMetadata] = useState(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingComplete, setStreamingComplete] = useState(false)
  
  // Authorization checker state
  const [authCheckForm, setAuthCheckForm] = useState({
    tokenId: '1'
  })
  const [authCheckResults, setAuthCheckResults] = useState(null)
  const [isCheckingAuth, setIsCheckingAuth] = useState(false)
  const [isTransferring, setIsTransferring] = useState(false)

  // Handle wallet connection
  const handleConnect = async () => {
    if (connectors[0]) {
      // First try to add 0G network
      await addZeroGNetwork()
      // Then connect
      connect({ connector: connectors[0] })
    }
  }

  // Handle mint INFT
  const handleMint = async (e) => {
    e.preventDefault()
    console.log('Mint button clicked', mintForm)
    
    if (!mintForm.recipient || !mintForm.encryptedURI || !mintForm.metadataHash) {
      alert('Please fill all fields')
      return
    }
    
    try {
      console.log('Calling mintINFT with:', {
        recipient: mintForm.recipient,
        encryptedURI: mintForm.encryptedURI,
        metadataHash: mintForm.metadataHash
      })
      
      const result = await mintINFT(
        mintForm.recipient,
        mintForm.encryptedURI,
        mintForm.metadataHash
      )
      
      console.log('Mint transaction submitted successfully, result:', result)
    } catch (error) {
      console.error('Mint failed:', error)
      console.error('Error details:', error.stack)
      alert('Mint failed: ' + error.message)
    }
  }

  // Handle authorize usage
  const handleAuthorize = async (e) => {
    e.preventDefault()
    console.log('Authorize button clicked', authorizeForm)
    
    if (!authorizeForm.tokenId || !authorizeForm.userAddress) {
      alert('Please fill all fields')
      return
    }
    
    try {
      console.log('🔐 Starting authorization process for:', {
        tokenId: authorizeForm.tokenId,
        userAddress: authorizeForm.userAddress
      })
      
      await authorizeUsage(authorizeForm.tokenId, authorizeForm.userAddress)
      
      console.log('✅ Authorization transaction submitted successfully')
      console.log('⏳ Please wait for transaction confirmation below...')
      
      // Clear the form on successful submission
      setAuthorizeForm({ tokenId: '', userAddress: '' })
    } catch (error) {
      console.error('❌ Authorization failed:', error)
      alert('Authorization failed: ' + error.message)
    }
  }

  // Handle inference
  const handleInference = (e) => {
    e.preventDefault()
    if (!inferForm.tokenId || !inferForm.input) {
      alert('Please fill all fields')
      return
    }
    
    // Clear previous results
    setInferenceError(null)
    setInferenceResult(null)
    setStreamingTokens([])
    setStreamingMetadata(null)
    setStreamingComplete(false)
    
    if (isStreamingMode) {
      // Handle streaming inference
      setIsStreaming(true)
      
      const handleToken = (tokenData) => {
        if (tokenData.type === 'start') {
          setStreamingMetadata(tokenData.metadata)
        } else if (tokenData.type === 'token') {
          setStreamingTokens(prev => [...prev, tokenData])
        }
      }
      
      const handleComplete = (completionData) => {
        setStreamingComplete(true)
        setIsStreaming(false)
        setInferenceResult({
          success: true,
          output: completionData.fullResponse,
          metadata: {
            ...streamingMetadata,
            totalTokens: completionData.totalTokens
          }
        })
      }
      
      const handleError = (error) => {
        console.warn('Streaming inference failed:', error.message)
        setInferenceError(error.message)
        setIsStreaming(false)
      }
      
      performStreamingInference(
        inferForm.tokenId, 
        inferForm.input, 
        handleToken, 
        handleComplete, 
        handleError
      ).catch(handleError)
      
    } else {
      // Handle regular inference
      const runInference = async () => {
        setIsInferring(true)
        
        try {
          const result = await performInference(inferForm.tokenId, inferForm.input)
          setInferenceResult(result)
          setInferenceError(null)
        } catch (error) {
          console.warn('Inference failed (handled):', error.message)
          setInferenceError(error.message)
          setInferenceResult(null)
        } finally {
          setIsInferring(false)
        }
      }
      
      runInference().catch((error) => {
        console.warn('Unhandled inference error:', error.message)
        setInferenceError(error.message || 'An unexpected error occurred')
        setIsInferring(false)
      })
    }
  }

  // Handle transfer with TEE mock integration
  const handleTransfer = async (e) => {
    e.preventDefault()
    
    // Validate form inputs
    if (!transferForm.from || !transferForm.to || !transferForm.tokenId) {
      alert('Please fill in all fields (from, to, tokenId)')
      return
    }

    // Prevent transfer during pending operations
    if (isTransferring) {
      return
    }

    // Validate addresses format
    const addressRegex = /^0x[a-fA-F0-9]{40}$/
    if (!addressRegex.test(transferForm.from)) {
      alert('Invalid from address format')
      return
    }
    if (!addressRegex.test(transferForm.to)) {
      alert('Invalid to address format') 
      return
    }

    // Validate token ID
    const tokenIdNum = parseInt(transferForm.tokenId)
    if (isNaN(tokenIdNum) || tokenIdNum < 1) {
      alert('Token ID must be a positive number')
      return
    }

    try {
      setIsTransferring(true)
      console.log('🔄 Starting INFT transfer process...')
      
      // Step 1: Prepare transfer data via our API (TEE mock)
      console.log('📡 Calling transfer API to generate TEE attestation and sealed key...')
      const transferResponse = await fetch('/api/transfer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          from: transferForm.from,
          to: transferForm.to,
          tokenId: transferForm.tokenId,
          originalKey: null // In production, would retrieve from secure storage
        }),
      })

      if (!transferResponse.ok) {
        const errorData = await transferResponse.json()
        throw new Error(errorData.message || 'Failed to prepare transfer data')
      }

      const { transferData } = await transferResponse.json()
      console.log('✅ Transfer data prepared:', {
        tokenId: transferData.tokenId,
        from: transferData.from,
        to: transferData.to,
        sealedKeyLength: transferData.sealedKey.length,
        proofLength: transferData.proof.length
      })

      // Step 2: Execute blockchain transfer
      console.log('⛓️  Executing blockchain transfer...')
      await transferINFT(
        transferData.from,
        transferData.to, 
        transferData.tokenId,
        transferData.sealedKey,
        transferData.proof
      )

      console.log('🎉 Transfer completed successfully!')
      alert(`✅ INFT Token ${transferData.tokenId} transferred successfully from ${transferData.from} to ${transferData.to}!`)
      
      // Reset form
      setTransferForm({
        from: '',
        to: '',
        tokenId: '1'
      })

    } catch (error) {
      console.error('❌ Transfer failed:', error)
      alert(`Transfer failed: ${error.message}`)
    } finally {
      setIsTransferring(false)
    }
  }

  // Authorization check using wagmi hooks (simple and reliable)
  const handleAuthCheck = async (e) => {
    e.preventDefault()
    
    if (!authCheckForm.tokenId) {
      alert('Please enter a token ID')
      return
    }
    
    console.log('🔍 Starting authorization check for token ID:', authCheckForm.tokenId)
    setIsCheckingAuth(true)
    setAuthCheckResults(null)
    
    try {
      // Use wagmi's built-in fetch capabilities
      const { readContract } = await import('viem/actions')
      const { createPublicClient, http } = await import('viem')
      const { defineChain } = await import('viem')
      
      // Define 0G chain
      const zeroGChain = defineChain({
        id: 16601,
        name: '0G Galileo Testnet',
        network: '0g-galileo',
        nativeCurrency: {
          name: '0G',
          symbol: '0G',
          decimals: 18,
        },
        rpcUrls: {
          default: {
            http: ['https://evmrpc-testnet.0g.ai'],
          },
        },
      })
      
      const client = createPublicClient({
        chain: zeroGChain,
        transport: http(),
      })
      
      // Get token owner (with proper error handling for non-existent tokens)
      let tokenOwner
      try {
        tokenOwner = await readContract(client, {
          address: CONTRACT_ADDRESSES.INFT,
          abi: INFT_ABI,
          functionName: 'ownerOf',
          args: [BigInt(authCheckForm.tokenId)],
        })
        console.log('Token owner:', tokenOwner)
      } catch (error) {
        // Check if this is the "token does not exist" error
        if (error.message.includes('0x7e273289') || 
            error.message.includes('ERC721NonexistentToken') ||
            error.message.includes('ERC721: invalid token ID') ||
            error.message.includes('nonexistent token')) {
          
          setAuthCheckResults({
            tokenId: authCheckForm.tokenId,
            error: `Token ID ${authCheckForm.tokenId} does not exist. This token has not been minted yet.`,
            exists: false,
            checkedAt: new Date().toLocaleTimeString()
          })
          setIsCheckingAuth(false)
          return
        }
        
        // Re-throw other errors
        throw error
      }
      
      // Get ALL authorized users using authorizedUsersOf (same as Hardhat script)
      console.log('Getting all authorized users...')
      
      let authorizedUsers = []
      try {
        authorizedUsers = await readContract(client, {
          address: CONTRACT_ADDRESSES.INFT,
          abi: INFT_ABI,
          functionName: 'authorizedUsersOf',
          args: [BigInt(authCheckForm.tokenId)],
        })
        
        console.log('All authorized users from contract:', authorizedUsers)
      } catch (error) {
        console.error('Failed to get authorized users list:', error)
        // Fall back to checking specific addresses
        authorizedUsers = [address, tokenOwner].filter(addr => addr)
      }
      
      // Also include current user and token owner if not in the list
      const allAddressesToCheck = [...authorizedUsers]
      if (address && !allAddressesToCheck.find(addr => addr.toLowerCase() === address.toLowerCase())) {
        allAddressesToCheck.push(address)
      }
      if (tokenOwner && !allAddressesToCheck.find(addr => addr.toLowerCase() === tokenOwner.toLowerCase())) {
        allAddressesToCheck.push(tokenOwner)
      }
      
      const authResults = []
      
      console.log('All addresses to process:', allAddressesToCheck)
      
      for (const checkAddr of allAddressesToCheck) {
        try {
          console.log(`Checking authorization for: ${checkAddr}`)
          
          // For addresses from authorizedUsersOf, they should be authorized
          const isFromAuthorizedList = authorizedUsers.find(addr => addr.toLowerCase() === checkAddr.toLowerCase())
          
          let isAuthorized = false
          if (isFromAuthorizedList) {
            isAuthorized = true
            console.log(`${checkAddr} is in authorized users list: true`)
          } else {
            // Double-check with isAuthorized function
            isAuthorized = await readContract(client, {
              address: CONTRACT_ADDRESSES.INFT,
              abi: INFT_ABI,
              functionName: 'isAuthorized',
              args: [BigInt(authCheckForm.tokenId), checkAddr],
            })
            console.log(`${checkAddr} authorization result:`, isAuthorized)
          }
          
          authResults.push({
            address: checkAddr,
            isAuthorized: !!isAuthorized,
            isOwner: checkAddr.toLowerCase() === tokenOwner.toLowerCase(),
            isCurrentUser: checkAddr.toLowerCase() === address?.toLowerCase()
          })
        } catch (error) {
          console.error(`Failed to check authorization for ${checkAddr}:`, error)
          authResults.push({
            address: checkAddr,
            isAuthorized: false,
            isOwner: checkAddr.toLowerCase() === tokenOwner.toLowerCase(),
            isCurrentUser: checkAddr.toLowerCase() === address?.toLowerCase(),
            error: error.message
          })
        }
      }
      
      setAuthCheckResults({
        tokenId: authCheckForm.tokenId,
        tokenOwner,
        authorizations: authResults,
        checkedAt: new Date().toLocaleTimeString()
      })
      
      console.log('✅ Authorization check completed:', authResults)
      
    } catch (error) {
      console.error('❌ Authorization check failed:', error)
      alert('Authorization check failed: ' + error.message)
    } finally {
      setIsCheckingAuth(false)
    }
  }

  // Show loading until component is mounted (fixes hydration error)
  if (!mounted) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <div className="text-center space-y-4">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-muted border-t-foreground mx-auto"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    )
  }

  if (!isConnected) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <div className="w-full max-w-md space-y-8">
          <div className="text-center space-y-4">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-accent mb-4">
              <Sparkles className="h-8 w-8 text-foreground" />
            </div>
            <div className="space-y-2">
              <h1 className="text-2xl font-semibold tracking-tight">0G INFT</h1>
              <p className="text-muted-foreground">
                Intelligent NFTs on 0G Galileo testnet
              </p>
            </div>
          </div>
          
          <Card className="border-border">
            <CardContent className="pt-6">
              <Button onClick={handleConnect} className="w-full" size="lg">
                <Wallet className="mr-2 h-4 w-4" />
                Connect Wallet
              </Button>
            </CardContent>
          </Card>
          
          <div className="text-center">
            <p className="text-xs text-muted-foreground">
              MetaMask or Web3 wallet required
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <header className="mb-12">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="inline-flex items-center justify-center w-10 h-10 rounded-xl bg-accent">
                <Sparkles className="h-5 w-5 text-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-semibold tracking-tight">0G INFT</h1>
                <p className="text-sm text-muted-foreground">Intelligent NFTs</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="hidden sm:block text-right space-y-1">
                <div className="text-xs text-muted-foreground">Connected</div>
                <div className="font-mono text-sm">{address?.slice(0, 6)}...{address?.slice(-4)}</div>
                <div className="text-xs text-muted-foreground">
                  {userBalance?.toString() || '0'} INFTs
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={() => disconnect()}>
                Disconnect
              </Button>
            </div>
          </div>
        </header>

        {/* Status Overview */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
          <Card className="border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent">
                  <Coins className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-2xl font-semibold">{currentTokenId?.toString() || '0'}</div>
                  <div className="text-xs text-muted-foreground">Next Token</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent">
                  <Users className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-2xl font-semibold">{userBalance?.toString() || '0'}</div>
                  <div className="text-xs text-muted-foreground">Owned INFTs</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent">
                  <Activity className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-sm font-medium">0G Galileo</div>
                  <div className="text-xs text-muted-foreground">Testnet</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* My Tokens Section */}
        {userBalance && userBalance > 0 && (
          <Card className="mb-8 border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Coins className="h-4 w-4" />
                My INFTs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <MyTokensList userAddress={address} />
            </CardContent>
          </Card>
        )}

        {/* Main Operations */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Mint INFT */}
          <Card className="border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Plus className="h-4 w-4" />
                Mint INFT
              </CardTitle>
              <CardDescription className="text-xs">
                Create a new Intelligent NFT
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <form onSubmit={handleMint} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="recipient" className="text-xs font-medium">Recipient</Label>
                  <Input
                    id="recipient"
                    value={mintForm.recipient}
                    onChange={(e) => setMintForm({...mintForm, recipient: e.target.value})}
                    placeholder="0x..."
                    className="text-sm"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="encryptedURI" className="text-xs font-medium">Encrypted URI</Label>
                  <Input
                    id="encryptedURI"
                    value={mintForm.encryptedURI}
                    onChange={(e) => setMintForm({...mintForm, encryptedURI: e.target.value})}
                    placeholder="0g://storage/..."
                    className="text-sm"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="metadataHash" className="text-xs font-medium">Metadata Hash</Label>
                  <Input
                    id="metadataHash"
                    value={mintForm.metadataHash}
                    onChange={(e) => setMintForm({...mintForm, metadataHash: e.target.value})}
                    placeholder="0x..."
                    className="text-sm"
                  />
                </div>
                <Button 
                  type="submit" 
                  className="w-full" 
                  size="sm"
                  disabled={isWritePending || isConfirming}
                >
                  {isWritePending || isConfirming ? 'Minting...' : 'Mint INFT'}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Authorize Usage */}
          <Card className="border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Settings className="h-4 w-4" />
                Authorize Access
              </CardTitle>
              <CardDescription className="text-xs">
                Grant inference permissions to other users
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <form onSubmit={handleAuthorize} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="authTokenId" className="text-xs font-medium">Token ID</Label>
                  <Input
                    id="authTokenId"
                    value={authorizeForm.tokenId}
                    onChange={(e) => setAuthorizeForm({...authorizeForm, tokenId: e.target.value})}
                    placeholder="1"
                    className="text-sm"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="userAddress" className="text-xs font-medium">User Address</Label>
                  <Input
                    id="userAddress"
                    value={authorizeForm.userAddress}
                    onChange={(e) => setAuthorizeForm({...authorizeForm, userAddress: e.target.value})}
                    placeholder="0x..."
                    className="text-sm"
                  />
                </div>
                <Button 
                  type="submit" 
                  className="w-full" 
                  size="sm"
                  disabled={isWritePending || isConfirming}
                >
                  {isWritePending || isConfirming ? 'Authorizing...' : 'Authorize User'}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Inference */}
          <Card className="border-border lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <MessageSquare className="h-4 w-4" />
                AI Inference
              </CardTitle>
              <CardDescription className="text-xs">
                Perform AI inference using your INFTs
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <form onSubmit={handleInference} className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="inferTokenId" className="text-xs font-medium">Token ID</Label>
                    <Input
                      id="inferTokenId"
                      value={inferForm.tokenId}
                      onChange={(e) => setInferForm({...inferForm, tokenId: e.target.value})}
                      placeholder="1"
                      className="text-sm"
                    />
                  </div>
                  <div className="flex items-end">
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={isStreamingMode}
                        onCheckedChange={setIsStreamingMode}
                        id="streaming-mode"
                      />
                      <Label htmlFor="streaming-mode" className="text-xs font-medium">
                        Stream
                      </Label>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="input" className="text-xs font-medium">Prompt</Label>
                  <Input
                    id="input"
                    value={inferForm.input}
                    onChange={(e) => setInferForm({...inferForm, input: e.target.value})}
                    placeholder="inspire me"
                    className="text-sm"
                  />
                </div>
                
                <Button 
                  type="submit" 
                  className="w-full" 
                  size="sm"
                  disabled={isInferring || isStreaming}
                >
                  {isStreaming ? (
                    <>
                      <Activity className="mr-2 h-3 w-3 animate-pulse" />
                      Streaming...
                    </>
                  ) : isInferring ? (
                    <>
                      <Activity className="mr-2 h-3 w-3 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-3 w-3" />
                      {isStreamingMode ? 'Start Stream' : 'Run Inference'}
                    </>
                  )}
                </Button>
                
                {/* Streaming Metadata Display */}
                {streamingMetadata && (
                  <div className="p-3 bg-accent/50 border border-border rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="h-3 w-3" />
                      <span className="text-xs font-medium">Streaming Session</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                      <div>Provider: <span className="font-mono">{streamingMetadata.provider}</span></div>
                      <div>Model: <span className="font-mono">{streamingMetadata.model}</span></div>
                      <div>Temperature: <span className="font-mono">{streamingMetadata.temperature}</span></div>
                      <div>Started: <span className="font-mono">{new Date(streamingMetadata.timestamp).toLocaleTimeString()}</span></div>
                    </div>
                  </div>
                )}

                {/* Streaming Tokens Display */}
                {(isStreaming || streamingTokens.length > 0) && (
                  <div className="p-4 bg-accent/30 border border-border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        {isStreaming ? (
                          <>
                            <Activity className="h-3 w-3 animate-pulse" />
                            <span className="text-xs font-medium">Streaming</span>
                          </>
                        ) : (
                          <>
                            <span className="text-xs font-medium">Complete</span>
                          </>
                        )}
                      </div>
                      {streamingTokens.length > 0 && (
                        <Badge variant="secondary" className="text-xs">
                          {streamingTokens.length} tokens
                        </Badge>
                      )}
                    </div>
                    
                    <div className="bg-card p-3 border border-border rounded min-h-[80px] max-h-60 overflow-y-auto">
                      <div className="whitespace-pre-wrap text-sm leading-relaxed">
                        {streamingTokens.map((token, index) => (
                          <span 
                            key={index} 
                            className="inline-block animate-in fade-in duration-200"
                            style={{ animationDelay: `${index * 50}ms` }}
                          >
                            {token.content}
                          </span>
                        ))}
                        {isStreaming && (
                          <span className="inline-block w-2 h-4 bg-foreground animate-pulse ml-1"></span>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Inference Success Result */}
                {inferenceResult && (
                  <div className="p-4 bg-accent/30 border border-border rounded-lg">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="flex items-center justify-center w-5 h-5 rounded-full bg-green-500/20">
                        <span className="text-green-600 text-xs">✓</span>
                      </div>
                      <span className="text-xs font-medium">Inference Complete</span>
                    </div>
                    <div className="bg-card p-3 border border-border rounded">
                      <p className="text-sm whitespace-pre-wrap leading-relaxed">{inferenceResult.output}</p>
                    </div>
                    
                    {/* Enhanced Metadata Display */}
                    {inferenceResult.metadata && (
                      <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                        {inferenceResult.metadata.provider && (
                          <div>Provider: <span className="font-mono">{inferenceResult.metadata.provider}</span></div>
                        )}
                        {inferenceResult.metadata.model && (
                          <div>Model: <span className="font-mono">{inferenceResult.metadata.model}</span></div>
                        )}
                        {inferenceResult.metadata.temperature && (
                          <div>Temperature: <span className="font-mono">{inferenceResult.metadata.temperature}</span></div>
                        )}
                        {inferenceResult.metadata.totalTokens && (
                          <div>Tokens: <span className="font-mono">{inferenceResult.metadata.totalTokens}</span></div>
                        )}
                      </div>
                    )}
                    
                    {inferenceResult.proof && (
                      <div className="text-xs text-muted-foreground mt-2">
                        Proof: <span className="font-mono">{inferenceResult.proof.slice(0, 50)}...</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Inference Error Display */}
                {inferenceError && (
                  <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="flex items-center justify-center w-5 h-5 rounded-full bg-destructive/20">
                        <span className="text-destructive text-xs">✕</span>
                      </div>
                      <span className="text-xs font-medium text-destructive">Inference Failed</span>
                    </div>
                    <p className="text-sm text-destructive/80 mb-3">{inferenceError}</p>
                    {inferenceError.includes('not authorized') && (
                      <div className="p-3 bg-accent/50 border border-border rounded text-xs text-muted-foreground">
                        <strong>Need access?</strong> Ask the token owner to authorize your address.
                      </div>
                    )}
                    {inferenceError.includes('Network error') && (
                      <div className="p-3 bg-accent/50 border border-border rounded text-xs text-muted-foreground">
                        <strong>Troubleshooting:</strong> Make sure the off-chain service is running on localhost:3000
                      </div>
                    )}
                  </div>
                )}
              </form>
            </CardContent>
          </Card>

          {/* Transfer */}
          <Card className="border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <ArrowUpRight className="h-4 w-4" />
                Transfer INFT
              </CardTitle>
              <CardDescription className="text-xs">
                Transfer ownership with TEE attestation
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <form onSubmit={handleTransfer} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="fromAddress" className="text-xs font-medium">From</Label>
                  <Input
                    id="fromAddress"
                    value={transferForm.from}
                    onChange={(e) => setTransferForm({...transferForm, from: e.target.value})}
                    placeholder="0x..."
                    className="text-sm"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="toAddress" className="text-xs font-medium">To</Label>
                  <Input
                    id="toAddress"
                    value={transferForm.to}
                    onChange={(e) => setTransferForm({...transferForm, to: e.target.value})}
                    placeholder="0x..."
                    className="text-sm"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="transferTokenId" className="text-xs font-medium">Token ID</Label>
                  <Input
                    id="transferTokenId"
                    value={transferForm.tokenId}
                    onChange={(e) => setTransferForm({...transferForm, tokenId: e.target.value})}
                    placeholder="1"
                    className="text-sm"
                  />
                </div>
                <Button type="submit" className="w-full" size="sm" disabled={isTransferring}>
                  {isTransferring ? 'Transferring...' : 'Transfer INFT'}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Authorization Checker */}
          <Card className="border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Shield className="h-4 w-4" />
                Check Access
              </CardTitle>
              <CardDescription className="text-xs">
                View authorization status for any token
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <form onSubmit={handleAuthCheck} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="checkTokenId" className="text-xs font-medium">Token ID</Label>
                  <Input
                    id="checkTokenId"
                    type="number"
                    value={authCheckForm.tokenId}
                    onChange={(e) => setAuthCheckForm({
                      ...authCheckForm,
                      tokenId: e.target.value
                    })}
                    placeholder="Enter token ID"
                    min="1"
                    className="text-sm"
                  />
                </div>
                
                <Button 
                  type="submit" 
                  className="w-full" 
                  size="sm"
                  disabled={isCheckingAuth}
                >
                  {isCheckingAuth ? 'Checking...' : 'Check Access'}
                </Button>
              </form>

              {/* Authorization Results */}
              {authCheckResults && (
                <div className="space-y-4">
                  <Separator />
                  <div>
                    <div className="flex items-center gap-2 mb-4">
                      <Shield className="h-4 w-4" />
                      <span className="text-sm font-medium">
                        Token #{authCheckResults.tokenId}
                      </span>
                    </div>
                    
                    {authCheckResults.error ? (
                      <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
                        <div className="text-xs font-medium text-destructive mb-2">Error</div>
                        <div className="text-xs text-destructive/80 mb-2">{authCheckResults.error}</div>
                        {authCheckResults.exists === false && (
                          <div className="text-xs text-muted-foreground">
                            Try minting a new token or check a different token ID
                          </div>
                        )}
                        <div className="text-xs text-muted-foreground mt-2">
                          Checked at {authCheckResults.checkedAt}
                        </div>
                      </div>
                    ) : authCheckResults.tokenOwner && (
                      <div className="mb-4 p-3 bg-accent/50 border border-border rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">Owner</div>
                        <div className="text-xs font-mono break-all">
                          {authCheckResults.tokenOwner}
                        </div>
                      </div>
                    )}

                    {!authCheckResults.error && (
                      <>
                        <div className="space-y-3">
                          <div className="text-xs font-medium">Access Status</div>
                          <div className="text-xs text-muted-foreground bg-accent/30 p-2 rounded">
                            <strong>Note:</strong> Token owners must explicitly authorize themselves for inference access.
                          </div>
                          {!authCheckResults.authorizations || authCheckResults.authorizations.length === 0 ? (
                            <div className="text-xs text-muted-foreground">No addresses checked</div>
                          ) : (
                            <div className="space-y-2">
                              {authCheckResults.authorizations.map((auth, index) => (
                                <div
                                  key={index}
                                  className={`p-3 rounded-lg border ${
                                auth.isAuthorized 
                                  ? 'bg-green-500/10 border-green-500/20' 
                                  : 'bg-accent/30 border-border'
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex-1">
                                  <div className="text-xs font-mono break-all">
                                    {auth.address}
                                  </div>
                                  <div className="flex items-center gap-2 mt-1">
                                    {auth.isOwner && (
                                      <Badge variant="secondary" className="text-xs h-5">
                                        Owner
                                      </Badge>
                                    )}
                                    {auth.isCurrentUser && (
                                      <Badge variant="outline" className="text-xs h-5">
                                        You
                                      </Badge>
                                    )}
                                  </div>
                                </div>
                                <div className="ml-4">
                                  {auth.isAuthorized ? (
                                    <span className="text-xs text-green-600 font-medium">✓ Authorized</span>
                                  ) : (
                                    <span className="text-xs text-muted-foreground">✕ Not Authorized</span>
                                  )}
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                          )}
                          
                          <div className="text-xs text-muted-foreground">
                            Checked at {authCheckResults.checkedAt}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Transaction Status */}
        {(isWritePending || isConfirming || isConfirmed) && (
          <Card className="mt-8 border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Activity className="h-4 w-4" />
                Transaction Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {isWritePending && (
                <div className="flex items-center gap-3">
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-muted border-t-foreground"></div>
                  <span className="text-sm">Transaction submitted...</span>
                </div>
              )}
              {isConfirming && (
                <div className="flex items-center gap-3">
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-muted border-t-foreground"></div>
                  <span className="text-sm">Waiting for confirmation...</span>
                </div>
              )}
              {isConfirmed && (
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="h-4 w-4 rounded-full bg-green-500 flex items-center justify-center">
                      <span className="text-white text-xs">✓</span>
                    </div>
                    <span className="text-sm font-medium">Transaction confirmed</span>
                  </div>
                  {hash && (
                    <div className="p-3 bg-accent/50 border border-border rounded">
                      <div className="text-xs text-muted-foreground mb-1">Transaction Hash</div>
                      <div className="text-xs font-mono break-all mb-2">{hash}</div>
                      <a 
                        href={`https://chainscan-galileo.0g.ai/tx/${hash}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-xs text-foreground hover:text-foreground/80"
                      >
                        View on Explorer
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Error Display */}
        {writeError && (
          <Card className="mt-8 border-destructive/20 bg-destructive/5">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base text-destructive">
                <span className="text-destructive">✕</span>
                Transaction Failed
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="p-3 bg-destructive/10 border border-destructive/20 rounded text-sm text-destructive">
                {writeError.message}
              </div>
              <div className="text-xs text-muted-foreground">
                Common causes: Insufficient gas, network issues, or user rejected transaction
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
