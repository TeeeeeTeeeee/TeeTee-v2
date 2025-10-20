import React, { useState, useEffect } from 'react'
import { useAccount, useConnect, useDisconnect, useReadContract, useWriteContract, useWaitForTransactionReceipt } from 'wagmi'

// 0G Network Configuration
const ZERO_G_NETWORK = {
  id: 16602,
  name: '0G Galileo Testnet',
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
  blockExplorers: {
    default: {
      name: '0G Galileo Block Explorer', 
      url: 'https://chainscan-galileo.0g.ai',
    },
  },
}

// Contract Addresses - Updated deployment on Galileo Chain ID 16602
const CONTRACT_ADDRESSES = {
  INFT: '0xB28dce039dDf7BC39aDE96984c8349DD5C6EcDC1',
  DATA_VERIFIER: '0xeD427A28Ffbd551178e12ab47cDccCc0ea9AE478',
  ORACLE_STUB: '0xc40DC9a5C20A758e2b0659b4CB739a25C2E3723d',
}

// Backend service URL
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001'

// INFT Contract ABI
const INFT_ABI = [
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

export default function INFTPage() {
  const [mounted, setMounted] = useState(false)
  const { address, isConnected, chain } = useAccount()
  const { connect, connectors } = useConnect()
  const { disconnect } = useDisconnect()
  const { writeContract, data: hash, isPending: isWritePending } = useWriteContract()
  const { isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({ hash })

  // Form states
  const [mintForm, setMintForm] = useState({
    recipient: '',
    encryptedURI: '0g://storage/demo-encrypted-quotes',
    metadataHash: '0x1f626cda1593594aea14fcc7edfd015e01fbd0a2eccc3032d553998e0a2a8f4b'
  })
  
  const [authorizeForm, setAuthorizeForm] = useState({
    tokenId: '1',
    userAddress: ''
  })
  
  const [inferForm, setInferForm] = useState({
    tokenId: '1',
    input: ''
  })

  // State management
  const [inferenceResult, setInferenceResult] = useState<any>(null)
  const [isInferring, setIsInferring] = useState(false)
  const [inferenceError, setInferenceError] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingTokens, setStreamingTokens] = useState<string[]>([])
  const [ownedTokens, setOwnedTokens] = useState<number[]>([])

  // Read contract data
  const { data: currentTokenId } = useReadContract({
    address: CONTRACT_ADDRESSES.INFT as `0x${string}`,
    abi: INFT_ABI,
    functionName: 'getCurrentTokenId',
  })

  const { data: userBalance } = useReadContract({
    address: CONTRACT_ADDRESSES.INFT as `0x${string}`,
    abi: INFT_ABI,
    functionName: 'balanceOf',
    args: address ? [address] : undefined,
  })

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (address && !mintForm.recipient) {
      setMintForm(prev => ({ ...prev, recipient: address }))
    }
  }, [address, mintForm.recipient])

  // Fetch owned tokens
  useEffect(() => {
    if (!address) return
    
    const fetchTokens = async () => {
      const tokens: number[] = []
      for (let i = 1; i <= 10; i++) {
        try {
          const response = await fetch('https://evmrpc-testnet.0g.ai', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              jsonrpc: '2.0',
              id: i,
              method: 'eth_call',
              params: [{
                to: CONTRACT_ADDRESSES.INFT,
                data: `0x6352211e${i.toString(16).padStart(64, '0')}`
              }, 'latest']
            })
          })
          const result = await response.json()
          if (result.result && result.result !== '0x') {
            const owner = '0x' + result.result.slice(-40)
            if (owner.toLowerCase() === address.toLowerCase()) {
              tokens.push(i)
            }
          }
        } catch (e) {
          break
        }
        await new Promise(r => setTimeout(r, 100))
      }
      setOwnedTokens(tokens)
    }
    
    fetchTokens()
  }, [address])

  const handleConnect = async () => {
    if (connectors[0]) {
      try {
        if (window.ethereum) {
          await (window as any).ethereum.request({
            method: 'wallet_addEthereumChain',
            params: [{
              chainId: `0x${ZERO_G_NETWORK.id.toString(16)}`,
              chainName: ZERO_G_NETWORK.name,
              nativeCurrency: ZERO_G_NETWORK.nativeCurrency,
              rpcUrls: ZERO_G_NETWORK.rpcUrls.default.http,
              blockExplorerUrls: [ZERO_G_NETWORK.blockExplorers.default.url],
            }],
          })
        }
      } catch (error) {
        console.log('Network already added or user rejected')
      }
      connect({ connector: connectors[0] })
    }
  }

  const handleMint = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!mintForm.recipient || !mintForm.encryptedURI || !mintForm.metadataHash) {
      alert('Please fill all fields')
      return
    }
    
    try {
      await writeContract({
        address: CONTRACT_ADDRESSES.INFT as `0x${string}`,
        abi: INFT_ABI,
        functionName: 'mint',
        args: [mintForm.recipient as `0x${string}`, mintForm.encryptedURI, mintForm.metadataHash as `0x${string}`],
      })
    } catch (error: any) {
      console.error('Mint failed:', error)
      alert('Mint failed: ' + (error?.message || 'Unknown error'))
    }
  }

  const handleAuthorize = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!authorizeForm.tokenId || !authorizeForm.userAddress) {
      alert('Please fill all fields')
      return
    }
    
    try {
      await writeContract({
        address: CONTRACT_ADDRESSES.INFT as `0x${string}`,
        abi: INFT_ABI,
        functionName: 'authorizeUsage',
        args: [BigInt(authorizeForm.tokenId), authorizeForm.userAddress as `0x${string}`],
      })
    } catch (error: any) {
      console.error('Authorize failed:', error)
      alert('Authorize failed: ' + (error?.message || 'Unknown error'))
    }
  }

  const handleInfer = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inferForm.tokenId || !inferForm.input) {
      alert('Please fill all fields')
      return
    }
    
    setIsInferring(true)
    setInferenceError(null)
    setInferenceResult(null)
    
    try {
      const response = await fetch(`${BACKEND_URL}/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tokenId: parseInt(inferForm.tokenId),
          input: inferForm.input,
          user: address,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`)
      }

      const result = await response.json()
      
      if (!result.success) {
        throw new Error(result.error || 'Inference failed')
      }

      setInferenceResult(result)
    } catch (error: any) {
      console.error('Inference error:', error)
      setInferenceError(error.message)
    } finally {
      setIsInferring(false)
    }
  }

  const handleStreamingInfer = async () => {
    if (!inferForm.tokenId || !inferForm.input) {
      alert('Please fill all fields')
      return
    }
    
    setIsStreaming(true)
    setInferenceError(null)
    setStreamingTokens([])
    
    try {
      const response = await fetch(`${BACKEND_URL}/infer/stream`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({
          tokenId: parseInt(inferForm.tokenId),
          input: inferForm.input,
          user: address,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      if (!response.body) {
        throw new Error('Response body is null')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split(/\r?\n\r?\n/)
        buffer = events.pop() || ''

        events.forEach(eventData => {
          const trimmed = eventData.trim()
          if (!trimmed) return
          
          const lines = trimmed.split('\n')
          let eventType = 'message'
          let data = ''
          
          lines.forEach(line => {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim()
            } else if (line.startsWith('data:')) {
              data = line.slice(5).trim()
            }
          })
          
          if (data) {
            try {
              const parsed = JSON.parse(data)
              if (eventType === 'token' && parsed.content) {
                setStreamingTokens(prev => [...prev, parsed.content])
              }
            } catch (e) {
              console.error('Parse error:', e)
            }
          }
        })
      }
    } catch (error: any) {
      console.error('Streaming error:', error)
      setInferenceError(error.message)
    } finally {
      setIsStreaming(false)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    alert('Copied to clipboard!')
  }

  if (!mounted) return null

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <h1>Intelligent NFTs (INFT)</h1>
      <p>ERC-7857 AI-Powered NFTs on 0G Network</p>
      <hr />

      {/* Wallet Connection */}
      <div style={{ marginBottom: '20px' }}>
        {!isConnected ? (
          <button onClick={handleConnect}>
            Connect Wallet
          </button>
        ) : (
          <div>
            <p>Connected: {address?.slice(0, 6)}...{address?.slice(-4)}</p>
            <p>Balance: {userBalance?.toString() || '0'} INFTs</p>
            <button onClick={() => disconnect()}>Disconnect</button>
          </div>
        )}
      </div>

      {isConnected && (
        <>
          {/* Stats */}
          <div style={{ marginBottom: '20px', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
            <div style={{ border: '1px solid #ccc', padding: '10px' }}>
              <h3>Total Supply</h3>
              <p>{currentTokenId?.toString() || '0'}</p>
            </div>
            
            <div style={{ border: '1px solid #ccc', padding: '10px' }}>
              <h3>Your INFTs</h3>
              <p>{ownedTokens.length}</p>
              {ownedTokens.length > 0 && (
                <div>
                  {ownedTokens.map(id => (
                    <span key={id} style={{ marginRight: '5px' }}>#{id}</span>
                  ))}
                </div>
              )}
            </div>
            
            <div style={{ border: '1px solid #ccc', padding: '10px' }}>
              <h3>Network</h3>
              <p>{chain?.name || 'Unknown'}</p>
              <p>Chain ID: {chain?.id || 'N/A'}</p>
            </div>
          </div>

          {/* Main Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
            {/* Mint INFT */}
            <div style={{ border: '1px solid #ccc', padding: '15px' }}>
              <h2>Mint INFT</h2>
              
              <form onSubmit={handleMint}>
                <div style={{ marginBottom: '10px' }}>
                  <label>Recipient Address</label><br />
                  <input
                    type="text"
                    value={mintForm.recipient}
                    onChange={(e) => setMintForm({...mintForm, recipient: e.target.value})}
                    style={{ width: '100%', padding: '5px' }}
                    placeholder="0x..."
                  />
                </div>
                
                <div style={{ marginBottom: '10px' }}>
                  <label>Encrypted URI</label><br />
                  <input
                    type="text"
                    value={mintForm.encryptedURI}
                    onChange={(e) => setMintForm({...mintForm, encryptedURI: e.target.value})}
                    style={{ width: '100%', padding: '5px' }}
                  />
                </div>
                
                <div style={{ marginBottom: '10px' }}>
                  <label>Metadata Hash</label><br />
                  <input
                    type="text"
                    value={mintForm.metadataHash}
                    onChange={(e) => setMintForm({...mintForm, metadataHash: e.target.value})}
                    style={{ width: '100%', padding: '5px' }}
                  />
                </div>
                
                <button
                  type="submit"
                  disabled={isWritePending || isConfirming}
                  style={{ width: '100%', padding: '10px' }}
                >
                  {isWritePending || isConfirming ? 'Processing...' : 'Mint INFT'}
                </button>
                
                {isConfirmed && (
                  <p style={{ color: 'green', marginTop: '10px' }}>Successfully minted!</p>
                )}
              </form>
            </div>

            {/* Authorize Usage */}
            <div style={{ border: '1px solid #ccc', padding: '15px' }}>
              <h2>Authorize Usage</h2>
              
              <form onSubmit={handleAuthorize}>
                <div style={{ marginBottom: '10px' }}>
                  <label>Token ID</label><br />
                  <input
                    type="number"
                    value={authorizeForm.tokenId}
                    onChange={(e) => setAuthorizeForm({...authorizeForm, tokenId: e.target.value})}
                    style={{ width: '100%', padding: '5px' }}
                    min="1"
                  />
                </div>
                
                <div style={{ marginBottom: '10px' }}>
                  <label>User Address</label><br />
                  <input
                    type="text"
                    value={authorizeForm.userAddress}
                    onChange={(e) => setAuthorizeForm({...authorizeForm, userAddress: e.target.value})}
                    style={{ width: '100%', padding: '5px' }}
                    placeholder="0x..."
                  />
                </div>
                
                <button
                  type="submit"
                  disabled={isWritePending || isConfirming}
                  style={{ width: '100%', padding: '10px' }}
                >
                  {isWritePending || isConfirming ? 'Processing...' : 'Authorize User'}
                </button>
                
                {isConfirmed && (
                  <p style={{ color: 'green', marginTop: '10px' }}>Authorization successful!</p>
                )}
              </form>
            </div>
          </div>

          {/* AI Inference */}
          <div style={{ border: '1px solid #ccc', padding: '15px', marginBottom: '20px' }}>
            <h2>AI Inference</h2>
            
            <form onSubmit={handleInfer}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '10px' }}>
                <div>
                  <label>Token ID</label><br />
                  <input
                    type="number"
                    value={inferForm.tokenId}
                    onChange={(e) => setInferForm({...inferForm, tokenId: e.target.value})}
                    style={{ width: '100%', padding: '5px' }}
                    min="1"
                  />
                </div>
                
                <div>
                  <label>Input Prompt</label><br />
                  <input
                    type="text"
                    value={inferForm.input}
                    onChange={(e) => setInferForm({...inferForm, input: e.target.value})}
                    style={{ width: '100%', padding: '5px' }}
                    placeholder="Enter your prompt..."
                  />
                </div>
              </div>
              
              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  type="submit"
                  disabled={isInferring || isStreaming}
                  style={{ flex: 1, padding: '10px' }}
                >
                  {isInferring ? 'Processing...' : 'Run Inference'}
                </button>
                
                <button
                  type="button"
                  onClick={handleStreamingInfer}
                  disabled={isInferring || isStreaming}
                  style={{ flex: 1, padding: '10px' }}
                >
                  {isStreaming ? 'Streaming...' : 'Stream Inference'}
                </button>
              </div>
              
              {/* Results */}
              {inferenceError && (
                <div style={{ marginTop: '10px', padding: '10px', border: '1px solid red', color: 'red' }}>
                  {inferenceError}
                </div>
              )}
              
              {inferenceResult && (
                <div style={{ marginTop: '10px', padding: '10px', border: '1px solid green' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <strong>Result:</strong>
                    <button onClick={() => copyToClipboard(inferenceResult.output)}>Copy</button>
                  </div>
                  <p>{inferenceResult.output}</p>
                  {inferenceResult.metadata && (
                    <div style={{ fontSize: '12px', color: '#666', marginTop: '10px' }}>
                      <div>Model: {inferenceResult.metadata.model}</div>
                      <div>Provider: {inferenceResult.metadata.provider}</div>
                    </div>
                  )}
                </div>
              )}
              
              {streamingTokens.length > 0 && (
                <div style={{ marginTop: '10px', padding: '10px', border: '1px solid blue' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <strong>Streaming Result:</strong>
                    <span>{streamingTokens.length} tokens</span>
                  </div>
                  <p>{streamingTokens.join('')}</p>
                </div>
              )}
            </form>
          </div>

          {/* Contract Info */}
          <div style={{ border: '1px solid #ccc', padding: '15px' }}>
            <h3>Contract Information</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', fontSize: '12px' }}>
              <div>
                <strong>INFT Contract:</strong>
                <p>{CONTRACT_ADDRESSES.INFT}</p>
              </div>
              <div>
                <strong>Data Verifier:</strong>
                <p>{CONTRACT_ADDRESSES.DATA_VERIFIER}</p>
              </div>
              <div>
                <strong>Oracle:</strong>
                <p>{CONTRACT_ADDRESSES.ORACLE_STUB}</p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
