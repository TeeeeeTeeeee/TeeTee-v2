import React, { useState, useEffect } from 'react'
import { useAccount, useConnect, useDisconnect, useReadContract } from 'wagmi'
import { useMintINFT, useAuthorizeINFT, useBurnINFT, CONTRACT_ADDRESSES, INFT_ABI } from '../hooks/useINFT'
import { useInference, useStreamingInference } from '../hooks/useInference'

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

export default function INFTPage() {
  const [mounted, setMounted] = useState(false)
  const { address, isConnected, chain } = useAccount()
  const { connect, connectors } = useConnect()
  const { disconnect } = useDisconnect()
  
  // Use custom hooks
  const { mint, isPending: isMintPending, isConfirming: isMintConfirming, isConfirmed: isMintConfirmed, error: mintError } = useMintINFT()
  const { authorize, revoke, isPending: isAuthPending, isConfirming: isAuthConfirming, isConfirmed: isAuthConfirmed, error: authError } = useAuthorizeINFT()
  const { infer, result: inferenceResult, isInferring, error: inferenceError } = useInference()
  const { streamInfer, tokens: streamingTokens, isStreaming, error: streamError } = useStreamingInference()

  // Form states
  const [mintForm, setMintForm] = useState({
    recipient: '',
    encryptedURI: '0g://storage/0xd1a827ee9216ffc63a3a2f239b7615eaecd8ede73b4c3d528c93d243740bff72',
    metadataHash: '0xcee513fb848d9c257cda4909f34d9bdc7097de1d70cb4ef322efd1cb2e6433ed'
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
    const success = await mint(mintForm.recipient, mintForm.encryptedURI, mintForm.metadataHash)
    if (!success && mintError) {
      alert('Mint failed: ' + mintError)
    }
  }

  const handleAuthorize = async (e: React.FormEvent) => {
    e.preventDefault()
    const success = await authorize(authorizeForm.tokenId, authorizeForm.userAddress)
    if (!success && authError) {
      alert('Authorize failed: ' + authError)
    }
  }


  const handleInfer = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inferForm.tokenId || !inferForm.input) {
      alert('Please fill all fields')
      return
    }
    
    try {
      await infer(parseInt(inferForm.tokenId), inferForm.input, address)
    } catch (error: any) {
      console.error('Inference error:', error)
    }
  }

  const handleStreamingInfer = async () => {
    if (!inferForm.tokenId || !inferForm.input) {
      alert('Please fill all fields')
      return
    }
    
    try {
      await streamInfer(parseInt(inferForm.tokenId), inferForm.input, address)
    } catch (error: any) {
      console.error('Streaming error:', error)
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

          {/* Info Box */}
          <div style={{ border: '2px solid #8b5cf6', padding: '15px', marginBottom: '20px', backgroundColor: '#f5f3ff' }}>
            <h3 style={{ marginTop: 0 }}>🔍 How to Identify Your INFTs</h3>
            <p style={{ margin: '10px 0', fontSize: '14px' }}>
              Your INFTs are identified by the <strong>minter address</strong> - the wallet that created them. 
              Only INFTs minted by <strong>your wallet address</strong> will be recognized in the chat interface.
            </p>
            <p style={{ margin: '10px 0', fontSize: '14px' }}>
              Current wallet: <code style={{ background: '#e9d5ff', padding: '2px 6px', borderRadius: '4px' }}>{address?.slice(0, 10)}...{address?.slice(-8)}</code>
            </p>
            <p style={{ margin: '10px 0', fontSize: '14px', color: '#666' }}>
              💡 <strong>Tip:</strong> Each INFT contains encrypted AI data stored on 0G Network. 
              The defaults below use your current encrypted data configuration.
            </p>
            <p style={{ margin: '10px 0', fontSize: '14px', color: '#10b981' }}>
              ⚡ <strong>New:</strong> Gas-optimized epoch-based authorization system - burn operations are now ultra-efficient (O(1) complexity)!
            </p>
          </div>

          {/* Main Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px', marginBottom: '20px' }}>
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
                  disabled={isMintPending || isMintConfirming}
                  style={{ width: '100%', padding: '10px' }}
                >
                  {isMintPending || isMintConfirming ? 'Processing...' : 'Mint INFT'}
                </button>
                
                {isMintConfirmed && (
                  <p style={{ color: 'green', marginTop: '10px' }}>Successfully minted!</p>
                )}
                {mintError && (
                  <p style={{ color: 'red', marginTop: '10px' }}>{mintError}</p>
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
                  disabled={isAuthPending || isAuthConfirming}
                  style={{ width: '100%', padding: '10px' }}
                >
                  {isAuthPending || isAuthConfirming ? 'Processing...' : 'Authorize User'}
                </button>
                
                {isAuthConfirmed && (
                  <p style={{ color: 'green', marginTop: '10px' }}>Authorization successful!</p>
                )}
                {authError && (
                  <p style={{ color: 'red', marginTop: '10px' }}>{authError}</p>
                )}
              </form>
            </div>

            {/* Revoke Usage */}
            <div style={{ border: '1px solid #ff4444', padding: '15px' }}>
              <h2 style={{ color: '#ff4444' }}>❌ Revoke Usage</h2>
              <p style={{ fontSize: '12px', color: '#666', marginTop: '0' }}>
                Remove authorization from a user
              </p>
              
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
                  type="button"
                  onClick={async () => {
                    const success = await revoke(authorizeForm.tokenId, authorizeForm.userAddress)
                    if (!success && authError) {
                      alert('Revoke failed: ' + authError)
                    }
                  }}
                  disabled={isAuthPending || isAuthConfirming}
                  style={{ width: '100%', padding: '10px', backgroundColor: '#ff4444', color: 'white', border: 'none', cursor: 'pointer' }}
                >
                  {isAuthPending || isAuthConfirming ? 'Processing...' : 'Revoke User'}
                </button>
                
                {isAuthConfirmed && (
                  <p style={{ color: 'green', marginTop: '10px' }}>Revoked successfully!</p>
                )}
                {authError && (
                  <p style={{ color: 'red', marginTop: '10px' }}>{authError}</p>
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
                    <button onClick={() => copyToClipboard(inferenceResult.output || '')}>Copy</button>
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
              
              {streamError && (
                <div style={{ marginTop: '10px', padding: '10px', border: '1px solid red', color: 'red' }}>
                  {streamError}
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
