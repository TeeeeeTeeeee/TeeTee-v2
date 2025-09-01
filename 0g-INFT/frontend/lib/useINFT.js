import { useAccount, useWriteContract, useReadContract, useWaitForTransactionReceipt, useConfig } from 'wagmi'
import { parseEther, isAddress } from 'viem'
import { waitForTransactionReceipt, writeContract as writeContractAction } from 'wagmi/actions'
import { CONTRACT_ADDRESSES, INFT_ABI, OFFCHAIN_SERVICE_URL } from './constants'

/**
 * Custom hook for interacting with INFT contract
 * Provides functions for mint, authorize, transfer, and inference operations
 */
export function useINFT() {
  const { address: account, chain } = useAccount()
  const config = useConfig()
  const { writeContract, data: hash, isPending: isWritePending, error: writeError } = useWriteContract()
  
  // Wait for transaction confirmation
  const { isLoading: isConfirming, isSuccess: isConfirmed } = useWaitForTransactionReceipt({
    hash,
  })



  // Read functions
  const { data: currentTokenId } = useReadContract({
    address: CONTRACT_ADDRESSES.INFT,
    abi: INFT_ABI,
    functionName: 'getCurrentTokenId',
  })

  const { data: userBalance } = useReadContract({
    address: CONTRACT_ADDRESSES.INFT,
    abi: INFT_ABI,
    functionName: 'balanceOf',
    args: account ? [account] : undefined,
    query: { enabled: !!account }
  })

  // Mint INFT function
  const mintINFT = async (recipient, encryptedURI, metadataHash) => {
    console.log('mintINFT called with:', { recipient, encryptedURI, metadataHash })
    
    if (!isAddress(recipient)) {
      throw new Error('Invalid recipient address')
    }
    
    console.log('About to call writeContract with:', {
      address: CONTRACT_ADDRESSES.INFT,
      functionName: 'mint',
      args: [recipient, encryptedURI, metadataHash],
    })
    
    try {
      const result = await writeContract({
        address: CONTRACT_ADDRESSES.INFT,
        abi: INFT_ABI,
        functionName: 'mint',
        args: [recipient, encryptedURI, metadataHash],
      })
      console.log('writeContract result:', result)
      return result
    } catch (error) {
      console.error('Mint error:', error)
      throw error
    }
  }

  // Authorize usage function
  const authorizeUsage = async (tokenId, userAddress) => {
    if (!isAddress(userAddress)) {
      throw new Error('Invalid user address')
    }
    
    try {
      await writeContract({
        address: CONTRACT_ADDRESSES.INFT,
        abi: INFT_ABI,
        functionName: 'authorizeUsage',
        args: [BigInt(tokenId), userAddress],
      })
    } catch (error) {
      console.error('Authorize error:', error)
      throw error
    }
  }

  // Revoke usage function
  const revokeUsage = async (tokenId, userAddress) => {
    if (!isAddress(userAddress)) {
      throw new Error('Invalid user address')
    }
    
    try {
      await writeContract({
        address: CONTRACT_ADDRESSES.INFT,
        abi: INFT_ABI,
        functionName: 'revokeUsage',
        args: [BigInt(tokenId), userAddress],
      })
    } catch (error) {
      console.error('Revoke error:', error)
      throw error
    }
  }

  // Transfer INFT function (requires sealedKey and proof from TEE)
  const transferINFT = async (from, to, tokenId, sealedKey, proof) => {
    if (!isAddress(from) || !isAddress(to)) {
      throw new Error('Invalid from/to address')
    }
    
    try {
      console.log('📤 Submitting transfer transaction...')
      
      // Submit the transaction and get hash using wagmi action
      const txHash = await writeContractAction(config, {
        address: CONTRACT_ADDRESSES.INFT,
        abi: INFT_ABI,
        functionName: 'transfer',
        args: [from, to, BigInt(tokenId), sealedKey, proof],
        gas: 150000, // Optimized gas limit based on fixed implementation (113K + buffer)
      })
      
      console.log('⏳ Transaction submitted, waiting for confirmation...', txHash)
      
      // Wait for transaction confirmation using the returned hash
      const receipt = await waitForTransactionReceipt(config, {
        hash: txHash,
        timeout: 180000, // 3 minute timeout for complex verification
      })
      
      if (receipt.status === 'reverted') {
        throw new Error(`Transaction failed: ${receipt.transactionHash}`)
      }
      
      console.log('✅ Transaction confirmed:', receipt.transactionHash)
      return receipt
      
    } catch (error) {
      console.error('Transfer error:', error)
      // Extract meaningful error message
      const errorMessage = error.message || error.toString()
      
      if (errorMessage.includes('User denied') || errorMessage.includes('rejected')) {
        throw new Error('Transaction was rejected by user')
      } else if (errorMessage.includes('insufficient funds')) {
        throw new Error('Insufficient funds for transaction')
      } else if (errorMessage.includes('execution reverted') || errorMessage.includes('out of gas')) {
        throw new Error('Transaction failed during execution - may need higher gas limit or check contract state')
      } else {
        throw new Error(`Transfer failed: ${errorMessage}`)
      }
    }
  }

  // Clone INFT function
  const cloneINFT = async (from, to, tokenId, sealedKey, proof) => {
    if (!isAddress(from) || !isAddress(to)) {
      throw new Error('Invalid from/to address')
    }
    
    try {
      await writeContract({
        address: CONTRACT_ADDRESSES.INFT,
        abi: INFT_ABI,
        functionName: 'clone',
        args: [from, to, BigInt(tokenId), sealedKey, proof],
      })
    } catch (error) {
      console.error('Clone error:', error)
      throw error
    }
  }

  // Inference function (calls off-chain service)
  const performInference = async (tokenId, input) => {
    try {
      const response = await fetch(`${OFFCHAIN_SERVICE_URL}/infer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tokenId: parseInt(tokenId),
          input: input,
          user: account, // Optional: specify user for authorization check
        }),
      })

      // Handle different HTTP error codes with user-friendly messages
      if (!response.ok) {
        let errorMessage = 'Inference request failed'
        
        switch (response.status) {
          case 403:
            errorMessage = 'Access denied: You are not authorized to use this INFT. Please check if you have been granted usage permissions for this token.'
            break
          case 404:
            errorMessage = 'INFT not found: The specified token ID does not exist or is not available.'
            break
          case 400:
            errorMessage = 'Invalid request: Please check your input and token ID.'
            break
          case 429:
            errorMessage = 'Rate limit exceeded: Too many requests. Please wait a moment and try again.'
            break
          case 500:
            errorMessage = 'Server error: The inference service is currently unavailable. Please try again later.'
            break
          case 503:
            errorMessage = 'Service unavailable: The inference service is temporarily down for maintenance.'
            break
          default:
            errorMessage = `Inference service error (${response.status}): Please try again or contact support.`
        }
        
        console.warn(`Inference failed with status ${response.status}:`, errorMessage)
        const error = new Error(errorMessage)
        error.handled = true // Mark as handled to prevent dev overlay
        throw error
      }

      const result = await response.json()
      
      if (!result.success) {
        const errorMsg = result.error || 'Inference processing failed'
        console.warn('Inference result error:', errorMsg)
        const error = new Error(errorMsg)
        error.handled = true
        throw error
      }

      return result
    } catch (error) {
      // Don't log the full stack trace for user-facing errors
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        const networkError = 'Network error: Cannot connect to inference service. Please check if the service is running.'
        console.warn('Network error during inference:', error.message)
        const netError = new Error(networkError)
        netError.handled = true
        throw netError
      }
      
      // Re-throw our custom error messages as-is, but mark as handled
      if (!error.handled) {
        console.warn('Inference error:', error.message)
        error.handled = true
      }
      throw error
    }
  }

  // Streaming inference function (calls off-chain service with SSE)
  const performStreamingInference = async (tokenId, input, onToken, onComplete, onError) => {
    try {
      const response = await fetch(`${OFFCHAIN_SERVICE_URL}/infer/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache'
        },
        body: JSON.stringify({
          tokenId: parseInt(tokenId),
          input: input,
          user: account,
        }),
      })

      if (!response.ok) {
        let errorMessage = 'Streaming inference request failed'
        
        switch (response.status) {
          case 403:
            errorMessage = 'Access denied: You are not authorized to use this INFT.'
            break
          case 404:
            errorMessage = 'INFT not found: The specified token ID does not exist.'
            break
          case 400:
            errorMessage = 'Invalid request: Please check your input and token ID.'
            break
          case 429:
            errorMessage = 'Rate limit exceeded: Too many requests. Please wait and try again.'
            break
          case 500:
            errorMessage = 'Server error: The inference service is currently unavailable.'
            break
          case 503:
            errorMessage = 'Service unavailable: The inference service is temporarily down.'
            break
          default:
            errorMessage = `Inference service error (${response.status}): Please try again.`
        }
        
        const error = new Error(errorMessage)
        error.handled = true
        throw error
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break
        
        const chunk = decoder.decode(value, { stream: true })
        // Debug: observe incoming SSE bytes in devtools
        if (process.env.NODE_ENV !== 'production') {
          try { console.debug('[SSE] chunk size', chunk.length) } catch {}
        }
        buffer += chunk

        // Process complete events (handle both \n\n and \r\n\r\n)
        const events = buffer.split(/\r?\n\r?\n/)
        buffer = events.pop() // Keep incomplete event in buffer

        events.forEach(eventData => {
          const trimmed = eventData.trim()
          if (trimmed) {
            if (process.env.NODE_ENV !== 'production') {
              try { console.debug('[SSE] event raw', trimmed) } catch {}
            }
            processSSEEvent(trimmed, onToken, onComplete, onError)
          }
        })
      }
      
    } catch (error) {
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        const networkError = 'Network error: Cannot connect to streaming inference service.'
        const netError = new Error(networkError)
        netError.handled = true
        onError(netError)
        return
      }
      
      if (!error.handled) {
        error.handled = true
      }
      onError(error)
    }
  }

  // Process Server-Sent Events
  const processSSEEvent = (eventData, onToken, onComplete, onError) => {
    const lines = eventData.split('\n')
    let eventType = 'message'
    let data = ''
    let id = ''
    
    lines.forEach(rawLine => {
      const line = rawLine.replace(/\r$/, '') // tolerate CRLF
      if (line.startsWith('event:')) {
        eventType = line.slice(6).trim() // support both 'event:' and 'event: '
      } else if (line.startsWith('data:')) {
        // Some servers may send multiple data lines per event. Join with newlines.
        const piece = line.slice(5).trimStart()
        data = data ? data + piece : piece
      } else if (line.startsWith('id:')) {
        id = line.slice(3).trim()
      }
    })
    
    if (!data) return
    
    try {
      const parsedData = JSON.parse(data)
      
      switch (eventType) {
        case 'start':
          onToken && onToken({
            type: 'start',
            metadata: {
              provider: parsedData.provider,
              model: parsedData.model,
              temperature: parsedData.temperature,
              requestId: parsedData.requestId,
              timestamp: parsedData.timestamp
            }
          })
          break
          
        case 'token':
          onToken && onToken({
            type: 'token',
            content: parsedData.content,
            tokenCount: parsedData.tokenCount,
            done: parsedData.done
          })
          break
          
        case 'completion':
          onComplete && onComplete({
            type: 'completion',
            fullResponse: parsedData.fullResponse,
            totalTokens: parsedData.totalTokens,
            done: parsedData.done
          })
          break
          
        case 'error':
          onError && onError(new Error(parsedData.error))
          break
        default:
          // Fallback: if server didn't set event name, infer from payload
          if (parsedData.type === 'start') {
            onToken && onToken({ type: 'start', metadata: parsedData })
          } else if (parsedData.type === 'token') {
            onToken && onToken({ type: 'token', content: parsedData.content, tokenCount: parsedData.tokenCount, done: parsedData.done })
          } else if (parsedData.type === 'completion') {
            onComplete && onComplete({ type: 'completion', fullResponse: parsedData.fullResponse, totalTokens: parsedData.totalTokens, done: parsedData.done })
          }
      }
      
    } catch (e) {
      onError && onError(new Error(`Parse error: ${data}`))
    }
  }

  // Check if user is authorized for a token
  const checkAuthorization = (tokenId, userAddress) => {
    return useReadContract({
      address: CONTRACT_ADDRESSES.INFT,
      abi: INFT_ABI,
      functionName: 'isAuthorized',
      args: [BigInt(tokenId), userAddress],
      query: { enabled: !!userAddress }
    })
  }

  // Get token owner
  const getTokenOwner = (tokenId) => {
    return useReadContract({
      address: CONTRACT_ADDRESSES.INFT,
      abi: INFT_ABI,
      functionName: 'ownerOf',
      args: [BigInt(tokenId)],
    })
  }

  // Get token metadata
  const getTokenMetadata = (tokenId) => {
    const { data: encryptedURI } = useReadContract({
      address: CONTRACT_ADDRESSES.INFT,
      abi: INFT_ABI,
      functionName: 'encryptedURI',
      args: [BigInt(tokenId)],
    })

    const { data: metadataHash } = useReadContract({
      address: CONTRACT_ADDRESSES.INFT,
      abi: INFT_ABI,
      functionName: 'metadataHash',
      args: [BigInt(tokenId)],
    })

    return { encryptedURI, metadataHash }
  }

  return {
    // Contract info
    currentTokenId,
    userBalance,
    
    // Write functions
    mintINFT,
    authorizeUsage,
    revokeUsage,
    transferINFT,
    cloneINFT,
    performInference,
    performStreamingInference,
    
    // Read functions
    checkAuthorization,
    getTokenOwner,
    getTokenMetadata,
    
    // Transaction state
    hash,
    isWritePending,
    isConfirming,
    isConfirmed,
    writeError,
  }
}
