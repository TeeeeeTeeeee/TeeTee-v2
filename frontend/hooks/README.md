# INFT Reusable Hooks & Utilities

This directory contains reusable React hooks and utility functions for interacting with Intelligent NFTs (INFT) on the 0G Network.

## Hooks

### `useINFT.ts`

Custom hooks for INFT contract interactions.

#### `useMintINFT()`

Hook for minting new INFT tokens.

```tsx
import { useMintINFT } from '../hooks/useINFT'

function MyComponent() {
  const { mint, isPending, isConfirming, isConfirmed, error } = useMintINFT()

  const handleMint = async () => {
    const success = await mint(
      '0x...recipient',
      '0g://storage/...',
      '0x...metadataHash'
    )
    
    if (success) {
      console.log('Minted successfully!')
    }
  }

  return (
    <button onClick={handleMint} disabled={isPending || isConfirming}>
      {isPending || isConfirming ? 'Minting...' : 'Mint INFT'}
    </button>
  )
}
```

**Returns:**
- `mint(recipient, encryptedURI, metadataHash)` - Function to mint a new INFT
- `isPending` - Transaction is being submitted
- `isConfirming` - Transaction is being confirmed
- `isConfirmed` - Transaction confirmed successfully
- `error` - Error message if mint failed
- `hash` - Transaction hash

---

#### `useAuthorizeINFT()`

Hook for authorizing and revoking INFT usage permissions.

```tsx
import { useAuthorizeINFT } from '../hooks/useINFT'

function MyComponent() {
  const { authorize, revoke, isPending, isConfirmed, error } = useAuthorizeINFT()

  const handleAuthorize = async () => {
    const success = await authorize(tokenId, userAddress)
    if (success) {
      console.log('User authorized!')
    }
  }

  const handleRevoke = async () => {
    const success = await revoke(tokenId, userAddress)
    if (success) {
      console.log('Authorization revoked!')
    }
  }

  return (
    <>
      <button onClick={handleAuthorize} disabled={isPending}>
        Authorize
      </button>
      <button onClick={handleRevoke} disabled={isPending}>
        Revoke
      </button>
    </>
  )
}
```

**Returns:**
- `authorize(tokenId, userAddress)` - Authorize user for token
- `revoke(tokenId, userAddress)` - Revoke user authorization
- `isPending` - Transaction is being submitted
- `isConfirming` - Transaction is being confirmed
- `isConfirmed` - Transaction confirmed successfully
- `error` - Error message if operation failed
- `hash` - Transaction hash

---

### `useInference.ts`

Custom hooks for running AI inference on INFT tokens.

#### `useInference()`

Hook for running non-streaming inference.

```tsx
import { useInference } from '../hooks/useInference'

function MyComponent() {
  const { infer, result, isInferring, error, reset } = useInference()

  const handleInfer = async () => {
    try {
      await infer(tokenId, 'inspire me', userAddress)
      // Result will be automatically set
    } catch (err) {
      console.error('Inference failed:', err)
    }
  }

  return (
    <div>
      <button onClick={handleInfer} disabled={isInferring}>
        Run Inference
      </button>
      
      {isInferring && <p>Processing...</p>}
      
      {result && (
        <div>
          <p>{result.output}</p>
          <small>Model: {result.metadata?.model}</small>
        </div>
      )}
      
      {error && <p style={{color: 'red'}}>{error}</p>}
    </div>
  )
}
```

**Returns:**
- `infer(tokenId, input, userAddress)` - Run inference
- `result` - Inference result object
- `isInferring` - Whether inference is running
- `error` - Error message if failed
- `reset()` - Reset state

---

#### `useStreamingInference()`

Hook for running streaming inference with real-time token updates.

```tsx
import { useStreamingInference } from '../hooks/useInference'

function MyComponent() {
  const { streamInfer, tokens, fullText, isStreaming, error } = useStreamingInference()

  const handleStream = async () => {
    try {
      await streamInfer(tokenId, 'tell me a story', userAddress)
      // Tokens will be updated in real-time
    } catch (err) {
      console.error('Streaming failed:', err)
    }
  }

  return (
    <div>
      <button onClick={handleStream} disabled={isStreaming}>
        Stream Inference
      </button>
      
      {isStreaming && <p>Streaming... ({tokens.length} tokens)</p>}
      
      {tokens.length > 0 && (
        <div>
          <p>{fullText}</p>
        </div>
      )}
    </div>
  )
}
```

**Returns:**
- `streamInfer(tokenId, input, userAddress)` - Run streaming inference
- `tokens` - Array of received tokens
- `fullText` - Concatenated full text
- `isStreaming` - Whether streaming is active
- `error` - Error message if failed
- `metadata` - Stream metadata (model, provider, etc.)
- `reset()` - Reset state

---

## Utilities

### `inftApi.ts`

Low-level API functions for backend communication.

#### `runInference(tokenId, input, userAddress)`

Direct API call for inference (non-streaming).

```tsx
import { runInference } from '../utils/inftApi'

const result = await runInference(1, 'inspire me', '0x...')
console.log(result.output)
```

#### `runStreamingInference(tokenId, input, userAddress, callbacks)`

Direct API call for streaming inference with callbacks.

```tsx
import { runStreamingInference } from '../utils/inftApi'

await runStreamingInference(1, 'tell a story', '0x...', {
  onStart: (metadata) => console.log('Started', metadata),
  onToken: (token, count) => console.log('Token:', token),
  onComplete: (full, count) => console.log('Done:', full),
  onError: (error) => console.error('Error:', error)
})
```

#### `checkLLMHealth()`

Check if the LLM backend is healthy.

```tsx
import { checkLLMHealth } from '../utils/inftApi'

const health = await checkLLMHealth()
console.log(health.ok ? 'Healthy' : 'Unhealthy')
```

#### `checkServiceHealth()`

Check if the backend service is healthy.

```tsx
import { checkServiceHealth } from '../utils/inftApi'

const health = await checkServiceHealth()
console.log(health.status) // 'healthy'
```

---

## Constants

### Contract Addresses

```tsx
import { CONTRACT_ADDRESSES } from '../hooks/useINFT'

console.log(CONTRACT_ADDRESSES.INFT)           // INFT contract
console.log(CONTRACT_ADDRESSES.DATA_VERIFIER)  // Data Verifier
console.log(CONTRACT_ADDRESSES.ORACLE_STUB)    // Oracle
```

### ABI

```tsx
import { INFT_ABI } from '../hooks/useINFT'
```

---

## Configuration

Set the backend URL in your `.env.local`:

```bash
NEXT_PUBLIC_BACKEND_URL=http://localhost:3001
```

---

## Example: Complete Flow

```tsx
import { useMintINFT, useAuthorizeINFT } from '../hooks/useINFT'
import { useInference } from '../hooks/useInference'

function CompleteExample() {
  const { mint, isConfirmed: mintConfirmed } = useMintINFT()
  const { authorize, isConfirmed: authConfirmed } = useAuthorizeINFT()
  const { infer, result, isInferring } = useInference()

  const runFullFlow = async () => {
    // 1. Mint INFT
    await mint(myAddress, encryptedURI, metadataHash)
    
    // 2. Authorize user
    await authorize(tokenId, myAddress)
    
    // 3. Run inference
    await infer(tokenId, 'inspire me', myAddress)
    
    console.log('Quote:', result?.output)
  }

  return <button onClick={runFullFlow}>Run Full Flow</button>
}
```

---

## Error Handling

All hooks return error messages that you can display to users:

```tsx
const { mint, error } = useMintINFT()

if (error) {
  alert('Mint failed: ' + error)
}
```

Inference hooks throw errors that you can catch:

```tsx
const { infer } = useInference()

try {
  await infer(tokenId, input, address)
} catch (err) {
  console.error('Failed:', err.message)
}
```


