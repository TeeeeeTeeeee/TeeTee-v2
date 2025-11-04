/**
 * INFT API Utilities
 * Reusable functions for interacting with the INFT backend service
 */

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001'

export interface InferenceRequest {
  tokenId: number
  input: string
  user?: string
}

export interface InferenceResponse {
  success: boolean
  output?: string
  proof?: string
  error?: string
  metadata?: {
    tokenId: number
    authorized: boolean
    timestamp: string
    proofHash: string
    provider?: string
    model?: string
    temperature?: number
    promptHash?: string
    contextHash?: string
    completionHash?: string
  }
}

export interface StreamingCallback {
  onToken?: (token: string, count: number) => void
  onComplete?: (fullResponse: string, totalTokens: number) => void
  onError?: (error: string) => void
  onStart?: (metadata: any) => void
}

/**
 * Run inference on an INFT token (non-streaming)
 */
export async function runInference(
  tokenId: number,
  input: string,
  userAddress?: string
): Promise<InferenceResponse> {
  if (!input) {
    throw new Error('Input is required')
  }

  // Convert to number and validate - tokenId 0 is valid
  let numericTokenId = Number(tokenId);
  
  // Only default to 1 if tokenId is actually invalid (not a number, NaN, or negative)
  // Note: tokenId 0 is valid!
  if (tokenId === null || tokenId === undefined || isNaN(numericTokenId) || !Number.isFinite(numericTokenId) || numericTokenId < 0) {
    console.warn('[INFT API] Invalid or missing tokenId, defaulting to 1');
    numericTokenId = 1;
  }
  
  // Ensure it's an integer
  numericTokenId = Math.floor(numericTokenId);

  console.log(`[INFT API] Sending inference request - TokenId: ${numericTokenId} (type: ${typeof numericTokenId})`);

  try {
    const response = await fetch(`${BACKEND_URL}/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tokenId: numericTokenId,
        input,
        user: userAddress,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`HTTP ${response.status}: ${errorText}`)
    }

    const result = await response.json()

    if (!result.success) {
      throw new Error(result.error || 'Inference failed')
    }

    return result
  } catch (error: any) {
    console.error('Inference error:', error)
    throw error
  }
}

/**
 * Run streaming inference on an INFT token
 */
export async function runStreamingInference(
  tokenId: number,
  input: string,
  userAddress: string | undefined,
  callbacks: StreamingCallback
): Promise<void> {
  if (!input) {
    throw new Error('Input is required')
  }

  // Convert to number and validate - tokenId 0 is valid
  let numericTokenId = Number(tokenId);
  
  // Only default to 1 if tokenId is actually invalid (not a number, NaN, or negative)
  // Note: tokenId 0 is valid!
  if (tokenId === null || tokenId === undefined || isNaN(numericTokenId) || !Number.isFinite(numericTokenId) || numericTokenId < 0) {
    console.warn('[INFT API] Invalid or missing tokenId, defaulting to 1');
    numericTokenId = 1;
  }
  
  // Ensure it's an integer
  numericTokenId = Math.floor(numericTokenId);

  console.log(`[INFT API] Streaming inference request - TokenId: ${numericTokenId} (type: ${typeof numericTokenId})`);

  try {
    const response = await fetch(`${BACKEND_URL}/infer/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({
        tokenId: numericTokenId,
        input,
        user: userAddress,
      }),
    })

    if (!response.ok) {
      let errorMsg = `HTTP ${response.status}`
      try {
        const errorData = await response.json()
        errorMsg = errorData.error || errorMsg
      } catch {
        // If response is not JSON, use status code
      }
      throw new Error(errorMsg)
    }

    if (!response.body) {
      throw new Error('Response body is null')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    const tokens: string[] = []

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

            if (eventType === 'start' && callbacks.onStart) {
              callbacks.onStart(parsed)
            } else if (eventType === 'token' && parsed.content) {
              tokens.push(parsed.content)
              if (callbacks.onToken) {
                callbacks.onToken(parsed.content, tokens.length)
              }
            } else if (eventType === 'completion' && callbacks.onComplete) {
              callbacks.onComplete(parsed.fullResponse || tokens.join(''), parsed.totalTokens || tokens.length)
            } else if (eventType === 'error' && callbacks.onError) {
              callbacks.onError(parsed.error || 'Unknown error')
            }
          } catch (e) {
            console.error('Parse error:', e)
          }
        }
      })
    }
  } catch (error: any) {
    console.error('Streaming error:', error)
    if (callbacks.onError) {
      callbacks.onError(error.message)
    }
    throw error
  }
}

/**
 * Check LLM health status
 */
export async function checkLLMHealth(): Promise<{
  provider: string
  model: string
  ok: boolean
  latency_ms?: number
  error?: string
}> {
  try {
    const response = await fetch(`${BACKEND_URL}/llm/health`)
    
    if (!response.ok) {
      const data = await response.json()
      return {
        provider: data.provider || 'unknown',
        model: data.model || 'unknown',
        ok: false,
        error: data.error || 'Service unavailable'
      }
    }

    return await response.json()
  } catch (error: any) {
    return {
      provider: 'unknown',
      model: 'unknown',
      ok: false,
      error: error.message
    }
  }
}

/**
 * Check service health
 */
export async function checkServiceHealth(): Promise<{
  status: string
  service: string
  timestamp: string
}> {
  try {
    const response = await fetch(`${BACKEND_URL}/health`)
    
    if (!response.ok) {
      throw new Error('Service unavailable')
    }

    return await response.json()
  } catch (error: any) {
    throw error
  }
}


