import { useState } from 'react'
import { runInference, runStreamingInference, InferenceResponse } from '../utils/inftApi'

/**
 * Custom hook for running INFT inference
 */
export function useInference() {
  const [result, setResult] = useState<InferenceResponse | null>(null)
  const [isInferring, setIsInferring] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const infer = async (tokenId: number, input: string, userAddress?: string) => {
    setIsInferring(true)
    setError(null)
    setResult(null)

    try {
      const response = await runInference(tokenId, input, userAddress)
      setResult(response)
      return response
    } catch (err: any) {
      const errorMessage = err?.message || 'Inference failed'
      setError(errorMessage)
      throw err
    } finally {
      setIsInferring(false)
    }
  }

  const reset = () => {
    setResult(null)
    setError(null)
    setIsInferring(false)
  }

  return {
    infer,
    result,
    isInferring,
    error,
    reset,
  }
}

/**
 * Custom hook for streaming INFT inference
 */
export function useStreamingInference() {
  const [tokens, setTokens] = useState<string[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [metadata, setMetadata] = useState<any>(null)

  const streamInfer = async (tokenId: number, input: string, userAddress?: string) => {
    setIsStreaming(true)
    setError(null)
    setTokens([])
    setMetadata(null)

    try {
      await runStreamingInference(tokenId, input, userAddress, {
        onStart: (meta) => {
          setMetadata(meta)
        },
        onToken: (token) => {
          setTokens(prev => [...prev, token])
        },
        onComplete: (fullResponse, totalTokens) => {
          console.log('Streaming complete:', { fullResponse, totalTokens })
        },
        onError: (err) => {
          setError(err)
        }
      })
    } catch (err: any) {
      const errorMessage = err?.message || 'Streaming inference failed'
      setError(errorMessage)
      throw err
    } finally {
      setIsStreaming(false)
    }
  }

  const reset = () => {
    setTokens([])
    setError(null)
    setIsStreaming(false)
    setMetadata(null)
  }

  return {
    streamInfer,
    tokens,
    fullText: tokens.join(''),
    isStreaming,
    error,
    metadata,
    reset,
  }
}

