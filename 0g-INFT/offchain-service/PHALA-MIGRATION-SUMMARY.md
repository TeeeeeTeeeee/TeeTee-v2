# Phala Network RedPill API Migration Summary

## Overview
Successfully migrated the 0G INFT Off-Chain Inference Service from local Ollama to Phala Network's RedPill API. This provides access to confidential AI computing with hardware-level TEE (Trusted Execution Environment) security.

## Key Changes Made

### 1. Environment Configuration Updates

**File: `.env.example`**
- Changed `LLM_PROVIDER` from `ollama` to `phala-redpill`
- Updated `LLM_HOST` from `http://localhost:11434` to `https://api.red-pill.ai`
- Changed `LLM_MODEL` from `llama3.2:3b-instruct-q4_K_M` to `phala/deepseek-r1-70b`
- Increased `LLM_REQUEST_TIMEOUT_MS` from `20000` to `30000` for network API calls
- Added new `REDPILL_API_KEY` environment variable
- Removed `LLM_SEED` (not used in OpenAI-compatible API)

### 2. Code Architecture Changes

**File: `index.ts`**

#### Type Interfaces
- Removed `OllamaGenerateResponse` interface
- Added OpenAI-compatible interfaces:
  - `OpenAIChoice`
  - `OpenAIChatResponse` 
  - `OpenAIChatStreamChunk`
- Updated `LLMConfig` interface to include `apiKey` field

#### API Integration
- **Non-streaming calls**: Replaced Ollama's `/api/generate` with OpenAI-compatible `/v1/chat/completions`
- **Streaming calls**: Updated to handle OpenAI-style Server-Sent Events (SSE) format
- **Authentication**: Added Bearer token authentication using `REDPILL_API_KEY`
- **Request format**: Changed from Ollama's prompt-based format to OpenAI's message-based format

#### Error Handling
- Enhanced error handling for HTTP status codes (401 Unauthorized, 429 Rate Limit)
- Updated error messages to reference RedPill API instead of Ollama
- Added API key validation on startup

#### Circuit Breaker
- Updated circuit breaker name from `LLM-Ollama-Circuit` to `LLM-RedPill-Circuit`
- Modified health check to use proper test prompt instead of "ping"

## Required Setup Steps

### 1. Get RedPill API Key
1. Contact the Phala Team to get a RedPill API key
2. Free, rate-limited developer API keys are available
3. Visit: https://docs.red-pill.ai/get-started/list-models for model information

### 2. Update Environment Variables
Update your `.env` file with:
```bash
# LLM Configuration - Phala Network RedPill Integration
LLM_PROVIDER=phala-redpill
LLM_HOST=https://api.red-pill.ai
LLM_MODEL=phala/deepseek-r1-70b
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=256
LLM_REQUEST_TIMEOUT_MS=30000
LLM_MAX_CONTEXT_QUOTES=25
LLM_DEV_FALLBACK=true
REDPILL_API_KEY=your_actual_redpill_api_key_here
```

### 3. Available Models
Based on documentation, available models include:
- `phala/deepseek-r1-70b` (currently configured)
- `o1-preview`
- Other models listed at https://docs.red-pill.ai/get-started/list-models

## Benefits of Migration

### 1. Enhanced Security
- **TEE Security**: LLM inference runs in GPU Trusted Execution Environment
- **Attestation Reports**: Cryptographic proof of TEE execution
- **Hardware-level Privacy**: Confidential computing with NVIDIA H100 GPUs

### 2. Improved Reliability
- **No Local Dependencies**: No need to run local Ollama instance
- **Professional Infrastructure**: Managed API service with high availability
- **Rate Limiting**: Built-in API rate limiting and scaling

### 3. Advanced Features
- **Remote Attestation**: Get cryptographic proofs of inference integrity
- **Signature Verification**: ECDSA signatures for inference results
- **Model Marketplace**: Access to various pre-configured models

## API Compatibility

The migration maintains **full backward compatibility** with existing function signatures:
- `handleInferRequest()` - Same request/response format
- `handleStreamingInferRequest()` - Same SSE event format
- `performInference()` - Same return interface
- All public APIs remain unchanged

## Testing the Migration

### 1. Health Check
```bash
curl http://localhost:3000/llm/health
```

### 2. Simple Inference
```bash
curl -X POST http://localhost:3000/infer \
  -H "Content-Type: application/json" \
  -d '{"tokenId": 1, "input": "inspire me"}'
```

### 3. Streaming Inference
```bash
curl -X POST http://localhost:3000/infer/stream \
  -H "Content-Type: application/json" \
  -d '{"tokenId": 1, "input": "tell me a story"}'
```

## Troubleshooting

### Common Issues

1. **"REDPILL_API_KEY environment variable is required"**
   - Solution: Add your RedPill API key to the `.env` file

2. **"Invalid RedPill API key"** (HTTP 401)
   - Solution: Verify your API key is correct and active

3. **"RedPill API rate limit exceeded"** (HTTP 429)
   - Solution: Wait before retrying or upgrade your API plan

4. **Connection timeout**
   - Solution: Check internet connectivity; increase `LLM_REQUEST_TIMEOUT_MS` if needed

### Rollback Plan
If you need to rollback to Ollama:
1. Restore original `.env.example` values
2. Use git to revert `index.ts` to previous commit
3. Ensure Ollama is running locally on port 11434

## Next Steps

1. **Set up your RedPill API key** in the environment configuration
2. **Test the integration** with the provided curl commands
3. **Monitor performance** through the health check endpoints
4. **Explore attestation features** for enhanced security verification

The migration is complete and the service is ready to use Phala Network's confidential AI infrastructure!
