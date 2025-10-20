# INFT Oracle Backend Service

Oracle backend service for Intelligent NFTs (INFT) on 0G Network. This service validates on-chain authorizations, fetches encrypted data from 0G Storage, and performs AI inference using LLM APIs.

## 🚀 Features

- ✅ **On-chain Authorization**: Validates ERC-7857 INFT usage permissions
- ✅ **0G Storage Integration**: Fetches and decrypts data from 0G Storage
- ✅ **LLM Inference**: Performs AI inference using Phala RedPill API
- ✅ **Streaming Support**: Real-time streaming inference with SSE
- ✅ **Oracle Proofs**: Generates cryptographic proofs for inference results
- ✅ **Rate Limiting**: 30 requests per minute per IP
- ✅ **Security**: Input sanitization and validation

## 📋 Prerequisites

- Node.js v18 or higher
- npm or yarn
- 0G Network RPC access
- Phala RedPill API key (get from [red-pill.ai](https://red-pill.ai))

## 🔧 Installation

1. **Install dependencies:**
   ```bash
   cd backend
   npm install
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` file:**
   - Add your `REDPILL_API_KEY`
   - Update contract addresses if needed
   - Configure other settings as required

## 🏃 Running the Service

### Development Mode (with auto-reload)
```bash
npm run dev
```

### Production Mode
```bash
npm run build
npm run serve
```

### Simple Start
```bash
npm start
```

The service will start on `http://localhost:3001` by default.

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "INFT Oracle Backend",
  "timestamp": "2025-01-20T12:00:00.000Z",
  "version": "1.0.0"
}
```

### LLM Health Check
```bash
GET /llm/health
```

**Response:**
```json
{
  "status": "healthy",
  "provider": "phala-redpill",
  "model": "phala/deepseek-r1-70b",
  "configured": true
}
```

### Run Inference
```bash
POST /infer
Content-Type: application/json

{
  "tokenId": 1,
  "input": "Tell me something motivational",
  "user": "0x1234...5678"
}
```

**Response:**
```json
{
  "success": true,
  "output": "The only way to do great work is to love what you do. - Steve Jobs",
  "proof": "{...}",
  "metadata": {
    "tokenId": 1,
    "authorized": true,
    "timestamp": "2025-01-20T12:00:00.000Z",
    "proofHash": "0x...",
    "provider": "phala-redpill",
    "model": "phala/deepseek-r1-70b",
    "temperature": 0.2
  }
}
```

### Streaming Inference
```bash
POST /infer/stream
Content-Type: application/json
Accept: text/event-stream

{
  "tokenId": 1,
  "input": "Tell me something motivational",
  "user": "0x1234...5678"
}
```

**Response (Server-Sent Events):**
```
event: start
data: {"provider":"phala-redpill","model":"phala/deepseek-r1-70b",...}

event: token
data: {"content":"The","tokenCount":1,"done":false}

event: token
data: {"content":" only","tokenCount":2,"done":false}

event: completion
data: {"fullResponse":"The only way...","totalTokens":20,"done":true}
```

## 🔐 Security Features

### Input Validation
- Maximum input length: 500 characters
- Character sanitization (removes control characters)
- Suspicious pattern detection

### Rate Limiting
- 30 requests per minute per IP
- Configurable via middleware

### Authorization
- On-chain validation using INFT contract
- Only authorized users can perform inference

### Data Protection
- Encryption keys never exposed in prompts
- AES-GCM decryption for stored data
- Proof generation with hashes only

## 🧪 Testing

### Test health endpoint:
```bash
curl http://localhost:3001/health
```

### Test LLM health:
```bash
curl http://localhost:3001/llm/health
```

### Test inference (requires authorization):
```bash
curl -X POST http://localhost:3001/infer \
  -H "Content-Type: application/json" \
  -d '{
    "tokenId": 1,
    "input": "motivation",
    "user": "0xYourAddress"
  }'
```

## 📊 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 3001 |
| `FRONTEND_URL` | CORS allowed origin | http://localhost:3000 |
| `GALILEO_RPC_URL` | 0G Network RPC | https://evmrpc-testnet.0g.ai |
| `INFT_CONTRACT_ADDRESS` | INFT contract address | 0x9C3F... |
| `LLM_PROVIDER` | LLM provider name | phala-redpill |
| `LLM_HOST` | LLM API host | https://api.red-pill.ai |
| `LLM_MODEL` | LLM model name | phala/deepseek-r1-70b |
| `LLM_TEMPERATURE` | LLM temperature | 0.2 |
| `LLM_MAX_TOKENS` | Max tokens per response | 256 |
| `LLM_REQUEST_TIMEOUT_MS` | Request timeout | 30000 |
| `REDPILL_API_KEY` | Phala RedPill API key | (required) |

## 🔗 Integration with Frontend

The frontend INFT page (`frontend/pages/inft.tsx`) connects to this backend:

1. **Update frontend environment:**
   ```bash
   # In frontend/.env.local
   NEXT_PUBLIC_BACKEND_URL=http://localhost:3001
   ```

2. **Start both services:**
   ```bash
   # Terminal 1: Backend
   cd backend
   npm start

   # Terminal 2: Frontend
   cd frontend
   npm run dev
   ```

3. **Access the UI:**
   Open `http://localhost:3000/inft` in your browser

## 🛠️ Troubleshooting

### "LLM API key not configured"
- Make sure you've set `REDPILL_API_KEY` in `.env`
- Get your API key from [red-pill.ai](https://red-pill.ai)

### "Access denied. You are not authorized"
- The wallet address must be authorized for the token
- Use the "Authorize Usage" form on the frontend to grant access

### "Failed to fetch encrypted data"
- Check that `0g-INFT/storage/dev-keys.json` exists
- Ensure the encrypted data file is available

### CORS errors
- Update `FRONTEND_URL` in `.env` to match your frontend URL
- Check that the frontend is making requests to the correct backend URL

## 📚 Architecture

```
┌─────────────┐         ┌─────────────────┐         ┌──────────────┐
│   Frontend  │────────▶│  Oracle Backend │────────▶│  0G Network  │
│  (Next.js)  │         │   (Express.js)  │         │  (Galileo)   │
└─────────────┘         └─────────────────┘         └──────────────┘
                                │
                                │
                                ▼
                        ┌─────────────────┐
                        │   Phala RedPill │
                        │   (LLM API)     │
                        └─────────────────┘
```

## 📝 License

MIT

## 🤝 Support

For issues or questions, please open an issue on the GitHub repository.



