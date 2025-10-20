# 🎨 Intelligent NFTs (INFT) - Complete System

> **ERC-7857 AI-Powered NFTs on 0G Network**

This is a complete implementation of Intelligent NFTs (INFT) that combines blockchain-based ownership with AI inference capabilities. The system allows you to mint NFTs that contain encrypted AI model data and perform verifiable AI inference through an oracle backend.

## 🌟 What is INFT?

**Intelligent NFTs** are a new standard (ERC-7857) that enables:

- 🤖 **AI Agent Ownership**: Each NFT represents an AI agent with unique capabilities
- 🔐 **Encrypted Storage**: AI model data stored securely on 0G Storage
- ✅ **On-chain Authorization**: Control who can use your AI without transferring ownership
- 🔍 **Verifiable Inference**: Cryptographic proofs for all AI operations
- 💡 **Transferable Intelligence**: Transfer complete AI agents between owners

## 📁 Project Structure

```
TeeTee-v2/
├── frontend/
│   └── pages/
│       └── inft.tsx              # 🎨 INFT UI (main interface)
│
├── backend/
│   ├── index.ts                  # 🔮 Oracle backend service
│   ├── package.json              # Backend dependencies
│   ├── tsconfig.json             # TypeScript config
│   └── README.md                 # Backend documentation
│
├── 0g-INFT/                      # 📚 Reference implementation
│   ├── contracts/                # Smart contracts (deployed)
│   ├── storage/                  # Encryption utilities
│   └── offchain-service/         # Original service
│
├── start-inft.sh                 # 🚀 Quick start (Mac/Linux)
├── start-inft.bat                # 🚀 Quick start (Windows)
├── INFT-SETUP-GUIDE.md           # 📖 Detailed setup guide
└── INFT-QUICK-REFERENCE.md       # 📋 Quick reference card
```

## 🚀 Quick Start (Choose One)

### Option 1: Automated Start (Recommended)

**Windows:**
```bash
start-inft.bat
```

**Mac/Linux:**
```bash
chmod +x start-inft.sh
./start-inft.sh
```

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
npm install
npm start
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### Option 3: Root Package Scripts

```bash
# Install all dependencies
npm run install:all

# Start both services
npm run dev
```

Then open: **http://localhost:3000/inft**

## ⚙️ Configuration

### 1. Backend Setup

Create `backend/.env`:
```env
# Get your API key from https://red-pill.ai
REDPILL_API_KEY=your_api_key_here

# Server settings
PORT=3001
FRONTEND_URL=http://localhost:3000

# 0G Network
GALILEO_RPC_URL=https://evmrpc-testnet.0g.ai
INFT_CONTRACT_ADDRESS=0x9C3FFe10e61B1750F61D2E0A64c6bBE8984BA268
```

### 2. Frontend Setup

Create `frontend/.env.local`:
```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:3001
```

## 🎯 How It Works

```
┌─────────────┐         ┌─────────────────┐         ┌──────────────┐
│   User UI   │────────▶│  Oracle Backend │────────▶│  0G Network  │
│  (Browser)  │         │   (Express.js)  │         │  Blockchain  │
└─────────────┘         └─────────────────┘         └──────────────┘
      │                         │
      │                         ▼
      │                 ┌─────────────────┐
      │                 │   Phala RedPill │
      └────────────────▶│   (LLM API)     │
                        └─────────────────┘
```

### Flow:

1. **User connects wallet** to 0G Galileo network
2. **Mints an INFT** with encrypted AI data reference
3. **Authorizes users** who can perform AI inference
4. **Requests inference** through the UI
5. **Backend validates** authorization on-chain
6. **Fetches and decrypts** AI data from 0G Storage
7. **Performs inference** using LLM API
8. **Returns result** with cryptographic proof

## 🎨 Features

### Frontend (`frontend/pages/inft.tsx`)

- ✅ **Wallet Connection**: Connect to 0G Network with MetaMask
- ✅ **Mint INFTs**: Create new intelligent NFTs
- ✅ **Authorization Management**: Grant/revoke usage permissions
- ✅ **AI Inference**: Standard and streaming inference
- ✅ **Token Management**: View owned tokens and stats
- ✅ **Real-time Updates**: Live transaction status
- ✅ **Beautiful UI**: Modern gradient design with animations

### Backend (`backend/index.ts`)

- ✅ **On-chain Validation**: ERC-7857 authorization checks
- ✅ **0G Storage Integration**: Encrypted data retrieval
- ✅ **LLM Inference**: Phala RedPill API integration
- ✅ **Streaming Support**: Server-Sent Events for real-time responses
- ✅ **Rate Limiting**: 30 requests/minute protection
- ✅ **Security**: Input sanitization and validation
- ✅ **Proof Generation**: Cryptographic proofs for all operations

## 📊 Smart Contracts (0G Galileo Testnet)

| Contract | Address | Purpose |
|----------|---------|---------|
| **INFT** | `0x9C3FFe10e61B1750F61D2E0A64c6bBE8984BA268` | Main ERC-7857 implementation |
| **DataVerifier** | `0xd84254b80e4C41A88aF309793F180a206421b450` | Oracle adapter with error handling |
| **Oracle** | `0x78aCb19366A0042dA3263747bda14BA43d68B0de` | Development oracle (stub) |

**Network Details:**
- Name: 0G Galileo Testnet
- Chain ID: 16602
- RPC: https://evmrpc-testnet.0g.ai
- Explorer: https://chainscan-galileo.0g.ai
- Faucet: https://faucet.0g.ai

## 🔐 Security Features

- **Input Validation**: Max 500 characters, sanitized
- **Rate Limiting**: 30 requests/minute per IP
- **On-chain Authorization**: Smart contract validation
- **Encryption**: AES-256-GCM for stored data
- **Proof Generation**: SHA-256 hashes for verification
- **No Key Exposure**: Encryption keys never in prompts

## 🧪 Testing

### Test Backend Health
```bash
curl http://localhost:3001/health
```

### Test LLM Connection
```bash
curl http://localhost:3001/llm/health
```

### Test Inference (requires authorization)
```bash
curl -X POST http://localhost:3001/infer \
  -H "Content-Type: application/json" \
  -d '{
    "tokenId": 1,
    "input": "Tell me something motivational",
    "user": "0xYourWalletAddress"
  }'
```

## 📖 Documentation

- **[INFT-SETUP-GUIDE.md](INFT-SETUP-GUIDE.md)** - Comprehensive setup instructions
- **[INFT-QUICK-REFERENCE.md](INFT-QUICK-REFERENCE.md)** - Quick reference card
- **[backend/README.md](backend/README.md)** - Backend API documentation
- **[0g-INFT/README.md](0g-INFT/README.md)** - Smart contract documentation

## 🎓 Usage Guide

### 1. Get Testnet Tokens
Visit [https://faucet.0g.ai](https://faucet.0g.ai) and request tokens

### 2. Connect Wallet
Click "Connect Wallet" on the INFT page

### 3. Mint Your First INFT
- Fill in recipient address (your wallet)
- Use default URI and hash (or customize)
- Click "Mint INFT"
- Approve transaction in MetaMask

### 4. Authorize Yourself
- Enter Token ID: 1
- Enter your wallet address
- Click "Authorize User"
- Approve transaction

### 5. Run AI Inference
- Enter Token ID: 1
- Enter prompt: "Tell me something inspiring"
- Click "Run Inference" or "Stream Inference"
- View AI-generated result

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend won't start | Check `REDPILL_API_KEY` in `.env` |
| Frontend can't connect | Ensure backend runs on port 3001 |
| "Access denied" error | Authorize your address first |
| "Insufficient funds" | Get tokens from faucet |
| Wallet connection fails | Add 0G network manually |
| Transaction pending | Wait longer, 0G may be slow |

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/llm/health` | LLM availability check |
| POST | `/infer` | Standard inference |
| POST | `/infer/stream` | Streaming inference (SSE) |

## 🔮 Advanced Features

### Streaming Inference
Real-time token-by-token responses using Server-Sent Events:
```javascript
POST /infer/stream
Accept: text/event-stream
```

### Custom Prompts
The system uses context-aware prompts with the encrypted dataset:
```
Input: "motivation"
Context: [25 inspirational quotes]
Output: AI-generated personalized response
```

### Proof Verification
Every inference includes cryptographic proof:
```json
{
  "proofHash": "0x...",
  "promptHash": "...",
  "contextHash": "...",
  "completionHash": "..."
}
```

## 🚧 Future Enhancements

- [ ] **Production Oracle**: Replace stub with TEE/ZKP oracle
- [ ] **Multiple Models**: Support different LLM providers
- [ ] **IPFS Integration**: Alternative storage backend
- [ ] **Token Marketplace**: Buy/sell INFTs
- [ ] **Advanced Analytics**: Usage statistics dashboard
- [ ] **Mobile Support**: Progressive Web App
- [ ] **Batch Operations**: Mint/authorize multiple tokens

## 🤝 Contributing

This is a reference implementation. Feel free to:
- Fork and customize for your needs
- Report issues or bugs
- Suggest improvements
- Share your implementations

## 📜 License

MIT License - See individual component licenses

## 🙏 Acknowledgments

- **0G Network** - High-performance blockchain infrastructure
- **Phala Network** - Confidential computing and LLM APIs
- **ERC-7857** - Intelligent NFT standard
- **OpenZeppelin** - Secure smart contract libraries

## 📞 Support

- **Documentation**: See markdown files in this project
- **0G Docs**: https://docs.0g.ai
- **Phala Docs**: https://docs.phala.network
- **Issues**: Open an issue on GitHub

## 🎉 Success!

If you see:
- ✅ Backend: "Server running on http://localhost:3001"
- ✅ Frontend: "Ready on http://localhost:3000"
- ✅ UI: Wallet connected and showing 0G Galileo
- ✅ Inference: AI responses appearing

**You're all set! Start building with Intelligent NFTs! 🚀**

---

**Quick Links:**
- [Setup Guide](INFT-SETUP-GUIDE.md) | [Quick Reference](INFT-QUICK-REFERENCE.md) | [Backend Docs](backend/README.md)
- [0G Network](https://0g.ai) | [Get Tokens](https://faucet.0g.ai) | [Explorer](https://chainscan-galileo.0g.ai)


