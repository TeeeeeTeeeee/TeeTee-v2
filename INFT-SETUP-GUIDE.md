# 🚀 INFT Complete Setup Guide

This guide will help you set up and run the complete INFT (Intelligent NFT) system on 0G Network.

## 📋 System Overview

```
TeeTee-v2/
├── frontend/          # Next.js frontend with INFT UI
├── backend/           # Oracle backend service for AI inference
└── 0g-INFT/          # Smart contracts and storage (reference)
```

## 🔧 Prerequisites

- **Node.js** v18+ and npm
- **MetaMask** or compatible Web3 wallet
- **0G Testnet Tokens** (get from [faucet.0g.ai](https://faucet.0g.ai))
- **Phala RedPill API Key** (get from [red-pill.ai](https://red-pill.ai))

## 📦 Installation

### 1. Install Backend Dependencies

```bash
cd backend
npm install
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
```

## ⚙️ Configuration

### 1. Backend Configuration

Create `backend/.env`:

```bash
cd backend
cp .env.example .env
```

Edit `backend/.env`:

```env
# Server Configuration
PORT=3001
FRONTEND_URL=http://localhost:3000

# 0G Network Configuration
GALILEO_RPC_URL=https://evmrpc-testnet.0g.ai
INFT_CONTRACT_ADDRESS=0x9C3FFe10e61B1750F61D2E0A64c6bBE8984BA268

# LLM Configuration
LLM_PROVIDER=phala-redpill
LLM_HOST=https://api.red-pill.ai
LLM_MODEL=phala/deepseek-r1-70b
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=256

# ⚠️ IMPORTANT: Add your Phala RedPill API key
REDPILL_API_KEY=your_actual_api_key_here
```

**Getting a Phala RedPill API Key:**
1. Visit [https://red-pill.ai](https://red-pill.ai)
2. Sign up for an account
3. Generate an API key
4. Copy it to your `.env` file

### 2. Frontend Configuration

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:3001
```

## 🚀 Running the System

### Step 1: Start the Backend (Terminal 1)

```bash
cd backend
npm start
```

**Expected output:**
```
╔════════════════════════════════════════════════════════════╗
║       🚀 INFT Oracle Backend Service Started              ║
╚════════════════════════════════════════════════════════════╝

✅ Server running on http://localhost:3001
✅ Connected to 0G Galileo (Chain ID: 16602)
✅ INFT Contract: 0x9C3FFe10e61B1750F61D2E0A64c6bBE8984BA268

📡 Endpoints:
   - GET  /health         - Service health check
   - GET  /llm/health     - LLM health check
   - POST /infer          - Run inference
   - POST /infer/stream   - Streaming inference

🔐 Security:
   - Rate limiting: 30 req/min
   - Input sanitization: enabled
   - Authorization: on-chain validation
```

### Step 2: Start the Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

**Expected output:**
```
ready - started server on 0.0.0.0:3000, url: http://localhost:3000
```

### Step 3: Access the INFT UI

Open your browser and navigate to:
```
http://localhost:3000/inft
```

## 🎯 Usage Guide

### 1. Connect Your Wallet

1. Click **"Connect Wallet"** button
2. Approve MetaMask connection
3. The app will automatically add 0G Galileo network to your wallet
4. Make sure you have some 0G testnet tokens (get from faucet)

### 2. Mint an INFT

1. Fill in the mint form:
   - **Recipient Address**: Your wallet address (auto-filled)
   - **Encrypted URI**: Leave default or use custom
   - **Metadata Hash**: Leave default or use custom
2. Click **"Mint INFT"**
3. Approve the transaction in MetaMask
4. Wait for confirmation

### 3. Authorize a User

Before someone (including yourself) can use an INFT for AI inference, they must be authorized:

1. Enter the **Token ID** you want to authorize
2. Enter the **User Address** to authorize
3. Click **"Authorize User"**
4. Approve the transaction in MetaMask

### 4. Perform AI Inference

Once authorized:

1. Enter the **Token ID** you're authorized for
2. Enter your **Input Prompt** (e.g., "Tell me something motivational")
3. Choose inference type:
   - **Run Inference**: Standard request/response
   - **Stream Inference**: Real-time token streaming
4. View the AI-generated response

## 🧪 Testing

### Test Backend Health

```bash
# Basic health check
curl http://localhost:3001/health

# LLM health check
curl http://localhost:3001/llm/health
```

### Test Inference (requires authorization)

```bash
curl -X POST http://localhost:3001/infer \
  -H "Content-Type: application/json" \
  -d '{
    "tokenId": 1,
    "input": "motivation",
    "user": "0xYourWalletAddress"
  }'
```

## 📊 Contract Addresses (0G Galileo Testnet)

| Contract | Address |
|----------|---------|
| **INFT** | `0x9C3FFe10e61B1750F61D2E0A64c6bBE8984BA268` |
| **DataVerifier** | `0xd84254b80e4C41A88aF309793F180a206421b450` |
| **Oracle** | `0x78aCb19366A0042dA3263747bda14BA43d68B0de` |

View on block explorer: [https://chainscan-galileo.0g.ai](https://chainscan-galileo.0g.ai)

## 🔍 Troubleshooting

### Backend Issues

**"LLM API key not configured"**
- Solution: Add your `REDPILL_API_KEY` to `backend/.env`

**"ECONNREFUSED" or connection errors**
- Solution: Check that backend is running on port 3001
- Solution: Verify `GALILEO_RPC_URL` is accessible

### Frontend Issues

**"Network error: Cannot connect to inference service"**
- Solution: Ensure backend is running
- Solution: Check `NEXT_PUBLIC_BACKEND_URL` in `frontend/.env.local`

**"Access denied. You are not authorized"**
- Solution: Use the "Authorize Usage" form to grant access to your address
- Solution: Make sure you're using the correct token ID

**Wallet connection issues**
- Solution: Make sure MetaMask is installed and unlocked
- Solution: Switch to 0G Galileo network manually if auto-add fails
- Solution: Add network manually:
  - Network Name: `0G Galileo Testnet`
  - RPC URL: `https://evmrpc-testnet.0g.ai`
  - Chain ID: `16602`
  - Currency Symbol: `0G`
  - Block Explorer: `https://chainscan-galileo.0g.ai`

### Transaction Issues

**"Insufficient funds for gas"**
- Solution: Get testnet tokens from [https://faucet.0g.ai](https://faucet.0g.ai)

**Transaction pending for too long**
- Solution: 0G Network may be congested, wait a bit longer
- Solution: Check transaction status on block explorer

## 🔐 Security Notes

⚠️ **Important Security Reminders:**

1. **Never share your private keys or mnemonic phrases**
2. **Store API keys securely** (never commit `.env` files)
3. **This is testnet** - Do not use mainnet private keys or real funds
4. **Rate limits apply** - 30 requests per minute per IP
5. **Input validation** - Max 500 characters per inference request

## 📁 Project Structure

```
backend/
├── index.ts              # Main backend service
├── package.json          # Dependencies
├── tsconfig.json         # TypeScript config
├── .env                  # Environment variables (create this)
├── .env.example          # Environment template
└── README.md            # Backend documentation

frontend/
├── pages/
│   └── inft.tsx         # INFT UI page
├── package.json         # Dependencies
├── .env.local           # Frontend environment (create this)
└── ...

0g-INFT/ (reference)
├── contracts/           # Smart contracts (deployed)
├── storage/            # Encryption utilities
└── offchain-service/   # Original service (reference)
```

## 🎓 Key Concepts

### ERC-7857 Intelligent NFTs
- NFTs that can store encrypted AI model data
- Support authorization for usage without transferring ownership
- Enable verifiable AI inference through oracle proofs

### Authorization Flow
1. Token owner authorizes a user address
2. User requests inference from backend
3. Backend validates authorization on-chain
4. If authorized, performs AI inference
5. Returns result with cryptographic proof

### Storage Architecture
- Encrypted data stored on 0G Storage network
- Only encrypted URI and hash stored on-chain
- Backend decrypts data for authorized users only

## 🆘 Getting Help

1. **Check logs**: Backend and frontend terminals show detailed logs
2. **Review documentation**: Each component has its own README
3. **Test endpoints**: Use curl to test backend directly
4. **Check contracts**: View transactions on block explorer

## 🎉 Success Indicators

You'll know everything is working when:

✅ Backend shows "Server running" message
✅ Frontend loads without errors
✅ Wallet connects and shows 0G Galileo network
✅ You can mint NFTs successfully
✅ Authorization transactions confirm
✅ AI inference returns responses
✅ Streaming inference shows real-time tokens

## 📚 Additional Resources

- **0G Network**: [https://0g.ai](https://0g.ai)
- **0G Faucet**: [https://faucet.0g.ai](https://faucet.0g.ai)
- **0G Explorer**: [https://chainscan-galileo.0g.ai](https://chainscan-galileo.0g.ai)
- **Phala RedPill**: [https://red-pill.ai](https://red-pill.ai)
- **ERC-7857 Spec**: See `0g-INFT/docs/` folder

## 🚀 Next Steps

Once you have the basic system running:

1. **Mint multiple INFTs** and test with different token IDs
2. **Authorize multiple users** to test access control
3. **Try different prompts** to test AI inference quality
4. **Compare streaming vs standard** inference modes
5. **Explore the codebase** to understand the architecture

Happy building with Intelligent NFTs! 🎨🤖


