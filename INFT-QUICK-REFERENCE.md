# 🚀 INFT Quick Reference Card

## 🎯 Quick Start (One Command)

### Windows:
```bash
start-inft.bat
```

### Mac/Linux:
```bash
chmod +x start-inft.sh
./start-inft.sh
```

## 📡 URLs

| Service | URL |
|---------|-----|
| **INFT UI** | http://localhost:3000/inft |
| **Main App** | http://localhost:3000 |
| **Backend API** | http://localhost:3001 |
| **Health Check** | http://localhost:3001/health |

## 🔑 Essential Environment Variables

### Backend (`backend/.env`):
```env
REDPILL_API_KEY=your_key_here   # ⚠️ REQUIRED - Get from red-pill.ai
PORT=3001
INFT_CONTRACT_ADDRESS=0x9C3FFe10e61B1750F61D2E0A64c6bBE8984BA268
```

### Frontend (`frontend/.env.local`):
```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:3001
```

## 🎨 Contract Addresses (0G Galileo)

```
INFT:         0x9C3FFe10e61B1750F61D2E0A64c6bBE8984BA268
DataVerifier: 0xd84254b80e4C41A88aF309793F180a206421b450
Oracle:       0x78aCb19366A0042dA3263747bda14BA43d68B0de

Network:      0G Galileo Testnet
Chain ID:     16602
RPC:          https://evmrpc-testnet.0g.ai
Explorer:     https://chainscan-galileo.0g.ai
Faucet:       https://faucet.0g.ai
```

## 🔄 Common Operations

### 1. Mint INFT
```
Connect Wallet → Mint INFT Tab → Fill Form → Mint
```

### 2. Authorize User
```
Connect Wallet → Authorize Usage Tab → Enter Token ID & Address → Authorize
```

### 3. Run AI Inference
```
Connect Wallet → AI Inference Tab → Enter Token ID & Prompt → Run Inference
```

## 💻 Manual Start Commands

### Start Backend:
```bash
cd backend
npm install  # First time only
npm start
```

### Start Frontend:
```bash
cd frontend
npm install  # First time only
npm run dev
```

## 🧪 Quick Tests

### Test Backend:
```bash
curl http://localhost:3001/health
```

### Test LLM:
```bash
curl http://localhost:3001/llm/health
```

### Test Inference:
```bash
curl -X POST http://localhost:3001/infer \
  -H "Content-Type: application/json" \
  -d '{"tokenId":1,"input":"motivation","user":"0xYourAddress"}'
```

## 🔧 Troubleshooting Quick Fixes

| Issue | Solution |
|-------|----------|
| **Backend won't start** | Check `.env` has `REDPILL_API_KEY` |
| **Frontend can't connect** | Check backend is running on 3001 |
| **"Access denied"** | Authorize your address first |
| **"Insufficient funds"** | Get tokens from faucet.0g.ai |
| **Wallet won't connect** | Add 0G Galileo network manually |
| **CORS errors** | Check `FRONTEND_URL` in backend `.env` |

## 🎯 Workflow Example

1. **Get Testnet Tokens**: https://faucet.0g.ai
2. **Connect Wallet**: Click "Connect Wallet" on INFT page
3. **Mint NFT**: Use mint form (recipient = your address)
4. **Authorize Yourself**: Token ID = 1, User = your address
5. **Test Inference**: Token ID = 1, Input = "inspire me"
6. **View Result**: See AI-generated response

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health |
| GET | `/llm/health` | LLM availability |
| POST | `/infer` | Run inference |
| POST | `/infer/stream` | Streaming inference |

## 🔐 Security Checklist

- [ ] Added `REDPILL_API_KEY` to backend `.env`
- [ ] Never commit `.env` files
- [ ] Using testnet only (not mainnet)
- [ ] Wallet has testnet tokens
- [ ] Both services running on localhost

## 📱 MetaMask Network Setup

If auto-add fails, add manually:

```
Network Name:       0G Galileo Testnet
RPC URL:            https://evmrpc-testnet.0g.ai
Chain ID:           16602
Currency Symbol:    0G
Block Explorer:     https://chainscan-galileo.0g.ai
```

## 🆘 Support Resources

- **Setup Guide**: `INFT-SETUP-GUIDE.md`
- **Backend Docs**: `backend/README.md`
- **Contract Docs**: `0g-INFT/README.md`
- **0G Docs**: https://docs.0g.ai

## 🎓 Key Concepts

- **INFT**: Intelligent NFT with AI capabilities
- **ERC-7857**: Standard for AI-powered NFTs
- **Authorization**: Permission to use INFT without owning it
- **Oracle**: Backend service that validates and performs inference
- **Proof**: Cryptographic proof of inference execution

## 🚀 Performance Tips

- Use **streaming inference** for real-time responses
- Keep prompts under **500 characters**
- Rate limit: **30 requests/minute**
- Backend timeout: **30 seconds**

---

**Ready to build? Start with:** `./start-inft.sh` (Mac/Linux) or `start-inft.bat` (Windows)


