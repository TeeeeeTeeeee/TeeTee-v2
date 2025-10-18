# 🚀 Complete Deployment Guide - 0G Galileo Testnet (Chain ID 16602)

This guide walks you through deploying all INFT contracts to the new 0G Galileo testnet.

---

## 📋 Prerequisites

### 1. Check Your Wallet Balance
- Visit: https://faucet.0g.ai
- Request testnet tokens for your deployer address
- Minimum required: **0.05 0G** (recommended: 0.1 0G)

### 2. Verify Environment Configuration
Your `.env` file should already be updated with chain ID 16602:

```bash
PRIVATE_KEY=your_private_key_here
GALILEO_RPC_URL=https://evmrpc-testnet.0g.ai
GALILEO_CHAIN_ID=16602
```

### 3. Check Network Connectivity
```bash
npx hardhat run scripts/ping.ts --network galileo
```

---

## 🎯 Deployment Options

### Option A: One-Command Full Deployment (Recommended)

Deploy all contracts with a single command:

```bash
cd /Users/marcus/Projects/TeeTee-v2/0g-INFT
npx hardhat run scripts/deploy-all.ts --network galileo
```

This script will:
1. ✅ Deploy OracleStub
2. ✅ Deploy DataVerifierAdapterFixed
3. ✅ Deploy INFTFixed
4. ✅ Verify all deployments
5. ✅ Save deployment data to `deployments/galileo.json`
6. ✅ Create a timestamped backup

**Estimated time:** 2-3 minutes  
**Estimated gas:** ~0.03 0G

---

### Option B: Step-by-Step Manual Deployment

If you prefer more control, deploy contracts individually:

#### Step 1: Deploy Oracle
```bash
npx hardhat run scripts/deployOracle.ts --network galileo
```

Expected output:
```
✅ OracleStub deployed successfully!
📧 Address: 0x...
```

#### Step 2: Deploy DataVerifier and INFT
```bash
npx hardhat run scripts/deployINFT.ts --network galileo
```

Expected output:
```
✅ DataVerifierAdapter deployed successfully!
✅ INFT contract deployed successfully!
```

#### Step 3: Or Use Fixed Contracts (Recommended)
```bash
npx hardhat run scripts/deploy-fixed-contracts.ts --network galileo
```

This deploys improved versions with better error handling.

---

## 📝 Post-Deployment Steps

### 1. Update Environment Variables

After deployment, update your `.env` file with the new addresses:

```bash
# Copy the addresses from deployment output
INFT_CONTRACT_ADDRESS=0x... # Your new INFT address
ORACLE_CONTRACT_ADDRESS=0x... # Your new Oracle address
```

Update both:
- `/Users/marcus/Projects/TeeTee-v2/0g-INFT/.env`
- `/Users/marcus/Projects/TeeTee-v2/0g-INFT/offchain-service/.env`

### 2. Update Frontend Configuration

Edit `frontend/lib/constants.js`:

```javascript
export const CONTRACT_ADDRESSES = {
  INFT: '0x...', // Your new INFT address
  DATA_VERIFIER: '0x...', // Your new DataVerifier address
  ORACLE_STUB: '0x...', // Your new Oracle address
}
```

### 3. Verify Deployment on Block Explorer

Visit the block explorer to confirm your contracts:
- **Explorer:** https://chainscan-galileo.0g.ai
- Search for your contract addresses
- Check that transactions are confirmed

---

## 🧪 Test Your Deployment

### Test 1: Mint a Token
```bash
npx hardhat run scripts/mint.ts --network galileo
```

Expected output:
```
✅ INFT minted successfully!
🎫 Token ID: 1
📧 Owner: 0x...
```

### Test 2: Authorize a User
```bash
npx hardhat run scripts/authorize.ts --network galileo
```

### Test 3: Start Services

#### Terminal 1: Off-chain Service
```bash
cd offchain-service
npm install  # if needed
npm start
```

Expected:
```
🚀 Off-chain service running on http://localhost:3000
✅ Connected to 0G Galileo (Chain ID: 16602)
```

#### Terminal 2: Frontend
```bash
cd frontend
npm install  # if needed
npm run dev
```

Expected:
```
Ready on http://localhost:3001
```

#### Terminal 3: Test the System
Visit http://localhost:3001 and:
1. Connect MetaMask
2. Verify network (should auto-add 0G Galileo)
3. Try minting an INFT
4. Test inference functionality

---

## 🔍 Troubleshooting

### Issue 1: "Insufficient funds for gas"
**Solution:**
- Visit https://faucet.0g.ai
- Request more testnet tokens
- Wait 1-2 minutes for confirmation

### Issue 2: "Network not responding"
**Solution:**
```bash
# Test RPC connectivity
curl -X POST https://evmrpc-testnet.0g.ai \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'

# Expected: {"jsonrpc":"2.0","id":1,"result":"0x40e2"}
# 0x40e2 = 16602 in hex
```

### Issue 3: "Oracle address not found"
**Solution:**
- Ensure you deployed OracleStub first
- Check `deployments/galileo.json` exists
- Run the deployment scripts in order

### Issue 4: Contract verification fails
**Solution:**
- Wait a few minutes for blocks to confirm
- Check gas price wasn't too low
- Verify you have enough balance
- Check block explorer for transaction status

### Issue 5: "Cannot find module" errors
**Solution:**
```bash
# Reinstall dependencies
npm install

# Rebuild contracts
npm run build
```

---

## 📊 Deployment Checklist

Use this checklist to track your deployment:

- [ ] Funded wallet with testnet tokens (min 0.05 0G)
- [ ] Updated `.env` with chain ID 16602
- [ ] Verified network connectivity with ping script
- [ ] Deployed contracts successfully
- [ ] Updated `.env` with new contract addresses
- [ ] Updated frontend constants with new addresses
- [ ] Verified contracts on block explorer
- [ ] Successfully minted a test token
- [ ] Started off-chain service
- [ ] Started frontend
- [ ] Tested full flow in browser

---

## 📞 Quick Reference

### Important Links
- **Faucet:** https://faucet.0g.ai
- **Block Explorer:** https://chainscan-galileo.0g.ai
- **RPC URL:** https://evmrpc-testnet.0g.ai
- **Chain ID:** 16602

### Contract Files
- **Oracle:** `contracts/OracleStub.sol`
- **DataVerifier:** `contracts/DataVerifierAdapterFixed.sol`
- **INFT:** `contracts/INFTFixed.sol`

### Deployment Scripts
- **All-in-one:** `scripts/deploy-all.ts`
- **Oracle only:** `scripts/deployOracle.ts`
- **INFT system:** `scripts/deployINFT.ts`
- **Fixed contracts:** `scripts/deploy-fixed-contracts.ts`

### Configuration Files
- **Network:** `hardhat.config.ts`
- **Environment:** `.env`
- **Frontend:** `frontend/lib/constants.js`
- **Service:** `offchain-service/.env`

---

## 🎓 Understanding the Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    0G Galileo Testnet                        │
│                     (Chain ID: 16602)                        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
         │ Oracle  │    │  Data   │    │  INFT   │
         │  Stub   │◄───┤Verifier │◄───┤Contract │
         └─────────┘    └─────────┘    └─────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Off-chain Service │
                    │  (Node.js/Express) │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Frontend dApp    │
                    │  (Next.js/React)   │
                    └───────────────────┘
```

### Component Roles:
1. **OracleStub**: Verifies proofs for INFT transfers (dev mode - always returns true)
2. **DataVerifierAdapter**: Wraps oracle with ERC-7857 interface
3. **INFT Contract**: Main ERC-7857 implementation for Intelligent NFTs
4. **Off-chain Service**: Handles 0G Storage, encryption, and LLM inference
5. **Frontend**: User interface for minting, transferring, and using INFTs

---

## 🆘 Need Help?

If you encounter issues:

1. **Check deployment logs** in `deployments/galileo.json`
2. **Verify contract addresses** on block explorer
3. **Review gas estimates** - ensure you have enough balance
4. **Test network connectivity** with ping script
5. **Check contract ABIs** are up to date with `npm run build`

---

## ✨ Success!

Once deployed, your INFT system is ready to:
- ✅ Mint Intelligent NFTs with AI models
- ✅ Store encrypted data on 0G Storage
- ✅ Perform AI inference through off-chain service
- ✅ Transfer INFTs with sealed key mechanism
- ✅ Authorize users for INFT usage

**Congratulations on deploying to 0G Galileo! 🎉**

