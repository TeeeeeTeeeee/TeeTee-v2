# Complete Deployment Guide for TeeTee

This guide walks you through deploying all smart contracts and updating the frontend.

## Prerequisites

- Node.js installed
- Wallet with testnet tokens
- Access to 0G Network testnet

## Step 1: Re-encrypt Data with New Configuration

```bash
cd 0g-INFT
npx ts-node storage/encrypt.ts
```

Save the output:
- `encryptedURI`
- `metadataHash`
- `encryptionKey`
- `iv`
- `tag`

## Step 2: Deploy INFT Contract

### 2a. Configure Environment

Create/update `0g-INFT/.env`:

```bash
# Deployment
PRIVATE_KEY=your_private_key_here
ZG_STORAGE_RPC=https://evmrpc-testnet.0g.ai
ZG_STORAGE_INDEXER=https://indexer-storage-testnet-turbo.0g.ai

# Contract Configuration
ORACLE_ADDRESS=0x0000000000000000000000000000000000000000  # Will be set later
```

### 2b. Deploy

```bash
cd 0g-INFT
npm install
npx hardhat compile
npx hardhat run scripts/deploy.ts --network 0gTestnet
```

**Save the output contract address!**

Example output:
```
INFT deployed to: 0x1234567890123456789012345678901234567890
```

## Step 3: Deploy Main TeeTee Contract

### 3a. Configure Environment

Create/update `smartcontract/.env`:

```bash
PRIVATE_KEY=your_private_key_here
ZEROG_TESTNET_RPC=https://evmrpc-testnet.0g.ai
```

### 3b. Deploy

```bash
cd smartcontract
npm install
npx hardhat compile
npx hardhat run scripts/deploy-creditusewithchat.js --network zerog_testnet
```

**Save the output contract address!**

Example output:
```
CreditUseWithChat deployed to: 0xABCDEF1234567890ABCDEF1234567890ABCDEF12
```

## Step 4: Mint INFT Token

Use the values from Step 1 to mint your first INFT:

```bash
cd 0g-INFT
# Edit scripts/mint-inft.ts with your values
npx ts-node scripts/mint-inft.ts
```

Or mint via Hardhat console:

```bash
npx hardhat console --network 0gTestnet

const INFT = await ethers.getContractFactory("INFT");
const inft = await INFT.attach("YOUR_INFT_CONTRACT_ADDRESS");

await inft.mint(
  "RECIPIENT_ADDRESS",
  "encryptedURI_from_step1",
  "metadataHash_from_step1"
);
```

**Save the Token ID** (usually starts at 0)

## Step 5: Authorize User for INFT

```bash
npx hardhat console --network 0gTestnet

const INFT = await ethers.getContractFactory("INFT");
const inft = await INFT.attach("YOUR_INFT_CONTRACT_ADDRESS");

await inft.authorizeUsage(
  TOKEN_ID,  // e.g., 0
  "USER_WALLET_ADDRESS"
);
```

## Step 6: Update Frontend Configuration

### 6a. Update Contract Addresses

Edit `frontend/utils/address.ts`:

```typescript
// Update with your deployed contract address from Step 3
export const CONTRACT_ADDRESS = '0xYourTeeTeeContractAddress';
```

### 6b. Update Network Configuration

Edit `frontend/lib/networkConfig.ts`:

```typescript
const CONTRACT_ADDRESSES = {
  testnet: {
    INFT: '0xYourINFTContractAddress',  // From Step 2
    CreditUse: '0xYourTeeTeeContractAddress',  // From Step 3
  },
  // ... rest of config
};
```

### 6c. Set Environment Variables

Create `frontend/.env.local`:

```bash
# Your wallet address that mints INFTs
# Only INFTs issued by this address will be recognized
NEXT_PUBLIC_INFT_ISSUER_ADDRESS=0xYourWalletAddressHere

# Network
NEXT_PUBLIC_NETWORK=testnet
```

## Step 7: Update Backend Configuration

Create/update `backend/.env`:

```bash
# Network
NETWORK=testnet
ZG_STORAGE_RPC=https://evmrpc-testnet.0g.ai
ZG_STORAGE_INDEXER=https://indexer-storage-testnet-turbo.0g.ai

# Contracts
INFT_CONTRACT_ADDRESS=0xYourINFTContractAddress
CREDITUSE_CONTRACT_ADDRESS=0xYourTeeTeeContractAddress

# LLM Configuration
REDPILL_API_KEY=your_redpill_api_key
LLM_PROVIDER=redpill
LLM_HOST=https://api.red-pill.ai
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
LLM_DEV_FALLBACK=true

# Server
PORT=3001
NODE_ENV=development
```

## Step 8: Start Services

### Backend

```bash
cd backend
npm install
npm run dev
```

Verify it's running on http://localhost:3001

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Access the app at http://localhost:3000

## Step 9: Test Everything

1. **Connect Wallet** to the frontend
2. **Check INFT Status** in chat page:
   - If you authorized your wallet: Should show "Use hosted INFT (No Tokens Used)"
   - If not authorized: Should show "You don't have a hosted model"
3. **Send a Test Message**: Try asking "What is blockchain?"
4. **Verify Backend Logs**: Check that inference is working

## Verification Checklist

- [ ] INFT contract deployed
- [ ] TeeTee contract deployed
- [ ] INFT minted with encrypted data
- [ ] User wallet authorized for INFT
- [ ] Frontend shows correct contract addresses
- [ ] Environment variable for issuer address set
- [ ] Backend is running and connected
- [ ] Frontend can connect to wallet
- [ ] Chat inference works with INFT
- [ ] Token-based chat works (without INFT)

## Troubleshooting

### "No models available" in chat

**Solution**: Make sure you've:
1. Deployed the TeeTee contract
2. Updated `frontend/utils/address.ts` with correct address
3. Restart frontend after updating addresses

### "You don't have a hosted model"

**Solutions**:
1. Check `NEXT_PUBLIC_INFT_ISSUER_ADDRESS` is set correctly
2. Verify you minted the INFT with your wallet
3. Verify you authorized the user wallet: `inft.authorizeUsage(tokenId, userAddress)`
4. Check network matches (testnet vs mainnet)

### Backend can't connect to contracts

**Solution**: 
1. Verify `backend/.env` has correct contract addresses
2. Check RPC endpoint is accessible
3. Verify network configuration matches

### LLM inference fails

**Solution**:
1. Check `REDPILL_API_KEY` is valid
2. Verify backend logs for specific error
3. Ensure fallback is enabled: `LLM_DEV_FALLBACK=true`

## Quick Deploy Script

Create `deploy-all.sh`:

```bash
#!/bin/bash

echo "🚀 TeeTee Complete Deployment Script"
echo "===================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Encrypt data
echo -e "${BLUE}Step 1: Encrypting data...${NC}"
cd 0g-INFT
npx ts-node storage/encrypt.ts
echo "✅ Save the encryptedURI and metadataHash!"
read -p "Press enter when you've saved the values..."

# Step 2: Deploy INFT
echo -e "${BLUE}Step 2: Deploying INFT contract...${NC}"
npx hardhat run scripts/deploy.ts --network 0gTestnet
echo "✅ Save the INFT contract address!"
read -p "Press enter when you've saved the address..."

# Step 3: Deploy TeeTee
echo -e "${BLUE}Step 3: Deploying TeeTee contract...${NC}"
cd ../smartcontract
npx hardhat run scripts/deploy-creditusewithchat.js --network zerog_testnet
echo "✅ Save the TeeTee contract address!"
read -p "Press enter when you've saved the address..."

# Step 4: Instructions
echo -e "${GREEN}Deployment Complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Update frontend/utils/address.ts with TeeTee contract address"
echo "2. Update frontend/lib/networkConfig.ts with both addresses"
echo "3. Set frontend/.env.local with NEXT_PUBLIC_INFT_ISSUER_ADDRESS"
echo "4. Update backend/.env with contract addresses"
echo "5. Mint INFT and authorize users"
echo "6. Start backend: cd backend && npm run dev"
echo "7. Start frontend: cd frontend && npm run dev"
echo ""
echo "🎉 Happy coding!"
```

Make it executable:
```bash
chmod +x deploy-all.sh
./deploy-all.sh
```

## Success!

Your TeeTee application is now fully deployed with:
- ✅ General purpose AI assistant
- ✅ INFT authorization system  
- ✅ Issuer filtering (only your INFTs recognized)
- ✅ Token-based and INFT-based inference
- ✅ 0G Storage integration

Enjoy your decentralized AI platform! 🚀

