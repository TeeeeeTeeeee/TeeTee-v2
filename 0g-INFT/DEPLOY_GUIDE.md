# 🚀 Quick Deployment Guide

## ✅ Your Configuration

- **Network**: Galileo Testnet (Chain ID: 16602)
- **Storage**: Real 0G Storage ✅
- **Wallet**: `0x9787cfF89D30bB6Ae87Aaad9B3a02E77B5caA8f1`
- **Latest Upload**:
  - URI: `0g://storage/0xd1a827ee9216ffc63a3a2f239b7615eaecd8ede73b4c3d528c93d243740bff72`
  - Metadata: `0xcee513fb848d9c257cda4909f34d9bdc7097de1d70cb4ef322efd1cb2e6433ed`
  - TX: `0xa866e211eaaf319f5112ec8ee9496e671e81476d75f722de48bf481ffc4b88ae`

## 📋 Which Script to Use?

Your `hardhat.config.ts` has these networks:
- `newton` - Chain ID 16600
- `galileo` - Chain ID 16602 ✅ **(Use this one!)**
- `mainnet` - Chain ID 16661

## 🎯 Deployment Options

### Option 1: Deploy Everything at Once (Recommended)

```powershell
cd 0g-INFT
npx hardhat run scripts/deploy-all.ts --network galileo
```

**This deploys:**
1. ✅ OracleStub (verification oracle)
2. ✅ DataVerifierAdapter (ERC-7857 interface)
3. ✅ INFT Contract (with new burn function!)

**Time**: ~2-3 minutes
**Cost**: ~0.05 0G tokens

---

### Option 2: Deploy Step-by-Step

#### Step 1: Deploy Oracle
```powershell
npx hardhat run scripts/deployOracle.ts --network galileo
```

#### Step 2: Deploy INFT
```powershell
npx hardhat run scripts/deployINFT.ts --network galileo
```

---

### Option 3: Use Fixed Contracts (Better Gas Optimization)
```powershell
npx hardhat run scripts/deploy-fixed-contracts.ts --network galileo
```

---

## 🎨 After Deployment

### 1. Mint Your First INFT

```powershell
npx hardhat run scripts/mint.ts --network galileo
```

This will prompt you for:
- **Recipient address**: Your wallet or user's wallet
- **Encrypted URI**: `0g://storage/0xd1a827ee9216ffc63a3a2f239b7615eaecd8ede73b4c3d528c93d243740bff72`
- **Metadata Hash**: `0xcee513fb848d9c257cda4909f34d9bdc7097de1d70cb4ef322efd1cb2e6433ed`

### 2. Authorize Users

```powershell
npx hardhat run scripts/authorize.ts --network galileo
```

You'll be prompted for:
- **Token ID**: The token you minted (usually starts at 0)
- **User Address**: Wallet address to authorize

### 3. Update Frontend

The deploy script will create a file: `deployments/galileo.json`

Use these addresses to update your frontend:
```typescript
// frontend/lib/networkConfig.ts
const CONTRACT_ADDRESSES = {
  testnet: {
    INFT: 'ADDRESS_FROM_DEPLOYMENT', // From galileo.json
    CreditUse: 'YOUR_TEETEE_CONTRACT',
  }
};
```

---

## 🆕 New Burn Function

The INFT contract now includes a `burn()` function:

```solidity
function burn(uint256 tokenId) external
```

**What it does:**
- ✅ Clears all user authorizations
- ✅ Removes metadata references
- ✅ Burns the NFT permanently
- ✅ Only owner or approved can burn

**Usage:**
```typescript
// Via ethers.js
await inftContract.burn(tokenId);
```

---

## 📊 Deployment Checklist

- [ ] Have 0.1+ 0G tokens in wallet
- [ ] `.env` file configured with `PRIVATE_KEY`
- [ ] Encrypted file ready at `storage/quotes.enc`
- [ ] Run deployment script: `deploy-all.ts`
- [ ] Save contract addresses from output
- [ ] Mint first INFT with real storage URI
- [ ] Authorize test users
- [ ] Update frontend configuration
- [ ] Test burn function (optional)

---

## 🔧 Useful Commands

### Check Deployment Status
```powershell
# View deployed contracts
Get-Content deployments/galileo.json | ConvertFrom-Json
```

### Verify on Explorer
```
https://chainscan-galileo.0g.ai/address/YOUR_CONTRACT_ADDRESS
```

### Check Your Balance
```powershell
npx hardhat run scripts/ping.ts --network galileo
```

---

## 🆘 Troubleshooting

### "Insufficient funds"
Get more tokens from faucet: https://faucet.0g.ai

### "Oracle not deployed"
Run `deployOracle.ts` first or use `deploy-all.ts`

### "Network not found"
Make sure you're using `--network galileo` (not `galileo-testnet` or `0gTestnet`)

### Check Contract Status
```powershell
npx hardhat run scripts/check-fixed-contract-status.ts --network galileo
```

---

## 🎯 Quick Start Command

**Just want to deploy everything now?**

```powershell
cd 0g-INFT
npx hardhat compile
npx hardhat run scripts/deploy-all.ts --network galileo
```

That's it! The script will:
1. Check your balance
2. Deploy all contracts
3. Save addresses to `deployments/galileo.json`
4. Show you next steps

---

## 📝 Next: Mint INFT with Real Storage

After deployment, mint using your **real 0G storage**:

```powershell
npx hardhat run scripts/mint.ts --network galileo
```

When prompted:
- **encryptedURI**: `0g://storage/0xd1a827ee9216ffc63a3a2f239b7615eaecd8ede73b4c3d528c93d243740bff72`
- **metadataHash**: `0xcee513fb848d9c257cda4909f34d9bdc7097de1d70cb4ef322efd1cb2e6433ed`

Your INFT will point to **real decentralized storage**! 🎉

---

## 🚀 Ready to Deploy?

```powershell
npx hardhat run scripts/deploy-all.ts --network galileo
```

Good luck! 🍀



