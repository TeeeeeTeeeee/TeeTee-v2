# INFT Configuration Guide

This guide explains how to configure your TeeTee application to only recognize INFTs issued by YOU, and how to update the data stored in your INFTs.

## Table of Contents
1. [Filtering INFTs by Issuer](#1-filtering-infts-by-issuer)
2. [Updating Storage Data (Quotes)](#2-updating-storage-data-quotes)

---

## 1. Filtering INFTs by Issuer

### Problem
By default, the chat page recognizes ANY INFT that a user is authorized for. You want to only recognize INFTs that YOU minted/issued.

### Solution Implemented

#### A. Smart Contract Changes (`0g-INFT/contracts/INFT.sol`)

Added tracking of who minted each INFT:

```solidity
/// @dev Mapping from token ID to original minter/issuer address
mapping(uint256 => address) public tokenMinter;
```

When minting, the contract now records the minter:
```solidity
// Track the original minter/issuer
tokenMinter[tokenId] = _msgSender();
```

#### B. Frontend Hook Changes (`frontend/hooks/useINFT.ts`)

Updated `useCheckINFTAuthorization` to accept an `allowedIssuer` parameter:

```typescript
export function useCheckINFTAuthorization(
  tokenId: number = 1, 
  userAddress?: string,
  allowedIssuer?: string // Only INFTs from this issuer are valid
)
```

The hook now:
1. Checks if user is authorized ✅
2. Checks if INFT was minted by allowed issuer ✅
3. Returns `true` only if BOTH conditions are met

#### C. Chat Page Integration (`frontend/pages/chat.tsx`)

The chat page now uses the issuer filter:

```typescript
const ALLOWED_INFT_ISSUER = process.env.NEXT_PUBLIC_INFT_ISSUER_ADDRESS;
const { isAuthorized: hasINFT } = useCheckINFTAuthorization(
  1, 
  address,
  ALLOWED_INFT_ISSUER // Only accept INFTs from this issuer
);
```

### Configuration Steps

#### Step 1: Set Your Wallet Address

Create a `frontend/.env.local` file:

```bash
# Your wallet address that mints INFTs
# Only INFTs issued by this address will be recognized
NEXT_PUBLIC_INFT_ISSUER_ADDRESS=0xYourWalletAddressHere

# Example:
# NEXT_PUBLIC_INFT_ISSUER_ADDRESS=0x1234567890123456789012345678901234567890
```

#### Step 2: Redeploy the Contract

The INFT contract now includes the `tokenMinter` mapping, so you need to redeploy:

```bash
cd 0g-INFT
npm install
npx hardhat compile
npx hardhat run scripts/deploy.ts --network 0gTestnet
```

Save the new contract address and update your frontend configuration.

#### Step 3: Mint INFTs

When you mint INFTs using the updated contract, the contract will automatically record you as the minter. Only these INFTs will be recognized in the chat interface.

```typescript
// Mint INFT (only contract owner can do this)
await inftContract.mint(
  recipientAddress,
  encryptedURI,
  metadataHash
);
// The contract automatically records msg.sender as tokenMinter
```

### Testing

1. **Connect with a wallet** that has an INFT minted by YOU
   - ✅ Should show "Use hosted INFT (No Tokens Used)"
   
2. **Connect with a wallet** that has an INFT from someone else
   - ✅ Should show "You don't have a hosted model" (disabled)

3. **Connect with a wallet** with no INFT
   - ✅ Should show "You don't have a hosted model" (disabled)

---

## 2. Updating Storage Data (Quotes)

### Quick Start

To change the quotes or data your INFT serves:

#### Step 1: Edit `0g-INFT/storage/quotes.json`

```json
{
  "version": "1.0.0",
  "quotes": [
    "Your custom quote 1",
    "Your custom quote 2",
    "Your custom quote 3"
  ],
  "metadata": {
    "created": "2025-11-03T00:00:00.000Z",
    "description": "My custom quotes",
    "totalQuotes": 3,
    "category": "custom"
  }
}
```

#### Step 2: Re-encrypt

```bash
cd 0g-INFT
npx ts-node storage/encrypt.ts
```

Output example:
```
🎯 PHASE 1 RESULTS:
============================================================
encryptedURI: 0g://storage/quotes_1730592000000.enc
metadataHash: 0x1f626cda...
encryptionKey: 0xe244c32a...
iv: 0xef0b00af...
tag: 0xb4ad0db8...
```

#### Step 3: Mint New INFT

Use the output values to mint a new INFT:

```typescript
await inftContract.mint(
  recipientAddress,
  "0g://storage/quotes_1730592000000.enc",  // new encryptedURI
  "0x1f626cda..."  // new metadataHash
);
```

### Custom Data Structures

You can store ANY JSON data, not just quotes:

#### Example: Knowledge Base
```json
{
  "version": "1.0.0",
  "knowledge": [
    {
      "topic": "Blockchain",
      "content": "Distributed ledger technology..."
    },
    {
      "topic": "AI",
      "content": "Machine learning and neural networks..."
    }
  ],
  "metadata": {
    "description": "Tech knowledge base",
    "category": "education"
  }
}
```

#### Example: Product Catalog
```json
{
  "version": "1.0.0",
  "products": [
    {
      "name": "Widget A",
      "description": "Best widget ever",
      "price": "$99"
    }
  ],
  "metadata": {
    "description": "Product catalog",
    "category": "ecommerce"
  }
}
```

### Backend Integration

The backend automatically loads from `0g-INFT/storage/quotes.enc`:

```typescript
// backend/index.ts line 664
const localPath = path.join(__dirname, '..', '0g-INFT', 'storage', 'quotes.enc');
```

After updating and re-encrypting, restart the backend:
```bash
cd backend
npm run dev
```

### Testing Your Changes

1. Update `quotes.json`
2. Run encryption script
3. Restart backend
4. Test via chat interface or API endpoint `/api/inft/infer`

---

## Summary

### What You've Configured

✅ **INFT Issuer Filtering**: Only INFTs minted by your wallet address are recognized

✅ **Minter Tracking**: Smart contract tracks who minted each INFT

✅ **Frontend Integration**: Chat page checks both authorization AND issuer

✅ **Data Customization**: You can update quotes/data anytime by editing JSON and re-encrypting

### Environment Variables Needed

```bash
# frontend/.env.local
NEXT_PUBLIC_INFT_ISSUER_ADDRESS=0xYourWalletAddressHere
```

### Files Modified

- ✅ `0g-INFT/contracts/INFT.sol` - Added `tokenMinter` mapping
- ✅ `frontend/hooks/useINFT.ts` - Added issuer validation
- ✅ `frontend/pages/chat.tsx` - Integrated issuer check

### Next Steps

1. **Set your wallet address** in `frontend/.env.local`
2. **Redeploy INFT contract** with minter tracking
3. **Mint your INFTs** - they'll be tracked as yours
4. **Customize your data** in `quotes.json` and re-encrypt

---

## Troubleshooting

### Issue: All users show "No hosted model"

**Solution**: Check that `NEXT_PUBLIC_INFT_ISSUER_ADDRESS` is set correctly in `.env.local`

### Issue: Wrong issuer detected

**Solution**: The contract records `msg.sender` when minting. Make sure YOU are calling the mint function, not a proxy contract.

### Issue: Backend not loading new quotes

**Solution**: 
1. Verify `quotes.enc` was updated (check file timestamp)
2. Restart the backend process
3. Check backend logs for encryption key loading

### Issue: Can't call mint function

**Solution**: Only the contract owner can mint. Make sure you're using the wallet that deployed the contract or was set as owner.

---

For more details:
- Storage/Quotes Update: See `0g-INFT/storage/UPDATE_QUOTES_GUIDE.md`
- Contract Documentation: See `0g-INFT/README.md`
- Backend API: See `backend/README.md`

