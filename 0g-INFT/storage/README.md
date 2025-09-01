# Phase 1 - Storage Payload & Encryption

This directory contains the implementation of Phase 1 of the 0G INFT Quote Generator project, focusing on data encryption and storage preparation according to ERC-7857 specifications.

## Overview

The storage system implements AES-GCM encryption for quote data and prepares it for upload to 0G Storage network, following the official 0G INFT documentation.

## Files

- `quotes.json` - Sample inspirational quotes dataset with metadata
- `encrypt.ts` - Main encryption script implementing AES-GCM with keccak256 hashing
- `quotes.enc` - Generated encrypted binary file (created after running encrypt.ts)
- `dev-keys.json` - Development keys and metadata (created after running encrypt.ts)

## Usage

### Prerequisites

Ensure you have the project dependencies installed:
```bash
npm install
```

### Running Encryption

Execute the encryption script from the project root:
```bash
# From /Users/marcus/Projects/0g-INFT
npx ts-node storage/encrypt.ts
```

### Expected Output

The script will:
1. 📖 Read the quotes.json file
2. 🔑 Generate a random 256-bit AES key
3. 🎲 Generate a random 96-bit IV
4. 🔐 Encrypt data using AES-GCM
5. 📊 Compute keccak256 metadata hash
6. 💾 Save encrypted file as quotes.enc
7. 🌐 Upload to storage (currently mocked)
8. 📤 Output encryptedURI and metadataHash

### Sample Output

```
🎯 PHASE 1 RESULTS:
============================================================
encryptedURI: 0g://storage/quotes_1755609538556.enc
metadataHash: 0x1f626cda1593594aea14fcc7edfd015e01fbd0a2eccc3032d553998e0a2a8f4b
encryptionKey: 0xe244c32a55d603b9aa4c4a9edef36f01f4d892120bfb53530d7b08d46a3a41f3
iv: 0xef0b00af897f36c63b3ccfa9
tag: 0xb4ad0db8001e4ae212979ea2d1077a23
```

## File Structure

### quotes.json
Contains the source data with:
- `version`: Schema version for compatibility
- `quotes`: Array of inspirational quote strings
- `metadata`: Descriptive information about the dataset

### quotes.enc
Binary file containing:
- IV (12 bytes) - Initialization Vector
- Tag (16 bytes) - Authentication tag
- Encrypted Data (variable) - AES-GCM encrypted quotes.json

### dev-keys.json
Development file containing:
- `encryptedURI`: Storage reference (mocked for now)
- `metadataHash`: keccak256 hash of encrypted data
- `key`: AES-256 encryption key (hex format)
- `iv`: Initialization vector (hex format)
- `tag`: Authentication tag (hex format)
- `timestamp`: Creation timestamp

## Encryption Details

### Algorithm: AES-256-GCM
- **Key Size**: 256 bits (32 bytes)
- **IV Size**: 96 bits (12 bytes) - Recommended for GCM
- **Tag Size**: 128 bits (16 bytes) - Authentication tag
- **AAD**: "0G-INFT-ERC7857" - Additional authenticated data

### Hash Function: keccak256
- Used for computing metadata hash per Ethereum standards
- Applied to the encrypted data blob
- Required by ERC-7857 specification

## 0G Storage Integration

### Current Status
🚨 **MOCK IMPLEMENTATION** - The upload function currently returns a mock URI.

### Next Steps
To integrate with actual 0G Storage:

1. **Install 0G Storage SDK**:
   ```bash
   npm install @0glabs/0g-storage-sdk
   ```

2. **Environment Configuration**:
   Add to `.env`:
   ```env
   # 0G Storage Configuration
   ZG_STORAGE_ENDPOINT=https://storage-testnet.0g.ai
   ZG_STORAGE_API_KEY=your_api_key_here
   ```

3. **Update Upload Function**:
   Replace the `uploadTo0GStorage` function in `encrypt.ts` with actual 0G Storage SDK calls.

### Environment Variables

For 0G Storage integration:
- `ZG_STORAGE_RPC` - 0G Storage RPC endpoint (default: https://evmrpc-testnet.0g.ai)
- `ZG_STORAGE_INDEXER` - 0G Storage indexer endpoint (default: https://indexer-storage-testnet-turbo.0g.ai)
- `ZG_STORAGE_PRIVATE_KEY` - Private key for storage transactions
- `PRIVATE_KEY` - Fallback private key if ZG_STORAGE_PRIVATE_KEY not set

### Wallet Requirements

⚠️ **Important**: Your wallet needs 0G testnet tokens to pay for storage fees.

- **Estimated cost**: ~0.003 0G tokens per file upload
- **Faucet**: Get testnet tokens from 0G faucet
- **Current wallet**: The script will show your wallet address for funding

## Security Notes

⚠️ **Development Only**: The current implementation saves encryption keys to `dev-keys.json` for development purposes. In production:

- Keys should be managed through secure key management systems
- Never commit actual encryption keys to version control
- Use environment variables or secure vaults for key storage
- Implement proper key rotation and access controls

## Integration with INFT Contract

The outputs from this phase are used in subsequent phases:

- `encryptedURI` → Stored in INFT contract for each token
- `metadataHash` → Used for on-chain verification
- `encryptionKey` → Used for re-encryption during transfers (TEE/ZKP)

## Phase 1 Success Criteria

✅ **Completed**:
- [x] Create storage/quotes.json with sample data
- [x] Implement AES-GCM encryption with 256-bit keys
- [x] Add keccak256 metadata hash computation
- [x] Generate encryptedURI and metadataHash output
- [x] Save encrypted file and development keys

🚧 **Pending**:
- [ ] Integrate real 0G Storage SDK
- [ ] Test blob retrieval from storage gateway
- [ ] Configure production environment variables

## Next: Phase 2

Once Phase 1 is complete with real 0G Storage integration, proceed to Phase 2: "ERC-7857 Contract (Spec Surface) + Oracle Interface" which will implement the on-chain INFT contract using these encrypted storage references.
