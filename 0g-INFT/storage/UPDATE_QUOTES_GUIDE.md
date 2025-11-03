# How to Update Quotes/Data in 0G Storage

## Quick Guide

### Step 1: Edit the Quotes Data

Edit the `quotes.json` file with your new quotes:

```json
{
  "version": "1.0.0",
  "quotes": [
    "Your first custom quote here",
    "Your second custom quote here",
    "Add as many as you want"
  ],
  "metadata": {
    "created": "2025-11-03T00:00:00.000Z",
    "description": "Your custom description",
    "totalQuotes": 3,
    "category": "your-category"
  }
}
```

### Step 2: Re-encrypt the Data

Run the encryption script to generate new encrypted data:

```bash
# From the 0g-INFT directory
cd 0g-INFT
npx ts-node storage/encrypt.ts
```

This will:
- ✅ Read your updated `quotes.json`
- ✅ Generate new encryption keys
- ✅ Create new `quotes.enc` (encrypted file)
- ✅ Create new `dev-keys.json` (development keys)
- ✅ Output new `encryptedURI` and `metadataHash`

### Step 3: Update Your INFT

Use the output to mint a new INFT or update existing one:

```typescript
// Example output from encrypt.ts
encryptedURI: "0g://storage/quotes_1730592000000.enc"
metadataHash: "0x1f626cda1593594aea14fcc7edfd015e01fbd0a2eccc3032d553998e0a2a8f4b"
```

Then mint a new INFT with these values:

```solidity
// On your deployed INFT contract
mint(
  recipientAddress,
  "0g://storage/quotes_1730592000000.enc",  // new encryptedURI
  "0x1f626cda1593594aea14fcc7edfd015e01fbd0a2eccc3032d553998e0a2a8f4b"  // new metadataHash
)
```

## Customization Examples

### Example 1: Tech Quotes
```json
{
  "version": "1.0.0",
  "quotes": [
    "Talk is cheap. Show me the code. - Linus Torvalds",
    "First, solve the problem. Then, write the code. - John Johnson",
    "Code is like humor. When you have to explain it, it's bad. - Cory House"
  ],
  "metadata": {
    "created": "2025-11-03T00:00:00.000Z",
    "description": "Programming wisdom",
    "totalQuotes": 3,
    "category": "tech"
  }
}
```

### Example 2: Business Advice
```json
{
  "version": "1.0.0",
  "quotes": [
    "The best time to plant a tree was 20 years ago. The second best time is now. - Chinese Proverb",
    "Don't be afraid to give up the good to go for the great. - John D. Rockefeller",
    "If you don't value your time, neither will others. - Kim Garst"
  ],
  "metadata": {
    "created": "2025-11-03T00:00:00.000Z",
    "description": "Business wisdom",
    "totalQuotes": 3,
    "category": "business"
  }
}
```

### Example 3: Any Custom Data Structure

You can actually store ANY JSON data, not just quotes:

```json
{
  "version": "1.0.0",
  "data": {
    "knowledge_base": [
      {
        "topic": "AI",
        "content": "Artificial Intelligence is..."
      },
      {
        "topic": "Blockchain",
        "content": "Blockchain technology..."
      }
    ]
  },
  "metadata": {
    "created": "2025-11-03T00:00:00.000Z",
    "description": "Custom knowledge base",
    "totalEntries": 2,
    "category": "knowledge"
  }
}
```

## Backend Integration

After re-encrypting, the backend will automatically use the new encrypted data. The backend code at `backend/index.ts` line 662 loads the encrypted file:

```typescript
const localPath = path.join(__dirname, '..', '0g-INFT', 'storage', 'quotes.enc');
```

So as long as you update `quotes.enc`, the backend will serve the new data!

## Testing Your Changes

1. **Update quotes.json** with your new data
2. **Run encrypt.ts** to generate new encrypted file
3. **Restart backend** to pick up new encrypted data
   ```bash
   cd backend
   npm run dev
   ```
4. **Test inference** via the frontend or API

## Important Notes

⚠️ **Security**: The `dev-keys.json` file contains your encryption keys. In production:
- Never commit this file to version control
- Use secure key management systems
- Rotate keys regularly

📝 **Versioning**: Consider updating the `version` field when you make significant changes to your data structure.

🔄 **Multiple INFTs**: You can create multiple INFTs with different data by:
1. Creating different JSON files (quotes1.json, quotes2.json, etc.)
2. Encrypting each separately
3. Minting separate INFTs with different encryptedURI values

