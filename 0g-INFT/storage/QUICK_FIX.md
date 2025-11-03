# Quick Fix: Insufficient Funds Error

## ✅ Good News!

Your encryption worked! The file `quotes.enc` was created successfully. The upload failure is **not a problem** for local development.

## What Happened?

The script tried to upload to 0G Storage but your wallet `0x32F91E4E2c60A9C16cAE736D3b42152B331c147F` needs testnet tokens (~0.003 OG).

## ✨ Solution: Skip Upload (Recommended for Now)

The backend already works with the local encrypted file! Just run the script again:

```bash
cd 0g-INFT
npx ts-node storage/encrypt.ts --skip-upload
```

Or set environment variable:
```bash
SKIP_UPLOAD=true npx ts-node storage/encrypt.ts
```

## What You'll Get

```
🎯 PHASE 1 RESULTS:
============================================================
encryptedURI: local://quotes.enc
storageRootHash: 0x6aa2f736069f0bd4508722f25a8365cc38919cb7d4bd342735aa88b477becbf5
metadataHash: 0x6aa2f736069f0bd4508722f25a8365cc38919cb7d4bd342735aa88b477becbf5
encryptionKey: 0x...
iv: 0x...
tag: 0x...

✅ Phase 1 completed successfully!
📁 Backend will use: 0g-INFT/storage/quotes.enc
```

## How Backend Works

The backend (`backend/index.ts` line 664) already checks for the local file first:

```typescript
const localPath = path.join(__dirname, '..', '0g-INFT', 'storage', 'quotes.enc');

if (fs.existsSync(localPath)) {
  console.log('Using local encrypted file');
  const encryptedData = fs.readFileSync(localPath);
  return this.decryptData(encryptedData);
}
```

So you can **proceed without uploading to 0G Storage**!

## Next Steps

### Option 1: Continue Without Upload (Easiest)

1. ✅ Your `quotes.enc` file already exists
2. ✅ Your `dev-keys.json` has the encryption keys  
3. ✅ Backend will automatically use the local file
4. **Just start the backend and frontend!**

```bash
# Terminal 1: Start Backend
cd backend
npm install
npm run dev

# Terminal 2: Start Frontend  
cd frontend
npm install
npm run dev
```

### Option 2: Get Testnet Tokens (For Production)

If you want to upload to 0G Storage later:

1. **Get testnet tokens** from 0G faucet
2. Send to your wallet: `0x32F91E4E2c60A9C16cAE736D3b42152B331c147F`
3. Run encryption **without** `--skip-upload` flag

```bash
# Check if you need more tokens
# Cost: ~0.003 OG per upload
npx ts-node storage/encrypt.ts
```

## Summary

✅ **Encryption**: SUCCESS  
✅ **Local file created**: `quotes.enc`  
✅ **Backend ready**: Will use local file  
⏭️ **Upload**: SKIPPED (not needed for development)  

You're good to go! 🚀

## Troubleshooting

### "File not found" error in backend

Make sure the path is correct:
```
0g-INFT/storage/quotes.enc
```

The backend looks for: `../0g-INFT/storage/quotes.enc` relative to `backend/` directory.

### Want to re-encrypt with new data?

1. Edit `0g-INFT/storage/quotes.json`
2. Run: `npx ts-node storage/encrypt.ts --skip-upload`
3. Restart backend

That's it!

