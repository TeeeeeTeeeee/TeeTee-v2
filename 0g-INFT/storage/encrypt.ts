import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { keccak256, JsonRpcProvider, Wallet } from 'ethers';
const { Indexer, ZgFile } = require('@0glabs/0g-ts-sdk');
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: path.join(__dirname, '..', '.env') });

/**
 * Phase 1 - Storage Payload & Encryption
 * 
 * This script implements AES-GCM encryption for the quotes.json payload
 * and computes the metadata hash as required by ERC-7857 INFT specification.
 * 
 * Functions:
 * - Generate random 256-bit AES key
 * - Encrypt quotes.json using AES-GCM
 * - Compute keccak256 hash of encrypted data
 * - Prepare for 0G Storage upload
 */

interface EncryptionResult {
  encryptedData: Buffer;
  iv: Buffer;
  tag: Buffer;
  key: Buffer;
  metadataHash: string;
}

/**
 * Encrypts the quotes.json file using AES-GCM encryption
 * @param inputPath Path to the input JSON file
 * @returns Encryption result with all necessary components
 */
function encryptQuotesData(inputPath: string): EncryptionResult {
  console.log('📖 Reading quotes data from:', inputPath);
  
  // Read the quotes.json file
  const quotesData = fs.readFileSync(inputPath, 'utf8');
  console.log('✅ Loaded quotes data:', quotesData.length, 'bytes');
  
  // Generate random 256-bit key for AES-GCM
  const key = crypto.randomBytes(32); // 256 bits
  console.log('🔑 Generated 256-bit encryption key');
  
  // Generate random 96-bit IV (recommended for GCM)
  const iv = crypto.randomBytes(12); // 96 bits
  console.log('🎲 Generated random IV');
  
  // Create cipher with proper GCM mode
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  
  // Encrypt the data
  let encryptedData = cipher.update(quotesData, 'utf8');
  encryptedData = Buffer.concat([encryptedData, cipher.final()]);
  
  // Get the authentication tag
  const tag = cipher.getAuthTag();
  
  console.log('🔐 Encryption completed:');
  console.log('  - Original size:', quotesData.length, 'bytes');
  console.log('  - Encrypted size:', encryptedData.length, 'bytes');
  console.log('  - IV length:', iv.length, 'bytes');
  console.log('  - Tag length:', tag.length, 'bytes');
  
  // Compute metadata hash using keccak256 (Ethereum standard)
  const metadataHash = keccak256(encryptedData);
  console.log('📊 Computed metadataHash (keccak256):', metadataHash);
  
  return {
    encryptedData,
    iv,
    tag,
    key,
    metadataHash
  };
}

/**
 * Saves the encrypted data to quotes.enc file
 * @param result Encryption result
 * @param outputDir Directory to save the encrypted file
 */
function saveEncryptedFile(result: EncryptionResult, outputDir: string): string {
  const encryptedPath = path.join(outputDir, 'quotes.enc');
  
  // Create a combined buffer: [IV][TAG][ENCRYPTED_DATA]
  const combinedBuffer = Buffer.concat([
    result.iv,
    result.tag,
    result.encryptedData
  ]);
  
  fs.writeFileSync(encryptedPath, combinedBuffer);
  console.log('💾 Saved encrypted file to:', encryptedPath);
  console.log('📏 Combined file size:', combinedBuffer.length, 'bytes');
  
  return encryptedPath;
}

/**
 * Uploads file to 0G Storage using the official SDK
 * @param encryptedFilePath Path to the encrypted file
 * @returns Storage URI with root hash
 */
async function uploadTo0GStorage(encryptedFilePath: string): Promise<{uri: string, rootHash: string, txHash: string}> {
  console.log('🌐 Uploading to 0G Storage...');
  
  // Get configuration from environment
  const evmRpc = process.env.ZG_STORAGE_RPC || 'https://evmrpc-testnet.0g.ai';
  const indexerRpc = process.env.ZG_STORAGE_INDEXER || 'https://indexer-storage-testnet-turbo.0g.ai';
  const privateKey = process.env.ZG_STORAGE_PRIVATE_KEY || process.env.PRIVATE_KEY;
  
  if (!privateKey) {
    throw new Error('Private key not found. Please set ZG_STORAGE_PRIVATE_KEY or PRIVATE_KEY in .env file');
  }
  
  console.log('📡 Connecting to 0G Storage network...');
  console.log('  - EVM RPC:', evmRpc);
  console.log('  - Indexer RPC:', indexerRpc);
  
  try {
    // Create provider and signer
    const provider = new JsonRpcProvider(evmRpc);
    const signer = new Wallet(privateKey, provider);
    
    console.log('👤 Using wallet address:', await signer.getAddress());
    
    // Create ZgFile from the encrypted file
    const file = await ZgFile.fromFilePath(encryptedFilePath);
    console.log('📁 Prepared file for upload:', encryptedFilePath);
    
    // Compute merkle tree and root hash
    const [tree, treeErr] = await file.merkleTree();
    if (treeErr) {
      throw new Error(`Failed to compute merkle tree: ${treeErr}`);
    }
    
    const rootHash = tree.rootHash();
    console.log('🌳 Computed merkle root hash:', rootHash);
    
    // Initialize indexer and upload
    const indexer = new Indexer(indexerRpc);
    console.log('📤 Starting upload to 0G Storage...');
    
    const [tx, uploadErr] = await indexer.upload(file, evmRpc, signer);
    
    if (uploadErr) {
      await file.close();
      // Check if it's a "too many data writing" error but transaction was successful
      if (uploadErr.toString().includes('too many data writing') && tx) {
        console.log('⚠️ Network congestion error, but transaction succeeded');
        console.log('✅ Upload transaction successful!');
        console.log('  - Transaction hash:', tx);
        console.log('  - Root hash:', rootHash);
        console.log('  - Status: File submitted to network (congestion during final verification)');
      } else {
        throw new Error(`Upload failed: ${uploadErr}`);
      }
    } else {
      console.log('✅ Upload successful!');
      console.log('  - Transaction hash:', tx);
      console.log('  - Root hash:', rootHash);
    }
    
    // Clean up
    await file.close();
    
    // Create URI in standard format
    const uri = `0g://storage/${rootHash}`;
    
    return { uri, rootHash, txHash: tx };
    
  } catch (error) {
    console.error('❌ 0G Storage upload failed:', error);
    throw error;
  }
}

/**
 * Main execution function
 */
async function main() {
  console.log('🚀 Starting Phase 1 - Storage Payload & Encryption');
  console.log('=' .repeat(60));
  
  try {
    const storageDir = __dirname;
    const quotesPath = path.join(storageDir, 'quotes.json');
    
    // Check if quotes.json exists
    if (!fs.existsSync(quotesPath)) {
      throw new Error(`quotes.json not found at ${quotesPath}`);
    }
    
    // Step 1: Encrypt the quotes data
    const encryptionResult = encryptQuotesData(quotesPath);
    
    // Step 2: Save encrypted file
    const encryptedPath = saveEncryptedFile(encryptionResult, storageDir);
    
    // Step 3: Upload to 0G Storage
    const uploadResult = await uploadTo0GStorage(encryptedPath);
    
    // Step 4: Output final results
    console.log('\n🎯 PHASE 1 RESULTS:');
    console.log('=' .repeat(60));
    console.log('encryptedURI:', uploadResult.uri);
    console.log('storageRootHash:', uploadResult.rootHash);
    console.log('transactionHash:', uploadResult.txHash);
    console.log('metadataHash:', encryptionResult.metadataHash);
    console.log('encryptionKey:', `0x${encryptionResult.key.toString('hex')}`);
    console.log('iv:', `0x${encryptionResult.iv.toString('hex')}`);
    console.log('tag:', `0x${encryptionResult.tag.toString('hex')}`);
    console.log('\n✅ Phase 1 completed successfully!');
    
    // Save key and metadata for development use
    const devDataPath = path.join(storageDir, 'dev-keys.json');
    const devData = {
      encryptedURI: uploadResult.uri,
      storageRootHash: uploadResult.rootHash,
      transactionHash: uploadResult.txHash,
      metadataHash: encryptionResult.metadataHash,
      key: `0x${encryptionResult.key.toString('hex')}`,
      iv: `0x${encryptionResult.iv.toString('hex')}`,
      tag: `0x${encryptionResult.tag.toString('hex')}`,
      timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync(devDataPath, JSON.stringify(devData, null, 2));
    console.log('🔧 Development keys saved to:', devDataPath);
    
  } catch (error) {
    console.error('❌ Error during encryption process:', error);
    process.exit(1);
  }
}

// Run the main function if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { encryptQuotesData, saveEncryptedFile, uploadTo0GStorage };
