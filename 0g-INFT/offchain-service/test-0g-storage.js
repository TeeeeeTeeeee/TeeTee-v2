#!/usr/bin/env node

/**
 * Test script for 0G Storage download functionality
 * 
 * This script tests the 0G Storage download implementation
 * by attempting to download a file using a known root hash.
 */

const { Indexer } = require('@0glabs/0g-ts-sdk');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

async function testDownload() {
  console.log('🧪 Testing 0G Storage Download Functionality');
  console.log('=' .repeat(50));

  try {
    // Configuration
    const indexerRpc = process.env.ZG_STORAGE_INDEXER || 'https://indexer-storage-testnet-turbo.0g.ai';
    const testRootHash = process.env.STORAGE_ROOT_HASH || '0xe3bf3e775364d9eb24fe11106ad035bebda6c2b0f2b0586ad4397246c864aedc';
    
    console.log('🔗 Indexer RPC:', indexerRpc);
    console.log('📋 Test Root Hash:', testRootHash);
    
    // Initialize indexer
    console.log('\n📡 Initializing 0G Storage Indexer...');
    const indexer = new Indexer(indexerRpc);
    console.log('✅ Indexer initialized successfully');
    
    // Create temp directory
    const tempDir = path.join(__dirname, 'temp');
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
      console.log('📁 Created temp directory');
    }
    
    // Download file
    const tempFile = path.join(tempDir, `test_download_${Date.now()}.enc`);
    console.log(`\n⬇️ Attempting to download file to: ${tempFile}`);
    console.log(`🔍 Download parameters: ${testRootHash}, ${tempFile}, true`);
    
    try {
      await indexer.download(testRootHash, tempFile, true);
    } catch (downloadError) {
      throw new Error(`Download failed: ${downloadError}`);
    }
    
    console.log('✅ Download completed successfully!');
    
    // Verify file exists and get stats
    if (fs.existsSync(tempFile)) {
      const stats = fs.statSync(tempFile);
      console.log(`📦 Downloaded file size: ${stats.size} bytes`);
      console.log(`📅 Modified: ${stats.mtime}`);
      
      // Read first few bytes to verify it's encrypted data
      const buffer = fs.readFileSync(tempFile);
      console.log(`🔍 First 16 bytes (hex): ${buffer.subarray(0, 16).toString('hex')}`);
      
      // Clean up
      fs.unlinkSync(tempFile);
      console.log('🧹 Cleaned up temp file');
    } else {
      throw new Error('Downloaded file not found');
    }
    
    console.log('\n✅ 0G Storage download test PASSED!');
    
  } catch (error) {
    console.error('\n❌ 0G Storage download test FAILED:');
    console.error('Error:', error.message);
    console.error('\nThis could be due to:');
    console.error('- Network connectivity issues');
    console.error('- Invalid root hash');
    console.error('- 0G Storage service unavailable');
    console.error('- File not found in storage');
    
    process.exit(1);
  }
}

// Run the test
if (require.main === module) {
  testDownload().catch(console.error);
}

module.exports = { testDownload };
