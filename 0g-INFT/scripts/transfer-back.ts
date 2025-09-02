import { ethers } from 'hardhat';

async function main() {
  console.log('🔄 Transferring Token Back to Original Owner...\n');

  // We need to use the second wallet that now owns the token
  const provider = ethers.provider;
  const privateKey = '0x8da4ef21b864d2cc526dbdb2a120bd2874c36c9d0a1fb7f8c63d7f7a8b41de8f'; // Second wallet private key
  const signer = new ethers.Wallet(privateKey, provider);
  
  console.log('Using wallet:', signer.address);
  
  // Contract address
  const INFT_ADDRESS = '0x67dDE9dF36Eb6725f265bc8A1908628e8d4AF9DA';
  
  // Get contract instance
  const inft = await ethers.getContractAt('INFT', INFT_ADDRESS);
  const inftWithSigner = inft.connect(signer);
  
  // Transfer parameters
  const from = signer.address; // Current owner
  const to = '0x32F91E4E2c60A9C16cAE736D3b42152B331c147F'; // Original owner
  const tokenId = 1;
  
  // Generate fresh optimized proof and sealed key
  const crypto = require('crypto');
  
  // Compact sealed key
  const sealedKeyData = {
    id: tokenId,
    to: to,
    ts: Date.now(),
    key: crypto.randomBytes(32).toString('hex'),
    nonce: crypto.randomBytes(12).toString('hex')
  };
  const sealedKey = '0x' + Buffer.from(JSON.stringify(sealedKeyData)).toString('hex');
  
  // Compact proof
  const proofData = {
    v: '1.0',
    type: 'TEE_STUB',
    ts: Date.now(),
    hash: crypto.createHash('sha256').update('transfer_back').digest('hex'),
    sig: crypto.randomBytes(16).toString('hex')
  };
  const proof = '0x' + Buffer.from(JSON.stringify(proofData)).toString('hex');
  
  console.log('📋 Transfer Parameters:');
  console.log('From:', from);
  console.log('To:', to);
  console.log('Token ID:', tokenId);
  console.log('Sealed Key Length:', sealedKey.length);
  console.log('Proof Length:', proof.length);
  
  try {
    // Check current ownership
    const currentOwner = await inft.ownerOf(tokenId);
    console.log('Current owner:', currentOwner);
    
    if (currentOwner.toLowerCase() !== from.toLowerCase()) {
      throw new Error(`Token not owned by expected wallet. Owner: ${currentOwner}, Expected: ${from}`);
    }
    
    // Execute the transfer
    console.log('\n🚀 Executing transfer back...');
    const tx = await inftWithSigner.transfer(from, to, tokenId, sealedKey, proof);
    
    console.log('Transaction hash:', tx.hash);
    console.log('Waiting for confirmation...');
    
    const receipt = await tx.wait();
    console.log('✅ Transfer back successful!');
    console.log('Gas used:', receipt.gasUsed.toString());
    console.log('Block number:', receipt.blockNumber);
    
    // Verify new ownership
    const newOwner = await inft.ownerOf(tokenId);
    console.log('New owner:', newOwner);
    
  } catch (error) {
    console.log('❌ Transfer failed:');
    console.log('Error:', error.message);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
