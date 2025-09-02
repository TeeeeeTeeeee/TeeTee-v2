import { ethers } from 'hardhat';

async function main() {
  console.log('🔍 Debugging Failed Transfer...\n');

  // The exact parameters from the failed frontend transfer
  const from = '0xAe7C6fDB1d03E8bc73A32D2C8B7BafA057d30437';
  const to = '0x32F91E4E2c60A9C16cAE736D3b42152B331c147F';
  const tokenId = 1;
  
  // Get current signer (should be the original owner to simulate)
  const [signer] = await ethers.getSigners();
  console.log('Current signer:', signer.address);
  console.log('Transfer from address:', from);
  console.log('Signer matches from address:', signer.address.toLowerCase() === from.toLowerCase());
  
  // Get contract instance
  const INFT_ADDRESS = '0x67dDE9dF36Eb6725f265bc8A1908628e8d4AF9DA';
  const inft = await ethers.getContractAt('INFT', INFT_ADDRESS);
  
  // Check current ownership
  console.log('\n📋 Current State:');
  const owner = await inft.ownerOf(tokenId);
  console.log('Token owner:', owner);
  console.log('From address:', from);
  console.log('Ownership match:', owner.toLowerCase() === from.toLowerCase());
  
  // Check if the current user can transfer (is owner or approved)
  console.log('Can current signer transfer:', owner.toLowerCase() === signer.address.toLowerCase());
  
  // Generate test data similar to frontend
  const crypto = require('crypto');
  
  const sealedKeyData = {
    id: tokenId,
    to: to,
    ts: Date.now(),
    key: crypto.randomBytes(32).toString('hex'),
    nonce: crypto.randomBytes(12).toString('hex')
  };
  const sealedKey = '0x' + Buffer.from(JSON.stringify(sealedKeyData)).toString('hex');
  
  const proofData = {
    v: '1.0',
    type: 'TEE_STUB',
    ts: Date.now(),
    hash: crypto.createHash('sha256').update('test_proof').digest('hex'),
    sig: crypto.randomBytes(16).toString('hex')
  };
  const proof = '0x' + Buffer.from(JSON.stringify(proofData)).toString('hex');
  
  console.log('\n🧪 Testing with fresh data:');
  console.log('Sealed key length:', sealedKey.length);
  console.log('Proof length:', proof.length);
  
  try {
    // Test static call first
    console.log('\n🔍 Testing static call...');
    await inft.transfer.staticCall(from, to, tokenId, sealedKey, proof);
    console.log('✅ Static call succeeded');
    
    // Try actual execution
    console.log('\n🚀 Testing actual execution...');
    const gasEstimate = await inft.transfer.estimateGas(from, to, tokenId, sealedKey, proof);
    console.log('Gas estimate:', gasEstimate.toString());
    
    const tx = await inft.transfer(from, to, tokenId, sealedKey, proof, {
      gasLimit: BigInt(Math.floor(Number(gasEstimate) * 1.2))
    });
    
    console.log('Transaction hash:', tx.hash);
    const receipt = await tx.wait();
    console.log('✅ Transfer successful!');
    console.log('Gas used:', receipt.gasUsed.toString());
    
  } catch (error) {
    console.log('❌ Transfer failed:');
    console.log('Error message:', error.message);
    
    // Check if this is a permission issue
    if (error.message.includes('not owner') || error.message.includes('not approved')) {
      console.log('\n🔍 Permission issue detected');
    }
    
    // Check if this is specific to the signer mismatch
    if (signer.address.toLowerCase() !== from.toLowerCase()) {
      console.log('\n⚠️  ISSUE: Current signer does not match the from address!');
      console.log('This means the frontend is connected to a different wallet than the transaction origin');
    }
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
