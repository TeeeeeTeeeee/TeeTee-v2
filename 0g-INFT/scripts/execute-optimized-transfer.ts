import { ethers } from 'hardhat';

async function main() {
  console.log('🚀 Executing Optimized Transfer...\n');

  const [signer] = await ethers.getSigners();
  
  // Contract addresses
  const INFT_ADDRESS = '0x67dDE9dF36Eb6725f265bc8A1908628e8d4AF9DA';
  
  // Get contract instance
  const inft = await ethers.getContractAt('INFT', INFT_ADDRESS);
  
  // Transfer parameters
  const from = signer.address;
  const to = '0xAe7C6fDB1d03E8bc73A32D2C8B7BafA057d30437';
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
    hash: crypto.createHash('sha256').update('test_proof').digest('hex'),
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
    // First estimate gas
    console.log('\n⛽ Estimating gas...');
    const gasEstimate = await inft.transfer.estimateGas(from, to, tokenId, sealedKey, proof);
    console.log('Gas estimate:', gasEstimate.toString());
    
    // Add 20% buffer to gas estimate
    const gasLimit = BigInt(Math.floor(Number(gasEstimate) * 1.2));
    console.log('Gas limit (with buffer):', gasLimit.toString());
    
    // Execute the transfer
    console.log('\n🚀 Executing transfer...');
    const tx = await inft.transfer(from, to, tokenId, sealedKey, proof, {
      gasLimit: gasLimit
    });
    
    console.log('Transaction hash:', tx.hash);
    console.log('Waiting for confirmation...');
    
    const receipt = await tx.wait();
    console.log('✅ Transfer successful!');
    console.log('Gas used:', receipt.gasUsed.toString());
    console.log('Block number:', receipt.blockNumber);
    
  } catch (error) {
    console.log('❌ Transfer failed:');
    console.log('Error:', error.message);
    
    if (error.data) {
      console.log('Error data:', error.data);
    }
    
    // Additional debugging
    if (error.message.includes('gas')) {
      console.log('\n⛽ Gas-related failure detected');
      try {
        const gasEstimate = await inft.transfer.estimateGas(from, to, tokenId, sealedKey, proof);
        console.log('Gas estimate would be:', gasEstimate.toString());
      } catch (estError) {
        console.log('Gas estimation also failed:', estError.message);
      }
    }
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
