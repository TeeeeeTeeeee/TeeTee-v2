import { ethers } from 'hardhat';

async function main() {
  console.log('🧪 Testing Fixed Transfer Implementation...\n');

  const [signer] = await ethers.getSigners();
  
  // Fixed contract addresses
  const INFT_FIXED_ADDRESS = '0x18db2ED477A25Aac615D803aE7be1d3598cdfF95';
  
  // Get contract instance
  const inftFixed = await ethers.getContractAt('INFTFixed', INFT_FIXED_ADDRESS);
  
  console.log('📋 Testing with Fixed Contracts:');
  console.log('INFT Fixed Address:', INFT_FIXED_ADDRESS);
  console.log('Signer:', signer.address);
  
  // First, mint a token for testing
  console.log('\n🎯 Minting test token...');
  const mintTx = await inftFixed.mint(
    signer.address,
    '0g://storage/test-fixed-token',
    ethers.keccak256(ethers.toUtf8Bytes('test-metadata'))
  );
  await mintTx.wait();
  console.log('✅ Token minted');
  
  const tokenId = await inftFixed.getCurrentTokenId();
  console.log('Token ID:', tokenId.toString());
  
  // Authorize the owner for usage
  console.log('\n🔐 Authorizing owner for usage...');
  const authTx = await inftFixed.authorizeUsage(tokenId, signer.address);
  await authTx.wait();
  console.log('✅ Owner authorized');
  
  // Test transfer parameters
  const from = signer.address;
  const to = '0xAe7C6fDB1d03E8bc73A32D2C8B7BafA057d30437';
  
  // Generate compact proof and sealed key
  const crypto = require('crypto');
  
  const sealedKeyData = {
    id: Number(tokenId),
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
    hash: crypto.createHash('sha256').update('test_proof_fixed').digest('hex'),
    sig: crypto.randomBytes(16).toString('hex')
  };
  const proof = '0x' + Buffer.from(JSON.stringify(proofData)).toString('hex');
  
  console.log('\n📊 Transfer Parameters:');
  console.log('From:', from);
  console.log('To:', to);
  console.log('Token ID:', tokenId.toString());
  console.log('Sealed Key Length:', sealedKey.length);
  console.log('Proof Length:', proof.length);
  
  try {
    // First estimate gas
    console.log('\n⛽ Estimating gas with fixed implementation...');
    const gasEstimate = await inftFixed.transfer.estimateGas(from, to, tokenId, sealedKey, proof);
    console.log('Gas estimate:', gasEstimate.toString());
    
    // Execute the transfer
    console.log('\n🚀 Executing fixed transfer...');
    const tx = await inftFixed.transfer(from, to, tokenId, sealedKey, proof, {
      gasLimit: BigInt(Math.floor(Number(gasEstimate) * 1.2))
    });
    
    console.log('Transaction hash:', tx.hash);
    const receipt = await tx.wait();
    
    console.log('✅ Fixed transfer successful!');
    console.log('Gas used:', receipt.gasUsed.toString());
    console.log('Gas estimate vs actual:', gasEstimate.toString(), 'vs', receipt.gasUsed.toString());
    
    // Verify transfer
    const newOwner = await inftFixed.ownerOf(tokenId);
    console.log('New owner:', newOwner);
    console.log('Transfer verified:', newOwner.toLowerCase() === to.toLowerCase());
    
  } catch (error) {
    console.log('❌ Transfer failed:');
    console.log('Error:', error.message);
    
    // Check if we get better error messages now
    if (error.data) {
      console.log('Error data:', error.data);
    }
    
    // Try to decode custom errors
    if (error.message.includes('CallerNotAuthorized')) {
      console.log('🔍 Decoded: Caller not authorized');
    } else if (error.message.includes('InvalidTransferProof')) {
      console.log('🔍 Decoded: Invalid transfer proof');
    } else if (error.message.includes('OracleCallFailed')) {
      console.log('🔍 Decoded: Oracle call failed');
    }
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
