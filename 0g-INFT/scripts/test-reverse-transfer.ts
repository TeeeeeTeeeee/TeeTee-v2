import { ethers } from 'hardhat';

async function main() {
  console.log('🔄 Testing Reverse Transfer...\n');

  // We need to use the wallet that owns the token now
  const provider = new ethers.JsonRpcProvider('https://evmrpc-testnet.0g.ai');
  const wallet = new ethers.Wallet(process.env.PRIVATE_KEY!, provider);
  
  console.log('Wallet address:', wallet.address);
  
  // Contract addresses
  const INFT_ADDRESS = '0x67dDE9dF36Eb6725f265bc8A1908628e8d4AF9DA';
  
  // Get contract instance
  const inft = await ethers.getContractAt('INFT', INFT_ADDRESS);
  const inftWithWallet = inft.connect(wallet);
  
  // Check current status
  const tokenId = 1;
  const currentOwner = await inft.ownerOf(tokenId);
  console.log('Current token owner:', currentOwner);
  console.log('Wallet address:', wallet.address);
  console.log('Ownership match:', currentOwner.toLowerCase() === wallet.address.toLowerCase());
  
  // Check authorization
  const isAuthorized = await inft.isAuthorized(tokenId, wallet.address);
  console.log('Wallet authorized:', isAuthorized);
  
  if (currentOwner.toLowerCase() !== wallet.address.toLowerCase()) {
    console.log('❌ Wallet does not own the token');
    return;
  }
  
  if (!isAuthorized) {
    console.log('🔐 Authorizing wallet first...');
    try {
      const authTx = await inftWithWallet.authorizeUsage(tokenId, wallet.address);
      console.log('Authorization tx:', authTx.hash);
      await authTx.wait();
      console.log('✅ Authorization complete');
    } catch (error) {
      console.log('❌ Authorization failed:', error.message);
      return;
    }
  }
  
  // Now try the transfer
  const from = wallet.address;
  const to = '0x32F91E4E2c60A9C16cAE736D3b42152B331c147F';
  
  // Generate fresh proof and sealed key
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
    hash: crypto.createHash('sha256').update('reverse_transfer').digest('hex'),
    sig: crypto.randomBytes(16).toString('hex')
  };
  const proof = '0x' + Buffer.from(JSON.stringify(proofData)).toString('hex');
  
  console.log('\n📋 Transfer Parameters:');
  console.log('From:', from);
  console.log('To:', to);
  console.log('Token ID:', tokenId);
  console.log('Sealed Key Length:', sealedKey.length);
  console.log('Proof Length:', proof.length);
  
  try {
    // Estimate gas first
    console.log('\n⛽ Estimating gas...');
    const gasEstimate = await inftWithWallet.transfer.estimateGas(from, to, tokenId, sealedKey, proof);
    console.log('Gas estimate:', gasEstimate.toString());
    
    // Execute the transfer
    console.log('\n🚀 Executing reverse transfer...');
    const tx = await inftWithWallet.transfer(from, to, tokenId, sealedKey, proof, {
      gasLimit: BigInt(Math.floor(Number(gasEstimate) * 1.2))
    });
    
    console.log('Transaction hash:', tx.hash);
    console.log('Waiting for confirmation...');
    
    const receipt = await tx.wait();
    console.log('✅ Reverse transfer successful!');
    console.log('Gas used:', receipt.gasUsed.toString());
    console.log('Block number:', receipt.blockNumber);
    
    // Verify the transfer
    const newOwner = await inft.ownerOf(tokenId);
    console.log('New token owner:', newOwner);
    
  } catch (error) {
    console.log('❌ Transfer failed:');
    console.log('Error:', error.message);
    
    if (error.data) {
      console.log('Error data:', error.data);
    }
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
