import { ethers } from 'hardhat';

async function main() {
  console.log('🔧 Setting up Token 1 for Transfer Testing...\n');

  const [signer] = await ethers.getSigners();
  
  // Fixed contract address
  const INFT_FIXED_ADDRESS = '0x18db2ED477A25Aac615D803aE7be1d3598cdfF95';
  
  // Get contract instance
  const inftFixed = await ethers.getContractAt('INFTFixed', INFT_FIXED_ADDRESS);
  
  const tokenId = 1;
  const currentOwner = '0xAe7C6fDB1d03E8bc73A32D2C8B7BafA057d30437';
  
  console.log('📋 Setup Details:');
  console.log('Token ID:', tokenId);
  console.log('Current Owner:', currentOwner);
  console.log('Your Address:', signer.address);
  
  // Option 1: Mint a new token for your wallet
  console.log('\n🎯 Minting Token 2 for your wallet...');
  try {
    const mintTx = await inftFixed.mint(
      signer.address,
      '0g://storage/test-token-2',
      ethers.keccak256(ethers.toUtf8Bytes('test-metadata-2'))
    );
    await mintTx.wait();
    console.log('✅ Token 2 minted for your wallet');
    
    // Authorize yourself for Token 2
    const authTx = await inftFixed.authorizeUsage(2, signer.address);
    await authTx.wait();
    console.log('✅ You are now authorized for Token 2');
    
    // Verify
    const newOwner = await inftFixed.ownerOf(2);
    const isAuthorized = await inftFixed.isAuthorized(2, signer.address);
    
    console.log('\n📊 Token 2 Status:');
    console.log('Owner:', newOwner);
    console.log('Your wallet authorized:', isAuthorized);
    console.log('Ready for transfer testing!');
    
  } catch (error) {
    console.log('❌ Setup failed:', error.message);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
