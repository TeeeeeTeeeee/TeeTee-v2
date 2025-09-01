import { ethers } from 'hardhat';

async function main() {
  console.log('🔐 Authorizing new token owner...\n');

  // Get the current token owner (0xAe7C...)
  const newOwnerAddress = '0xAe7C6fDB1d03E8bc73A32D2C8B7BafA057d30437';
  const tokenId = 1;
  
  // But we need to authorize from the contract deployer account
  const [deployer] = await ethers.getSigners();
  console.log('Deployer address:', deployer.address);
  
  // Contract addresses  
  const INFT_ADDRESS = '0x67dDE9dF36Eb6725f265bc8A1908628e8d4AF9DA';
  
  // Get contract instance
  const inft = await ethers.getContractAt('INFT', INFT_ADDRESS);
  
  try {
    // Check current owner
    const currentOwner = await inft.ownerOf(tokenId);
    console.log('Current token owner:', currentOwner);
    console.log('New owner to authorize:', newOwnerAddress);
    console.log('Owner match:', currentOwner.toLowerCase() === newOwnerAddress.toLowerCase());
    
    if (currentOwner.toLowerCase() !== newOwnerAddress.toLowerCase()) {
      console.log('❌ Address mismatch! Token owner has changed.');
      return;
    }
    
    // Check current authorization status
    const isAuthorized = await inft.isAuthorized(tokenId, newOwnerAddress);
    console.log('Current authorization status:', isAuthorized);
    
    if (isAuthorized) {
      console.log('✅ Already authorized!');
      return;
    }
    
    // The owner needs to authorize themselves
    // Since we're using deployer account, we need to simulate what the new owner would do
    console.log('\n🔑 Note: The new owner needs to authorize themselves using their own wallet');
    console.log('Command for new owner to run:');
    console.log(`npx hardhat run scripts/authorize-self.ts --network galileo`);
    
    // Let's create that script
    console.log('\n📝 Creating authorization script for new owner...');
    
  } catch (error) {
    console.log('❌ Authorization check failed:', error.message);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
