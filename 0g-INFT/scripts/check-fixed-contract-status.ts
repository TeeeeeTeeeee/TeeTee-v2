import { ethers } from 'hardhat';

async function main() {
  console.log('🔍 Checking Fixed Contract Status...\n');

  const [signer] = await ethers.getSigners();
  
  // Fixed contract address
  const INFT_FIXED_ADDRESS = '0x18db2ED477A25Aac615D803aE7be1d3598cdfF95';
  
  // Get contract instance
  const inftFixed = await ethers.getContractAt('INFTFixed', INFT_FIXED_ADDRESS);
  
  console.log('📋 Contract Info:');
  console.log('Fixed INFT Address:', INFT_FIXED_ADDRESS);
  console.log('Your Address:', signer.address);
  
  try {
    // Check current token ID
    const currentTokenId = await inftFixed.getCurrentTokenId();
    console.log('\n🎯 Token Status:');
    console.log('Current Token ID:', currentTokenId.toString());
    
    if (currentTokenId > 0) {
      // Check Token 1 details
      const tokenId = 1;
      const owner = await inftFixed.ownerOf(tokenId);
      const encryptedURI = await inftFixed.encryptedURI(tokenId);
      const metadataHash = await inftFixed.metadataHash(tokenId);
      
      console.log('\n�� Token 1 Details:');
      console.log('Owner:', owner);
      console.log('Encrypted URI:', encryptedURI);
      console.log('Metadata Hash:', metadataHash);
      
      // Check authorization
      const isAuthorized = await inftFixed.isAuthorized(tokenId, owner);
      console.log('Owner Authorized:', isAuthorized);
      
      // Get all authorized users
      const authorizedUsers = await inftFixed.authorizedUsersOf(tokenId);
      console.log('All Authorized Users:', authorizedUsers);
      
      console.log('\n✅ Token 1 is ready for testing!');
      console.log('You can use this token for transfer testing.');
      
    } else {
      console.log('\n❌ No tokens minted yet on fixed contract');
      console.log('Need to mint a new token first.');
    }
    
  } catch (error) {
    console.log('❌ Error checking contract:', error.message);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
