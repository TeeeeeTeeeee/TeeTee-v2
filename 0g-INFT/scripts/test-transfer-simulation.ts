import { ethers } from 'hardhat';

async function main() {
  console.log('🧪 Testing Transfer with Optimized Proof...\n');

  const [signer] = await ethers.getSigners();
  
  // Contract addresses
  const INFT_ADDRESS = '0x67dDE9dF36Eb6725f265bc8A1908628e8d4AF9DA';
  
  // Get contract instance
  const inft = await ethers.getContractAt('INFT', INFT_ADDRESS);
  
  // Test parameters (matching the failed transaction)
  const from = '0x32F91E4E2c60A9C16cAE736D3b42152B331c147F';
  const to = '0xAe7C6fDB1d03E8bc73A32D2C8B7BafA057d30437';
  const tokenId = 1;
  
  // Optimized sealed key and proof (compact format)
  const sealedKey = '0x7b226964223a312c22746f223a22307841653743366644423164303345386263373341333244324338423742616641303537643330343337222c227473223a313735353730313033303432362c226b6579223a22323335663838323331613464633a632366623336313339303863383836643565626136333461323338333438363333373064316232393062333365353333376636663665633330363831363338356232326532222c226e6f6e6365223a223631386236336330623636656563333635373364373633227d';
  
  const proof = '0x7b2276223a22312e30222c2274797065223a225445455f53545542222c227473223a313735353730313033303432362c2268617368223a2230666663356164396162396361386565633064353431623136346662393238336331313634383933383465643731626235396666333535333463333265643633222c22736967223a223963303132393762326664616539386465353136346436323438303665386561227d';
  
  try {
    console.log('📋 Testing transfer call...');
    console.log('From:', from);
    console.log('To:', to);
    console.log('Token ID:', tokenId);
    console.log('Sealed Key Length:', sealedKey.length);
    console.log('Proof Length:', proof.length);
    
    // First, let's test if we can call the function and see detailed error
    const result = await inft.transfer.staticCall(from, to, tokenId, sealedKey, proof);
    console.log('✅ Static call succeeded:', result);
    
  } catch (error) {
    console.log('❌ Static call failed:');
    console.log('Error:', error.message);
    
    // Try to decode the error
    if (error.data) {
      console.log('Error data:', error.data);
    }
    
    // Check specific failure conditions
    console.log('\n🔍 Checking failure conditions:');
    
    // Check ownership
    try {
      const owner = await inft.ownerOf(tokenId);
      console.log('Token owner:', owner);
      console.log('From address:', from);
      console.log('Ownership match:', owner.toLowerCase() === from.toLowerCase());
    } catch (e) {
      console.log('Failed to check ownership:', e.message);
    }
    
    // Check authorization
    try {
      const isAuthorized = await inft.isAuthorized(tokenId, signer.address);
      console.log('Caller authorized:', isAuthorized);
    } catch (e) {
      console.log('Failed to check authorization:', e.message);
    }
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
