import { ethers } from 'hardhat';

async function main() {
  console.log('🔍 Checking authorization status for Token 1...\n');

  const INFT_ADDRESS = '0x67dDE9dF36Eb6725f265bc8A1908628e8d4AF9DA';
  const inft = await ethers.getContractAt('INFT', INFT_ADDRESS);
  
  const tokenId = 1;
  
  // Wallets to check
  const wallets = [
    { name: 'Owner', address: '0x32F91E4E2c60A9C16cAE736D3b42152B331c147F' },
    { name: 'Wallet 2', address: '0xAe7C6fDB1d03E8bc73A32D2C8B7BafA057d30437' },
    { name: 'Current User', address: '0x17f5afc78EFF841fB8c7384150658777405a3736' }
  ];
  
  console.log('📋 Token Information:');
  console.log('Token ID:', tokenId);
  console.log('Token Owner:', await inft.ownerOf(tokenId));
  
  console.log('\n🔍 Authorization Check:');
  for (const wallet of wallets) {
    const isAuthorized = await inft.isAuthorized(tokenId, wallet.address);
    console.log(`${wallet.name} (${wallet.address}): ${isAuthorized ? '✅ Authorized' : '❌ Not Authorized'}`);
  }
  
  // Get all authorized users
  console.log('\n📊 All Authorized Users:');
  try {
    const authorizedUsers = await inft.authorizedUsersOf(tokenId);
    console.log('Total count:', authorizedUsers.length);
    authorizedUsers.forEach((user, index) => {
      console.log(`  ${index + 1}. ${user}`);
    });
  } catch (error) {
    console.log('Error getting authorized users:', error.message);
  }
  
  // Check if we need to authorize the current user
  const currentUserAuthorized = await inft.isAuthorized(tokenId, '0x17f5afc78EFF841fB8c7384150658777405a3736');
  if (!currentUserAuthorized) {
    console.log('\n🔐 Authorizing current user wallet...');
    try {
      const [signer] = await ethers.getSigners();
      const authTx = await inft.authorizeUsage(tokenId, '0x17f5afc78EFF841fB8c7384150658777405a3736');
      console.log('Authorization transaction:', authTx.hash);
      await authTx.wait();
      console.log('✅ Current user wallet authorized successfully!');
    } catch (error) {
      console.log('❌ Authorization failed:', error.message);
    }
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
