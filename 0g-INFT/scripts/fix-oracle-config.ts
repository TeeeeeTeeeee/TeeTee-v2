import { ethers } from 'hardhat';

async function main() {
  console.log('🔧 Fixing Oracle Configuration in DataVerifier...\n');

  // Get signer
  const [signer] = await ethers.getSigners();
  console.log('Signer address:', signer.address);

  // Contract addresses
  const CORRECT_ORACLE_ADDRESS = '0x567e70a52AB420c525D277b0020260a727A735dB';
  const DATA_VERIFIER_ADDRESS = '0x9C3FFe10e61B1750F61D2E0A64c6bBE8984BA268';

  // Get contract instance
  const dataVerifier = await ethers.getContractAt('DataVerifierAdapter', DATA_VERIFIER_ADDRESS);

  console.log('🔍 Current Configuration:');
  const currentOracle = await dataVerifier.getOracleAddress();
  console.log('Current oracle:', currentOracle);
  console.log('Correct oracle:', CORRECT_ORACLE_ADDRESS);
  console.log('Match:', currentOracle.toLowerCase() === CORRECT_ORACLE_ADDRESS.toLowerCase());

  if (currentOracle.toLowerCase() !== CORRECT_ORACLE_ADDRESS.toLowerCase()) {
    console.log('\n🔄 Updating oracle address...');
    
    try {
      // Note: This will only work if the DataVerifier has a function to update the oracle
      // If not, we may need to redeploy or check if there's an owner function
      const tx = await dataVerifier.setOracleAddress(CORRECT_ORACLE_ADDRESS);
      console.log('Transaction hash:', tx.hash);
      
      await tx.wait();
      console.log('✅ Oracle address updated successfully!');
      
      // Verify the update
      const newOracle = await dataVerifier.getOracleAddress();
      console.log('New oracle address:', newOracle);
      
    } catch (error) {
      console.log('❌ Failed to update oracle address:', error.message);
      console.log('\n💡 This likely means the DataVerifier contract doesn\'t have an updateable oracle address.');
      console.log('We may need to:');
      console.log('1. Redeploy the DataVerifier with the correct oracle address, or');
      console.log('2. Update the INFT contract to use the correct DataVerifier, or');
      console.log('3. Deploy a new oracle at the expected address');
    }
  } else {
    console.log('✅ Oracle address is already correct!');
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Fix failed:', error);
    process.exit(1);
  });
