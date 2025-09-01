import { ethers } from 'hardhat';

async function main() {
  console.log('🔮 Testing Oracle Verification with Optimized Proof...\n');

  const [signer] = await ethers.getSigners();
  
  // Contract addresses
  const ORACLE_ADDRESS = '0x567e70a52AB420c525D277b0020260a727A735dB';
  const DATA_VERIFIER_ADDRESS = '0xc13C532A60467c66bf0FFbeF52cD851bF1bC7fC6';
  
  // Get contract instances
  const oracle = await ethers.getContractAt('OracleStub', ORACLE_ADDRESS);
  const dataVerifier = await ethers.getContractAt('DataVerifierAdapter', DATA_VERIFIER_ADDRESS);
  
  // Test with the optimized proof format used in transfer
  const optimizedProof = '0x7b2276223a22312e30222c2274797065223a225445455f53545542222c227473223a313735353730313033303432362c2268617368223a2230666663356164396162396361386565633064353431623136346662393238336331313634383933383465643731626235396666333535333463333265643633222c22736967223a223963303132393762326664616539386465353136346436323438303665386561227d';
  
  console.log('📋 Testing Oracle:');
  console.log('Proof length:', optimizedProof.length);
  
  try {
    console.log('Testing oracle.verifyProof...');
    const oracleResult = await oracle.verifyProof.staticCall(optimizedProof);
    console.log('✅ Oracle verifyProof result:', oracleResult);
    
    console.log('Testing dataVerifier.verifyTransferValidity...');
    const dataVerifierResult = await dataVerifier.verifyTransferValidity.staticCall(optimizedProof);
    console.log('✅ DataVerifier verifyTransferValidity result:', dataVerifierResult);
    
    // Check oracle status
    const verificationEnabled = await oracle.verificationEnabled();
    console.log('Oracle verification enabled:', verificationEnabled);
    
  } catch (error) {
    console.log('❌ Verification failed:');
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
