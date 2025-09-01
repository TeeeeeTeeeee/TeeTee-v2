import { ethers } from 'hardhat';

async function main() {
  console.log('🔍 Debugging Transfer Issue...\n');

  // Get signer
  const [signer] = await ethers.getSigners();
  console.log('Signer address:', signer.address);

  // Contract addresses
  const INFT_ADDRESS = '0x67dDE9dF36Eb6725f265bc8A1908628e8d4AF9DA';
  const ORACLE_ADDRESS = '0x567e70a52AB420c525D277b0020260a727A735dB';
  const DATA_VERIFIER_ADDRESS = '0xc13C532A60467c66bf0FFbeF52cD851bF1bC7fC6';

  // Get contract instances
  const inft = await ethers.getContractAt('INFT', INFT_ADDRESS);
  const oracle = await ethers.getContractAt('OracleStub', ORACLE_ADDRESS);

  console.log('\n📋 Current Status:');
  
  // Check token ownership
  const tokenId = 1;
  const currentOwner = await inft.ownerOf(tokenId);
  console.log(`Token ${tokenId} owner:`, currentOwner);
  console.log('Your address:    ', signer.address);
  console.log('Ownership match: ', currentOwner.toLowerCase() === signer.address.toLowerCase());

  // Check oracle status
  console.log('\n🔮 Oracle Status:');
  const oracleOwner = await oracle.owner();
  const verificationEnabled = await oracle.verificationEnabled();
  console.log('Oracle owner:', oracleOwner);
  console.log('Verification enabled:', verificationEnabled);

  // Test oracle verification with dummy data
  console.log('\n🧪 Testing Oracle Verification:');
  const dummyProof = ethers.toUtf8Bytes('test proof data');
  try {
    const verifyResult = await oracle.verifyProof(dummyProof);
    console.log('Oracle verifyProof result:', verifyResult);
  } catch (error) {
    console.log('Oracle verifyProof failed:', error.message);
  }

  // Check data verifier configuration
  console.log('\n🔧 Data Verifier Configuration:');
  const dataVerifier = await ethers.getContractAt('DataVerifierAdapter', DATA_VERIFIER_ADDRESS);
  try {
    const configuredOracle = await dataVerifier.getOracleAddress();
    console.log('Configured oracle address:', configuredOracle);
    console.log('Oracle address match:', configuredOracle.toLowerCase() === ORACLE_ADDRESS.toLowerCase());
  } catch (error) {
    console.log('Failed to get oracle address:', error.message);
  }

  // Test data verifier with dummy proof
  console.log('\n🔬 Testing Data Verifier:');
  try {
    const transferValidityResult = await dataVerifier.verifyTransferValidity(dummyProof);
    console.log('DataVerifier verifyTransferValidity result:', transferValidityResult);
  } catch (error) {
    console.log('DataVerifier verifyTransferValidity failed:', error.message);
  }

  console.log('\n✅ Debug complete');
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Debug failed:', error);
    process.exit(1);
  });
