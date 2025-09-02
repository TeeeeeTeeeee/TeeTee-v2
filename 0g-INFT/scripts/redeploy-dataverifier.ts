import { ethers } from 'hardhat';
import fs from 'fs';

async function main() {
  console.log('🚀 Redeploying DataVerifier with correct oracle address...\n');

  // Get signer
  const [signer] = await ethers.getSigners();
  console.log('Deployer address:', signer.address);

  // Correct oracle address
  const CORRECT_ORACLE_ADDRESS = '0x567e70a52AB420c525D277b0020260a727A735dB';
  const INFT_ADDRESS = '0xF170237160314f5D8526f981b251b56e25347Ed9';

  console.log('Oracle address:', CORRECT_ORACLE_ADDRESS);

  // Deploy new DataVerifier
  console.log('\n📦 Deploying new DataVerifier...');
  const DataVerifierAdapter = await ethers.getContractFactory('DataVerifierAdapter');
  const dataVerifier = await DataVerifierAdapter.deploy(CORRECT_ORACLE_ADDRESS);
  await dataVerifier.waitForDeployment();

  const dataVerifierAddress = await dataVerifier.getAddress();
  console.log('✅ New DataVerifier deployed at:', dataVerifierAddress);

  // Verify the oracle address is correct
  const configuredOracle = await dataVerifier.getOracleAddress();
  console.log('Configured oracle:', configuredOracle);
  console.log('Oracle match:', configuredOracle.toLowerCase() === CORRECT_ORACLE_ADDRESS.toLowerCase());

  // Update the INFT contract to use the new DataVerifier
  console.log('\n🔄 Updating INFT contract...');
  const inft = await ethers.getContractAt('INFT', INFT_ADDRESS);
  
  try {
    const updateTx = await inft.setDataVerifier(dataVerifierAddress);
    console.log('Update transaction hash:', updateTx.hash);
    await updateTx.wait();
    console.log('✅ INFT contract updated with new DataVerifier!');
  } catch (error) {
    console.log('❌ Failed to update INFT contract:', error.message);
    console.log('💡 This likely means the INFT contract doesn\'t have a setDataVerifier function.');
    console.log('The INFT contract will need to be redeployed or the dataVerifier address is immutable.');
  }

  // Update deployment file
  console.log('\n📄 Updating deployment file...');
  const deploymentFile = './deployments/galileo.json';
  const deployment = JSON.parse(fs.readFileSync(deploymentFile, 'utf8'));
  
  // Add new dataVerifier entry
  deployment.dataVerifierNew = {
    address: dataVerifierAddress,
    deployer: signer.address,
    deploymentTx: dataVerifier.deploymentTransaction()?.hash || 'unknown',
    timestamp: new Date().toISOString(),
    contractName: 'DataVerifierAdapter',
    oracleAddress: CORRECT_ORACLE_ADDRESS,
    network: {
      name: 'galileo',
      chainId: '16601'
    },
    note: 'Redeployed with correct oracle address for transfer functionality'
  };

  fs.writeFileSync(deploymentFile, JSON.stringify(deployment, null, 2));
  console.log('✅ Deployment file updated');

  console.log('\n🎯 Summary:');
  console.log('Old DataVerifier:', deployment.dataVerifier.address);
  console.log('New DataVerifier:', dataVerifierAddress);
  console.log('Oracle Address:', CORRECT_ORACLE_ADDRESS);
  console.log('\n⚠️  Manual Step Required:');
  console.log('Update frontend constants.js with new DataVerifier address:');
  console.log(`DATA_VERIFIER: '${dataVerifierAddress}'`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Redeployment failed:', error);
    process.exit(1);
  });
