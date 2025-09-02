import { ethers } from 'hardhat';
import fs from 'fs';

async function main() {
  console.log('🚀 Redeploying INFT with correct DataVerifier address...\n');

  // Get signer
  const [signer] = await ethers.getSigners();
  console.log('Deployer address:', signer.address);

  // Contract addresses
  const NEW_DATA_VERIFIER_ADDRESS = '0xbE6567dD6EB5c3B1d33F1f95373b0D0EC4F0F762';
  
  console.log('DataVerifier address:', NEW_DATA_VERIFIER_ADDRESS);

  // Deploy new INFT contract
  console.log('\n📦 Deploying new INFT contract...');
  const INFT = await ethers.getContractFactory('INFT');
  const inft = await INFT.deploy(
    '0G Intelligent NFTs',    // name
    '0G-INFT',               // symbol  
    NEW_DATA_VERIFIER_ADDRESS, // dataVerifier
    signer.address           // initialOwner
  );
  await inft.waitForDeployment();

  const inftAddress = await inft.getAddress();
  console.log('✅ New INFT deployed at:', inftAddress);

  // Test that everything is connected properly
  console.log('\n🔍 Verifying deployment...');
  
  // Check token name and symbol
  const name = await inft.name();
  const symbol = await inft.symbol();
  console.log('Token name:', name);
  console.log('Token symbol:', symbol);
  
  // Check initial token ID
  const currentTokenId = await inft.getCurrentTokenId();
  console.log('Current token ID:', currentTokenId.toString());

  // Update deployment file
  console.log('\n📄 Updating deployment file...');
  const deploymentFile = './deployments/galileo.json';
  const deployment = JSON.parse(fs.readFileSync(deploymentFile, 'utf8'));
  
  // Add new INFT entry
  deployment.inftNew = {
    address: inftAddress,
    deployer: signer.address,
    deploymentTx: inft.deploymentTransaction()?.hash || 'unknown',
    timestamp: new Date().toISOString(),
    contractName: 'INFT',
    dataVerifierAddress: NEW_DATA_VERIFIER_ADDRESS,
    tokenName: '0G Intelligent NFTs',
    tokenSymbol: '0G-INFT',
    network: {
      name: 'galileo',
      chainId: '16601'
    },
    note: 'Redeployed with correct DataVerifier for transfer functionality'
  };

  fs.writeFileSync(deploymentFile, JSON.stringify(deployment, null, 2));
  console.log('✅ Deployment file updated');

  console.log('\n🎯 Summary:');
  console.log('Old INFT:', deployment.inft.address);
  console.log('New INFT:', inftAddress);
  console.log('DataVerifier:', NEW_DATA_VERIFIER_ADDRESS);
  
  console.log('\n⚠️  Manual Steps Required:');
  console.log('1. Update frontend constants.js:');
  console.log(`   INFT: '${inftAddress}'`);
  console.log(`   DATA_VERIFIER: '${NEW_DATA_VERIFIER_ADDRESS}'`);
  console.log('');
  console.log('2. You will need to:');
  console.log('   - Mint new tokens on the new contract');
  console.log('   - Update any saved token IDs in your frontend');
  console.log('   - The old tokens will remain on the old contract');
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('INFT redeployment failed:', error);
    process.exit(1);
  });
