import { ethers } from 'hardhat';

async function main() {
  console.log('🎨 Minting test token on new INFT contract...\n');

  // Get signer
  const [signer] = await ethers.getSigners();
  console.log('Minter address:', signer.address);

  // New INFT contract address
  const NEW_INFT_ADDRESS = '0x660bee924261bf25239B77B340A0B8fd2069d0FE';
  
  // Get contract instance
  const inft = await ethers.getContractAt('INFT', NEW_INFT_ADDRESS);

  // Check initial state
  const currentTokenId = await inft.getCurrentTokenId();
  console.log('Current token ID before mint:', currentTokenId.toString());

  // Mint a token to the signer
  console.log('\n🎯 Minting token...');
  const encryptedURI = '0g://storage/test-uri-for-transfers';
  const metadataHash = ethers.keccak256(ethers.toUtf8Bytes('test metadata for transfers'));
  
  const mintTx = await inft.mint(
    signer.address,
    encryptedURI,
    metadataHash
  );
  
  console.log('Mint transaction hash:', mintTx.hash);
  const receipt = await mintTx.wait();
  console.log('✅ Token minted successfully!');
  console.log('Gas used:', receipt.gasUsed.toString());

  // Check new token ID
  const newTokenId = await inft.getCurrentTokenId();
  console.log('New current token ID:', newTokenId.toString());
  
  // Verify ownership
  const tokenId = Number(newTokenId) - 1; // The token that was just minted
  const owner = await inft.ownerOf(tokenId);
  console.log(`Token ${tokenId} owner:`, owner);
  console.log('Owner matches signer:', owner.toLowerCase() === signer.address.toLowerCase());

  // Authorize yourself for inference
  console.log('\n🔐 Authorizing token for inference...');
  const authTx = await inft.authorizeUsage(tokenId, signer.address);
  console.log('Authorization transaction hash:', authTx.hash);
  await authTx.wait();
  console.log('✅ Token authorized for inference!');

  console.log('\n🎯 Summary:');
  console.log(`Contract: ${NEW_INFT_ADDRESS}`);
  console.log(`Token ID: ${tokenId}`);
  console.log(`Owner: ${signer.address}`);
  console.log(`✅ Ready for transfer testing!`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Minting failed:', error);
    process.exit(1);
  });
