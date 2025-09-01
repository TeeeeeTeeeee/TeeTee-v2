import { ethers } from 'hardhat';

async function main() {
  const INFT_ADDRESS = '0x660bee924261bf25239B77B340A0B8fd2069d0FE';
  const YOUR_ADDRESS = '0x32F91E4E2c60A9C16cAE736D3b42152B331c147F';
  
  const inft = await ethers.getContractAt('INFT', INFT_ADDRESS);
  
  try {
    console.log('🔍 Checking contract state...');
    
    // Check current token ID
    const currentTokenId = await inft.getCurrentTokenId();
    console.log('Current token ID:', currentTokenId.toString());
    
    // Check your balance
    const balance = await inft.balanceOf(YOUR_ADDRESS);
    console.log('Your balance:', balance.toString());
    
    // Check token 1 owner
    try {
      const token1Owner = await inft.ownerOf(1);
      console.log('Token 1 owner:', token1Owner);
      console.log('You own token 1:', token1Owner.toLowerCase() === YOUR_ADDRESS.toLowerCase());
    } catch (error) {
      console.log('Token 1 does not exist');
    }
    
    // Check if token 2 exists
    try {
      const token2Owner = await inft.ownerOf(2);
      console.log('Token 2 owner:', token2Owner);
    } catch (error) {
      console.log('Token 2 does not exist');
    }
    
  } catch (error) {
    console.error('Error:', error.message);
  }
}

main().catch(console.error);
