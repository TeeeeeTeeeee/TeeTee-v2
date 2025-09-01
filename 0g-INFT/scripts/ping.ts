import { ethers } from "hardhat";
import * as dotenv from "dotenv";

dotenv.config();

/**
 * Ping script to test 0G network connectivity
 * This script connects to the specified 0G testnet and fetches basic network information
 * to verify that the hardhat configuration is working correctly.
 */
async function main() {
  console.log("🔍 Testing 0G Network Connectivity...\n");
  
  try {
    // Get the current network configuration
    const network = await ethers.provider.getNetwork();
    console.log(`📡 Connected to network: ${network.name}`);
    console.log(`🔗 Chain ID: ${network.chainId}`);
    
    // Verify it matches expected 0G chain IDs
    const expectedChains = [16600, 16601, 31337]; // Newton, Galileo, Hardhat local
    if (!expectedChains.includes(Number(network.chainId))) {
      console.warn(`⚠️  Warning: Chain ID ${network.chainId} is not a known 0G testnet`);
    } else {
      const chainName = network.chainId === 16600n ? "Newton" : 
                       network.chainId === 16601n ? "Galileo" : 
                       network.chainId === 31337n ? "Hardhat Local" : "Unknown";
      console.log(`✅ Confirmed 0G ${chainName} testnet connection`);
    }
    
    // Get latest block to verify connectivity
    const blockNumber = await ethers.provider.getBlockNumber();
    console.log(`📦 Latest block number: ${blockNumber}`);
    
    // Get block details
    const block = await ethers.provider.getBlock(blockNumber);
    if (block) {
      console.log(`⏰ Block timestamp: ${new Date(block.timestamp * 1000).toISOString()}`);
      console.log(`⛽ Block gas limit: ${block.gasLimit.toString()}`);
    }
    
    // Test account balance if private key is provided
    if (process.env.PRIVATE_KEY) {
      const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, ethers.provider);
      const balance = await ethers.provider.getBalance(wallet.address);
      console.log(`💰 Wallet address: ${wallet.address}`);
      console.log(`💎 Balance: ${ethers.formatEther(balance)} OG`);
      
      if (balance === 0n) {
        console.log(`\n🚨 Warning: Wallet has 0 balance. Get testnet tokens from:`);
        console.log(`   Newton/Galileo faucet: https://faucet.0g.ai`);
      }
    } else {
      console.log(`🔑 No private key provided - skipping wallet balance check`);
      console.log(`   Add PRIVATE_KEY to .env file to test wallet connectivity`);
    }
    
    console.log(`\n✅ Network connectivity test successful!`);
    
  } catch (error) {
    console.error(`❌ Network connectivity test failed:`);
    console.error(error);
    
    // Provide helpful debugging information
    console.log(`\n🔧 Debug Information:`);
    console.log(`   Current network config: ${JSON.stringify(await ethers.provider.getNetwork(), null, 2)}`);
    console.log(`   Check your .env file and hardhat.config.ts`);
    console.log(`   Verify RPC URLs are accessible:`);
    console.log(`   - Newton: https://evmrpc-testnet.0g.ai (Chain ID: 16600)`);
    console.log(`   - Galileo: https://evmrpc-testnet.0g.ai (Chain ID: 16601)`);
    
    process.exit(1);
  }
}

// Execute the ping test
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
