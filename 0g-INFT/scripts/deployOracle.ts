import { ethers } from "hardhat";
import { writeFileSync, readFileSync, existsSync } from "fs";
import { join } from "path";

/**
 * Script to deploy OracleStub contract to 0G testnet
 * 
 * This deploys the development oracle stub that always returns true for proof verification.
 * In production, this should be replaced with a real TEE or ZKP oracle.
 * 
 * Usage:
 * npx hardhat run scripts/deployOracle.ts --network galileo
 * npx hardhat run scripts/deployOracle.ts --network newton
 */
async function main() {
  const [deployer] = await ethers.getSigners();
  const network = await ethers.provider.getNetwork();
  
  console.log("🚀 Deploying OracleStub contract...");
  console.log("📍 Network:", network.name, "Chain ID:", network.chainId.toString());
  console.log("👤 Deployer address:", deployer.address);
  
  // Check deployer balance
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("💰 Deployer balance:", ethers.formatEther(balance), "ETH");
  
  if (balance < ethers.parseEther("0.01")) {
    console.warn("⚠️  Warning: Low balance. You may need to fund your wallet from the 0G faucet:");
    console.warn("   https://faucet.0g.ai");
  }
  
  // Deploy OracleStub contract
  console.log("\n📄 Deploying OracleStub...");
  const OracleStub = await ethers.getContractFactory("OracleStub");
  const oracleStub = await OracleStub.deploy();
  
  // Wait for deployment to be mined
  await oracleStub.waitForDeployment();
  const oracleAddress = await oracleStub.getAddress();
  
  console.log("✅ OracleStub deployed successfully!");
  console.log("📧 Address:", oracleAddress);
  console.log("⛽ Deployment transaction:", oracleStub.deploymentTransaction()?.hash);
  
  // Verify deployment by calling a function
  console.log("\n🔍 Verifying deployment...");
  const owner = await oracleStub.owner();
  const verificationEnabled = await oracleStub.verificationEnabled();
  
  console.log("👤 Oracle owner:", owner);
  console.log("✓ Verification enabled:", verificationEnabled);
  
  // Save deployment info to JSON file
  const networkName = network.name || network.chainId.toString();
  const deploymentFile = join(__dirname, `../deployments/${networkName}.json`);
  
  let deploymentData: any = {};
  if (existsSync(deploymentFile)) {
    deploymentData = JSON.parse(readFileSync(deploymentFile, "utf8"));
  }
  
  deploymentData.oracle = {
    address: oracleAddress,
    deployer: deployer.address,
    deploymentTx: oracleStub.deploymentTransaction()?.hash,
    blockNumber: await ethers.provider.getBlockNumber(),
    timestamp: new Date().toISOString(),
    contractName: "OracleStub",
    network: {
      name: networkName,
      chainId: network.chainId.toString()
    }
  };
  
  writeFileSync(deploymentFile, JSON.stringify(deploymentData, null, 2));
  console.log("💾 Deployment info saved to:", deploymentFile);
  
  // Display next steps
  console.log("\n🎯 Next Steps:");
  console.log("1. Deploy INFT contract with oracle address:");
  console.log(`   npx hardhat run scripts/deployINFT.ts --network ${networkName}`);
  console.log("2. Verify contracts on block explorer if desired");
  console.log("3. Fund the oracle if needed for production usage");
  
  console.log("\n📋 Summary:");
  console.log(`OracleStub: ${oracleAddress}`);
}

// Execute deployment script
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exit(1);
  });
