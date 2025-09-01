import { ethers } from "hardhat";
import { writeFileSync, readFileSync, existsSync } from "fs";
import { join } from "path";

/**
 * Script to deploy INFT and DataVerifierAdapter contracts to 0G testnet
 * 
 * This deploys the complete ERC-7857 INFT system including:
 * - DataVerifierAdapter (wraps OracleStub with ERC-7857 interface)
 * - INFT contract (main ERC-7857 implementation)
 * 
 * Prerequisites:
 * - OracleStub must be deployed first (run deployOracle.ts)
 * 
 * Usage:
 * npx hardhat run scripts/deployINFT.ts --network galileo
 * npx hardhat run scripts/deployINFT.ts --network newton
 */
async function main() {
  const [deployer] = await ethers.getSigners();
  const network = await ethers.provider.getNetwork();
  const networkName = network.name || network.chainId.toString();
  
  console.log("🚀 Deploying INFT system contracts...");
  console.log("📍 Network:", network.name, "Chain ID:", network.chainId.toString());
  console.log("👤 Deployer address:", deployer.address);
  
  // Check deployer balance
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("💰 Deployer balance:", ethers.formatEther(balance), "ETH");
  
  if (balance < ethers.parseEther("0.02")) {
    console.warn("⚠️  Warning: Low balance. You may need to fund your wallet from the 0G faucet:");
    console.warn("   https://faucet.0g.ai");
  }
  
  // Load existing deployment data
  const deploymentFile = join(__dirname, `../deployments/${networkName}.json`);
  if (!existsSync(deploymentFile)) {
    throw new Error(`❌ No deployment file found at ${deploymentFile}. Please deploy Oracle first.`);
  }
  
  const deploymentData = JSON.parse(readFileSync(deploymentFile, "utf8"));
  if (!deploymentData.oracle?.address) {
    throw new Error("❌ Oracle address not found in deployment file. Please deploy Oracle first.");
  }
  
  const oracleAddress = deploymentData.oracle.address;
  console.log("🔗 Using Oracle address:", oracleAddress);
  
  // Deploy DataVerifierAdapter
  console.log("\n📄 Deploying DataVerifierAdapter...");
  const DataVerifierAdapter = await ethers.getContractFactory("DataVerifierAdapter");
  const dataVerifier = await DataVerifierAdapter.deploy(oracleAddress);
  
  await dataVerifier.waitForDeployment();
  const dataVerifierAddress = await dataVerifier.getAddress();
  
  console.log("✅ DataVerifierAdapter deployed successfully!");
  console.log("📧 Address:", dataVerifierAddress);
  console.log("⛽ Deployment transaction:", dataVerifier.deploymentTransaction()?.hash);
  
  // Verify DataVerifierAdapter deployment
  console.log("\n🔍 Verifying DataVerifierAdapter...");
  const wrappedOracleAddress = await dataVerifier.getOracleAddress();
  console.log("🔗 Wrapped Oracle address:", wrappedOracleAddress);
  
  if (wrappedOracleAddress.toLowerCase() !== oracleAddress.toLowerCase()) {
    throw new Error("❌ DataVerifierAdapter oracle address mismatch!");
  }
  
  // Deploy INFT contract
  console.log("\n📄 Deploying INFT contract...");
  const INFT = await ethers.getContractFactory("INFT");
  
  // INFT constructor parameters
  const tokenName = "0G Intelligent NFTs";
  const tokenSymbol = "0G-INFT";
  const initialOwner = deployer.address;
  
  const inft = await INFT.deploy(
    tokenName,
    tokenSymbol,
    dataVerifierAddress,
    initialOwner
  );
  
  await inft.waitForDeployment();
  const inftAddress = await inft.getAddress();
  
  console.log("✅ INFT contract deployed successfully!");
  console.log("📧 Address:", inftAddress);
  console.log("⛽ Deployment transaction:", inft.deploymentTransaction()?.hash);
  
  // Verify INFT deployment
  console.log("\n🔍 Verifying INFT deployment...");
  const contractName = await inft.name();
  const contractSymbol = await inft.symbol();
  const contractDataVerifier = await inft.dataVerifier();
  const contractOwner = await inft.owner();
  const currentTokenId = await inft.getCurrentTokenId();
  
  console.log("📝 Token name:", contractName);
  console.log("🏷️  Token symbol:", contractSymbol);
  console.log("🔍 DataVerifier address:", contractDataVerifier);
  console.log("👤 Contract owner:", contractOwner);
  console.log("🆔 Current token ID:", currentTokenId.toString());
  
  // Verify addresses match
  if (contractDataVerifier.toLowerCase() !== dataVerifierAddress.toLowerCase()) {
    throw new Error("❌ INFT dataVerifier address mismatch!");
  }
  
  if (contractOwner.toLowerCase() !== deployer.address.toLowerCase()) {
    throw new Error("❌ INFT owner address mismatch!");
  }
  
  // Update deployment data
  deploymentData.dataVerifier = {
    address: dataVerifierAddress,
    deployer: deployer.address,
    deploymentTx: dataVerifier.deploymentTransaction()?.hash,
    blockNumber: await ethers.provider.getBlockNumber(),
    timestamp: new Date().toISOString(),
    contractName: "DataVerifierAdapter",
    oracleAddress: oracleAddress,
    network: {
      name: networkName,
      chainId: network.chainId.toString()
    }
  };
  
  deploymentData.inft = {
    address: inftAddress,
    deployer: deployer.address,
    deploymentTx: inft.deploymentTransaction()?.hash,
    blockNumber: await ethers.provider.getBlockNumber(),
    timestamp: new Date().toISOString(),
    contractName: "INFT",
    dataVerifierAddress: dataVerifierAddress,
    tokenName: tokenName,
    tokenSymbol: tokenSymbol,
    network: {
      name: networkName,
      chainId: network.chainId.toString()
    }
  };
  
  writeFileSync(deploymentFile, JSON.stringify(deploymentData, null, 2));
  console.log("💾 Deployment info updated in:", deploymentFile);
  
  // Display next steps
  console.log("\n🎯 Next Steps:");
  console.log("1. Mint your first INFT:");
  console.log(`   npx hardhat run scripts/mint.ts --network ${networkName}`);
  console.log("2. Test authorization and transfer functions");
  console.log("3. Integrate with off-chain service for inference");
  console.log("4. Consider upgrading to production oracle services");
  
  console.log("\n📋 Deployment Summary:");
  console.log(`Oracle (Stub):        ${oracleAddress}`);
  console.log(`DataVerifierAdapter:  ${dataVerifierAddress}`);
  console.log(`INFT Contract:        ${inftAddress}`);
  console.log(`\n🌐 Block Explorer:`);
  
  // Provide block explorer links based on network
  if (network.chainId === 16601n) {
    console.log(`Galileo Explorer: https://chainscan-galileo.0g.ai`);
  } else if (network.chainId === 16600n) {
    console.log(`Newton Explorer: https://chainscan-newton.0g.ai`);
  }
  
  console.log("\n✨ ERC-7857 INFT system deployment completed successfully!");
}

// Execute deployment script
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exit(1);
  });
