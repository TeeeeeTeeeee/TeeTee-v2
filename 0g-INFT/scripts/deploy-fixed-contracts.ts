import { ethers } from "hardhat";
import { writeFileSync, readFileSync } from "fs";

/**
 * Deploy fixed contracts with improved error handling and gas optimization
 */
async function main() {
  const [deployer] = await ethers.getSigners();
  const network = await ethers.provider.getNetwork();
  
  console.log("🚀 Deploying Fixed Contracts...");
  console.log("📍 Network:", network.name, "Chain ID:", network.chainId.toString());
  console.log("👤 Deployer address:", deployer.address);
  
  // Check deployer balance
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("💰 Deployer balance:", ethers.formatEther(balance), "OG");
  
  // Use existing oracle
  const ORACLE_ADDRESS = '0x567e70a52AB420c525D277b0020260a727A735dB';
  console.log("🔮 Using existing Oracle at:", ORACLE_ADDRESS);
  
  // Deploy fixed DataVerifierAdapter
  console.log("\n📄 Deploying Fixed DataVerifierAdapter...");
  const DataVerifierAdapterFixed = await ethers.getContractFactory("DataVerifierAdapterFixed");
  const dataVerifierFixed = await DataVerifierAdapterFixed.deploy(ORACLE_ADDRESS);
  await dataVerifierFixed.waitForDeployment();
  const dataVerifierAddress = await dataVerifierFixed.getAddress();
  
  console.log("✅ Fixed DataVerifierAdapter deployed!");
  console.log("📧 Address:", dataVerifierAddress);
  
  // Deploy fixed INFT
  console.log("\n📄 Deploying Fixed INFT...");
  const INFTFixed = await ethers.getContractFactory("INFTFixed");
  const inftFixed = await INFTFixed.deploy(
    dataVerifierAddress,
    "0G Intelligent NFTs (Fixed)",
    "0G-INFT-V2"
  );
  await inftFixed.waitForDeployment();
  const inftAddress = await inftFixed.getAddress();
  
  console.log("✅ Fixed INFT deployed!");
  console.log("📧 Address:", inftAddress);
  
  // Verify deployments
  console.log("\n🔍 Verifying deployments...");
  
  // Check DataVerifier
  const configuredOracle = await dataVerifierFixed.getOracleAddress();
  console.log("DataVerifier oracle:", configuredOracle);
  console.log("Oracle match:", configuredOracle.toLowerCase() === ORACLE_ADDRESS.toLowerCase());
  
  // Check INFT
  const inftName = await inftFixed.name();
  const inftSymbol = await inftFixed.symbol();
  const inftDataVerifier = await inftFixed.dataVerifier();
  console.log("INFT name:", inftName);
  console.log("INFT symbol:", inftSymbol);
  console.log("INFT dataVerifier:", inftDataVerifier);
  console.log("DataVerifier match:", inftDataVerifier.toLowerCase() === dataVerifierAddress.toLowerCase());
  
  // Update deployment data
  const deploymentData = {
    network: {
      name: network.name,
      chainId: network.chainId.toString()
    },
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      oracle: ORACLE_ADDRESS,
      dataVerifierFixed: dataVerifierAddress,
      inftFixed: inftAddress
    },
    improvements: [
      "Proper error bubbling in DataVerifierAdapter",
      "Custom errors instead of string requires",
      "Removed try/catch gas overhead",
      "Better revert reason preservation"
    ]
  };
  
  // Write to file
  writeFileSync(
    "deployments/fixed-contracts.json",
    JSON.stringify(deploymentData, null, 2)
  );
  
  console.log("\n✅ Fixed contracts deployed successfully!");
  console.log("📄 Deployment data saved to deployments/fixed-contracts.json");
  console.log("\n📋 Summary:");
  console.log("  Oracle (existing):", ORACLE_ADDRESS);
  console.log("  DataVerifier (fixed):", dataVerifierAddress);
  console.log("  INFT (fixed):", inftAddress);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
