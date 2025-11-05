import { ethers } from "hardhat";
import { writeFileSync } from "fs";
import { join } from "path";

/**
 * Master deployment script for complete INFT system
 * 
 * Deploys in order:
 * 1. OracleStub (development oracle)
 * 2. DataVerifierAdapterFixed (improved error handling)
 * 3. INFTFixed (improved gas efficiency)
 * 
 * Usage:
 * npx hardhat run scripts/deploy-all.ts --network galileo
 */

async function main() {
  const [deployer] = await ethers.getSigners();
  const network = await ethers.provider.getNetwork();
  const networkName = network.name || 'galileo';
  
  console.log("╔════════════════════════════════════════════════════════════╗");
  console.log("║     🚀 DEPLOYING COMPLETE INFT SYSTEM TO 0G NETWORK       ║");
  console.log("╚════════════════════════════════════════════════════════════╝\n");
  
  console.log("📍 Network Information:");
  console.log("   Name:", network.name);
  console.log("   Chain ID:", network.chainId.toString());
  console.log("   👤 Deployer:", deployer.address);
  
  // Check deployer balance
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("   💰 Balance:", ethers.formatEther(balance), "0G");
  
  if (balance < ethers.parseEther("0.05")) {
    console.warn("\n⚠️  WARNING: Low balance detected!");
    console.warn("   You may need to fund your wallet from the 0G faucet:");
    console.warn("   🔗 https://faucet.0g.ai");
    console.warn("   Required: ~0.05 0G for deployment");
    
    // Give user 10 seconds to cancel
    console.log("\n⏳ Starting deployment in 10 seconds... (Ctrl+C to cancel)");
    await new Promise(resolve => setTimeout(resolve, 10000));
  }
  
  console.log("\n" + "═".repeat(60));
  console.log("STEP 1/3: Deploying OracleStub");
  console.log("═".repeat(60));
  
  // Deploy OracleStub
  console.log("\n📄 Deploying OracleStub contract...");
  const OracleStub = await ethers.getContractFactory("OracleStub");
  const oracleStub = await OracleStub.deploy();
  await oracleStub.waitForDeployment();
  const oracleAddress = await oracleStub.getAddress();
  
  console.log("✅ OracleStub deployed!");
  console.log("   📧 Address:", oracleAddress);
  console.log("   ⛽ Tx Hash:", oracleStub.deploymentTransaction()?.hash);
  
  // Verify Oracle
  const oracleOwner = await oracleStub.owner();
  const verificationEnabled = await oracleStub.verificationEnabled();
  console.log("   🔍 Verified - Owner:", oracleOwner);
  console.log("   🔍 Verified - Verification Enabled:", verificationEnabled);
  
  console.log("\n" + "═".repeat(60));
  console.log("STEP 2/3: Deploying DataVerifierAdapterFixed");
  console.log("═".repeat(60));
  
  // Deploy DataVerifierAdapterFixed
  console.log("\n📄 Deploying DataVerifierAdapterFixed...");
  const DataVerifierAdapterFixed = await ethers.getContractFactory("DataVerifierAdapterFixed");
  const dataVerifier = await DataVerifierAdapterFixed.deploy(oracleAddress);
  await dataVerifier.waitForDeployment();
  const dataVerifierAddress = await dataVerifier.getAddress();
  
  console.log("✅ DataVerifierAdapterFixed deployed!");
  console.log("   📧 Address:", dataVerifierAddress);
  console.log("   ⛽ Tx Hash:", dataVerifier.deploymentTransaction()?.hash);
  
  // Verify DataVerifier
  const configuredOracle = await dataVerifier.getOracleAddress();
  console.log("   🔍 Verified - Oracle Address:", configuredOracle);
  console.log("   🔍 Verified - Oracle Match:", configuredOracle.toLowerCase() === oracleAddress.toLowerCase() ? "✓" : "✗");
  
  console.log("\n" + "═".repeat(60));
  console.log("STEP 3/3: Deploying INFT");
  console.log("═".repeat(60));
  
  // Deploy INFT (updated with ownerAuthorizeUsage)
  console.log("\n📄 Deploying INFT contract...");
  const INFT = await ethers.getContractFactory("INFT");
  const inft = await INFT.deploy(
    "0G Intelligent NFTs",
    "0G-INFT",
    dataVerifierAddress,
    deployer.address // initialOwner
  );
  await inft.waitForDeployment();
  const inftAddress = await inft.getAddress();
  
  console.log("✅ INFT deployed!");
  console.log("   📧 Address:", inftAddress);
  console.log("   ⛽ Tx Hash:", inft.deploymentTransaction()?.hash);
  
  // Verify INFT
  const inftName = await inft.name();
  const inftSymbol = await inft.symbol();
  const inftDataVerifier = await inft.dataVerifier();
  const currentTokenId = await inft.getCurrentTokenId();
  
  console.log("   🔍 Verified - Name:", inftName);
  console.log("   🔍 Verified - Symbol:", inftSymbol);
  console.log("   🔍 Verified - DataVerifier:", inftDataVerifier);
  console.log("   🔍 Verified - DataVerifier Match:", inftDataVerifier.toLowerCase() === dataVerifierAddress.toLowerCase() ? "✓" : "✗");
  console.log("   🔍 Verified - Current Token ID:", currentTokenId.toString());
  
  // Save deployment data
  console.log("\n" + "═".repeat(60));
  console.log("💾 Saving Deployment Data");
  console.log("═".repeat(60));
  
  const deploymentData = {
    network: {
      name: networkName,
      chainId: network.chainId.toString(),
      rpcUrl: "https://evmrpc-testnet.0g.ai",
      explorer: "https://chainscan-galileo.0g.ai"
    },
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      oracle: {
        address: oracleAddress,
        name: "OracleStub",
        txHash: oracleStub.deploymentTransaction()?.hash
      },
      dataVerifier: {
        address: dataVerifierAddress,
        name: "DataVerifierAdapterFixed",
        txHash: dataVerifier.deploymentTransaction()?.hash,
        oracleAddress: oracleAddress
      },
      inft: {
        address: inftAddress,
        name: "INFT",
        txHash: inft.deploymentTransaction()?.hash,
        dataVerifierAddress: dataVerifierAddress,
        tokenName: inftName,
        tokenSymbol: inftSymbol
      }
    },
    features: [
      "ERC-7857 Intelligent NFT Standard",
      "Backend authorization with ownerAuthorizeUsage",
      "Token IDs start at 1 (never 0)",
      "Epoch-based authorization system (O(1) clearing)",
      "0G Storage integration ready"
    ]
  };
  
  // Save to deployments folder
  const deploymentFile = join(__dirname, `../deployments/${networkName}.json`);
  writeFileSync(deploymentFile, JSON.stringify(deploymentData, null, 2));
  console.log("✅ Deployment data saved to:", deploymentFile);
  
  // Also save to a timestamped file for backup
  const backupFile = join(__dirname, `../deployments/${networkName}-${Date.now()}.json`);
  writeFileSync(backupFile, JSON.stringify(deploymentData, null, 2));
  console.log("✅ Backup saved to:", backupFile);
  
  // Print final summary
  console.log("\n╔════════════════════════════════════════════════════════════╗");
  console.log("║              ✨ DEPLOYMENT SUCCESSFUL! ✨                  ║");
  console.log("╚════════════════════════════════════════════════════════════╝\n");
  
  console.log("📋 Deployed Contracts:");
  console.log("┌────────────────────────────────────────────────────────────┐");
  console.log("│ Contract                    │ Address                      │");
  console.log("├────────────────────────────────────────────────────────────┤");
  console.log(`│ OracleStub                  │ ${oracleAddress.substring(0, 20)}... │`);
  console.log(`│ DataVerifierAdapterFixed    │ ${dataVerifierAddress.substring(0, 20)}... │`);
  console.log(`│ INFT                        │ ${inftAddress.substring(0, 20)}... │`);
  console.log("└────────────────────────────────────────────────────────────┘");
  
  console.log("\n🌐 Block Explorer Links:");
  if (network.chainId === 16602n) {
    console.log(`   Oracle: https://chainscan-galileo.0g.ai/address/${oracleAddress}`);
    console.log(`   DataVerifier: https://chainscan-galileo.0g.ai/address/${dataVerifierAddress}`);
    console.log(`   INFT: https://chainscan-galileo.0g.ai/address/${inftAddress}`);
  }
  
  console.log("\n🎯 Next Steps:");
  console.log("   1. Update your .env file with the new contract addresses:");
  console.log(`      INFT_CONTRACT_ADDRESS=${inftAddress}`);
  console.log(`      ORACLE_CONTRACT_ADDRESS=${oracleAddress}`);
  console.log("");
  console.log("   2. Update frontend configuration (if needed):");
  console.log("      frontend/lib/constants.js");
  console.log("");
  console.log("   3. Test the deployment:");
  console.log(`      npx hardhat run scripts/mint.ts --network ${networkName}`);
  console.log("");
  console.log("   4. Start the off-chain service:");
  console.log("      cd offchain-service && npm start");
  console.log("");
  console.log("   5. Launch the frontend:");
  console.log("      cd frontend && npm run dev");
  
  console.log("\n💡 Tip: Fund your wallet from the faucet if needed:");
  console.log("   🔗 https://faucet.0g.ai");
  
  console.log("\n" + "═".repeat(60));
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ DEPLOYMENT FAILED!");
    console.error("Error:", error);
    process.exit(1);
  });

