import { readFileSync, writeFileSync } from "fs";
import { join } from "path";

/**
 * Helper script to update .env files with newly deployed contract addresses
 * 
 * Usage:
 * npx ts-node scripts/update-env.ts
 * or
 * npx hardhat run scripts/update-env.ts
 */

async function main() {
  console.log("🔄 Updating .env files with deployed contract addresses...\n");
  
  // Read deployment data
  const deploymentFile = join(__dirname, '../deployments/galileo.json');
  
  try {
    const deploymentData = JSON.parse(readFileSync(deploymentFile, 'utf8'));
    
    const oracleAddress = deploymentData.contracts.oracle?.address || 
                         deploymentData.oracle?.address;
    const inftAddress = deploymentData.contracts.inft?.address || 
                       deploymentData.contracts.inftFixed ||
                       deploymentData.inft?.address;
    
    if (!oracleAddress || !inftAddress) {
      console.error("❌ Could not find contract addresses in deployment file");
      console.error("   Make sure you've deployed contracts first!");
      process.exit(1);
    }
    
    console.log("📋 Found deployed contracts:");
    console.log("   Oracle:", oracleAddress);
    console.log("   INFT:", inftAddress);
    
    // Update root .env
    updateEnvFile(join(__dirname, '../.env'), oracleAddress, inftAddress);
    
    // Update offchain-service .env
    updateEnvFile(join(__dirname, '../offchain-service/.env'), oracleAddress, inftAddress);
    
    console.log("\n✅ Successfully updated .env files!");
    console.log("\n🎯 Next steps:");
    console.log("   1. Update frontend/lib/constants.js manually if needed");
    console.log("   2. Restart any running services (frontend, offchain-service)");
    console.log("   3. Test with: npx hardhat run scripts/mint.ts --network galileo");
    
  } catch (error) {
    console.error("❌ Error reading deployment file:", error);
    console.error("   Make sure you've run deployment scripts first!");
    process.exit(1);
  }
}

function updateEnvFile(envPath: string, oracleAddress: string, inftAddress: string) {
  try {
    let envContent = readFileSync(envPath, 'utf8');
    
    // Update ORACLE_CONTRACT_ADDRESS
    if (envContent.includes('ORACLE_CONTRACT_ADDRESS=')) {
      envContent = envContent.replace(
        /ORACLE_CONTRACT_ADDRESS=.*/,
        `ORACLE_CONTRACT_ADDRESS=${oracleAddress}`
      );
      console.log(`   ✓ Updated ORACLE_CONTRACT_ADDRESS in ${envPath}`);
    } else {
      envContent += `\nORACLE_CONTRACT_ADDRESS=${oracleAddress}\n`;
      console.log(`   ✓ Added ORACLE_CONTRACT_ADDRESS to ${envPath}`);
    }
    
    // Update INFT_CONTRACT_ADDRESS
    if (envContent.includes('INFT_CONTRACT_ADDRESS=')) {
      envContent = envContent.replace(
        /INFT_CONTRACT_ADDRESS=.*/,
        `INFT_CONTRACT_ADDRESS=${inftAddress}`
      );
      console.log(`   ✓ Updated INFT_CONTRACT_ADDRESS in ${envPath}`);
    } else {
      envContent += `INFT_CONTRACT_ADDRESS=${inftAddress}\n`;
      console.log(`   ✓ Added INFT_CONTRACT_ADDRESS to ${envPath}`);
    }
    
    writeFileSync(envPath, envContent);
    
  } catch (error) {
    console.warn(`   ⚠️  Could not update ${envPath}:`, error);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });

