import { ethers } from "hardhat";
import * as fs from "fs";
import * as path from "path";

/**
 * Phase 5: INFT Authorization Management Script
 * 
 * This script manages usage authorizations for INFT tokens using the ERC-7857
 * compliant authorizeUsage function. It provides functionality to:
 * - Grant authorization to users for specific tokens
 * - Revoke authorization from users
 * - Query authorization status
 * - List all authorized users for a token
 * 
 * Based on ERC-7857 specification with simple boolean authorization model.
 */

interface AuthorizationConfig {
  tokenId: number;
  userAddress: string;
  action: 'grant' | 'revoke' | 'check';
}

interface DeploymentData {
  inft: {
    address: string;
    contractName: string;
  };
  mintedTokens: Array<{
    tokenId: number;
    owner: string;
  }>;
}

async function loadDeploymentData(): Promise<DeploymentData> {
  const network = await ethers.provider.getNetwork();
  const networkName = network.chainId === BigInt(16601) ? 'galileo' : 'newton';
  
  const deploymentPath = path.join(__dirname, '..', 'deployments', `${networkName}.json`);
  
  if (!fs.existsSync(deploymentPath)) {
    throw new Error(`Deployment file not found: ${deploymentPath}`);
  }
  
  return JSON.parse(fs.readFileSync(deploymentPath, 'utf8'));
}

async function authorizeUsage(config: AuthorizationConfig) {
  console.log(`\n🔐 Phase 5: INFT Authorization Management`);
  console.log(`========================================`);
  
  // Load deployment data
  const deploymentData = await loadDeploymentData();
  const inftAddress = deploymentData.inft.address;
  
  console.log(`📋 Configuration:`);
  console.log(`   INFT Contract: ${inftAddress}`);
  console.log(`   Token ID: ${config.tokenId}`);
  console.log(`   User Address: ${config.userAddress}`);
  console.log(`   Action: ${config.action}`);
  
  // Get signer and connect to contract
  const [signer] = await ethers.getSigners();
  console.log(`   Signer: ${signer.address}`);
  
  const INFT = await ethers.getContractFactory("INFT");
  const inftContract = INFT.attach(inftAddress);
  
  // Verify token exists and get owner
  let tokenOwner: string;
  try {
    tokenOwner = await inftContract.ownerOf(config.tokenId);
    console.log(`   Token Owner: ${tokenOwner}`);
  } catch (error) {
    throw new Error(`Token ${config.tokenId} does not exist or is not accessible`);
  }
  
  // Verify signer has permission to manage authorizations
  if (signer.address.toLowerCase() !== tokenOwner.toLowerCase()) {
    // Check if signer is approved for this token
    const isApproved = await inftContract.isApprovedForAll(tokenOwner, signer.address) ||
                      await inftContract.getApproved(config.tokenId) === signer.address;
    
    if (!isApproved) {
      throw new Error(`Signer ${signer.address} is not the owner or approved operator for token ${config.tokenId}`);
    }
  }
  
  console.log(`\n⚡ Executing ${config.action} authorization...`);
  
  try {
    switch (config.action) {
      case 'grant':
        await grantAuthorization(inftContract, config.tokenId, config.userAddress);
        break;
        
      case 'revoke': 
        await revokeAuthorization(inftContract, config.tokenId, config.userAddress);
        break;
        
      case 'check':
        await checkAuthorization(inftContract, config.tokenId, config.userAddress);
        break;
        
      default:
        throw new Error(`Unknown action: ${config.action}`);
    }
    
    // Show current authorization status
    await showAuthorizationStatus(inftContract, config.tokenId);
    
  } catch (error: any) {
    console.error(`❌ Error during ${config.action} operation:`, error.message);
    throw error;
  }
}

async function grantAuthorization(inftContract: any, tokenId: number, userAddress: string) {
  // Check if already authorized
  const isCurrentlyAuthorized = await inftContract.isAuthorized(tokenId, userAddress);
  
  if (isCurrentlyAuthorized) {
    console.log(`✅ User ${userAddress} is already authorized for token ${tokenId}`);
    return;
  }
  
  console.log(`📝 Granting authorization...`);
  
  // Estimate gas
  const gasEstimate = await inftContract.authorizeUsage.estimateGas(tokenId, userAddress);
  console.log(`   Estimated gas: ${gasEstimate.toString()}`);
  
  // Execute transaction
  const tx = await inftContract.authorizeUsage(tokenId, userAddress);
  console.log(`   Transaction hash: ${tx.hash}`);
  
  // Wait for confirmation
  const receipt = await tx.wait();
  console.log(`   ✅ Authorization granted! Gas used: ${receipt.gasUsed.toString()}`);
  
  // Parse events
  const authEvent = receipt.logs.find((log: any) => {
    try {
      const parsed = inftContract.interface.parseLog(log);
      return parsed?.name === 'AuthorizedUsage';
    } catch {
      return false;
    }
  });
  
  if (authEvent) {
    const parsed = inftContract.interface.parseLog(authEvent);
    console.log(`   📢 Event: AuthorizedUsage(tokenId=${parsed.args.tokenId}, user=${parsed.args.user}, authorized=${parsed.args.authorized})`);
  }
}

async function revokeAuthorization(inftContract: any, tokenId: number, userAddress: string) {
  // Check if currently authorized
  const isCurrentlyAuthorized = await inftContract.isAuthorized(tokenId, userAddress);
  
  if (!isCurrentlyAuthorized) {
    console.log(`✅ User ${userAddress} is not currently authorized for token ${tokenId}`);
    return;
  }
  
  console.log(`📝 Revoking authorization...`);
  
  // Estimate gas
  const gasEstimate = await inftContract.revokeUsage.estimateGas(tokenId, userAddress);
  console.log(`   Estimated gas: ${gasEstimate.toString()}`);
  
  // Execute transaction
  const tx = await inftContract.revokeUsage(tokenId, userAddress);
  console.log(`   Transaction hash: ${tx.hash}`);
  
  // Wait for confirmation
  const receipt = await tx.wait();
  console.log(`   ✅ Authorization revoked! Gas used: ${receipt.gasUsed.toString()}`);
  
  // Parse events
  const authEvent = receipt.logs.find((log: any) => {
    try {
      const parsed = inftContract.interface.parseLog(log);
      return parsed?.name === 'AuthorizedUsage';
    } catch {
      return false;
    }
  });
  
  if (authEvent) {
    const parsed = inftContract.interface.parseLog(authEvent);
    console.log(`   📢 Event: AuthorizedUsage(tokenId=${parsed.args.tokenId}, user=${parsed.args.user}, authorized=${parsed.args.authorized})`);
  }
}

async function checkAuthorization(inftContract: any, tokenId: number, userAddress: string) {
  console.log(`🔍 Checking authorization status...`);
  
  const isAuthorized = await inftContract.isAuthorized(tokenId, userAddress);
  
  if (isAuthorized) {
    console.log(`   ✅ User ${userAddress} IS authorized for token ${tokenId}`);
  } else {
    console.log(`   ❌ User ${userAddress} is NOT authorized for token ${tokenId}`);
  }
}

async function showAuthorizationStatus(inftContract: any, tokenId: number) {
  console.log(`\n📊 Current Authorization Status for Token ${tokenId}:`);
  
  try {
    const authorizedUsers = await inftContract.authorizedUsersOf(tokenId);
    
    if (authorizedUsers.length === 0) {
      console.log(`   No users currently authorized`);
    } else {
      console.log(`   Authorized users (${authorizedUsers.length}):`);
      authorizedUsers.forEach((user: string, index: number) => {
        console.log(`   ${index + 1}. ${user}`);
      });
    }
  } catch (error: any) {
    console.log(`   Error retrieving authorized users: ${error.message}`);
  }
}

// Script execution
async function main() {
  // Parse arguments from environment variable or command line
  let args: string[];
  
  if (process.env.HARDHAT_SCRIPT_ARGS) {
    args = process.env.HARDHAT_SCRIPT_ARGS.split(' ');
  } else {
    args = process.argv.slice(2);
  }
  
  if (args.length < 3) {
    console.log(`Usage Option 1 (Environment Variable):`);
    console.log(`  HARDHAT_SCRIPT_ARGS="<tokenId> <userAddress> <action>" npx hardhat run scripts/authorize.ts --network <network>`);
    console.log(`\nUsage Option 2 (Direct):`);
    console.log(`  npx hardhat run scripts/authorize.ts --network <network> -- <tokenId> <userAddress> <action>`);
    console.log(`\nActions: grant, revoke, check`);
    console.log(`\nExamples:`);
    console.log(`  HARDHAT_SCRIPT_ARGS="1 0x742d35Cc6635C0532925a3b8D84d7E7b0000000 check" npx hardhat run scripts/authorize.ts --network galileo`);
    console.log(`  HARDHAT_SCRIPT_ARGS="1 0x742d35Cc6635C0532925a3b8D84d7E7b0000000 grant" npx hardhat run scripts/authorize.ts --network galileo`);
    process.exit(1);
  }
  
  const config: AuthorizationConfig = {
    tokenId: parseInt(args[0]),
    userAddress: args[1],
    action: args[2] as 'grant' | 'revoke' | 'check'
  };
  
  // Validate inputs
  if (isNaN(config.tokenId) || config.tokenId < 0) {
    throw new Error(`Invalid token ID: ${args[0]}`);
  }
  
  if (!ethers.isAddress(config.userAddress)) {
    throw new Error(`Invalid user address: ${config.userAddress}`);
  }
  
  if (!['grant', 'revoke', 'check'].includes(config.action)) {
    throw new Error(`Invalid action: ${config.action}. Must be grant, revoke, or check`);
  }
  
  try {
    await authorizeUsage(config);
    console.log(`\n🎉 Phase 5 authorization operation completed successfully!`);
  } catch (error: any) {
    console.error(`\n💥 Phase 5 authorization failed:`, error.message);
    process.exit(1);
  }
}

// Handle script execution
if (require.main === module) {
  main().catch((error) => {
    console.error("Script execution failed:", error);
    process.exit(1);
  });
}

export { authorizeUsage, AuthorizationConfig };
