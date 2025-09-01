import { ethers } from 'hardhat';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

/**
 * Phase 7 - Transfer INFT (TEE Path via Oracle)
 * 
 * This script implements the TEE transfer path for ERC-7857 INFT tokens.
 * It demonstrates the secure transfer process where:
 * 1. A sealed key is re-encrypted for the new owner (simulated via TEE)
 * 2. A cryptographic proof (TEE attestation) is generated
 * 3. The oracle verifies the proof before allowing the transfer
 * 4. Token ownership is transferred with proper event emission
 * 
 * Key Features:
 * - TEE attestation simulation for development
 * - Oracle-verified transfer process
 * - Sealed key re-encryption handling
 * - Comprehensive verification and tracking
 */

interface DeploymentData {
  oracle: any;
  dataVerifier: any;
  inft: any;
  mintedTokens?: Array<{
    tokenId: number;
    owner: string;
    encryptedURI: string;
    metadataHash: string;
    phase1Storage?: {
      encryptionKey: string;
      iv: string;
      tag: string;
    };
  }>;
}

interface TransferParams {
  from: string;      // Current owner address
  to: string;        // New owner address  
  tokenId: number;   // Token ID to transfer
  sealedKey: string; // Re-encrypted key for new owner (hex string)
  proof: string;     // TEE attestation proof (hex string)
}

interface TransferResult {
  tokenId: number;
  from: string;
  to: string;
  txHash: string;
  gasUsed: string;
  proofHash: string;
  sealedKey: string;
}

/**
 * Load deployment data from galileo.json
 */
function loadDeploymentData(): DeploymentData {
  const deploymentPath = path.join(__dirname, '..', 'deployments', 'galileo.json');
  
  if (!fs.existsSync(deploymentPath)) {
    throw new Error(`Deployment file not found: ${deploymentPath}`);
  }
  
  const deploymentData = JSON.parse(fs.readFileSync(deploymentPath, 'utf8'));
  
  // Validate required contracts are deployed
  if (!deploymentData.inft?.address) {
    throw new Error('INFT contract address not found in deployment data');
  }
  
  console.log('📋 Loaded deployment data:');
  console.log('  - INFT Contract:', deploymentData.inft.address);
  console.log('  - DataVerifier:', deploymentData.dataVerifier?.address || 'N/A');
  console.log('  - Oracle:', deploymentData.oracle?.address || 'N/A');
  
  return deploymentData;
}

/**
 * Get network information and validate connection
 */
async function getNetworkInfo() {
  const [signer] = await ethers.getSigners();
  const provider = signer.provider;
  const network = await provider.getNetwork();
  const balance = await provider.getBalance(signer.address);
  
  console.log('🌐 Network Information:');
  console.log('  - Chain ID:', network.chainId.toString());
  console.log('  - Network Name:', network.name);
  console.log('  - Signer Address:', signer.address);
  console.log('  - Balance:', ethers.formatEther(balance), 'OG');
  
  // Verify we're on 0G Galileo testnet
  if (network.chainId !== 16601n) {
    throw new Error(`Expected 0G Galileo testnet (Chain ID: 16601), but connected to Chain ID: ${network.chainId}`);
  }
  
  // Check minimum balance for transaction
  const minBalance = ethers.parseEther('0.01'); // 0.01 OG minimum
  if (balance < minBalance) {
    throw new Error(`Insufficient balance. Required: ${ethers.formatEther(minBalance)} OG, Available: ${ethers.formatEther(balance)} OG`);
  }
  
  return { signer, provider, network };
}

/**
 * Validate transfer parameters
 */
function validateTransferParams(params: TransferParams): void {
  if (!params.from || !ethers.isAddress(params.from)) {
    throw new Error(`Invalid from address: ${params.from}`);
  }
  
  if (!params.to || !ethers.isAddress(params.to)) {
    throw new Error(`Invalid to address: ${params.to}`);
  }
  
  if (params.from.toLowerCase() === params.to.toLowerCase()) {
    throw new Error('Cannot transfer to the same address');
  }
  
  if (!params.tokenId || params.tokenId < 1) {
    throw new Error(`Invalid token ID: ${params.tokenId}`);
  }
  
  if (!params.sealedKey || params.sealedKey.length < 10) {
    throw new Error('Sealed key cannot be empty or too short');
  }
  
  if (!params.proof || params.proof.length < 10) {
    throw new Error('Proof cannot be empty or too short');
  }
  
  console.log('✅ Transfer parameters validated');
}

/**
 * Verify token exists and current ownership
 */
async function verifyTokenOwnership(
  inftContract: any,
  tokenId: number,
  expectedOwner: string
): Promise<void> {
  
  console.log(`🔍 Verifying token ${tokenId} ownership...`);
  
  try {
    // Check if token exists
    const actualOwner = await inftContract.ownerOf(tokenId);
    
    if (actualOwner.toLowerCase() !== expectedOwner.toLowerCase()) {
      throw new Error(`Token ${tokenId} is not owned by ${expectedOwner}. Actual owner: ${actualOwner}`);
    }
    
    // Get token metadata for verification
    const encryptedURI = await inftContract.encryptedURI(tokenId);
    const metadataHash = await inftContract.metadataHash(tokenId);
    
    console.log('✅ Token ownership verified:');
    console.log('  - Token ID:', tokenId);
    console.log('  - Current Owner:', actualOwner);
    console.log('  - Encrypted URI:', encryptedURI);
    console.log('  - Metadata Hash:', metadataHash);
    
  } catch (error) {
    if (error.message && error.message.includes('nonexistent token')) {
      throw new Error(`Token ${tokenId} does not exist`);
    }
    throw error;
  }
}

/**
 * Generate TEE attestation proof stub for development
 * In production, this would be replaced with actual TEE enclave attestation
 */
function generateTEEAttestationProof(
  tokenId: number,
  from: string,
  to: string,
  sealedKey: string,
  originalEncryptionKey?: string
): string {
  
  console.log('🔒 Generating TEE attestation proof (development stub)...');
  
  // Simulate TEE attestation data structure
  const attestationData = {
    version: '1.0.0',
    type: 'TEE_ATTESTATION',
    timestamp: new Date().toISOString(),
    
    // TEE Environment Info (simulated)
    enclaveInfo: {
      measurement: crypto.createHash('sha256').update('0g-inft-tee-enclave-v1').digest('hex'),
      vendor: '0G-Labs-TEE-Simulator',
      version: '1.0.0'
    },
    
    // Transfer Operation Data
    operation: {
      type: 'INFT_TRANSFER',
      tokenId,
      from,
      to,
      sealedKeyHash: crypto.createHash('sha256').update(sealedKey).digest('hex')
    },
    
    // Re-encryption Evidence (simulated)
    reEncryption: {
      originalKeyHash: originalEncryptionKey ? 
        crypto.createHash('sha256').update(originalEncryptionKey).digest('hex') : 
        'simulated_original_key_hash',
      newSealedKey: sealedKey,
      reEncryptionProof: 'simulated_re_encryption_proof_' + crypto.randomBytes(16).toString('hex')
    },
    
    // TEE Signature (simulated)
    signature: {
      algorithm: 'ECDSA_P256',
      signature: 'simulated_tee_signature_' + crypto.randomBytes(32).toString('hex'),
      publicKey: 'simulated_tee_pubkey_' + crypto.randomBytes(33).toString('hex')
    }
  };
  
  // Create attestation hash
  const attestationHash = crypto
    .createHash('sha256')
    .update(JSON.stringify(attestationData))
    .digest('hex');
  
  // Create final proof payload
  const proofPayload = {
    attestation: attestationData,
    attestationHash: '0x' + attestationHash,
    proofType: 'TEE_ATTESTATION_STUB'
  };
  
  const proofHex = '0x' + Buffer.from(JSON.stringify(proofPayload)).toString('hex');
  
  console.log('✅ TEE attestation proof generated:');
  console.log('  - Proof Length:', proofHex.length, 'characters');
  console.log('  - Attestation Hash:', '0x' + attestationHash);
  console.log('  - Enclave Measurement:', attestationData.enclaveInfo.measurement);
  
  return proofHex;
}

/**
 * Generate sealed key for new owner (simulated re-encryption)
 * In production, this would be done within a TEE with actual re-encryption
 */
function generateSealedKey(
  tokenId: number,
  newOwner: string,
  originalKey?: string
): string {
  
  console.log('🔐 Generating sealed key for new owner (simulated re-encryption)...');
  
  // Simulate re-encryption process
  const reEncryptionData = {
    tokenId,
    newOwner,
    timestamp: new Date().toISOString(),
    originalKeyHash: originalKey ? 
      crypto.createHash('sha256').update(originalKey).digest('hex') : 
      'simulated_original_key',
    
    // Simulated re-encrypted key (in reality, this would be the original key
    // re-encrypted with the new owner's public key)
    reEncryptedKey: crypto.randomBytes(32).toString('hex'),
    
    // Re-encryption metadata
    algorithm: 'AES-256-GCM',
    keyDerivation: 'HKDF-SHA256',
    nonce: crypto.randomBytes(12).toString('hex')
  };
  
  const sealedKeyHex = '0x' + Buffer.from(JSON.stringify(reEncryptionData)).toString('hex');
  
  console.log('✅ Sealed key generated:');
  console.log('  - New Owner:', newOwner);
  console.log('  - Sealed Key Length:', sealedKeyHex.length, 'characters');
  console.log('  - Re-encrypted Key Hash:', crypto.createHash('sha256').update(sealedKeyHex).digest('hex'));
  
  return sealedKeyHex;
}

/**
 * Execute the transfer transaction
 */
async function executeTransfer(
  inftContract: any,
  params: TransferParams,
  signer: any
): Promise<TransferResult> {
  
  console.log('🚀 Executing INFT transfer...');
  console.log('  - From:', params.from);
  console.log('  - To:', params.to);
  console.log('  - Token ID:', params.tokenId);
  console.log('  - Sealed Key Length:', params.sealedKey.length, 'characters');
  console.log('  - Proof Length:', params.proof.length, 'characters');
  
  // Submit the transfer transaction
  console.log('📤 Submitting transfer transaction...');
  const tx = await inftContract.transfer(
    params.from,
    params.to,
    params.tokenId,
    params.sealedKey,
    params.proof
  );
  
  console.log('⏳ Transaction submitted:', tx.hash);
  console.log('⏳ Waiting for confirmation...');
  
  // Wait for transaction to be mined
  const receipt = await tx.wait();
  
  if (receipt.status !== 1) {
    throw new Error(`Transfer transaction failed with status: ${receipt.status}`);
  }
  
  // Parse events from the transaction
  const events = receipt.logs.map((log: any) => {
    try {
      return inftContract.interface.parseLog(log);
    } catch {
      return null;
    }
  }).filter(Boolean);
  
  console.log('📋 Transaction events:');
  events.forEach((event: any, index: number) => {
    console.log(`  ${index + 1}. ${event.name}:`, event.args);
  });
  
  // Calculate proof hash
  const proofHash = ethers.keccak256(params.proof);
  
  console.log('✅ Transfer completed successfully!');
  console.log('  - Transaction Hash:', receipt.hash);
  console.log('  - Block Number:', receipt.blockNumber);
  console.log('  - Gas Used:', receipt.gasUsed.toString());
  console.log('  - Proof Hash:', proofHash);
  
  return {
    tokenId: params.tokenId,
    from: params.from,
    to: params.to,
    txHash: receipt.hash,
    gasUsed: receipt.gasUsed.toString(),
    proofHash,
    sealedKey: params.sealedKey
  };
}

/**
 * Verify transfer completion
 */
async function verifyTransferCompletion(
  inftContract: any,
  transferResult: TransferResult
): Promise<void> {
  
  console.log('🔍 Verifying transfer completion...');
  
  // Check new ownership
  const newOwner = await inftContract.ownerOf(transferResult.tokenId);
  if (newOwner.toLowerCase() !== transferResult.to.toLowerCase()) {
    throw new Error(`Transfer verification failed. Expected owner: ${transferResult.to}, Actual: ${newOwner}`);
  }
  
  // Verify token metadata is unchanged
  const encryptedURI = await inftContract.encryptedURI(transferResult.tokenId);
  const metadataHash = await inftContract.metadataHash(transferResult.tokenId);
  
  console.log('✅ Transfer verification successful:');
  console.log('  - Token ID:', transferResult.tokenId);
  console.log('  - New Owner:', newOwner);
  console.log('  - Encrypted URI preserved:', encryptedURI);
  console.log('  - Metadata Hash preserved:', metadataHash);
  console.log('  - Transaction Hash:', transferResult.txHash);
}

/**
 * Update deployment tracking with transfer info
 */
function updateDeploymentTracking(
  deploymentData: DeploymentData,
  transferResult: TransferResult
): void {
  
  console.log('📝 Updating deployment tracking...');
  
  const deploymentPath = path.join(__dirname, '..', 'deployments', 'galileo.json');
  
  // Add transfer history
  const updatedDeployment = {
    ...deploymentData,
    transferHistory: deploymentData.transferHistory || [],
  };
  
  // Create transfer record
  const transferRecord = {
    tokenId: transferResult.tokenId,
    from: transferResult.from,
    to: transferResult.to,
    sealedKey: transferResult.sealedKey,
    proofHash: transferResult.proofHash,
    transaction: {
      hash: transferResult.txHash,
      gasUsed: transferResult.gasUsed,
      timestamp: new Date().toISOString()
    },
    transferType: 'TEE_ATTESTATION'
  };
  
  updatedDeployment.transferHistory.push(transferRecord);
  
  // Update minted token owner if exists
  if (updatedDeployment.mintedTokens) {
    const tokenIndex = updatedDeployment.mintedTokens.findIndex(
      (token: any) => token.tokenId === transferResult.tokenId
    );
    
    if (tokenIndex !== -1) {
      updatedDeployment.mintedTokens[tokenIndex].owner = transferResult.to;
    }
  }
  
  // Write updated deployment data
  fs.writeFileSync(deploymentPath, JSON.stringify(updatedDeployment, null, 2));
  
  console.log('✅ Deployment tracking updated');
  console.log('  - Added transfer record for Token ID:', transferResult.tokenId);
}

/**
 * Find available token for transfer (from deployment data)
 */
function findAvailableToken(deploymentData: DeploymentData, ownerAddress: string): any {
  if (!deploymentData.mintedTokens || deploymentData.mintedTokens.length === 0) {
    throw new Error('No minted tokens found in deployment data. Please run mint script first.');
  }
  
  // Find a token owned by the specified address
  const ownedToken = deploymentData.mintedTokens.find(
    (token: any) => token.owner.toLowerCase() === ownerAddress.toLowerCase()
  );
  
  if (!ownedToken) {
    throw new Error(`No tokens found owned by address: ${ownerAddress}`);
  }
  
  return ownedToken;
}

/**
 * Main execution function
 */
async function main() {
  console.log('🚀 Starting Phase 7 - INFT Transfer (TEE Path via Oracle)');
  console.log('=' .repeat(70));
  
  try {
    // Step 1: Load deployment data and setup
    const deploymentData = loadDeploymentData();
    const { signer } = await getNetworkInfo();
    
    // Step 2: Find an available token to transfer
    const availableToken = findAvailableToken(deploymentData, signer.address);
    console.log('🎯 Found available token:');
    console.log('  - Token ID:', availableToken.tokenId);
    console.log('  - Current Owner:', availableToken.owner);
    
    // Step 3: Setup transfer parameters
    // For demonstration, we'll transfer to a different address (we can use the same address for testing)
    const recipientAddress = '0x1234567890123456789012345678901234567890'; // Example recipient
    
    // Get original encryption key if available
    const originalKey = availableToken.phase1Storage?.encryptionKey;
    
    // Generate sealed key and proof
    const sealedKey = generateSealedKey(availableToken.tokenId, recipientAddress, originalKey);
    const proof = generateTEEAttestationProof(
      availableToken.tokenId, 
      signer.address, 
      recipientAddress, 
      sealedKey,
      originalKey
    );
    
    const transferParams: TransferParams = {
      from: signer.address,
      to: recipientAddress,
      tokenId: availableToken.tokenId,
      sealedKey,
      proof
    };
    
    // Step 4: Validate parameters
    validateTransferParams(transferParams);
    
    // Step 5: Connect to INFT contract and verify current state
    const INFT = await ethers.getContractFactory('INFT');
    const inftContract = INFT.attach(deploymentData.inft.address).connect(signer);
    
    await verifyTokenOwnership(inftContract, transferParams.tokenId, transferParams.from);
    
    // Step 6: Execute the transfer
    const transferResult = await executeTransfer(inftContract, transferParams, signer);
    
    // Step 7: Verify transfer completion
    await verifyTransferCompletion(inftContract, transferResult);
    
    // Step 8: Update deployment tracking
    updateDeploymentTracking(deploymentData, transferResult);
    
    // Step 9: Output final results
    console.log('\n🎯 PHASE 7 RESULTS:');
    console.log('=' .repeat(70));
    console.log('Transfer Type: TEE Attestation Path');
    console.log('Token ID:', transferResult.tokenId);
    console.log('From:', transferResult.from);
    console.log('To:', transferResult.to);
    console.log('Transaction Hash:', transferResult.txHash);
    console.log('Gas Used:', transferResult.gasUsed);
    console.log('Proof Hash:', transferResult.proofHash);
    console.log('Sealed Key Length:', transferResult.sealedKey.length, 'characters');
    
    console.log('\n✅ Phase 7 completed successfully!');
    console.log('🔗 Transfer verified on 0G Galileo testnet');
    console.log('🔒 TEE attestation proof validated by oracle');
    console.log('🎉 INFT ownership successfully transferred with sealed key re-encryption');
    
  } catch (error) {
    console.error('❌ Error during transfer process:', error);
    process.exit(1);
  }
}

/**
 * Transfer with custom parameters
 */
async function transferWithCustomParams(
  tokenId?: number,
  from?: string,
  to?: string,
  sealedKey?: string,
  proof?: string
) {
  console.log('🚀 Starting Custom INFT Transfer');
  console.log('=' .repeat(50));
  
  try {
    const deploymentData = loadDeploymentData();
    const { signer } = await getNetworkInfo();
    
    // Use provided parameters or generate them
    const transferParams: TransferParams = {
      tokenId: tokenId || 1,
      from: from || signer.address,
      to: to || '0x1234567890123456789012345678901234567890',
      sealedKey: sealedKey || generateSealedKey(tokenId || 1, to || '0x1234567890123456789012345678901234567890'),
      proof: proof || generateTEEAttestationProof(tokenId || 1, from || signer.address, to || '0x1234567890123456789012345678901234567890', sealedKey || 'dummy')
    };
    
    validateTransferParams(transferParams);
    
    const INFT = await ethers.getContractFactory('INFT');
    const inftContract = INFT.attach(deploymentData.inft.address).connect(signer);
    
    await verifyTokenOwnership(inftContract, transferParams.tokenId, transferParams.from);
    
    const transferResult = await executeTransfer(inftContract, transferParams, signer);
    
    await verifyTransferCompletion(inftContract, transferResult);
    
    updateDeploymentTracking(deploymentData, transferResult);
    
    console.log('\n✅ Custom transfer completed successfully!');
    return transferResult;
    
  } catch (error) {
    console.error('❌ Error during custom transfer:', error);
    throw error;
  }
}

// Run the main function if this script is executed directly
if (require.main === module) {
  // Check for command line arguments
  const args = process.argv.slice(2);
  if (args.length > 0) {
    const [tokenId, from, to, sealedKey, proof] = args;
    transferWithCustomParams(
      tokenId ? parseInt(tokenId) : undefined,
      from,
      to,
      sealedKey,
      proof
    ).catch(console.error);
  } else {
    main().catch(console.error);
  }
}

export { 
  main, 
  transferWithCustomParams, 
  generateTEEAttestationProof,
  generateSealedKey,
  loadDeploymentData
};
