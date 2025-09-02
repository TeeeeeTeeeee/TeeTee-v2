/**
 * INFT Transfer API Endpoint
 * Integrates TEE mock transfer functionality from scripts/transfer.ts
 * Provides the frontend with proper TEE attestation and sealed key generation
 */

import crypto from 'crypto'

/**
 * Generate TEE attestation proof stub for development
 * This is extracted from scripts/transfer.ts lines 184-254
 */
function generateTEEAttestationProof(tokenId, from, to, sealedKey, originalEncryptionKey) {
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
  // Create a more compact proof payload for gas optimization
  const proofPayload = {
    v: '1.0', // version shortened
    type: 'TEE_STUB', // type shortened
    ts: Date.now(), // timestamp as number
    hash: attestationHash, // just hash without 0x prefix
    sig: attestationData.signature.signature.slice(-32) // shortened signature
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
 * This is extracted from scripts/transfer.ts lines 256-295
 */
function generateSealedKey(tokenId, newOwner, originalKey) {
  console.log('🔐 Generating sealed key for new owner (simulated re-encryption)...');
  
  // Create compact re-encryption data for gas optimization
  const reEncryptionData = {
    id: tokenId,
    to: newOwner,
    ts: Date.now(),
    key: crypto.randomBytes(32).toString('hex'),
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
 * Validate Ethereum addresses
 */
function isValidAddress(address) {
  return /^0x[a-fA-F0-9]{40}$/.test(address);
}

/**
 * API Handler for INFT transfer preparation
 */
export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ 
      error: 'Method not allowed',
      message: 'Only POST requests are supported'
    });
  }

  try {
    const { from, to, tokenId, originalKey } = req.body;

    // Validate required parameters
    if (!from || !to || !tokenId) {
      return res.status(400).json({
        error: 'Missing required parameters',
        message: 'from, to, and tokenId are required'
      });
    }

    // Validate addresses
    if (!isValidAddress(from)) {
      return res.status(400).json({
        error: 'Invalid from address',
        message: 'from must be a valid Ethereum address'
      });
    }

    if (!isValidAddress(to)) {
      return res.status(400).json({
        error: 'Invalid to address', 
        message: 'to must be a valid Ethereum address'
      });
    }

    // Validate token ID
    const tokenIdNum = parseInt(tokenId);
    if (isNaN(tokenIdNum) || tokenIdNum < 1) {
      return res.status(400).json({
        error: 'Invalid token ID',
        message: 'tokenId must be a positive integer'
      });
    }

    console.log(`🔄 Processing transfer request for Token ${tokenIdNum} from ${from} to ${to}`);

    // Step 1: Validate token ownership
    try {
      console.log('🔍 Validating token ownership...');
      
      const ownershipResponse = await fetch('https://evmrpc-testnet.0g.ai', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'eth_call',
          params: [{
            to: '0x18db2ED477A25Aac615D803aE7be1d3598cdfF95', // Fixed INFT contract
            data: '0x6352211e' + tokenIdNum.toString(16).padStart(64, '0') // ownerOf(tokenId)
          }, 'latest'],
          id: 1
        })
      });

      const ownershipResult = await ownershipResponse.json();
      
      if (ownershipResult.error) {
        throw new Error(`Failed to check ownership: ${ownershipResult.error.message}`);
      }

      // Check if the result is null or empty (token doesn't exist)
      if (!ownershipResult.result || ownershipResult.result === '0x') {
        return res.status(404).json({
          error: 'Token does not exist',
          message: `Token ${tokenIdNum} has not been minted yet`,
          details: {
            tokenId: tokenIdNum,
            requestedFrom: from
          }
        });
      }

      const actualOwner = '0x' + ownershipResult.result.slice(-40).toLowerCase();
      const fromLower = from.toLowerCase();

      if (actualOwner !== fromLower) {
        return res.status(403).json({
          error: 'Ownership validation failed',
          message: `Token ${tokenIdNum} is owned by ${actualOwner}, not ${from}`,
          details: {
            tokenId: tokenIdNum,
            actualOwner,
            requestedFrom: from
          }
        });
      }

      console.log(`✅ Ownership validated: ${from} owns Token ${tokenIdNum}`);
      
    } catch (ownershipError) {
      console.error('❌ Ownership validation error:', ownershipError);
      
      const errorMessage = ownershipError instanceof Error ? ownershipError.message : String(ownershipError);
      
      // Check if this is a "token doesn't exist" error
      if (errorMessage.includes('execution reverted') || 
          errorMessage.includes('ERC721NonexistentToken') ||
          errorMessage.includes('invalid token ID') ||
          errorMessage.includes('nonexistent token')) {
        return res.status(404).json({
          error: 'Token does not exist',
          message: `Token ${tokenIdNum} has not been minted yet`,
          details: {
            tokenId: tokenIdNum,
            requestedFrom: from
          }
        });
      }
      
      return res.status(500).json({
        error: 'Ownership validation failed',
        message: 'Unable to verify token ownership',
        details: errorMessage
      });
    }

    // Step 2: Generate sealed key for new owner
    const sealedKey = generateSealedKey(tokenIdNum, to, originalKey);

    // Step 3: Generate TEE attestation proof
    const proof = generateTEEAttestationProof(tokenIdNum, from, to, sealedKey, originalKey);

    // Return the transfer data needed for the blockchain transaction
    const response = {
      success: true,
      transferData: {
        from,
        to,
        tokenId: tokenIdNum,
        sealedKey,
        proof
      },
      metadata: {
        timestamp: new Date().toISOString(),
        proofLength: proof.length,
        sealedKeyLength: sealedKey.length,
        proofType: 'TEE_ATTESTATION_STUB'
      }
    };

    console.log('✅ Transfer preparation completed successfully');
    
    return res.status(200).json(response);

  } catch (error) {
    console.error('❌ Transfer API error:', error);
    
    return res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to prepare transfer data',
      details: error instanceof Error ? error.message : String(error)
    });
  }
}
