/**
 * TEE Verification Utilities
 * Functions to verify Phala TEE endpoints and attestations
 */

interface HealthCheckResult {
  success: boolean;
  error?: string;
}

interface HashCheckResult {
  success: boolean;
  attestationHash?: string;
  note?: string;
  rtmrs?: string[];
  error?: string;
}

/**
 * Check if the TEE endpoint health endpoint is accessible
 */
export const healthCheck = async (shardUrl: string): Promise<HealthCheckResult> => {
  try {
    const baseUrl = shardUrl.endsWith('/') ? shardUrl : `${shardUrl}/`;
    
    const response = await fetch(`${baseUrl}health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.ok) {
      return {
        success: false,
        error: 'Health check failed. TEE endpoint is not accessible.'
      };
    }
    
    return { success: true };
    
  } catch (error: any) {
    return {
      success: false,
      error: error.message || 'Failed to connect to health endpoint.'
    };
  }
};

/**
 * Get attestation from TEE endpoint and generate SHA-256 hash of the note
 */
export const hashCheckModel = async (shardUrl: string): Promise<HashCheckResult> => {
  try {
    const baseUrl = shardUrl.endsWith('/') ? shardUrl : `${shardUrl}/`;
    
    const response = await fetch(`${baseUrl}attest/quick`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.ok) {
      return {
        success: false,
        error: 'Attestation request failed.'
      };
    }
    
    const attestationData = await response.json();
    
    // Verify attestation structure
    if (!attestationData.success) {
      return {
        success: false,
        error: 'Attestation was not successful.'
      };
    }
    
    if (!attestationData.note) {
      return {
        success: false,
        error: 'Attestation note not found.'
      };
    }
    
    // Generate SHA-256 hash of the note
    const encoder = new TextEncoder();
    const data = encoder.encode(attestationData.note);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    
    return {
      success: true,
      attestationHash: hashHex,
      note: attestationData.note,
      rtmrs: attestationData.rtmrs
    };
    
  } catch (error: any) {
    return {
      success: false,
      error: error.message || 'Failed to verify attestation.'
    };
  }
};

/**
 * Complete TEE endpoint verification (health check + hash verification)
 */
export const verifyTEEEndpoint = async (shardUrl: string): Promise<{
  success: boolean;
  attestationHash?: string;
  error?: string;
}> => {
  // Expected attestation hash to verify against
  const EXPECTED_HASH = 'df99bee2e34b5f653722df6fb654c0d906c57173b6be3fab104815e58ce064bc';
  
  // Step 1: Health check first
  const healthResult = await healthCheck(shardUrl);
  if (!healthResult.success) {
    return {
      success: false,
      error: healthResult.error || 'Health check failed'
    };
  }
  
  // Step 2: Hash check model (attestation)
  const hashResult = await hashCheckModel(shardUrl);
  if (!hashResult.success) {
    return {
      success: false,
      error: hashResult.error || 'Attestation check failed'
    };
  }
  
  // Step 3: Verify the hash matches the expected value
  if (hashResult.attestationHash !== EXPECTED_HASH) {
    return {
      success: false,
      error: `Attestation hash mismatch. Expected: ${EXPECTED_HASH}, Got: ${hashResult.attestationHash}`
    };
  }
  
  // Log verification details
  console.log('TEE Verification Success:', {
    note: hashResult.note,
    hash: hashResult.attestationHash,
    rtmrs: hashResult.rtmrs
  });
  
  return {
    success: true,
    attestationHash: hashResult.attestationHash
  };
};

