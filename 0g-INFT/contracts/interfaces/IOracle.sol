// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title IOracle
 * @dev Oracle interface for verifying proofs in ERC-7857 INFT transfers
 * This interface follows the 0G INFT integration pattern where proof verification
 * is delegated to external oracle services (TEE or ZKP-based)
 */
interface IOracle {
    /**
     * @dev Verifies a cryptographic proof for INFT metadata transfer
     * @param proof The cryptographic proof to verify (can be TEE attestation or ZKP)
     * @return bool True if the proof is valid, false otherwise
     * 
     * The proof format depends on the oracle implementation:
     * - TEE Oracle: proof contains enclave attestation + re-encryption proof
     * - ZKP Oracle: proof contains zero-knowledge proof of correct re-encryption
     */
    function verifyProof(bytes calldata proof) external returns (bool);
    
    /**
     * @dev Verifies a proof against a specific metadata hash
     * @param proof The cryptographic proof to verify
     * @param metadataHash The hash of the original metadata being transferred
     * @return bool True if the proof is valid for the given metadata hash
     */
    function verifyProofWithHash(bytes calldata proof, bytes32 metadataHash) external returns (bool);
}
