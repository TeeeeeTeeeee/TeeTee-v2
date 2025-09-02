// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./interfaces/IOracle.sol";

/**
 * @title OracleStub
 * @dev Development-only oracle stub that always returns true for proof verification
 * WARNING: This is for development and testing purposes only!
 * In production, this should be replaced with a real TEE or ZKP oracle
 * that properly verifies cryptographic proofs for INFT metadata transfers.
 */
contract OracleStub is IOracle {
    address public owner;
    bool public verificationEnabled;
    
    // Events for development tracking
    event ProofVerified(bytes proof, bool result);
    event ProofVerifiedWithHash(bytes proof, bytes32 metadataHash, bool result);
    event VerificationToggled(bool enabled);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "OracleStub: caller is not the owner");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        verificationEnabled = true;
    }
    
    /**
     * @dev Always returns true for development purposes
     * @param proof The proof to "verify" (ignored in stub)
     * @return bool Always returns true when verification is enabled
     */
    function verifyProof(bytes calldata proof) external override returns (bool) {
        bool result = verificationEnabled;
        emit ProofVerified(proof, result);
        return result;
    }
    
    /**
     * @dev Always returns true for development purposes  
     * @param proof The proof to "verify" (ignored in stub)
     * @param metadataHash The metadata hash (ignored in stub)
     * @return bool Always returns true when verification is enabled
     */
    function verifyProofWithHash(bytes calldata proof, bytes32 metadataHash) external override returns (bool) {
        bool result = verificationEnabled;
        emit ProofVerifiedWithHash(proof, metadataHash, result);
        return result;
    }
    
    /**
     * @dev Toggle verification for testing failure scenarios
     * @param enabled Whether to enable or disable verification
     */
    function setVerificationEnabled(bool enabled) external onlyOwner {
        verificationEnabled = enabled;
        emit VerificationToggled(enabled);
    }
    
    /**
     * @dev Transfer ownership of the stub
     * @param newOwner The new owner address
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "OracleStub: new owner is the zero address");
        owner = newOwner;
    }
}
