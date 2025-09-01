// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./IOracle.sol";

/**
 * @title IDataVerifierAdapter
 * @dev Adapter interface that bridges our advanced IOracle interface with the ERC-7857 IDataVerifier specification
 * 
 * This adapter wraps our existing IOracle.sol (which supports verifyProof and verifyProofWithHash)
 * and exposes the standard ERC-7857 IDataVerifier interface functions as required by the EIP Draft.
 * 
 * ERC-7857 IDataVerifier specification requires:
 * - verifyOwnership(bytes calldata data) -> bool
 * - verifyTransferValidity(bytes calldata data) -> bool
 * 
 * Our IOracle provides more granular proof verification, so this adapter maps
 * the ERC-7857 standard calls to our underlying oracle implementation.
 */
interface IDataVerifierAdapter {
    /**
     * @dev Verifies ownership of an INFT token according to ERC-7857 specification
     * @param data Encoded verification data containing proof and context
     * @return bool True if ownership verification succeeds
     * 
     * Implementation delegates to IOracle.verifyProof() or verifyProofWithHash()
     * based on the data structure provided.
     */
    function verifyOwnership(bytes calldata data) external returns (bool);
    
    /**
     * @dev Verifies the validity of an INFT transfer according to ERC-7857 specification  
     * @param data Encoded verification data containing transfer proof and metadata
     * @return bool True if transfer validity verification succeeds
     * 
     * Implementation delegates to IOracle.verifyProof() or verifyProofWithHash()
     * for cryptographic verification of re-encryption proofs (TEE attestation or ZKP).
     */
    function verifyTransferValidity(bytes calldata data) external returns (bool);
    
    /**
     * @dev Returns the underlying oracle contract address for transparency
     * @return address The address of the wrapped IOracle implementation
     */
    function getOracleAddress() external view returns (address);
}

/**
 * @title DataVerifierAdapter
 * @dev Concrete implementation of IDataVerifierAdapter that wraps an IOracle contract
 * 
 * This adapter enables ERC-7857 compliance while preserving the advanced functionality
 * of our existing oracle infrastructure. It handles the encoding/decoding of ERC-7857
 * standard data formats to our oracle's proof verification methods.
 */
contract DataVerifierAdapter is IDataVerifierAdapter {
    /// @dev The underlying oracle contract that performs actual verification
    IOracle public immutable oracle;
    
    /// @dev Emitted when the adapter successfully verifies ownership
    event OwnershipVerified(bytes32 indexed dataHash, bool result);
    
    /// @dev Emitted when the adapter successfully verifies transfer validity
    event TransferValidityVerified(bytes32 indexed dataHash, bool result);
    
    /**
     * @dev Constructor sets the oracle contract address
     * @param _oracle Address of the IOracle implementation to wrap
     */
    constructor(address _oracle) {
        require(_oracle != address(0), "Oracle address cannot be zero");
        oracle = IOracle(_oracle);
    }
    
    /**
     * @dev Implements ERC-7857 verifyOwnership by delegating to oracle.verifyProof()
     * @param data Encoded verification data - expects (bytes proof) or (bytes proof, bytes32 hash)
     * @return bool Verification result from underlying oracle
     */
    function verifyOwnership(bytes calldata data) external override returns (bool) {
        bytes32 dataHash = keccak256(data);
        bool result;
        
        // Try to decode as (proof, metadataHash) first, fallback to just (proof)
        if (data.length >= 64) { // Minimum size for proof + hash
            try this._verifyWithHash(data) returns (bool success) {
                result = success;
            } catch {
                // Fallback to simple proof verification
                result = oracle.verifyProof(data);
            }
        } else {
            // Simple proof verification for smaller data
            result = oracle.verifyProof(data);
        }
        
        emit OwnershipVerified(dataHash, result);
        return result;
    }
    
    /**
     * @dev Implements ERC-7857 verifyTransferValidity by delegating to oracle verification
     * @param data Encoded transfer verification data
     * @return bool Verification result from underlying oracle
     */
    function verifyTransferValidity(bytes calldata data) external override returns (bool) {
        bytes32 dataHash = keccak256(data);
        bool result;
        
        // Transfer validity typically includes metadata hash for re-encryption verification
        if (data.length >= 64) {
            try this._verifyWithHash(data) returns (bool success) {
                result = success;
            } catch {
                result = oracle.verifyProof(data);
            }
        } else {
            result = oracle.verifyProof(data);
        }
        
        emit TransferValidityVerified(dataHash, result);
        return result;
    }
    
    /**
     * @dev Returns the oracle contract address for transparency
     * @return address The wrapped oracle contract address
     */
    function getOracleAddress() external view override returns (address) {
        return address(oracle);
    }
    
    /**
     * @dev Internal helper to attempt verification with metadata hash
     * @param data Encoded data containing proof and hash
     * @return bool Verification result
     */
    function _verifyWithHash(bytes calldata data) external returns (bool) {
        // Decode as (bytes proof, bytes32 metadataHash)
        // This is a simplified approach - in production, use proper ABI encoding
        require(data.length >= 64, "Invalid data format");
        
        // Extract last 32 bytes as metadata hash
        bytes32 metadataHash = bytes32(data[data.length - 32:]);
        
        // Extract remaining bytes as proof
        bytes memory proof = data[:data.length - 32];
        
        return oracle.verifyProofWithHash(proof, metadataHash);
    }
}
