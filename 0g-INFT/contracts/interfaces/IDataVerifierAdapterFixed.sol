// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./IOracle.sol";

/**
 * @title IDataVerifierAdapter (Fixed Version)
 * @dev Improved adapter that bubbles up revert data and uses custom errors for gas efficiency
 */
interface IDataVerifierAdapter {
    function verifyOwnership(bytes calldata data) external returns (bool);
    function verifyTransferValidity(bytes calldata data) external returns (bool);
    function getOracleAddress() external view returns (address);
}

/**
 * @title DataVerifierAdapterFixed  
 * @dev Fixed implementation with proper error bubbling and gas optimization
 */
contract DataVerifierAdapterFixed is IDataVerifierAdapter {
    IOracle public immutable oracle;
    
    // Custom errors for gas efficiency and better debugging
    error OracleCallFailed(bytes lowLevelData);
    error InvalidDataFormat();
    error VerificationFailed();
    
    // Events for tracking
    event OwnershipVerified(bytes32 indexed dataHash, bool result);
    event TransferValidityVerified(bytes32 indexed dataHash, bool result);
    
    constructor(address _oracle) {
        require(_oracle != address(0), "Oracle address cannot be zero");
        oracle = IOracle(_oracle);
    }
    
    /**
     * @dev Implements ERC-7857 verifyOwnership with proper error bubbling
     */
    function verifyOwnership(bytes calldata data) external override returns (bool) {
        bytes32 dataHash = keccak256(data);
        bool result = _verifyWithBubbling(data);
        emit OwnershipVerified(dataHash, result);
        return result;
    }
    
    /**
     * @dev Implements ERC-7857 verifyTransferValidity with proper error bubbling
     */
    function verifyTransferValidity(bytes calldata data) external override returns (bool) {
        bytes32 dataHash = keccak256(data);
        bool result = _verifyWithBubbling(data);
        emit TransferValidityVerified(dataHash, result);
        return result;
    }
    
    /**
     * @dev Internal verification with proper error bubbling
     * @param data Verification data to pass to oracle
     * @return bool Verification result
     */
    function _verifyWithBubbling(bytes calldata data) internal returns (bool) {
        // Direct call to oracle with proper error bubbling
        (bool success, bytes memory returnData) = address(oracle).call(
            abi.encodeWithSelector(IOracle.verifyProof.selector, data)
        );
        
        if (!success) {
            // Bubble up the original revert reason if available
            if (returnData.length > 0) {
                assembly {
                    revert(add(returnData, 0x20), mload(returnData))
                }
            } else {
                revert OracleCallFailed(returnData);
            }
        }
        
        return abi.decode(returnData, (bool));
    }
    
    /**
     * @dev Returns the oracle contract address for transparency
     */
    function getOracleAddress() external view override returns (address) {
        return address(oracle);
    }
}
