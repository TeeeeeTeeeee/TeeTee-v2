// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./interfaces/IDataVerifierAdapter.sol";

/**
 * @title INFT (Fixed Version)
 * @dev Fixed implementation with custom errors for better gas efficiency and error reporting
 */
contract INFTFixed is ERC721, Ownable, ReentrancyGuard {
    
    // Custom errors for gas efficiency and better debugging
    error CallerNotAuthorized();
    error InvalidFromAddress();
    error TransferToZeroAddress();
    error EmptySealedKey();
    error InvalidTransferProof();
    error TokenNotFound();
    error UserAlreadyAuthorized();
    error UserNotAuthorized();
    
    // State variables
    IDataVerifierAdapter public immutable dataVerifier;
    uint256 private _currentTokenId;
    
    // Token metadata storage
    mapping(uint256 => string) private _encryptedURIs;
    mapping(uint256 => bytes32) private _metadataHashes;
    
    // Authorization system for ERC-7857
    mapping(uint256 => mapping(address => bool)) private _authorizedUsers;
    mapping(uint256 => address[]) private _authorizedUsersList;
    
    // Events
    event Transferred(
        address indexed from,
        address indexed to,
        uint256 indexed tokenId,
        string sealedKey,
        bytes32 proofHash
    );
    
    event PublishedSealedKey(
        uint256 indexed tokenId,
        string sealedKey,
        address indexed recipient
    );
    
    event AuthorizedUsage(
        uint256 indexed tokenId,
        address indexed user,
        bool authorized
    );
    
    constructor(
        address _dataVerifier,
        string memory _name,
        string memory _symbol
    ) ERC721(_name, _symbol) Ownable(msg.sender) {
        require(_dataVerifier != address(0), "DataVerifier address cannot be zero");
        dataVerifier = IDataVerifierAdapter(_dataVerifier);
        _currentTokenId = 0;
    }
    
    /**
     * @dev Transfer an INFT with sealed key and proof verification (ERC-7857 required)
     * @param from Current owner address
     * @param to New owner address
     * @param tokenId Token ID to transfer
     * @param sealedKey Encrypted key for the new owner (from TEE/ZKP re-encryption)
     * @param proof Cryptographic proof of valid re-encryption (TEE attestation or ZKP)
     */
    function transfer(
        address from,
        address to,
        uint256 tokenId,
        bytes calldata sealedKey,
        bytes calldata proof
    ) external nonReentrant {
        // Custom error replacements for gas efficiency
        if (!_isAuthorized(ownerOf(tokenId), _msgSender(), tokenId)) {
            revert CallerNotAuthorized();
        }
        if (from != ownerOf(tokenId)) {
            revert InvalidFromAddress();
        }
        if (to == address(0)) {
            revert TransferToZeroAddress();
        }
        if (sealedKey.length == 0) {
            revert EmptySealedKey();
        }
        
        // Verify the re-encryption proof through oracle
        // This will now bubble up proper error messages from the fixed adapter
        if (!dataVerifier.verifyTransferValidity(proof)) {
            revert InvalidTransferProof();
        }
        
        // Clear all authorizations for this token on transfer
        _clearAuthorizations(tokenId);
        
        // Perform the actual transfer
        _transfer(from, to, tokenId);
        
        // Emit ERC-7857 required event
        bytes32 proofHash = keccak256(proof);
        emit Transferred(from, to, tokenId, string(sealedKey), proofHash);
        
        // Emit sealed key publication event
        emit PublishedSealedKey(tokenId, string(sealedKey), to);
    }
    
    /**
     * @dev Mint a new INFT token
     */
    function mint(
        address to,
        string memory encryptedURI,
        bytes32 metadataHash
    ) external onlyOwner returns (uint256) {
        _currentTokenId++;
        uint256 tokenId = _currentTokenId;
        
        _mint(to, tokenId);
        _encryptedURIs[tokenId] = encryptedURI;
        _metadataHashes[tokenId] = metadataHash;
        
        return tokenId;
    }
    
    /**
     * @dev Authorize a user to perform inference with this INFT
     */
    function authorizeUsage(uint256 tokenId, address user) external {
        if (!_isAuthorized(ownerOf(tokenId), _msgSender(), tokenId)) {
            revert CallerNotAuthorized();
        }
        if (_authorizedUsers[tokenId][user]) {
            revert UserAlreadyAuthorized();
        }
        
        _authorizedUsers[tokenId][user] = true;
        _authorizedUsersList[tokenId].push(user);
        
        emit AuthorizedUsage(tokenId, user, true);
    }
    
    /**
     * @dev Revoke a user's authorization for this INFT
     */
    function revokeUsage(uint256 tokenId, address user) external {
        if (!_isAuthorized(ownerOf(tokenId), _msgSender(), tokenId)) {
            revert CallerNotAuthorized();
        }
        if (!_authorizedUsers[tokenId][user]) {
            revert UserNotAuthorized();
        }
        
        _authorizedUsers[tokenId][user] = false;
        
        // Remove from list
        address[] storage usersList = _authorizedUsersList[tokenId];
        for (uint256 i = 0; i < usersList.length; i++) {
            if (usersList[i] == user) {
                usersList[i] = usersList[usersList.length - 1];
                usersList.pop();
                break;
            }
        }
        
        emit AuthorizedUsage(tokenId, user, false);
    }
    
    /**
     * @dev Check if a user is authorized to use this INFT
     */
    function isAuthorized(uint256 tokenId, address user) external view returns (bool) {
        return _authorizedUsers[tokenId][user];
    }
    
    /**
     * @dev Get all authorized users for a token
     */
    function authorizedUsersOf(uint256 tokenId) external view returns (address[] memory) {
        return _authorizedUsersList[tokenId];
    }
    
    /**
     * @dev Get current token ID counter
     */
    function getCurrentTokenId() external view returns (uint256) {
        return _currentTokenId;
    }
    
    /**
     * @dev Get encrypted URI for a token
     */
    function encryptedURI(uint256 tokenId) external view returns (string memory) {
        return _encryptedURIs[tokenId];
    }
    
    /**
     * @dev Get metadata hash for a token
     */
    function metadataHash(uint256 tokenId) external view returns (bytes32) {
        return _metadataHashes[tokenId];
    }
    
    /**
     * @dev Override tokenURI to return encrypted URI
     */
    function tokenURI(uint256 tokenId) public view override returns (string memory) {
        return _encryptedURIs[tokenId];
    }
    
    /**
     * @dev Clear all authorizations for a token (called on transfer)
     */
    function _clearAuthorizations(uint256 tokenId) internal {
        address[] memory users = _authorizedUsersList[tokenId];
        for (uint256 i = 0; i < users.length; i++) {
            _authorizedUsers[tokenId][users[i]] = false;
            emit AuthorizedUsage(tokenId, users[i], false);
        }
        delete _authorizedUsersList[tokenId];
    }
}
