// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "./interfaces/IDataVerifierAdapter.sol";

/**
 * @title INFT - Intelligent NFT Implementation (ERC-7857 Compliant)
 * @dev Implementation of ERC-7857 Intelligent NFTs with 0G Storage integration
 * 
 * This contract implements the ERC-7857 specification for Intelligent NFTs that can:
 * - Store encrypted AI agent metadata on 0G Storage (off-chain references only)
 * - Support secure transfer of both ownership and encrypted metadata via oracle proofs
 * - Enable authorized usage of AI agents without transferring ownership
 * - Maintain privacy through cryptographic proofs (TEE attestations or ZKPs)
 * 
 * Key ERC-7857 Features:
 * - transfer() with sealed key and proof verification
 * - clone() to create copies with re-encrypted metadata
 * - authorizeUsage() for temporary usage permissions
 * - Integration with IDataVerifier (oracle) for proof verification
 * 
 * Storage Model (per ERC-7857 spec):
 * - On-chain: Only references (encryptedURI, metadataHash) and authorizations
 * - Off-chain (0G Storage): Encrypted AI agent bundles (models, memory, config)
 */
contract INFT is ERC721, Ownable, ReentrancyGuard {
    
    /// @dev Current token ID counter for minting
    uint256 private _tokenIdCounter;
    
    /// @dev Mapping from token ID to encrypted data URI on 0G Storage
    mapping(uint256 => string) public encryptedURI;
    
    /// @dev Mapping from token ID to metadata hash for verification
    mapping(uint256 => bytes32) public metadataHash;
    
    /// @dev Mapping from token ID to mapping of authorized users
    mapping(uint256 => mapping(address => bool)) public authorizedUsers;
    
    /// @dev Mapping from token ID to array of authorized user addresses for enumeration
    mapping(uint256 => address[]) private _authorizedUsersArray;
    
    /// @dev Oracle/verifier contract for proof verification
    IDataVerifierAdapter public immutable dataVerifier;
    
    // ================================
    // ERC-7857 REQUIRED EVENTS
    // ================================
    
    /**
     * @dev Emitted when usage authorization is granted or revoked
     * @param tokenId The INFT token ID
     * @param user The authorized user address
     * @param authorized True if authorized, false if revoked
     */
    event AuthorizedUsage(uint256 indexed tokenId, address indexed user, bool authorized);
    
    /**
     * @dev Emitted when an INFT is transferred with proof verification
     * @param from Previous owner address
     * @param to New owner address 
     * @param tokenId The INFT token ID
     * @param sealedKey Encrypted key for the new owner
     * @param proofHash Hash of the verification proof used
     */
    event Transferred(
        address indexed from, 
        address indexed to, 
        uint256 indexed tokenId, 
        bytes sealedKey, 
        bytes32 proofHash
    );
    
    /**
     * @dev Emitted when an INFT is cloned (copied with re-encrypted metadata)
     * @param originalTokenId The source token ID
     * @param clonedTokenId The new cloned token ID
     * @param to The owner of the cloned token
     * @param sealedKey Encrypted key for the clone owner
     * @param proofHash Hash of the verification proof used
     */
    event Cloned(
        uint256 indexed originalTokenId,
        uint256 indexed clonedTokenId, 
        address indexed to,
        bytes sealedKey,
        bytes32 proofHash
    );
    
    /**
     * @dev Emitted when a sealed key is published for a token
     * @param tokenId The INFT token ID
     * @param sealedKey The published sealed key
     * @param recipient The intended recipient of the sealed key
     */
    event PublishedSealedKey(uint256 indexed tokenId, bytes sealedKey, address indexed recipient);
    
    // ================================
    // CONSTRUCTOR
    // ================================
    
    /**
     * @dev Constructor initializes the INFT contract
     * @param _name Token collection name
     * @param _symbol Token collection symbol
     * @param _dataVerifier Address of the IDataVerifierAdapter contract
     * @param _initialOwner Initial owner of the contract (for Ownable)
     */
    constructor(
        string memory _name,
        string memory _symbol,
        address _dataVerifier,
        address _initialOwner
    ) ERC721(_name, _symbol) Ownable(_initialOwner) {
        require(_dataVerifier != address(0), "DataVerifier address cannot be zero");
        dataVerifier = IDataVerifierAdapter(_dataVerifier);
        _tokenIdCounter = 1; // Start token IDs from 1
    }
    
    // ================================
    // ERC-7857 CORE FUNCTIONS
    // ================================
    
    /**
     * @dev Transfer an INFT with sealed key and proof verification (ERC-7857 required)
     * @param from Current owner address
     * @param to New owner address
     * @param tokenId Token ID to transfer
     * @param sealedKey Encrypted key for the new owner (from TEE/ZKP re-encryption)
     * @param proof Cryptographic proof of valid re-encryption (TEE attestation or ZKP)
     * 
     * This function verifies the proof through the oracle before transferring ownership.
     * The proof attests that the metadata was correctly re-encrypted for the new owner.
     */
    function transfer(
        address from,
        address to,
        uint256 tokenId,
        bytes calldata sealedKey,
        bytes calldata proof
    ) external nonReentrant {
        require(_isAuthorized(ownerOf(tokenId), _msgSender(), tokenId), "Caller is not owner nor approved");
        require(from == ownerOf(tokenId), "From address is not the owner");
        require(to != address(0), "Cannot transfer to zero address");
        require(sealedKey.length > 0, "Sealed key cannot be empty");
        
        // Verify the re-encryption proof through oracle
        require(
            dataVerifier.verifyTransferValidity(proof), 
            "Invalid transfer proof"
        );
        
        // Clear all authorizations for this token on transfer
        _clearAuthorizations(tokenId);
        
        // Perform the actual transfer
        _transfer(from, to, tokenId);
        
        // Emit ERC-7857 required event
        bytes32 proofHash = keccak256(proof);
        emit Transferred(from, to, tokenId, sealedKey, proofHash);
        
        // Emit sealed key publication event
        emit PublishedSealedKey(tokenId, sealedKey, to);
    }
    
    /**
     * @dev Clone an INFT to create a copy with re-encrypted metadata (ERC-7857 required)
     * @param from Original token owner (must be caller or approved)
     * @param to New clone owner address
     * @param tokenId Original token ID to clone
     * @param sealedKey Encrypted key for the clone owner
     * @param proof Cryptographic proof of valid re-encryption for cloning
     * @return newTokenId The token ID of the newly created clone
     * 
     * Cloning creates a new token with the same metadata reference but re-encrypted
     * for the clone owner. This enables "copying" AI agents with proper access control.
     */
    function clone(
        address from,
        address to,
        uint256 tokenId,
        bytes calldata sealedKey,
        bytes calldata proof
    ) external nonReentrant returns (uint256 newTokenId) {
        require(_isAuthorized(ownerOf(tokenId), _msgSender(), tokenId), "Caller is not owner nor approved");
        require(from == ownerOf(tokenId), "From address is not the owner");
        require(to != address(0), "Cannot clone to zero address");
        _requireOwned(tokenId);
        require(sealedKey.length > 0, "Sealed key cannot be empty");
        
        // Verify the re-encryption proof for cloning
        require(
            dataVerifier.verifyTransferValidity(proof),
            "Invalid clone proof"
        );
        
        // Create new token ID
        newTokenId = _tokenIdCounter++;
        
        // Copy metadata references from original token
        encryptedURI[newTokenId] = encryptedURI[tokenId];
        metadataHash[newTokenId] = metadataHash[tokenId];
        
        // Mint the cloned token to the new owner
        _mint(to, newTokenId);
        
        // Emit ERC-7857 required events
        bytes32 proofHash = keccak256(proof);
        emit Cloned(tokenId, newTokenId, to, sealedKey, proofHash);
        emit PublishedSealedKey(newTokenId, sealedKey, to);
        
        return newTokenId;
    }
    
    /**
     * @dev Authorize or revoke usage permissions for an INFT (ERC-7857 required)
     * @param tokenId Token ID to authorize usage for
     * @param user Address to grant or revoke authorization
     * 
     * Note: ERC-7857 specification only requires (tokenId, user) parameters.
     * This implementation follows the standard. If permissions data is needed,
     * it can be handled through separate function calls or off-chain mechanisms.
     */
    function authorizeUsage(uint256 tokenId, address user) external {
        require(user != address(0), "Cannot authorize zero address");
        require(_isAuthorized(ownerOf(tokenId), _msgSender(), tokenId), "Caller is not owner nor approved");
        
        // Grant authorization if not already authorized
        if (!authorizedUsers[tokenId][user]) {
            authorizedUsers[tokenId][user] = true;
            _authorizedUsersArray[tokenId].push(user);
            emit AuthorizedUsage(tokenId, user, true);
        }
    }
    
    /**
     * @dev Revoke usage authorization for an INFT 
     * @param tokenId Token ID to revoke authorization for
     * @param user Address to revoke authorization from
     */
    function revokeUsage(uint256 tokenId, address user) external {
        require(_isAuthorized(ownerOf(tokenId), _msgSender(), tokenId), "Caller is not owner nor approved");
        
        if (authorizedUsers[tokenId][user]) {
            authorizedUsers[tokenId][user] = false;
            _removeFromAuthorizedArray(tokenId, user);
            emit AuthorizedUsage(tokenId, user, false);
        }
    }
    
    // ================================
    // ERC-7857 VIEW FUNCTIONS
    // ================================
    
    /**
     * @dev Get all authorized users for a token (ERC-7857 helper function)
     * @param tokenId Token ID to query
     * @return Array of authorized user addresses
     */
    function authorizedUsersOf(uint256 tokenId) external view returns (address[] memory) {
        _requireOwned(tokenId); // This will throw ERC721NonexistentToken if token doesn't exist
        
        // Filter out revoked users (where authorized = false)
        address[] memory allUsers = _authorizedUsersArray[tokenId];
        uint256 count = 0;
        
        // Count active authorizations
        for (uint256 i = 0; i < allUsers.length; i++) {
            if (authorizedUsers[tokenId][allUsers[i]]) {
                count++;
            }
        }
        
        // Build result array with only active authorizations
        address[] memory result = new address[](count);
        uint256 index = 0;
        for (uint256 i = 0; i < allUsers.length; i++) {
            if (authorizedUsers[tokenId][allUsers[i]]) {
                result[index] = allUsers[i];
                index++;
            }
        }
        
        return result;
    }
    
    /**
     * @dev Check if a user is authorized to use a token
     * @param tokenId Token ID to check
     * @param user User address to check
     * @return True if user is authorized
     */
    function isAuthorized(uint256 tokenId, address user) external view returns (bool) {
        return authorizedUsers[tokenId][user];
    }
    
    // ================================
    // MINTING FUNCTIONS (Phase 1 Integration)
    // ================================
    
    /**
     * @dev Mint a new INFT with encrypted metadata reference
     * @param to Owner of the new token
     * @param _encryptedURI URI pointing to encrypted metadata on 0G Storage
     * @param _metadataHash Hash of the encrypted metadata for verification
     * @return tokenId The ID of the newly minted token
     * 
     * This function integrates with Phase 1 storage pipeline outputs.
     */
    function mint(
        address to,
        string memory _encryptedURI,
        bytes32 _metadataHash
    ) external onlyOwner returns (uint256 tokenId) {
        require(to != address(0), "Cannot mint to zero address");
        require(bytes(_encryptedURI).length > 0, "Encrypted URI cannot be empty");
        require(_metadataHash != bytes32(0), "Metadata hash cannot be zero");
        
        tokenId = _tokenIdCounter++;
        
        // Store metadata references (off-chain pointers only, per ERC-7857)
        encryptedURI[tokenId] = _encryptedURI;
        metadataHash[tokenId] = _metadataHash;
        
        // Mint the token
        _mint(to, tokenId);
        
        return tokenId;
    }
    
    // ================================
    // INTERNAL HELPER FUNCTIONS
    // ================================
    
    /**
     * @dev Clear all authorizations for a token (used during transfers)
     * @param tokenId Token ID to clear authorizations for
     */
    function _clearAuthorizations(uint256 tokenId) internal {
        address[] memory users = _authorizedUsersArray[tokenId];
        for (uint256 i = 0; i < users.length; i++) {
            if (authorizedUsers[tokenId][users[i]]) {
                authorizedUsers[tokenId][users[i]] = false;
                emit AuthorizedUsage(tokenId, users[i], false);
            }
        }
        // Clear the array
        delete _authorizedUsersArray[tokenId];
    }
    
    /**
     * @dev Remove a user from the authorized users array
     * @param tokenId Token ID
     * @param user User address to remove
     */
    function _removeFromAuthorizedArray(uint256 tokenId, address user) internal {
        address[] storage users = _authorizedUsersArray[tokenId];
        for (uint256 i = 0; i < users.length; i++) {
            if (users[i] == user) {
                // Replace with last element and pop
                users[i] = users[users.length - 1];
                users.pop();
                break;
            }
        }
    }
    
    /**
     * @dev Check if a token exists
     * @param tokenId Token ID to check
     * @return True if token exists
     */
    function _exists(uint256 tokenId) internal view returns (bool) {
        return _ownerOf(tokenId) != address(0);
    }
    
    // ================================
    // OVERRIDE FUNCTIONS
    // ================================
    
    /**
     * @dev Override _beforeTokenTransfer to clear authorizations on regular transfers
     */
    function _update(address to, uint256 tokenId, address auth) internal virtual override returns (address) {
        address from = _ownerOf(tokenId);
        
        // Clear authorizations on transfer (but not on mint)
        if (from != address(0) && to != address(0)) {
            _clearAuthorizations(tokenId);
        }
        
        return super._update(to, tokenId, auth);
    }
    
    /**
     * @dev Override tokenURI to return the encrypted URI from 0G Storage
     * @param tokenId Token ID to get URI for
     * @return The encrypted URI pointing to 0G Storage
     */
    function tokenURI(uint256 tokenId) public view virtual override returns (string memory) {
        require(_exists(tokenId), "URI query for nonexistent token");
        return encryptedURI[tokenId];
    }
    
    /**
     * @dev Get the current token ID counter value
     * @return Current token ID counter
     */
    function getCurrentTokenId() external view returns (uint256) {
        return _tokenIdCounter;
    }
}
