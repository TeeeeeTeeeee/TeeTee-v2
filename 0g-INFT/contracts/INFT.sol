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
 * - authorizeUsage() for temporary usage permissions (using O(1) epoch-based system)
 * - Integration with IDataVerifier (oracle) for proof verification
 * 
 * Storage Model (per ERC-7857 spec):
 * - On-chain: Only references (encryptedURI, metadataHash) and authorizations
 * - Off-chain (0G Storage): Encrypted AI agent bundles (models, memory, config)
 * 
 * Authorization uses an epoch-based system for gas efficiency:
 * - Clearing all authorizations is O(1) instead of O(n)
 * - Each token has an epoch counter that increments on clear
 * - User authorizations are valid only for the current epoch
 */
contract INFT is ERC721, Ownable, ReentrancyGuard {
    
    /// @dev Current token ID counter for minting
    uint256 private _tokenIdCounter;
    
    /// @dev Mapping from token ID to encrypted data URI on 0G Storage
    mapping(uint256 => string) public encryptedURI;
    
    /// @dev Mapping from token ID to metadata hash for verification
    mapping(uint256 => bytes32) public metadataHash;
    
    /// @dev Mapping from token ID to current authorization epoch
    mapping(uint256 => uint64) public authEpoch;
    
    /// @dev Mapping from token ID to user address to their authorization epoch
    mapping(uint256 => mapping(address => uint64)) public userAuthEpoch;
    
    /// @dev Mapping from token ID to original minter/issuer address
    mapping(uint256 => address) public tokenMinter;
    
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
     * @dev Emitted when all authorizations for a token are cleared (epoch bumped)
     * @param tokenId The INFT token ID
     * @param newEpoch The new epoch number
     */
    event AuthorizationsCleared(uint256 indexed tokenId, uint64 newEpoch);
    
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
        
        // Clear all authorizations for this token on transfer (O(1) operation)
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
        
        // Initialize new token's epoch to 1 (0 is used for "never authorized")
        authEpoch[newTokenId] = 1;
        
        // Mint the cloned token to the new owner
        _mint(to, newTokenId);
        
        // Emit ERC-7857 required events
        bytes32 proofHash = keccak256(proof);
        emit Cloned(tokenId, newTokenId, to, sealedKey, proofHash);
        emit PublishedSealedKey(newTokenId, sealedKey, to);
        
        return newTokenId;
    }
    
    /**
     * @dev Authorize or grant usage permissions for an INFT (ERC-7857 required)
     * @param tokenId Token ID to authorize usage for
     * @param user Address to grant authorization
     * 
     * Uses epoch-based authorization for O(1) clearing.
     */
    function authorizeUsage(uint256 tokenId, address user) external {
        require(user != address(0), "Cannot authorize zero address");
        require(_isAuthorized(ownerOf(tokenId), _msgSender(), tokenId), "Caller is not owner nor approved");
        
        uint64 currentEpoch = authEpoch[tokenId];
        if (currentEpoch == 0) {
            // First authorization for this token, initialize epoch
            authEpoch[tokenId] = 1;
            currentEpoch = 1;
        }
        
        // Grant authorization by setting user's epoch to current epoch
        if (userAuthEpoch[tokenId][user] != currentEpoch) {
            userAuthEpoch[tokenId][user] = currentEpoch;
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
        
        uint64 currentEpoch = authEpoch[tokenId];
        // Only revoke if user is currently authorized
        if (userAuthEpoch[tokenId][user] == currentEpoch && currentEpoch != 0) {
            userAuthEpoch[tokenId][user] = 0; // Set to 0 to revoke
            emit AuthorizedUsage(tokenId, user, false);
        }
    }
    
    // ================================
    // ERC-7857 VIEW FUNCTIONS
    // ================================
    
    /**
     * @dev Check if a user is authorized to use a token
     * @param tokenId Token ID to check
     * @param user User address to check
     * @return True if user is authorized for the current epoch
     */
    function isAuthorized(uint256 tokenId, address user) external view returns (bool) {
        uint64 currentEpoch = authEpoch[tokenId];
        if (currentEpoch == 0) return false; // No authorizations exist
        return userAuthEpoch[tokenId][user] == currentEpoch && userAuthEpoch[tokenId][user] != 0;
    }
    
    /**
     * @dev Get all authorized users for a token
     * @param tokenId Token ID to query
     * @return Empty array - enumeration not supported with epoch-based system
     * 
     * NOTE: With epoch-based authorization, on-chain enumeration is not efficient.
     * Listen to AuthorizedUsage events off-chain to track authorized users.
     */
    function authorizedUsersOf(uint256 tokenId) external view returns (address[] memory) {
        _requireOwned(tokenId); // Verify token exists
        // Return empty array - enumeration should be done off-chain via events
        return new address[](0);
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
        
        // Track the original minter/issuer
        tokenMinter[tokenId] = _msgSender();
        
        // Initialize epoch to 1 (0 means no authorizations)
        authEpoch[tokenId] = 1;
        
        // Mint the token
        _mint(to, tokenId);
        
        return tokenId;
    }
    
    /**
     * @dev Burn an INFT token
     * @param tokenId Token ID to burn
     * 
     * Requirements:
     * - Caller must be the token owner or approved
     * - Clears all authorizations in O(1) time (epoch-based)
     * - Removes all metadata references
     */
    function burn(uint256 tokenId) external {
        address tokenOwner = ownerOf(tokenId);
        
        // Only token owner or approved can burn their own token
        require(
            _isAuthorized(tokenOwner, _msgSender(), tokenId),
            "Caller is not token owner nor approved"
        );
        
        // Clear all authorizations before burning (O(1) operation!)
        _clearAuthorizations(tokenId);
        
        // Clear metadata references
        delete encryptedURI[tokenId];
        delete metadataHash[tokenId];
        delete tokenMinter[tokenId];
        delete authEpoch[tokenId];
        
        // Burn the token (automatically emits Transfer event)
        _burn(tokenId);
    }
    
    /**
     * @dev Owner-only function to "burn" any token by transferring to blackhole
     * @param tokenId Token ID to burn
     * 
     * This is a simpler alternative to burn() that avoids potential issues by
     * transferring the token to a blackhole address (0x...dEaD) instead of calling _burn().
     * 
     * Requirements:
     * - Caller must be the contract owner
     * - Clears all authorizations in O(1) time (epoch-based)
     */
    function ownerBurn(uint256 tokenId) external onlyOwner {
        address tokenOwner = ownerOf(tokenId);
        address blackhole = 0x000000000000000000000000000000000000dEaD;
        
        // Clear all authorizations (O(1) operation!)
        _clearAuthorizations(tokenId);
        
        // Transfer to blackhole address instead of burning
        // This is simpler and avoids any potential _burn() issues
        _transfer(tokenOwner, blackhole, tokenId);
    }
    
    // ================================
    // INTERNAL HELPER FUNCTIONS
    // ================================
    
    /**
     * @dev Clear all authorizations for a token in O(1) time
     * @param tokenId Token ID to clear authorizations for
     * 
     * This increments the authorization epoch, invalidating all previous authorizations.
     * No loops, no per-user writes - just one storage operation!
     */
    function _clearAuthorizations(uint256 tokenId) internal {
        uint64 currentEpoch = authEpoch[tokenId];
        if (currentEpoch == 0) {
            authEpoch[tokenId] = 1; // Initialize if needed
        } else {
            authEpoch[tokenId] = currentEpoch + 1; // Bump epoch
        }
        emit AuthorizationsCleared(tokenId, authEpoch[tokenId]);
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
     * @dev Override _update to clear authorizations on regular transfers
     */
    function _update(address to, uint256 tokenId, address auth) internal virtual override returns (address) {
        address from = _ownerOf(tokenId);
        
        // Clear authorizations on transfer (but not on mint or burn) - O(1)!
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
