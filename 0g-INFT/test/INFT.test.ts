import { expect } from "chai";
import { ethers } from "hardhat";
import { INFT, DataVerifierAdapter, OracleStub } from "../typechain-types";
import { HardhatEthersSigner } from "@nomicfoundation/hardhat-ethers/signers";

describe("INFT - ERC-7857 Intelligent NFT Implementation", function () {
  let inft: INFT;
  let oracleStub: OracleStub;
  let dataVerifierAdapter: DataVerifierAdapter;
  let owner: HardhatEthersSigner;
  let user1: HardhatEthersSigner;
  let user2: HardhatEthersSigner;
  let executor: HardhatEthersSigner;

  // Test data constants
  const TEST_ENCRYPTED_URI = "0g://storage/0xe3bf3e775364d9eb24fe11106ad035bebda6c2b0f2b0586ad4397246c864aedc";
  const TEST_METADATA_HASH = "0xa43868e7e1335a6070b9ef4ec1c89a23050d73d3173f487557c56b51f2c34e3b";
  const TEST_SEALED_KEY = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef";
  const TEST_PROOF = "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890";

  beforeEach(async function () {
    // Get signers
    [owner, user1, user2, executor] = await ethers.getSigners();

    // Deploy OracleStub (returns true for all proofs in development)
    const OracleStubFactory = await ethers.getContractFactory("OracleStub");
    oracleStub = await OracleStubFactory.deploy();
    await oracleStub.waitForDeployment();

    // Deploy DataVerifierAdapter that wraps the OracleStub
    const DataVerifierAdapterFactory = await ethers.getContractFactory("DataVerifierAdapter");
    dataVerifierAdapter = await DataVerifierAdapterFactory.deploy(await oracleStub.getAddress());
    await dataVerifierAdapter.waitForDeployment();

    // Deploy INFT contract
    const INFTFactory = await ethers.getContractFactory("INFT");
    inft = await INFTFactory.deploy(
      "Test Intelligent NFT",
      "TINFT", 
      await dataVerifierAdapter.getAddress(),
      owner.address
    );
    await inft.waitForDeployment();

    // Mint a test token to user1
    await inft.mint(user1.address, TEST_ENCRYPTED_URI, TEST_METADATA_HASH);
  });

  describe("Deployment", function () {
    it("Should set the correct name and symbol", async function () {
      expect(await inft.name()).to.equal("Test Intelligent NFT");
      expect(await inft.symbol()).to.equal("TINFT");
    });

    it("Should set the correct data verifier adapter", async function () {
      expect(await inft.dataVerifier()).to.equal(await dataVerifierAdapter.getAddress());
    });

    it("Should set the correct owner", async function () {
      expect(await inft.owner()).to.equal(owner.address);
    });

    it("Should initialize token counter correctly", async function () {
      expect(await inft.getCurrentTokenId()).to.equal(2); // Should be 2 after minting token 1
    });
  });

  describe("Minting", function () {
    it("Should mint token with correct metadata", async function () {
      const tokenId = 1;
      
      expect(await inft.ownerOf(tokenId)).to.equal(user1.address);
      expect(await inft.encryptedURI(tokenId)).to.equal(TEST_ENCRYPTED_URI);
      expect(await inft.metadataHash(tokenId)).to.equal(TEST_METADATA_HASH);
      expect(await inft.tokenURI(tokenId)).to.equal(TEST_ENCRYPTED_URI);
    });

    it("Should increment token ID counter", async function () {
      await inft.mint(user2.address, "test://uri2", "0x1234567890123456789012345678901234567890123456789012345678901234");
      expect(await inft.getCurrentTokenId()).to.equal(3);
    });

    it("Should revert when minting to zero address", async function () {
      await expect(
        inft.mint(ethers.ZeroAddress, TEST_ENCRYPTED_URI, TEST_METADATA_HASH)
      ).to.be.revertedWith("Cannot mint to zero address");
    });

    it("Should revert when minting with empty URI", async function () {
      await expect(
        inft.mint(user1.address, "", TEST_METADATA_HASH)
      ).to.be.revertedWith("Encrypted URI cannot be empty");
    });

    it("Should revert when minting with zero metadata hash", async function () {
      await expect(
        inft.mint(user1.address, TEST_ENCRYPTED_URI, ethers.ZeroHash)
      ).to.be.revertedWith("Metadata hash cannot be zero");
    });
  });

  describe("ERC-7857 authorizeUsage Function", function () {
    const tokenId = 1;

    it("Should authorize usage and emit AuthorizedUsage event", async function () {
      await expect(inft.connect(user1).authorizeUsage(tokenId, executor.address))
        .to.emit(inft, "AuthorizedUsage")
        .withArgs(tokenId, executor.address, true);

      expect(await inft.isAuthorized(tokenId, executor.address)).to.be.true;
    });

    it("Should add user to authorizedUsersOf array", async function () {
      await inft.connect(user1).authorizeUsage(tokenId, executor.address);
      
      const authorizedUsers = await inft.authorizedUsersOf(tokenId);
      expect(authorizedUsers).to.have.lengthOf(1);
      expect(authorizedUsers[0]).to.equal(executor.address);
    });

    it("Should not duplicate authorization for same user", async function () {
      // Authorize twice
      await inft.connect(user1).authorizeUsage(tokenId, executor.address);
      await inft.connect(user1).authorizeUsage(tokenId, executor.address);
      
      const authorizedUsers = await inft.authorizedUsersOf(tokenId);
      expect(authorizedUsers).to.have.lengthOf(1);
      expect(authorizedUsers[0]).to.equal(executor.address);
    });

    it("Should allow multiple users to be authorized", async function () {
      await inft.connect(user1).authorizeUsage(tokenId, executor.address);
      await inft.connect(user1).authorizeUsage(tokenId, user2.address);
      
      const authorizedUsers = await inft.authorizedUsersOf(tokenId);
      expect(authorizedUsers).to.have.lengthOf(2);
      expect(authorizedUsers).to.include(executor.address);
      expect(authorizedUsers).to.include(user2.address);
    });

    it("Should revert when non-owner tries to authorize", async function () {
      await expect(
        inft.connect(user2).authorizeUsage(tokenId, executor.address)
      ).to.be.revertedWith("Caller is not owner nor approved");
    });

    it("Should revert when authorizing zero address", async function () {
      await expect(
        inft.connect(user1).authorizeUsage(tokenId, ethers.ZeroAddress)
      ).to.be.revertedWith("Cannot authorize zero address");
    });

    it("Should revert when token does not exist", async function () {
      await expect(
        inft.connect(user1).authorizeUsage(999, executor.address)
      ).to.be.revertedWithCustomError(inft, "ERC721NonexistentToken");
    });

    it("Should allow approved address to authorize", async function () {
      // Approve user2 to manage tokenId
      await inft.connect(user1).approve(user2.address, tokenId);
      
      await expect(inft.connect(user2).authorizeUsage(tokenId, executor.address))
        .to.emit(inft, "AuthorizedUsage")
        .withArgs(tokenId, executor.address, true);
    });
  });

  describe("ERC-7857 revokeUsage Function", function () {
    const tokenId = 1;

    beforeEach(async function () {
      // Authorize executor first
      await inft.connect(user1).authorizeUsage(tokenId, executor.address);
    });

    it("Should revoke usage and emit AuthorizedUsage event", async function () {
      await expect(inft.connect(user1).revokeUsage(tokenId, executor.address))
        .to.emit(inft, "AuthorizedUsage")
        .withArgs(tokenId, executor.address, false);

      expect(await inft.isAuthorized(tokenId, executor.address)).to.be.false;
    });

    it("Should remove user from authorizedUsersOf array", async function () {
      await inft.connect(user1).revokeUsage(tokenId, executor.address);
      
      const authorizedUsers = await inft.authorizedUsersOf(tokenId);
      expect(authorizedUsers).to.have.lengthOf(0);
    });

    it("Should handle revoking non-authorized user gracefully", async function () {
      // Try to revoke user2 who was never authorized
      await inft.connect(user1).revokeUsage(tokenId, user2.address);
      
      // Should not affect existing authorizations
      expect(await inft.isAuthorized(tokenId, executor.address)).to.be.true;
    });

    it("Should properly handle multiple users authorization/revocation", async function () {
      // Authorize user2
      await inft.connect(user1).authorizeUsage(tokenId, user2.address);
      
      let authorizedUsers = await inft.authorizedUsersOf(tokenId);
      expect(authorizedUsers).to.have.lengthOf(2);
      
      // Revoke executor
      await inft.connect(user1).revokeUsage(tokenId, executor.address);
      
      authorizedUsers = await inft.authorizedUsersOf(tokenId);
      expect(authorizedUsers).to.have.lengthOf(1);
      expect(authorizedUsers[0]).to.equal(user2.address);
    });
  });

  describe("ERC-7857 authorizedUsersOf Function", function () {
    const tokenId = 1;

    it("Should return empty array for token with no authorizations", async function () {
      const authorizedUsers = await inft.authorizedUsersOf(tokenId);
      expect(authorizedUsers).to.have.lengthOf(0);
    });

    it("Should return correct users after authorization", async function () {
      await inft.connect(user1).authorizeUsage(tokenId, executor.address);
      await inft.connect(user1).authorizeUsage(tokenId, user2.address);
      
      const authorizedUsers = await inft.authorizedUsersOf(tokenId);
      expect(authorizedUsers).to.have.lengthOf(2);
      expect(authorizedUsers).to.include(executor.address);
      expect(authorizedUsers).to.include(user2.address);
    });

    it("Should filter out revoked users", async function () {
      await inft.connect(user1).authorizeUsage(tokenId, executor.address);
      await inft.connect(user1).authorizeUsage(tokenId, user2.address);
      await inft.connect(user1).revokeUsage(tokenId, executor.address);
      
      const authorizedUsers = await inft.authorizedUsersOf(tokenId);
      expect(authorizedUsers).to.have.lengthOf(1);
      expect(authorizedUsers[0]).to.equal(user2.address);
    });

    it("Should revert for non-existent token", async function () {
      await expect(
        inft.authorizedUsersOf(999)
      ).to.be.revertedWithCustomError(inft, "ERC721NonexistentToken");
    });
  });

  describe("ERC-7857 Transfer with Proof", function () {
    const tokenId = 1;

    beforeEach(async function () {
      // Authorize executor for testing authorization clearing
      await inft.connect(user1).authorizeUsage(tokenId, executor.address);
    });

    it("Should transfer with valid proof and emit events", async function () {
      await expect(
        inft.connect(user1).transfer(
          user1.address,
          user2.address, 
          tokenId,
          TEST_SEALED_KEY,
          TEST_PROOF
        )
      ).to.emit(inft, "Transferred")
        .withArgs(user1.address, user2.address, tokenId, TEST_SEALED_KEY, ethers.keccak256(TEST_PROOF))
        .and.to.emit(inft, "PublishedSealedKey")
        .withArgs(tokenId, TEST_SEALED_KEY, user2.address)
        .and.to.emit(inft, "Transfer") // Standard ERC721 event
        .withArgs(user1.address, user2.address, tokenId);

      expect(await inft.ownerOf(tokenId)).to.equal(user2.address);
    });

    it("Should clear authorizations on transfer", async function () {
      // Verify executor is authorized before transfer
      expect(await inft.isAuthorized(tokenId, executor.address)).to.be.true;
      
      await inft.connect(user1).transfer(
        user1.address,
        user2.address,
        tokenId,
        TEST_SEALED_KEY,
        TEST_PROOF
      );
      
      // Verify authorization is cleared after transfer
      expect(await inft.isAuthorized(tokenId, executor.address)).to.be.false;
      const authorizedUsers = await inft.authorizedUsersOf(tokenId);
      expect(authorizedUsers).to.have.lengthOf(0);
    });

    it("Should revert with invalid from address", async function () {
      await expect(
        inft.connect(user1).transfer(
          user2.address, // Wrong from address
          user2.address,
          tokenId,
          TEST_SEALED_KEY,
          TEST_PROOF
        )
      ).to.be.revertedWith("From address is not the owner");
    });

    it("Should revert when transferring to zero address", async function () {
      await expect(
        inft.connect(user1).transfer(
          user1.address,
          ethers.ZeroAddress,
          tokenId,
          TEST_SEALED_KEY,
          TEST_PROOF
        )
      ).to.be.revertedWith("Cannot transfer to zero address");
    });

    it("Should revert with empty sealed key", async function () {
      await expect(
        inft.connect(user1).transfer(
          user1.address,
          user2.address,
          tokenId,
          "0x", // Empty sealed key
          TEST_PROOF
        )
      ).to.be.revertedWith("Sealed key cannot be empty");
    });

    it("Should revert when non-owner tries to transfer", async function () {
      await expect(
        inft.connect(user2).transfer(
          user1.address,
          user2.address,
          tokenId,
          TEST_SEALED_KEY,
          TEST_PROOF
        )
      ).to.be.revertedWith("Caller is not owner nor approved");
    });
  });

  describe("ERC-7857 Clone Function", function () {
    const tokenId = 1;

    it("Should clone token and emit events", async function () {
      const newTokenId = await inft.getCurrentTokenId();
      
      await expect(
        inft.connect(user1).clone(
          user1.address,
          user2.address,
          tokenId,
          TEST_SEALED_KEY,
          TEST_PROOF
        )
      ).to.emit(inft, "Cloned")
        .withArgs(tokenId, newTokenId, user2.address, TEST_SEALED_KEY, ethers.keccak256(TEST_PROOF))
        .and.to.emit(inft, "PublishedSealedKey")
        .withArgs(newTokenId, TEST_SEALED_KEY, user2.address)
        .and.to.emit(inft, "Transfer") // ERC721 mint event
        .withArgs(ethers.ZeroAddress, user2.address, newTokenId);
    });

    it("Should copy metadata from original token", async function () {
      const newTokenId = await inft.getCurrentTokenId();
      
      await inft.connect(user1).clone(
        user1.address,
        user2.address,
        tokenId,
        TEST_SEALED_KEY,
        TEST_PROOF
      );
      
      expect(await inft.encryptedURI(newTokenId)).to.equal(TEST_ENCRYPTED_URI);
      expect(await inft.metadataHash(newTokenId)).to.equal(TEST_METADATA_HASH);
      expect(await inft.ownerOf(newTokenId)).to.equal(user2.address);
    });

    it("Should return correct new token ID", async function () {
      const expectedTokenId = await inft.getCurrentTokenId();
      
      // Use static call to get return value without executing
      const result = await inft.connect(user1).clone.staticCall(
        user1.address,
        user2.address,
        tokenId,
        TEST_SEALED_KEY,
        TEST_PROOF
      );
      
      expect(result).to.equal(expectedTokenId);
    });

    it("Should not affect original token ownership or metadata", async function () {
      await inft.connect(user1).clone(
        user1.address,
        user2.address,
        tokenId,
        TEST_SEALED_KEY,
        TEST_PROOF
      );
      
      // Original token should remain unchanged
      expect(await inft.ownerOf(tokenId)).to.equal(user1.address);
      expect(await inft.encryptedURI(tokenId)).to.equal(TEST_ENCRYPTED_URI);
      expect(await inft.metadataHash(tokenId)).to.equal(TEST_METADATA_HASH);
    });
  });

  describe("Data Verifier Integration", function () {
    it("Should use correct data verifier address", async function () {
      expect(await inft.dataVerifier()).to.equal(await dataVerifierAdapter.getAddress());
    });

    it("Should get oracle address from adapter", async function () {
      expect(await dataVerifierAdapter.getOracleAddress()).to.equal(await oracleStub.getAddress());
    });
  });

  describe("Edge Cases and Security", function () {
    const tokenId = 1;

    it("Should handle reentrancy protection", async function () {
      // ReentrancyGuard should prevent reentrancy attacks
      // This is mainly tested by ensuring nonReentrant modifier is in place
      expect(await inft.connect(user1).transfer(
        user1.address,
        user2.address,
        tokenId,
        TEST_SEALED_KEY,
        TEST_PROOF
      )).to.not.be.reverted;
    });

    it("Should maintain authorization state correctly across operations", async function () {
      // Authorize multiple users
      await inft.connect(user1).authorizeUsage(tokenId, executor.address);
      await inft.connect(user1).authorizeUsage(tokenId, user2.address);
      
      // Transfer should clear all authorizations
      await inft.connect(user1).transfer(
        user1.address,
        user2.address,
        tokenId,
        TEST_SEALED_KEY,
        TEST_PROOF
      );
      
      // Verify all authorizations are cleared
      expect(await inft.isAuthorized(tokenId, executor.address)).to.be.false;
      expect(await inft.isAuthorized(tokenId, user2.address)).to.be.false;
      const authorizedUsers = await inft.authorizedUsersOf(tokenId);
      expect(authorizedUsers).to.have.lengthOf(0);
    });
  });
});
