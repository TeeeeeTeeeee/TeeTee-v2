
# INFT (Intelligent NFTs) — Deep Technical Walkthrough (GitHub‑compatible)

This version fixes Mermaid syntax so diagrams render on GitHub.

---

## 1) System Architecture (Flowchart)

```mermaid
flowchart LR
  subgraph OnChain["0G Chain - EVM"]
    C["ERC-7857 Contract"]
  end

  subgraph OffChain["Off-chain - Oracles"]
    O1["TEE Oracle - Re-encrypt + Attest"]
    O2["ZKP Prover - Re-encrypt Proof"]
  end

  subgraph Storage["0G Storage"]
    M["Encrypted AI Metadata - weights, config, memory"]
  end

  U1["Current Owner - PubKey A"]
  U2["New Owner - PubKey B"]
  COMP["0G Compute - verifiable inference"]
  DA["0G Data Availability"]

  U1 -->|owns| C
  C -->|URI + hash| M
  C -->|verify proofs| O1
  C -->|verify proofs| O2
  C -->|authorize usage| COMP
  M -->|availability| DA
  U2 -->|provides pubkey| C

  O1 -. sealed key & attestation .-> C
  O2 -. zk proof .-> C
```

---

## 2) Lifecycle: Minting

```mermaid
sequenceDiagram
  autonumber
  participant Dev as Dev/Creator
  participant Enc as Encryptor
  participant Store as 0G Storage
  participant C as ERC-7857

  Note over Dev: "Train or assemble the AI agent"
  Dev->>Enc: Produce payload (weights, config, state)
  Enc->>Enc: Encrypt with AES-GCM
  Enc->>Enc: Seal key to Owner pubkey
  Enc->>Store: Upload encrypted blob
  Store-->>Dev: Return encryptedURI + hash
  Dev->>C: mint(tokenId, encryptedURI, hash, owner)
  C-->>Dev: Token minted (INFT)
```

---

## 3) Lifecycle: Authorized Usage (AI-as-a-Service without transfer)

```mermaid
sequenceDiagram
  autonumber
  participant Owner as Owner
  participant C as ERC-7857
  participant Exec as Executor (dApp/Service)
  participant Comp as 0G Compute

  Owner->>C: authorizeUsage(tokenId, Exec, permissions)
  Exec->>Comp: runInference(tokenId, input, proofMode)
  Comp-->>Exec: Output plus verification proof
```

---

## 4) Lifecycle: Transfer (TEE path)

```mermaid
sequenceDiagram
  autonumber
  participant A as Seller (PubKey A)
  participant B as Buyer (PubKey B)
  participant TEE as TEE Oracle
  participant C as ERC-7857
  participant S as 0G Storage

  A->>TEE: Provide encryptedURI_A + sealedKey_A + B pubkey
  TEE->>TEE: Re-encrypt payload
  TEE->>TEE: Seal new key to B
  TEE-->>A: sealedKey_B + attestation proof + new encryptedURI
  A->>C: transfer(tokenId, sealedKey_B, proof)
  C->>C: Verify attestation + update URI/hash
  C-->>B: Ownership changed
  B->>S: Fetch encryptedURI + decrypt using sealedKey_B
```

---

## 5) Lifecycle: Transfer (ZKP path, conceptual)

```mermaid
sequenceDiagram
  autonumber
  participant A as Seller
  participant B as Buyer
  participant Prover as ZKP Prover
  participant C as ERC-7857

  A->>Prover: Provide old ciphertext + B pubkey
  Prover->>Prover: Re-encrypt payload
  Prover->>Prover: Generate zk proof of correctness
  Prover-->>A: Return re-encrypted material + zk proof
  A->>C: transfer(tokenId, sealedKey_B, zkProof)
  C->>C: Verify zk proof + update state
  C-->>B: Ownership changed
```

---

## 6) Data Model (Concise)

- **On-chain (ERC-7857):** `encryptedURI`, `metadataHash`, usage authorizations.  
- **Off-chain (0G Storage):** encrypted bundle `{modelWeights, tokenizer, config, memory, adapters, provenance, version}`.  
- **Crypto:** AES‑GCM for payload; asymmetric key wrapping to the owner’s public key; oracle proof (TEE attestation or ZK proof) verified by the contract.

---

## 7) Minimal Integration Steps

### Minting
1. Serialize agent artifacts.  
2. Encrypt payload; wrap symmetric key to owner public key.  
3. Upload encrypted blob to 0G Storage → `encryptedURI`, `metadataHash`.  
4. Mint ERC‑7857 with URI + hash.  

### Authorized Usage
1. `authorizeUsage(tokenId, executor, permissions)`.  
2. Executor calls 0G Compute with `tokenId` and inputs.  
3. Receive outputs + proof.  

### Transfer
1. Recipient shares public key.  
2. TEE or ZKP service re‑encrypts payload for recipient and returns proof.  
3. Contract verifies proof and updates state.  
4. Recipient decrypts with sealed key.  

---

## 8) On-Chain vs Off-Chain Responsibilities

### On-Chain (0G Chain, Solidity)
- Token ownership (ERC-721 base).  
- ERC-7857 extensions: `transfer`, `clone`, `authorizeUsage`.  
- Stores lightweight references: `encryptedURI` and `metadataHash`.  
- Verifies proofs (TEE attestations or ZKPs).  
- Updates state on valid transfer or authorization.  

```solidity
contract ERC7857 is ERC721 {
    mapping(uint256 => string) public encryptedURI;
    mapping(uint256 => bytes32) public metadataHash;

    function transfer(
        address from,
        address to,
        uint256 tokenId,
        bytes calldata sealedKeyForTo,
        bytes calldata proof
    ) external {
        require(verifyProof(proof, metadataHash[tokenId]), "Invalid proof");
        _transfer(from, to, tokenId);
    }
}
```

### Off-Chain (Services + Storage + Compute)
- **0G Storage:** hosts encrypted AI bundles permanently.  
- **TEE/ZKP Oracle:** performs re-encryption, produces proof of correctness.  
- **0G Compute:** executes inference jobs when authorized.  

```typescript
function reencrypt(oldCipher, oldKey, newPubKey) {
    // inside TEE or as ZKP circuit
    decrypted = decrypt(oldCipher, oldKey);
    newCipher = encrypt(decrypted, newPubKey);

    if (TEE) {
        return { sealedKey: newCipher, proof: enclaveAttestation() };
    } else {
        return { sealedKey: newCipher, proof: generateZKProof(oldCipher, newCipher) };
    }
}
```

**Rule of Thumb:**  
- On-chain = ownership + references + proof checks.  
- Off-chain = heavy lifting (storage, encryption, inference, proof generation).  
