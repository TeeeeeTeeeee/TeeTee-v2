# INFT Quote Generator — Phased Plan (Strictly 0G‑Aligned)

This plan is split into clear **phases** you can hand to Cursor. It follows the **0G INFT (ERC‑7857) docs**:

- INFT Overview: https://docs.0g.ai/developer-hub/building-on-0g/inft/inft-overview
- ERC‑7857 Spec: https://docs.0g.ai/developer-hub/building-on-0g/inft/erc7857
- Integration Guide: https://docs.0g.ai/developer-hub/building-on-0g/inft/integration

> Notes:
>
> - We adopt the docs’ contract surface: `transfer(from,to,tokenId, sealedKey, proof)`, `clone(...)`, `authorizeUsage(tokenId, executor, permissions)` and an **oracle-based** proof verification (`IOracle.verifyProof`).
> - Storage uses **0G Storage** (preferred per docs). You can start with IPFS for local dev, then swap to 0G Storage SDK following the Integration guide.
> - Compute/Oracle are pluggable: TEE or ZKP, verified through the on‑chain **oracle** interface.
> - Use **bytes `permissions`** for `authorizeUsage` (not bool).

---

## Phase 0 — Workspace & Network Setup

**Goal:** Have a ready Hardhat TS project and testnet config per 0G docs.

### Tasks

1. Create a Hardhat (TypeScript) project with OpenZeppelin.
2. Add `.env` with `PRIVATE_KEY`, `RPC_URL`, `CHAIN_ID`.
3. Configure **0G Testnet** (choose one per docs):
   - **Newton** Chain ID `16600`
   - **Galileo** Chain ID `16601` (RPC often `https://evmrpc-testnet.0g.ai` — confirm in docs)
4. Add minimal NPM scripts: build, test, deploy.
5. Add a placeholder **OracleStub** contract for development (returns `true` from `verifyProof`), to be swapped with a real oracle endpoint later per Integration guide.

### Deliverables

- `hardhat.config.ts` with 0G testnet network
- `.env.example` + `.env`
- `contracts/OracleStub.sol`

### Success Criteria

- `npx hardhat compile` succeeds
- `npx hardhat run scripts/ping.ts --network 0gTestnet` can fetch chain ID

---

## Phase 1 — Storage Payload & Encryption (0G Storage)

**Goal:** Prepare an **encrypted** payload and store it with 0G Storage per docs.

### Tasks

1. Create `storage/quotes.json` with a small array of strings + version field.
2. Implement `storage/encrypt.ts` to:
   - Generate a random 256‑bit key (AES‑GCM)
   - Encrypt `quotes.json` → outputs `quotes.enc`, `iv`, `tag`
   - Compute `metadataHash = keccak256(quotes.enc)`
3. Upload `quotes.enc` via **0G Storage SDK** and record `encryptedURI`.
   - (Dev fallback: IPFS Kubo; but keep the code path ready for 0G Storage as per Integration guide.)

### Deliverables

- `storage/quotes.json`
- `storage/encrypt.ts` (Node crypto, prints `encryptedURI` + `metadataHash`)
- `storage/README.md` (how to run, env vars for 0G Storage client if needed)

### Success Criteria

- Running `node storage/encrypt.ts` prints a valid `encryptedURI` and `metadataHash`
- The blob is retrievable from the configured storage gateway

---

## Phase 2 — ERC‑7857 Contract (Spec Surface) + Oracle Interface

**Goal:** Implement the on‑chain INFT per **ERC‑7857** and plug the **oracle** verification pattern described by 0G.

### Tasks

1. Create `contracts/interfaces/IOracle.sol` with:
   ```solidity
   interface IOracle {
     function verifyProof(bytes calldata proof) external view returns (bool);
   }
   ```
2. Create `contracts/INFT.sol`:
   - Inherit `ERC721`
   - Store for each `tokenId`:
     - `encryptedURI` (string)
     - `metadataHash` (bytes32)
   - Expose **spec‑aligned** functions:
     - `transfer(address from, address to, uint256 tokenId, bytes calldata sealedKey, bytes calldata proof)`
     - `clone(address from, address to, uint256 tokenId, bytes calldata sealedKey, bytes calldata proof) external returns (uint256 newTokenId)`
     - `authorizeUsage(uint256 tokenId, address executor, bytes calldata permissions)`
   - Route proof checks through `IOracle(oracle).verifyProof(proof)` (do **not** embed zk verifier logic in this contract).
   - On valid `transfer`, update on‑chain state if the proof attests to an update (e.g., metadata pointer/hash/keys as per your oracle semantics), then call `_transfer`.
3. Add events per your needs (e.g., `UsageAuthorized`, `TransferredWithProof`).

> The **exact on‑chain state fields updated** by transfers depends on your oracle semantics (TEE vs ZK). Keep only **references** and **hashes** on‑chain, per docs.

### Deliverables

- `contracts/interfaces/IOracle.sol`
- `contracts/INFT.sol` (ERC‑7857 surface; uses oracle for proof)

### Success Criteria

- Compiles cleanly
- Unit test confirms `authorizeUsage` writes `permissions` bytes and emits an event

---

## Phase 3 — Deployment Scripts

**Goal:** Scripts to deploy the Oracle (stub) and INFT on 0G testnet.

### Tasks

1. `scripts/deployOracle.ts` → deploy `OracleStub` (dev only).
2. `scripts/deployINFT.ts` → deploy `INFT` with constructor param `oracle` set to stub address.
3. Write to a local `deployments/<network>.json` with addresses.

### Deliverables

- `scripts/deployOracle.ts`
- `scripts/deployINFT.ts`
- `deployments/` folder JSON

### Success Criteria

- One‑shot deploy works:  
  `npx hardhat run scripts/deployOracle.ts --network 0gTestnet`  
  `npx hardhat run scripts/deployINFT.ts --network 0gTestnet`

---

## Phase 4 — Mint

**Goal:** Mint an INFT bound to the encrypted storage reference per docs.

### Tasks

1. `scripts/mint.ts`:
   - Inputs: `tokenId`, `to`, `encryptedURI`, `metadataHash`
   - Calls your `mint(...)` helper on the INFT (use an external `mint` function or constructor logic consistent with ERC‑721 patterns; the spec focuses on transfer/clone/authorizeUsage and doesn’t mandate a mint signature).
2. Record the minted token in `deployments/<network>.json`.

### Deliverables

- `scripts/mint.ts`

### Success Criteria

- A token exists with `encryptedURI` + `metadataHash` retrievable via view functions

---

## Phase 5 — Authorized Usage (Spec: bytes permissions)

**Goal:** Enable an executor to run inference without transferring ownership, per ERC‑7857 docs.

### Tasks

1. `scripts/authorize.ts`:
   - Input: `tokenId`, `executor`, `permissions` (bytes; e.g., a compact ABI‑encoded struct)
   - Calls `authorizeUsage(tokenId, executor, permissions)`
2. Add a view helper `getAuthorization(tokenId, executor) → bytes` to your INFT contract if useful for the off‑chain service.

### Deliverables

- `scripts/authorize.ts`

### Success Criteria

- Authorization is written and readable; event emitted

---

## Phase 6 — Off‑Chain Service (Integration Guide Pattern)

**Goal:** Build a minimal service that follows the **Integration** doc shape: it consumes `authorizeUsage`, fetches from **0G Storage**, does inference, and returns a result plus a **proof** (stub at first; later TEE/ZK).

### Tasks

1. `offchain-service/` (Node.js/TS + Express):
   - `/infer` POST `{ tokenId, input }` →
     - Check authorization on chain (read `permissions`).
     - Fetch `encryptedURI` → download blob from 0G Storage.
     - Decrypt with your local demo key (dev only).
     - Return `{ output, proof }` where `proof` is the oracle stub payload.
2. Add a **listener** for `UsageAuthorized` (optional) just to log.

> This service structure mirrors the docs’ pattern: off‑chain compute + an oracle proof verified on‑chain.

### Deliverables

- `offchain-service/index.ts`
- `.env` for RPC + contract address

### Success Criteria

- `curl` to `/infer` returns a random quote and a non‑empty `proof` (from the stub)

---

## Phase 7 — Transfer (TEE path via Oracle)

**Goal:** Implement the **TEE** transfer path as the first real oracle integration (per docs).

### Tasks

1. Extend Oracle interface and your stub to accept `{ sealedKey, attestation }` and return `true` in dev.
2. Update client script `scripts/transfer.ts` to call:
   ```
   transfer(from, to, tokenId, sealedKey, proof)
   ```
   where `proof` encodes the enclave **attestation** (per Integration guide; keep format flexible).
3. Contract verifies via `IOracle(oracle).verifyProof(proof)`. On success, `_transfer`.

### Deliverables

- `contracts/OracleStub.sol` supporting attestation bytes
- `scripts/transfer.ts`

### Success Criteria

- Transfer succeeds through the oracle stub
- State remains consistent and token ownership updates

---

## Phase 8 — Transfer (ZKP path via Oracle) — Optional

**Goal:** Add the **ZKP** re‑encryption route, still via the same `IOracle.verifyProof` interface.

### Tasks

1. Stand up a ZK prover service that produces a **proof artifact** per the oracle’s expected format (follow 0G’s Integration guide; public signals should be **hashes/commitments**, not raw secrets).
2. Oracle verifies the zk proof off‑chain and returns `true` to the INFT contract.
3. Reuse `transfer(...)` with `{ sealedKey, proof }`.

### Deliverables

- ZK prover (service script) and a mocked **oracle** that accepts its proof format

### Success Criteria

- Transfer succeeds with the ZK oracle path (dev/proto)

---

## Phase 9 — Frontend (Optional)

**Goal:** Provide a tiny UI to exercise mint, authorize, infer, and transfer.

### Tasks

1. Build a simple React page:
   - Connect wallet (0G testnet)
   - Buttons: Mint, Authorize, Infer, Transfer
   - Show latest output and tx hashes
2. Use Wagmi/RainbowKit for wallet UX.

### Deliverables

- `frontend/` app (Vite + React)

### Success Criteria

- You can run the full loop from the browser against 0G testnet

---

## Phase 10 — Verification & Handover

**Goal:** Ensure everything matches **0G docs** before iterating further.

### Tasks

- Confirm function signatures and oracle usage match the **ERC‑7857** & **Integration** pages.
- Ensure **only references/hashes** sit on‑chain; no plaintext model data.
- Confirm `permissions` is **bytes**, not bool.
- Confirm testnet RPC, Chain ID, faucet per the **0G Testnet Overview**.

### Final Checklist

- [ ] Storage: 0G Storage SDK used; `encryptedURI` + `metadataHash` set
- [ ] Contract: ERC‑7857‑aligned functions; oracle‑based proof verification
- [ ] Authorize: `bytes permissions` written and read correctly
- [ ] Off‑chain: `/infer` works end‑to‑end
- [ ] Transfer: works with oracle stub (TEE path); ZKP path optional
- [ ] Frontend: optional demo UI verified on 0G testnet

---

## Appendix — File Index

```
contracts/
  INFT.sol                     # ERC-7857 surface; oracle-based verification
  interfaces/IOracle.sol
  OracleStub.sol               # dev-only stub

scripts/
  deployOracle.ts
  deployINFT.ts
  mint.ts
  authorize.ts
  transfer.ts

offchain-service/
  index.ts                     # /infer (reads permissions; fetches/decrypts; returns proof)

storage/
  quotes.json
  encrypt.ts                   # AES-GCM; prints encryptedURI + metadataHash
  README.md

frontend/ (optional)
  ...
```

---

## Cursor Usage

Ask Cursor to implement **phase by phase**. For example:

> **“Implement Phase 2 exactly. Use the ERC‑7857 function names (`transfer`, `clone`, `authorizeUsage`) and route proof checks through `IOracle.verifyProof`. Keep only `encryptedURI` and `metadataHash` on-chain. Do not embed zk code in the INFT contract. Add events and basic views.”**

Then proceed to phases 3 → 4 → …

---

## References (keep these open while coding)

- INFT Overview — https://docs.0g.ai/developer-hub/building-on-0g/inft/inft-overview
- ERC‑7857 Spec — https://docs.0g.ai/developer-hub/building-on-0g/inft/erc7857
- Integration Guide — https://docs.0g.ai/developer-hub/building-on-0g/inft/integration
- Testnet Overview — https://docs.0g.ai/developer-hub/network/testnet-overview
