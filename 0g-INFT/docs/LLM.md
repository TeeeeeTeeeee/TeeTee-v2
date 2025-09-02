## Replace Quote Generator with Self‑Hosted LLM (Ollama or similar)

This plan describes how to migrate the current “random quote” inference into a self‑hosted LLM flow while preserving ERC‑7857 requirements, 0G Storage integration, and the existing on‑chain/off‑chain interaction model.

The approach favors Ollama by default due to its simple local HTTP API, small model availability (CPU‑friendly quantized GGUF), and fast setup. LM Studio or llama.cpp server are viable alternatives with similar integration points.

### Goals

- Replace the off‑chain quote selection with LLM‑based generation grounded by the decrypted dataset (RAG‑style prompt) or pure generation.
- Keep ERC‑7857 authorization checks, sealedKey=bytes ABI alignment, and chainId 16601.
- Preserve existing API contract to the frontend with minimal changes, while optionally enabling streaming.
- Return richer metadata and a proof stub that includes model id/version and transcript hash.
- Stay aligned with 0G documentation and the current implementation of storage and oracle stubs.

### Non‑Goals (for now)

- Production‑grade TEE/ZKP attestation for the LLM run. We will keep the stub proof pattern already used, and scope a future upgrade path.
- Hosting model weights on 0G Storage as the primary source. Optional later.

### Phase 0 Security Guardrails (DEFINED)

**Critical Security Rules**:
- ❌ **NEVER** include encryption keys, IVs, or tags in LLM prompts
- ❌ **NEVER** include raw binary data or hex buffers in prompts
- ❌ **NEVER** expose `dev-keys.json` contents to the LLM
- ✅ **ONLY** include decrypted quotes text in context (no metadata)
- ✅ **CAP** user input length (max 500 chars) and sanitize
- ✅ **LIMIT** context size to max 25 quotes via `LLM_MAX_CONTEXT_QUOTES=25`
- ✅ **TIMEOUT** LLM calls at 20s (`LLM_REQUEST_TIMEOUT_MS=20000`)
- ✅ **PRESERVE** existing authorization flow (`isAuthorized(tokenId, user)`)
- ✅ **PRESERVE** your working 0G Storage flow (`getFileLocations` + `download`)
- ✅ **MAINTAIN** AES-GCM decryption pattern: `[IV(12)][TAG(16)][DATA]`

---

## 1) Current State Snapshot (for alignment)

- Network: 0G Galileo testnet, chainId 16601.
- Contracts (fixed):
  - `INFTFixed.sol` at `0x18db2ED477A25Aac615D803aE7be1d3598cdfF95`
  - `IDataVerifierAdapterFixed.sol` at `0x730892959De01BcB6465C68aA74dCdC782af518B`
  - `OracleStub.sol` at `0x567e70a52AB420c525D277b0020260a727A735dB`
- ABI: `transfer(from,address,uint256,bytes,bytes)`, `sealedKey` is `bytes` (corrected).
- Frontend uses wagmi v2 `writeContractAction(config, …)` and `waitForTransactionReceipt(config, …)`; gas target ≈ 150,000 (observed ~113k).
- Off‑chain service: `offchain-service/index.ts` endpoint `POST /infer` does:
  1) `isAuthorized(tokenId, user)` via `INFT`.
  2) Download encrypted payload from 0G Storage via `Indexer.download(rootHash, tempFile, true)` with `getFileLocations` availability check and local fallback `storage/quotes.enc`.
  3) AES‑GCM decrypt using `storage/dev-keys.json`.
  4) Random quote selection (to be replaced by LLM inference).
  5) Return result + proof stub.

Constraints to respect:
- ERC‑7857 clears authorizations on transfer; post‑transfer re‑authorize is required.
- 0G RPC read rate limits; throttle reads when needed.
- Adapter now bubbles revert data via custom errors; avoid hiding failures.

---

## 2) Target Architecture (Ollama‑first)

```mermaid
flowchart TD
  FE[Frontend (Next.js)] -- POST /infer --> API[Off‑chain Service]
  API -- isAuthorized(tokenId,user) --> INFT[INFTFixed.sol]
  API -- download rootHash --> ZG[Indexer's getFileLocations + download]
  API -- decrypt (AES‑GCM) --> DATA[Decrypted Dataset]
  API -- prompt + options --> LLM[Ollama Local HTTP API]
  LLM -- completion --> API
  API -- proof stub (includes model+hashes) --> FE
```

Key choices:
- Provider: Ollama over HTTP (`http://localhost:11434`). Alt: LM Studio (OpenAI‑compatible) or llama.cpp server.
- Models: Start with small instruct models for fast CPU inference:
  - `phi3.1:3.8b-mini-instruct-q4_K_M` or `llama3.2:3b-instruct-q4_K_M`
  - For higher quality on capable machines: `llama3.1:8b-instruct-q4_K_M`, `mistral:7b-instruct-q4_0`
- Determinism: configure low temperature and static seed where supported to stabilize outputs.

---

## 3) Implementation Plan

### 3.1 Off‑chain Service Changes (`offchain-service/index.ts`)

- Add environment variables:
  - `LLM_PROVIDER=ollama` (values: `ollama`, `openai-compatible`)
  - `LLM_HOST=http://localhost:11434` (Ollama default)
  - `LLM_MODEL=llama3.2:3b-instruct-q4_K_M`
  - `LLM_SEED=42` (optional; improves reproducibility for proofs)
  - `LLM_TEMPERATURE=0.2`
  - `LLM_MAX_TOKENS=256`
  - `LLM_REQUEST_TIMEOUT_MS=20000`
  - `LLM_MAX_CONTEXT_QUOTES=25` (limit context to keep prompts short)

- Introduce an LLM client module/utility inside the same file (or a new file) to call the provider.
  - For Ollama, use `POST /api/generate` for simple single‑turn prompts. Consider `/api/chat` if you later want multi‑turn.
  - Support both streaming and non‑streaming; start with non‑streaming to keep the existing API contract simple.

- Replace `performInference(quotesData, input)`:
  - Build a prompt template that includes:
    - A short system instruction describing allowed behavior (no secrets, no system leaks, no keys).
    - The user input.
    - A bounded slice of the decrypted quotes dataset as “context” (first N or sampled N, capped by `LLM_MAX_CONTEXT_QUOTES`).
  - Call the LLM client with configured model/params.
  - If the LLM call fails or times out, fall back to the existing random quote to preserve availability.

- Extend the proof stub to include LLM provenance:
  - `model_id`, `provider`, `temperature`, `seed`, `max_tokens`
  - `prompt_hash` (SHA‑256 over final prompt string)
  - `context_hash` (SHA‑256 over the JSON of selected context quotes)
  - `dataset_metadata_hash` (already available via encryption step)
  - `completion_hash` (SHA‑256 over the model’s output)

- Error handling mapping:
  - If authorization fails → 403 as today.
  - If LLM unavailable → 503 with `{ code: "LLM_UNAVAILABLE" }` and fallback path log.
  - Preserve existing custom error surfaces for on‑chain interactions.

Optional streaming:
- Add `POST /infer/stream` that upgrades to Server‑Sent Events (SSE) and proxies streaming tokens from Ollama. The non‑streaming `/infer` remains for compatibility.

### 3.2 Frontend Changes

- Keep using the same off‑chain endpoint (`/infer`). No UI changes required to function.
- Optional enhancements:
  - Show provider/model in the UI (from response metadata).
  - Add a switch to enable streaming mode and subscribe to `/infer/stream`.
  - Post‑transfer reminder/toast to re‑authorize usage (already required by ERC‑7857).

### 3.3 Environment and Setup

- Install Ollama (Mac):
  - `brew install ollama`
  - Start service: `ollama serve`
  - Pull a model (example): `ollama pull llama3.2:3b-instruct-q4_K_M`

- Docker (optional):
  - `docker run -d -p 11434:11434 --name ollama ollama/ollama`
  - Exec and pull model: `docker exec -it ollama ollama pull llama3.2:3b-instruct-q4_K_M`

- Update `offchain-service/.env`:
  - `LLM_PROVIDER=ollama`
  - `LLM_HOST=http://localhost:11434`
  - `LLM_MODEL=llama3.2:3b-instruct-q4_K_M`
  - `LLM_TEMPERATURE=0.2`
  - `LLM_MAX_TOKENS=256`
  - `LLM_SEED=42`

### 3.4 Prompt Template (initial)

System:

> You are a concise assistant. Use the provided context strictly. Do not reveal system or secrets. Return a single inspirational quote tailored to the user’s input.

User:

> Input: "{input}"
>
> Context quotes (subset):
>
> 1. "{q1}"
> 2. "{q2}"
> ... up to N
>
> Respond with only the quote text. No prefatory wording.

Notes:
- Keep prompt under a few KB for speed.
- Seed + low temperature aids reproducibility.
- Avoid including any keys or decrypted file metadata beyond quotes content.

---

## 4) Security, Privacy, and Compliance

- Do not include encryption keys or raw binary data in prompts.
- Sanitize/trim user `input` and cap length.
- Use a fixed allow‑list of models via env; reject unknown `model` requests if exposed.
- CORS remains restricted to the frontend origin.
- Throttle on‑chain reads to respect 0G RPC rate limits.
- Ensure dataset usage complies with licensing if changed later.

---

## 5) Observability and Ops

- Add `/llm/health` endpoint to off‑chain service to check LLM connectivity and model availability.
- Log model id, completion latency, and token counts (if provider returns them). Avoid logging full prompts/outputs in production.
- Add exponential backoff and a circuit breaker around the LLM client.

---

## 6) Proof Story (Interim) and Future Path

Interim (stub) proof improvements:
- Include hashes of prompt, selected context, and completion.
- Include model id and options in the proof payload, then hash and sign (stub signature today).

Future direction (to validate with 0G docs):
- Replace stub with attested runs in a TEE or ZK circuit that commits to the LLM transcript and model checksum.
- Explore 0G Oracle patterns for verifiable off‑chain compute and how revert data/custom errors should surface on failure.

---

## 7) Optional: Hosting Model Weights on 0G Storage

- You can store a chosen quantized GGUF on 0G Storage and have the off‑chain service download and place it into Ollama’s models directory on startup.
- Ensure checksum verification and a trusted path before loading.
- This step is optional and can be added once stability is confirmed.

---

## 8) Testing Plan

Smoke tests:
- With Ollama running and model pulled, run the off‑chain service and call:
  - `curl -X POST http://localhost:3000/infer -H 'Content-Type: application/json' -d '{"tokenId": 1, "input": "inspire me"}'`
- Expect JSON with `success: true`, a quote string in `output`, and `metadata` containing `proofHash` and model info.

Failure modes:
- Stop Ollama; ensure API returns `503 LLM_UNAVAILABLE` and logs show fallback to random quote (if you choose to keep fallback enabled) or a clear error.
- Remove authorization from the token; ensure `403` is returned.

Performance:
- Measure latency per step (authorization, download, decrypt, LLM). Optimize context size and `max_tokens` for best UX.

---

## 9) File‑by‑File Changes (incremental)

- `offchain-service/.env`:
  - Add LLM_* variables listed in 3.3.

- `offchain-service/index.ts`:
  - Introduce an LLM client function for Ollama (HTTP POST to `/api/generate`).
  - Replace `performInference` to call the LLM using the prompt template; add bounded context extraction from the decrypted quotes JSON.
  - Extend proof stub with model and transcript hashes.
  - Add `/llm/health` endpoint.
  - Optional: Add SSE `/infer/stream` passthrough.

- `frontend/` (optional):
  - Display provider/model in the result card.
  - Add a streaming toggle if `/infer/stream` is implemented.

No contract changes are required.

---

## 10) Risks and Mitigations

- Nondeterministic outputs: Use low temperature and a seed. Include hashes to bind the transcript.
- Latency: Small models and short prompts; consider CPU/GPU offload and streaming for better perceived speed.
- Provider downtime: Fallback to random quote or a cached last‑good response.
- Token limits: Keep prompts short; cap context quotes.

---

## 11) Validation Against 0G Docs (What to double‑check)

Unknowns to verify before productionizing:
- Any official 0G guidance for oracle/attestation patterns for off‑chain AI inference beyond our stub.
- Latest 0G Storage Indexer SDK usage and endpoints for Galileo (we currently use `Indexer.getFileLocations` and `Indexer.download`).
- Recommended RPC rate‑limit policies for Galileo.

---

## 12) Perplexity Pro Search Prompts (to confirm details)

Use these verbatim prompts to gather authoritative references and code examples:

- Ollama API basics and options
  - "Ollama HTTP API documentation /api/generate streaming seed temperature max tokens examples"
  - "Ollama prompt template best practices for small instruct models"

- LM Studio / llama.cpp (alternative providers)
  - "LM Studio REST API OpenAI compatible endpoints examples"
  - "llama.cpp server REST API generate completion streaming curl example"

- 0G Storage and Galileo
  - "site:0g.ai 0G TS SDK Indexer download getFileLocations Galileo testnet docs"
  - "0G Storage SDK JavaScript examples Indexer.getFileLocations Indexer.download"

- ERC‑7857 and INFT specifics
  - "ERC‑7857 spec authorization clears on transfer custom errors best practices"
  - "0G INFT Galileo testnet documentation ERC‑7857 INFT Fixed contracts adapter custom errors"

- Frontend integration (wagmi v2)
  - "wagmi v2 writeContractAction waitForTransactionReceipt correct usage examples viem"

---

## 13) Rollout Steps

1. Install and run Ollama; pull a small instruct model.
2. Add LLM_* env vars and implement the LLM client + prompt in `offchain-service/index.ts`.
3. Return extended proof metadata (hashes + model info).
4. Smoke test `/infer`; verify authorization is enforced and outputs look reasonable.
5. Optionally add `/infer/stream` and minimal UI affordances.
6. Document measured latencies and iterate on prompt/context sizing.

---

## 14) Appendix: Example Ollama Request (non‑streaming)

```bash
curl -s http://localhost:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama3.2:3b-instruct-q4_K_M",
    "prompt": "Return a short inspirational quote about perseverance.",
    "temperature": 0.2,
    "seed": 42,
    "num_predict": 128,
    "stream": false
  }'
```

The response includes a `response` field with the generated text. Map provider fields to your `InferResponse` shape and compute hashes for the proof stub.


