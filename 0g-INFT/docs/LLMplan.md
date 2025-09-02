## Phased Plan: Replace Quote Generator with Self‑Hosted LLM (Ollama)

This plan outlines concrete phases to migrate the off‑chain "Quote generator" to a local LLM (Ollama by default) while preserving ERC‑7857, 0G Storage integration, and current on‑chain flows on 0G Galileo (16601).

Each phase lists scope, tasks, deliverables, acceptance criteria, risks/mitigations, and validation queries for Perplexity Pro to confirm unknowns against official docs.

---

### Phase 0 — Discovery & Guardrails

Scope: Validate assumptions; define security/perf guardrails.

Tasks:
- Review current codepaths:
  - `offchain-service/index.ts` (authorization, 0G download, AES‑GCM decrypt, quote selection, proof stub)
  - `storage/encrypt.ts` and `storage/dev-keys.json` usage
  - Frontend flow calling `POST /infer` (no change required initially)
- Define guardrails:
  - No keys/IV/tags in prompts
  - Bounded prompt size and user input length caps
  - Rate limiting/throttling for 0G RPC reads
  - LLM timeouts and deterministic settings (seed + low temperature) where possible

Deliverables:
- Agreed constraints documented in `docs/LLM.md` (already added) and referenced by this plan.

Acceptance Criteria:
- We have explicit rules for prompt building, context limits, and failure behavior.

Risks/Mitigations:
- Unknown LLM determinism → use seed + low temperature and add transcript hashes to proof stub.

Perplexity Pro Queries:
- "0G INFT Galileo testnet documentation ERC‑7857 INFT Fixed contracts adapter custom errors"
- "wagmi v2 writeContractAction waitForTransactionReceipt correct usage examples viem"

---

### Phase 1 — Local LLM Setup (Ollama)

Scope: Install and validate a lightweight local LLM.

Tasks:
- Install and run Ollama locally:
  - macOS: `brew install ollama && ollama serve`
  - or Docker: `docker run -d -p 11434:11434 --name ollama ollama/ollama`
- Pull a small instruct model (fast on CPU):
  - `ollama pull llama3.2:3b-instruct-q4_K_M` (or `phi3.1:3.8b-mini-instruct-q4_K_M`)
- Smoke test the API:
  - `curl -s http://localhost:11434/api/generate -H 'Content-Type: application/json' -d '{"model":"llama3.2:3b-instruct-q4_K_M","prompt":"Say hi","stream":false}'`

Deliverables:
- Running Ollama service with at least one model available.

Acceptance Criteria:
- `curl` to `/api/generate` returns a non‑error JSON with a `response` field.

Risks/Mitigations:
- CPU‑only hosts may be slow → use smallest quantized models and low max tokens.

Perplexity Pro Queries:
- "Ollama HTTP API documentation /api/generate streaming seed temperature max tokens examples"

---

### Phase 2 — Off‑Chain LLM Integration (Non‑Streaming)

Scope: Replace random quote selection with an LLM call while preserving the existing `/infer` contract.

Tasks:
- Add env vars in `offchain-service/.env`:
  - `LLM_PROVIDER=ollama`
  - `LLM_HOST=http://localhost:11434`
  - `LLM_MODEL=llama3.2:3b-instruct-q4_K_M`
  - `LLM_TEMPERATURE=0.2`
  - `LLM_MAX_TOKENS=256`
  - `LLM_SEED=42` (if supported) 
  - `LLM_REQUEST_TIMEOUT_MS=20000`
  - `LLM_MAX_CONTEXT_QUOTES=25`
- In `offchain-service/index.ts`:
  - Implement an LLM client function to call Ollama `/api/generate` (non‑streaming) with timeout handling.
  - Build a prompt:
    - System instruction: “concise, safe, return only the quote text.”
    - User `input`
    - A bounded subset of decrypted quotes (N from `LLM_MAX_CONTEXT_QUOTES`)
  - Replace `performInference(...)` to call the LLM client and return the model’s completion.
  - Fallback policy (configurable): on LLM failure, return 503 with `{ code: "LLM_UNAVAILABLE" }` or fallback to the previous random quote path for dev only (controlled via `LLM_DEV_FALLBACK=true`).
  - Extend proof stub to include:
    - `provider`, `model_id`, `temperature`, `seed`, `max_tokens`
    - `prompt_hash`, `context_hash`, `completion_hash`
    - `dataset_metadata_hash` (already produced in Phase 1 encryption)

Deliverables:
- Updated `offchain-service/index.ts` with LLM call and extended proof stub.
- Updated `offchain-service/.env` example.

Acceptance Criteria:
- `POST /infer` returns `success: true` with an LLM‑generated quote when authorized.
- On LLM unavailability, the service responds per the chosen policy (503 or dev fallback).
- Proof stub contains model/option fields and transcript hashes.

Risks/Mitigations:
- Prompt too long → cap context length and sanitize input.
- Latency spikes → keep `max_tokens` low, use small models.

Perplexity Pro Queries:
- "Ollama prompt template best practices for small instruct models"
- "site:0g.ai 0G TS SDK Indexer download getFileLocations Galileo testnet docs"

---

### Phase 3 — Health, Observability, and Security

Scope: Make the service diagnosable; add basic hardening.

Tasks:
- Add `GET /llm/health` to `offchain-service/index.ts` returning `{ provider, model, ok }` and timing info.
- Log: model id, total LLM latency, token estimate (if available), and decision for fallback; avoid logging full prompts/outputs in prod.
- Input validation: length caps and character filtering for `input`.
- Timeouts and circuit breaker around the LLM client to avoid resource exhaustion.
- Basic rate limiting on `/infer` if necessary.

Deliverables:
- New health endpoint and structured logs.

Acceptance Criteria:
- `/llm/health` returns 200 with correct model and provider when LLM is reachable.
- Logs show timing and decision paths for success/failure without leaking secrets.

Risks/Mitigations:
- Excess logging → ensure PII redaction and truncation.

Perplexity Pro Queries:
- "Node.js axios/fetch timeout and abort controller best practices"

---

### Phase 4 — Optional Streaming Endpoint and UI Enhancements

Scope: Add streaming for better UX; surface model info in UI.

Tasks:
- Implement `POST /infer/stream` using Server‑Sent Events (SSE) that proxies Ollama streaming tokens.
- Frontend (optional):
  - Display model/provider from `metadata` in existing result card.
  - Add a toggle for streaming mode; switch to SSE endpoint when enabled.

Deliverables:
- SSE endpoint and optional UI updates.

Acceptance Criteria:
- Streaming endpoint delivers progressive tokens and completes cleanly.
- Non‑streaming path remains the default and stable.

Risks/Mitigations:
- SSE disconnects → implement client reconnect/backoff.

Perplexity Pro Queries:
- "Ollama /api/generate streaming example SSE"
- "Next.js SSE client pattern examples"

---

### Phase 5 — End‑to‑End (E2E) Validation on Galileo

Scope: Validate full flow with on‑chain authorization and 0G Storage.

Tasks:
- Ensure `offchain-service/.env` uses fixed contract addresses and Galileo RPC as already configured.
- Run end‑to‑end tests:
  - Authorized path: user authorized via `authorizeUsage(tokenId, user)` → `/infer` returns LLM quote.
  - Unauthorized path: `/infer` returns 403.
  - 0G Storage unavailable path: verify local fallback loads `storage/quotes.enc` and the flow still works.
- Post‑transfer flow: ensure the UI prompts to re‑authorize as ERC‑7857 clears authorizations.

Deliverables:
- Test logs and sample responses captured.

Acceptance Criteria:
- All three paths (authorized, unauthorized, storage‑fallback) behave as expected.
- Returned `metadata.proofHash` is stable for identical prompts under seeded conditions.

Risks/Mitigations:
- 0G RPC rate limiting → throttle reads and add retry with backoff.

Perplexity Pro Queries:
- "0G Storage SDK JavaScript examples Indexer.getFileLocations Indexer.download"

---

### Phase 6 — Performance Tuning

Scope: Optimize latency and output quality.

Tasks:
- Measure step latencies (auth, download, decrypt, LLM) and set budgets.
- Tune: `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`, context size, and possibly switch to a faster/smaller model.
- Consider shallow streaming for faster perceived response.

Deliverables:
- A short tuning report appended to `docs/LLM.md`.

Acceptance Criteria:
- Median `/infer` latency within target (e.g., < 3.5s depending on storage download speed), with stable quality.

Risks/Mitigations:
- Model swap can change style → keep prompts explicit and test regressions.

---

### Phase 7 — Production Hardening (Optional)

Scope: Operationalize the service.

Tasks:
- Containerize `offchain-service` and Ollama; pin model tags.
- Add process manager (PM2) or systemd for restarts.
- Enforce TLS termination and auth between FE ↔ off‑chain if exposed beyond localhost.
- Add alerts for LLM unavailability and slow responses.

Deliverables:
- Container artifacts and deployment instructions.

Acceptance Criteria:
- Service restarts cleanly; health checks pass; dashboards show basic SLOs.

Risks/Mitigations:
- Model disk size → pre‑warm, cache, or provision storage accordingly.

Perplexity Pro Queries:
- "Best practices for running Ollama in production Docker compose GPU support"

---

### Phase 8 — Future Work: Verifiable Inference

Scope: Replace stub proof with attestable execution.

Tasks (research + spike):
- Survey TEE or ZK approaches for binding model checksum + prompt + completion.
- Explore 0G Oracle patterns for verifiable off‑chain compute.
- Define the on‑chain verification path and failure surfaces (custom errors already supported).

Deliverables:
- RFC documenting options and a prototype timeline.

Acceptance Criteria:
- A chosen path with feasibility notes and next steps.

Perplexity Pro Queries:
- "TEE attestation for LLM inference proof of computation open source"
- "ZK proofs for language model inference state of the art survey"

---

## Implementation Checklist (by file)

- `offchain-service/.env`:
  - Add `LLM_PROVIDER`, `LLM_HOST`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_SEED`, `LLM_REQUEST_TIMEOUT_MS`, `LLM_MAX_CONTEXT_QUOTES`, optional `LLM_DEV_FALLBACK`.

- `offchain-service/index.ts`:
  - Add LLM client for Ollama `/api/generate` (non‑streaming first, optional streaming later).
  - Build prompt with bounded context extracted from decrypted quotes JSON.
  - Replace `performInference` to call LLM; add fallback policy; extend proof stub with hashes and model metadata.
  - Add `GET /llm/health`; optional `POST /infer/stream`.

- `frontend/` (optional):
  - Display provider/model in UI; add streaming toggle if implemented.

No Solidity or ABI changes required.

---

## Rollback Plan

- Toggle `LLM_DEV_FALLBACK=true` to revert to random quote path (dev only) while keeping the LLM code in place.
- Or set `LLM_PROVIDER=disabled` to bypass LLM calls entirely and use the previous behavior.
- Keep a feature flag in the off‑chain service to switch paths without redeploy.

---

## Test Commands

- Health:
```bash
curl -s http://localhost:3000/health | jq
curl -s http://localhost:3000/llm/health | jq
```

- Inference (authorized):
```bash
curl -s -X POST http://localhost:3000/infer \
  -H 'Content-Type: application/json' \
  -d '{"tokenId": 1, "input": "inspire me"}' | jq
```

- Ollama smoke test:
```bash
curl -s http://localhost:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.2:3b-instruct-q4_K_M","prompt":"Say hi","stream":false}' | jq
```

---

## Open Questions (to resolve before prod)

- Confirm latest 0G Storage Indexer SDK APIs for Galileo (we use `getFileLocations` + `download`).
- Confirm Ollama `seed` behavior stability across versions and models.
- Decide default fallback policy (503 vs. dev random quote) for production.


