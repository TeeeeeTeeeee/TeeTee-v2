import { ethers } from 'ethers';
import axios from 'axios';
import dotenv from 'dotenv';
import { createRequire } from 'module';

dotenv.config({ path: '../.env' });

/**
 * Expected environment variables:
 * - RPC_URL: JSON-RPC endpoint for target chain
 * - PRIVATE_KEY: Private key of the CreditUse contract owner (required because reportDowntime is onlyOwner)
 * - CONTRACT_ADDRESS: Deployed CreditUse contract address
 */

class DowntimeOracle {
  constructor() {
    const rpcUrl = process.env.RPC_URL;
    const privateKey = process.env.PRIVATE_KEY;
    const contractAddress = process.env.CONTRACT_ADDRESS;

    if (!rpcUrl || !privateKey || !contractAddress) {
      throw new Error('Missing required env vars: RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS');
    }

    this.provider = new ethers.JsonRpcProvider(rpcUrl);
    this.wallet = new ethers.Wallet(privateKey, this.provider);
    this.contractAddress = contractAddress;

    const require = createRequire(import.meta.url);
      const creditArtifact = require('../../artifacts/contracts/creditUse.sol/CreditUse.json');
      const abi = creditArtifact.abi;
      console.log('Loaded ABI from artifacts.');
    
    this.contract = new ethers.Contract(contractAddress, abi, this.wallet);

    // In-memory state
    this.llmConfigs = new Map(); // id -> { id, shardUrl1, shardUrl2, modelName }
    this.downtimeAccumulator = new Map(); // id -> { host1: minutes, host2: minutes }
    this.lastCheckTime = new Map(); // id -> unixSeconds

    // Interval management
    this.intervalMs = parseInt(process.env.CHECK_INTERVAL_MS || '60000', 10);
    if (!Number.isFinite(this.intervalMs) || this.intervalMs <= 0) this.intervalMs = 60000;

    this.isTicking = false; // prevent overlapping runs

    console.log(`Oracle initialized:`);
    console.log(` - Wallet: ${this.wallet.address}`);
    console.log(` - Contract: ${contractAddress}`);
    console.log(` - Interval: ${this.intervalMs} ms`);
  }

  async addLLMConfig(config) {
    this.llmConfigs.set(config.id, config);
    if (!this.downtimeAccumulator.has(config.id)) {
      this.downtimeAccumulator.set(config.id, { host1: 0, host2: 0 });
    }
    if (!this.lastCheckTime.has(config.id)) {
      const now = Math.floor(Date.now() / 1000);
      const approxLast = now - Math.max(60, Math.floor(this.intervalMs / 1000));
      this.lastCheckTime.set(config.id, approxLast);
    }
    console.log(`Added LLM config for ID ${config.id}: ${config.modelName}`);
  }

  async checkHealthWithRetry(url, maxRetries = 3) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const start = Date.now();
        const res = await axios.get(`${url}/health`, {
          timeout: 10000,
          headers: { 'User-Agent': 'Downtime-Oracle/1.0' }
        });
        const responseTime = Date.now() - start;
        return { isHealthy: res.status === 200, responseTime };
      } catch (err) {
        if (attempt === maxRetries) {
          return { isHealthy: false, responseTime: 0, error: err?.message || String(err) };
        }
        // simple linear backoff: 1s, 2s, ...
        await new Promise((r) => setTimeout(r, 1000 * attempt));
      }
    }
    return { isHealthy: false, responseTime: 0, error: 'Max retries exceeded' };
  }

  async monitorLLM(llmId) {
    const config = this.llmConfigs.get(llmId);
    if (!config) {
      console.warn(`No config for LLM ${llmId}`);
      return;
    }

    const now = Math.floor(Date.now() / 1000);
    const last = this.lastCheckTime.get(llmId) ?? now - Math.floor(this.intervalMs / 1000);
    const secondsElapsed = Math.max(60, now - last); // clamp to at least 60 seconds per spec
    const minutesElapsed = Math.floor(secondsElapsed / 60) || 1; // at least 1 minute

    console.log(`Checking LLM ${llmId} (${config.modelName})...`);

    const [host1Result, host2Result] = await Promise.all([
      this.checkHealthWithRetry(config.shardUrl1),
      this.checkHealthWithRetry(config.shardUrl2)
    ]);

    const acc = this.downtimeAccumulator.get(llmId) || { host1: 0, host2: 0 };

    if (!host1Result.isHealthy) {
      acc.host1 += minutesElapsed; // accumulate minutes of downtime
      console.log(` - Host1 DOWN (${config.shardUrl1}). +${minutesElapsed}min => ${acc.host1}min`);
    } else {
      console.log(` - Host1 UP (${config.shardUrl1}). rt=${host1Result.responseTime}ms`);
    }

    if (!host2Result.isHealthy) {
      acc.host2 += minutesElapsed; // accumulate minutes of downtime
      console.log(` - Host2 DOWN (${config.shardUrl2}). +${minutesElapsed}min => ${acc.host2}min`);
    } else {
      console.log(` - Host2 UP (${config.shardUrl2}). rt=${host2Result.responseTime}ms`);
    }

    this.downtimeAccumulator.set(llmId, acc);

    // Report if we hit threshold (5 minutes or more)
    if (acc.host1 >= 5 || acc.host2 >= 5) {
      try {
        console.log(`Reporting downtime for LLM ${llmId}: host1=${acc.host1}min, host2=${acc.host2}min`);
        const tx = await this.contract.reportDowntime(
          llmId,
          BigInt(acc.host1),
          BigInt(acc.host2)
        );
        const receipt = await tx.wait();
        console.log(` - Reported. txHash=${receipt.hash}`);
        // reset accumulators after success
        this.downtimeAccumulator.set(llmId, { host1: 0, host2: 0 });
      } catch (err) {
        console.error(` - Failed to report downtime for LLM ${llmId}:`, err?.message || String(err));
      }
    }

    this.lastCheckTime.set(llmId, now);
  }

  async loadLLMsFromContract() {
    console.log('Loading LLMs from contract...');
    let count = 0;
    // Heuristic: try indices 0..49 until a call reverts or an empty entry is found
    for (let i = 0; i < 50; i++) {
      try {
        const llm = await this.contract.getHostedLLM(i);
        if (!llm || !llm.host1 || llm.host1 === ethers.ZeroAddress) {
          break;
        }
        await this.addLLMConfig({
          id: i,
          shardUrl1: llm.shardUrl1,
          shardUrl2: llm.shardUrl2,
          modelName: llm.modelName
        });
        count += 1;
      } catch (_) {
        break; // assume no more entries
      }
    }
    console.log(`Loaded ${count} LLM(s).`);
  }

  async preflight() {
    try {
      const code = await this.provider.getCode(this.contractAddress);
      if (!code || code === '0x') {
        console.error(`No contract code found at ${this.contractAddress}. Is CONTRACT_ADDRESS correct and on the same network as RPC_URL?`);
        return;
      }
    } catch (e) {
      console.warn('Failed to fetch contract code:', e?.message || e);
    }

    if (typeof this.contract.owner === 'function') {
      try {
        const onChainOwner = await this.contract.owner();
        if (onChainOwner.toLowerCase() !== this.wallet.address.toLowerCase()) {
          console.error(`Owner mismatch: contract owner ${onChainOwner} != oracle wallet ${this.wallet.address}. reportDowntime will likely revert.`);
        } else {
          console.log(`Owner preflight OK. Contract owner = oracle wallet (${onChainOwner}).`);
        }
      } catch (e) {
        console.warn('owner() call failed (ABI mismatch or non-ownable contract):', e?.message || e);
      }
    } else {
      console.warn('owner() not in ABI; skipping owner preflight. Ensure the oracle key is the contract owner.');
    }
  }

  async start() {
    console.log('🚀 Starting downtime oracle...');

    await this.preflight();
    await this.loadLLMsFromContract();

    // Initial pass
    for (const llmId of this.llmConfigs.keys()) {
      await this.monitorLLM(llmId);
    }

    // Schedule periodic checks
    setInterval(async () => {
      if (this.isTicking) {
        console.log('Previous tick still running, skipping this interval.');
        return;
      }
      this.isTicking = true;
      try {
        console.log(`\n🔍 Tick at ${new Date().toISOString()}`);
        for (const llmId of this.llmConfigs.keys()) {
          await this.monitorLLM(llmId);
        }
      } finally {
        this.isTicking = false;
      }
    }, this.intervalMs);
  }
}

// Bootstrap
const oracle = new DowntimeOracle();
oracle.start().catch((e) => {
  console.error('Oracle failed to start:', e);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down oracle...');
  process.exit(0);
});
