// Server-side utilities for 0G storage operations
// NOTE: This module uses Node.js APIs (fs, path) and should be used on the server only.

import { ZgFile, Indexer, Batcher, KvClient } from '@0glabs/0g-ts-sdk';
import { ethers } from 'ethers';
import fs from 'node:fs';
import path from 'node:path';

// Network Constants (allow override via env if needed)
const RPC_URL: string = process.env.ZG_RPC_URL || 'https://evmrpc-testnet.0g.ai/';
const INDEXER_RPC: string = process.env.ZG_INDEXER_RPC || 'https://indexer-storage-testnet-turbo.0g.ai';

// Initialize indexer once (stateless client)
const indexer: Indexer = new Indexer(INDEXER_RPC);

function getPrivateKey(): string {
  const pk = process.env.PRIVATE_KEY;
  if (!pk) {
    throw new Error(
      'Missing PRIVATE_KEY environment variable. Set PRIVATE_KEY in your environment (e.g., .env.local).',
    );
  }
  return pk.startsWith('0x') ? pk : `0x${pk}`;
}

function getSigner(): { provider: ethers.JsonRpcProvider; signer: ethers.Wallet } {
  const provider = new ethers.JsonRpcProvider(RPC_URL);
  const signer = new ethers.Wallet(getPrivateKey(), provider);
  return { provider, signer };
}

// Ensure the parent directory of outputPath exists
async function ensureDirForFile(filePath: string): Promise<void> {
  const dir = path.dirname(filePath);
  await fs.promises.mkdir(dir, { recursive: true });
}

function extractRootHashFromTree(tree: any): string | undefined {
  if (!tree) return undefined;
  const rh = (tree as any).rootHash;
  if (typeof rh === 'function') {
    try { return rh(); } catch { return undefined; }
  }
  if (typeof rh === 'string') return rh;
  return undefined;
}

function normalizeTxHash(tx: any): string {
  if (!tx) return '';
  if (typeof tx === 'string') return tx;
  // Try common shapes
  if (typeof tx.hash === 'string') return tx.hash; // Ethers TxResponse
  if (typeof tx.txHash === 'string') return tx.txHash; // SDK may return { txHash }
  try {
    return JSON.stringify(tx);
  } catch {
    return String(tx);
  }
}