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

export async function uploadFile(
  filePath: string,
): Promise<{ rootHash: string | undefined; txHash: string }> {
  if (!filePath || typeof filePath !== 'string') {
    throw new Error('uploadFile: invalid filePath');
  }

  const { signer } = getSigner();

  // Create file object from file path
  const file = await ZgFile.fromFilePath(filePath);

  try {
    // Generate Merkle tree for verification
    const [tree, treeErr]: [unknown, unknown] = await file.merkleTree();
    if (treeErr !== null && treeErr !== undefined) {
      throw new Error(`Error generating Merkle tree: ${String(treeErr)}`);
    }

    const rootHash = extractRootHashFromTree(tree);

    // Upload to network
    // Cast signer to any to avoid CJS/ESM Wallet type incompatibility from ethers in different builds
    const [tx, uploadErr]: [any, any] = await indexer.upload(file, RPC_URL, signer as unknown as any);
    if (uploadErr !== null && uploadErr !== undefined) {
      throw new Error(`Upload error: ${String(uploadErr)}`);
    }

    const txHash = normalizeTxHash(tx);

    return { rootHash, txHash };
  } finally {
    // Always close the file when done
    await file.close();
  }
}

export async function downloadFile(rootHash: string, outputPath: string): Promise<void> {
  if (!rootHash || typeof rootHash !== 'string') {
    throw new Error('downloadFile: invalid rootHash');
  }
  if (!outputPath || typeof outputPath !== 'string') {
    throw new Error('downloadFile: invalid outputPath');
  }

  await ensureDirForFile(outputPath);

  // withProof = true enables Merkle proof verification
  const err: unknown = await indexer.download(rootHash, outputPath, true);
  if (err !== null && err !== undefined) {
    throw new Error(`Download error: ${String(err)}`);
  }
}

export const _internal = { getSigner };