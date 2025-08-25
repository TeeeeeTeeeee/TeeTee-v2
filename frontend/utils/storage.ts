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