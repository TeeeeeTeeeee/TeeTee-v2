import type { NextApiRequest, NextApiResponse, PageConfig } from 'next';
import os from 'node:os';
import fs from 'node:fs';
import path from 'node:path';
import { downloadFile } from '../../utils/storage';

export const config: PageConfig = {
  api: {
    // Allow streaming large files without a response size cap
    responseLimit: false,
  },
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
): Promise<void> {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    res.status(405).json({ error: 'Method Not Allowed' });
    return;
  }

  try {
    const { rootHash, filename } = req.query as {
      rootHash?: string | string[];
      filename?: string | string[];
    };

    const rootHashStr = Array.isArray(rootHash) ? rootHash[0] : rootHash;
    const rawFilename = Array.isArray(filename) ? filename[0] : filename;

    if (!rootHashStr || typeof rootHashStr !== 'string') {
      res.status(400).json({ error: 'Invalid or missing rootHash' });
      return;
    }

    // Sanitize filename; default to a short hash prefix
    const safeName = typeof rawFilename === 'string' && rawFilename.trim()
      ? rawFilename.replace(/[^a-zA-Z0-9._-]/g, '_')
      : `${rootHashStr.slice(0, 10)}.bin`;

    // Download to a temporary file first (with Merkle proof verification enabled in downloadFile)
    const tmpPath = path.join(
      os.tmpdir(),
      `zg_${Date.now()}_${Math.random().toString(36).slice(2)}.bin`,
    );

    await downloadFile(rootHashStr, tmpPath);

    const stat = await fs.promises.stat(tmpPath);

    // Prepare headers for file download
    res.statusCode = 200;
    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Content-Disposition', `attachment; filename="${safeName}"`);
    res.setHeader('Content-Length', String(stat.size));

    // TODO: stream file to client & cleanup temp file
    res.end(`File ready at ${tmpPath}`);
  } catch (err: any) {
    // eslint-disable-next-line no-console
    console.error('download-stream error:', err);
    if (!res.headersSent) {
      res.status(500).json({ error: err?.message || 'Internal Server Error' });
      return;
    }
    res.end();
  }
}
