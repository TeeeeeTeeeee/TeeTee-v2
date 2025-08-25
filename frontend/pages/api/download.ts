import type { NextApiRequest, NextApiResponse, PageConfig } from 'next';

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

    // For now, just respond with the computed safe filename
    res.status(200).json({ safeName });
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
