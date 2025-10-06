import type { NextApiRequest, NextApiResponse } from 'next';
import { downloadFile } from '../../utils/storage';
import os from 'node:os';
import fs from 'node:fs';
import path from 'node:path';

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
};

type SuccessResponse = {
  messages: ChatMessage[];
};

type ErrorResponse = { error: string };

type ResponseBody = SuccessResponse | ErrorResponse;

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ResponseBody>,
) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const { rootHash } = req.query;

    if (!rootHash || typeof rootHash !== 'string') {
      return res.status(400).json({ error: 'Root hash is required' });
    }

    // Download the file from 0G storage to temp location
    const tmpPath = path.join(
      os.tmpdir(),
      `chat_${Date.now()}_${Math.random().toString(36).slice(2)}.txt`,
    );

    await downloadFile(rootHash, tmpPath);

    // Read the file content
    const content = await fs.promises.readFile(tmpPath, 'utf-8');

    // Clean up temp file
    await fs.promises.unlink(tmpPath).catch(() => {});

    // Parse the transcript format: [timestamp] ROLE: content
    const lines = content.split('\n').filter(line => line.trim());
    const messages: ChatMessage[] = [];

    for (const line of lines) {
      // Match pattern: [2024-01-01T12:00:00.000Z] USER: message content
      const match = line.match(/^\[([^\]]+)\]\s+(USER|ASSISTANT):\s+(.+)$/);
      if (match) {
        const [, timestamp, role, content] = match;
        messages.push({
          role: role.toLowerCase() as 'user' | 'assistant',
          content: content.trim(),
          timestamp: new Date(timestamp).getTime(),
        });
      }
    }

    return res.status(200).json({ messages });
  } catch (err: any) {
    console.error('get-chat-messages error:', err);
    return res.status(500).json({ error: err?.message || 'Internal Server Error' });
  }
}

