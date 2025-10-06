import type { NextApiRequest, NextApiResponse } from 'next';
import { Readable } from 'node:stream';
import { uploadStream } from '../../utils/storage';
import { connectToDatabase } from '../../utils/mongodb';

// Types expected from the client
type ChatMessage = {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: number;
};

type SuccessResponse = {
  filename: string;
  rootHash: string | undefined;
  txHash: string;
  saved: boolean;
};

type ErrorResponse = { error: string };

type ResponseBody = SuccessResponse | ErrorResponse;

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ResponseBody>,
) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method Not Allowed' });
  }

  try {
    const { messages, filename, walletAddress } = req.body as {
      messages?: ChatMessage[];
      filename?: string;
      walletAddress?: string;
    };

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Invalid or empty messages array' });
    }

    if (!walletAddress || typeof walletAddress !== 'string') {
      return res.status(400).json({ error: 'Wallet address is required' });
    }

    // Build a simple plaintext transcript
    const lines: string[] = [];
    for (const msg of messages) {
      const role = (msg?.role || 'unknown').toUpperCase();
      const time = msg?.timestamp ? new Date(msg.timestamp).toISOString() : new Date().toISOString();
      const content = typeof msg?.content === 'string' ? msg.content : JSON.stringify(msg?.content ?? '');
      lines.push(`[${time}] ${role}: ${content}`);
    }
    const transcript = lines.join('\n') + '\n';

    const fname = typeof filename === 'string' && filename.trim() ? filename.trim() : `chat_${Date.now()}.txt`;

    // Create a readable stream from the transcript string
    const stream = Readable.from(transcript, { encoding: 'utf-8' });

    // Upload to 0G storage
    const { rootHash, txHash } = await uploadStream(stream, fname);

    // Save metadata to MongoDB
    let saved = false;
    try {
      const { db } = await connectToDatabase();
      const collection = db.collection('chatSessions');
      
      await collection.insertOne({
        walletAddress: walletAddress.toLowerCase(), // Normalize to lowercase
        filename: fname,
        rootHash: rootHash || null,
        txHash,
        messageCount: messages.length,
        createdAt: new Date(),
        updatedAt: new Date(),
      });
      
      saved = true;
    } catch (dbError: any) {
      console.error('MongoDB save error:', dbError);
      // Continue even if MongoDB fails - the file is still uploaded to 0G
    }

    return res.status(200).json({ filename: fname, rootHash, txHash, saved });
  } catch (err: any) {
    // eslint-disable-next-line no-console
    console.error('chat-session save error:', err);
    return res.status(500).json({ error: err?.message || 'Internal Server Error' });
  }
}
