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
  sessionId?: string;
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
    const { messages, filename, walletAddress, sessionId } = req.body as {
      messages?: ChatMessage[];
      filename?: string;
      walletAddress?: string;
      sessionId?: string;
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

    // Extract preview from first user message
    const firstUserMessage = messages.find(msg => msg.role === 'user');
    const preview = firstUserMessage?.content 
      ? (firstUserMessage.content.length > 60 
          ? firstUserMessage.content.substring(0, 60).trim() 
          : firstUserMessage.content.trim())
      : undefined;

    // Create a readable stream from the transcript string
    const stream = Readable.from(transcript, { encoding: 'utf-8' });

    // Upload to 0G storage
    const { rootHash, txHash } = await uploadStream(stream, fname);

    // Save or update metadata in MongoDB only
    let saved = false;
    let returnedSessionId = sessionId;
    
    const { db } = await connectToDatabase();
    const chatSessionsCollection = db.collection('chatSessions');
    
    if (sessionId) {
      // Update existing session in MongoDB
      const result = await chatSessionsCollection.updateOne(
        { sessionId: sessionId },
        {
          $set: {
            rootHash: rootHash || null,
            txHash,
            messageCount: messages.length,
            filename: fname,
            preview,
            updatedAt: new Date().toISOString(),
          }
        }
      );
      saved = result.modifiedCount > 0;
      returnedSessionId = sessionId;
      console.log(`Updated MongoDB session ${sessionId} with new root hash: ${rootHash}`);
    } else {
      // Generate new session ID
      const { randomUUID } = await import('crypto');
      returnedSessionId = randomUUID();
      
      // Insert new session into MongoDB
      await chatSessionsCollection.insertOne({
        sessionId: returnedSessionId,
        walletAddress: walletAddress.toLowerCase(),
        filename: fname,
        rootHash: rootHash || null,
        txHash,
        messageCount: messages.length,
        preview,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
      saved = true;
      console.log(`Created MongoDB session ${returnedSessionId} with root hash: ${rootHash}`);
    }

    return res.status(200).json({ 
      filename: fname, 
      rootHash, 
      txHash, 
      saved,
      sessionId: returnedSessionId 
    });
  } catch (err: any) {
    // eslint-disable-next-line no-console
    console.error('chat-session save error:', err);
    return res.status(500).json({ error: err?.message || 'Internal Server Error' });
  }
}
