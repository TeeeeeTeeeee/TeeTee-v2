import type { NextApiRequest, NextApiResponse } from 'next';
import { Readable } from 'node:stream';
import { uploadStream } from '../../utils/storage';
import { addSession, updateSession } from '../../utils/json-storage';

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

    // Save or update metadata in local JSON storage
    // Important: Always replace old entry with new one containing latest root hash
    let saved = false;
    let returnedSessionId = sessionId;
    
    try {
      if (sessionId) {
        // Update existing session with NEW root hash and tx hash
        // This removes the old hash and keeps only the latest version
        const updated = updateSession(sessionId, {
          rootHash: rootHash || null,
          txHash,
          messageCount: messages.length,
          filename: fname, // Keep filename updated too
          preview, // Add preview field
        });
        saved = !!updated;
        
        console.log(`Updated session ${sessionId} with new root hash: ${rootHash}`);
      } else {
        // Create new session
        const newSession = addSession({
          walletAddress: walletAddress.toLowerCase(), // Normalize to lowercase
          filename: fname,
          rootHash: rootHash || null,
          txHash,
          messageCount: messages.length,
          preview, // Add preview field
        });
        returnedSessionId = newSession._id;
        saved = true;
        
        console.log(`Created new session ${returnedSessionId} with root hash: ${rootHash}`);
      }
    } catch (storageError: any) {
      console.error('JSON storage save error:', storageError);
      // Continue even if local storage fails - the file is still uploaded to 0G
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
